"""Experiment-only adapters. No application imports this module or selects these backends."""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]


def stabilize_graph_order(storage):
    """Remove backend traversal order from LightRAG's stable rank/weight ties.

    LightRAG 1.5.7 sorts local relations by rank/weight without an ID tie-break.
    Canonicalize undirected adjacency before that sort; preserve scores, all
    edges and endpoint orientation. Apply equally to both comparison backends.
    """
    original = storage.get_nodes_edges_batch

    async def ordered(node_ids):
        result = await original(node_ids)
        return {node: sorted(edges, key=lambda pair: tuple(sorted(pair)))
                for node, edges in result.items()}

    storage.get_nodes_edges_batch = ordered


def endpoint_config(prefix: str = "LIGHTRAG_EVAL") -> dict:
    """Read explicit experiment settings or an existing enabled project endpoint.

    Never rotate, rewrite config, print credentials, or include them in reports.
    Judge settings are optional; otherwise use the same model and label that limitation.
    """
    from dotenv import dotenv_values
    local_path = Path(__file__).resolve().parent / ".env.local"
    local = dotenv_values(local_path, encoding="utf-8-sig", interpolate=False) if local_path.exists() else {}
    names = {field: f"{prefix}_{suffix}" for field, suffix in
             (("key", "API_KEY"), ("base", "BASE_URL"), ("model", "MODEL"))}
    explicit = {field: os.environ.get(name, local.get(name)) for field, name in names.items()}
    if any(value is not None for value in explicit.values()):
        if not all(isinstance(value, str) and value.strip() for value in explicit.values()):
            # An incomplete experiment config must not silently use production credentials.
            raise ValueError(f"{prefix} 配置不完整：请在实验 .env.local 中填写 API_KEY、BASE_URL、MODEL。")
        base = explicit["base"].strip().rstrip("/")
        url = base if base.endswith("/chat/completions") else base + ("" if base.endswith("/v1") else "/v1") + "/chat/completions"
        return {"key": explicit["key"].strip(), "url": url, "model": explicit["model"].strip()}
    if prefix != "LIGHTRAG_EVAL":
        return endpoint_config()
    config_path = Path(os.getenv("FINSIGHT_CONFIG_DIR", str(ROOT))) / "user_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    candidates = [e for e in config.get("llm_endpoints", [])
                  if e.get("enabled", True) and e.get("api_key")]
    source = candidates[0] if candidates else {
        "api_key": config.get("llm_api_key"), "api_base": config.get("llm_api_base"),
        "model": config.get("llm_model"),
    }
    key = os.getenv(f"{prefix}_API_KEY") or source.get("api_key") or os.getenv("OPENAI_COMPATIBLE_API_KEY")
    base = os.getenv(f"{prefix}_BASE_URL") or source.get("api_base") or os.getenv("OPENAI_COMPATIBLE_API_BASE")
    model = os.getenv(f"{prefix}_MODEL") or source.get("model") or os.getenv("OPENAI_COMPATIBLE_MODEL")
    if not key or not base or not model:
        raise ValueError("LLM 未配置：请设置 LIGHTRAG_EVAL_API_KEY/BASE_URL/MODEL，或使用项目现有设置。")
    base = base.rstrip("/")
    url = base if base.endswith("/chat/completions") else base + ("" if base.endswith("/v1") else "/v1") + "/chat/completions"
    return {"key": key, "url": url, "model": model}


class CallBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def acquire(self):
        if self.used >= self.limit:
            raise RuntimeError("实验 LLM 调用总预算已用尽")
        self.used += 1


class ChatClient:
    def __init__(self, config: dict, budget: CallBudget, timeout: float):
        import httpx
        self.config = config
        self.budget = budget
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self.phase = "preflight"
        self.calls: list[dict] = []
        # DeepSeek defaults to thinking mode; use one explicit non-thinking
        # setting for extraction, all answer arms and the judge in this experiment.
        self.request_options = ({"thinking": {"type": "disabled"}}
                                if urlparse(config["url"]).hostname == "api.deepseek.com" else {})

    async def complete(self, prompt: str, system_prompt: str | None = None,
                       history_messages: list | None = None, **kwargs) -> str:
        self.budget.acquire()
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        messages += list(history_messages or [])
        messages.append({"role": "user", "content": prompt})
        body = {"model": self.config["model"], "messages": messages,
                "temperature": 0, "max_tokens": kwargs.get("max_tokens", 2400), "stream": False}
        body.update(self.request_options)
        if kwargs.get("response_format") == {"type": "json_object"} or kwargs.get("keyword_extraction"):
            body["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        record = {"phase": self.phase, "model": self.config["model"], "ok": False,
                  "prompt_tokens": None, "completion_tokens": None}
        try:
            response = await asyncio.wait_for(self.client.post(
                self.config["url"], headers={"Authorization": f"Bearer {self.config['key']}"},
                json=body), timeout=self.timeout)
            if response.status_code != 200:
                # Provider bodies may echo keys or private endpoint details.
                raise RuntimeError(f"LLM HTTP {response.status_code}; 响应正文未写入日志")
            payload = response.json()
            usage = payload.get("usage") or {}
            record.update(prompt_tokens=usage.get("prompt_tokens"),
                          completion_tokens=usage.get("completion_tokens"))
            choice = payload["choices"][0]
            if choice.get("finish_reason") == "length":
                raise RuntimeError("LLM 输出超过 token 上限，不能作为完整结果评分")
            answer = choice["message"].get("content")
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("LLM 返回空文本")
            record.update(ok=True)
            return answer
        finally:
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
            self.calls.append(record)

    async def close(self):
        await self.client.aclose()


class SharedEmbedding:
    """Same real dense encoder for both engines; legacy lexical scoring is preserved.

    No hash fallback is allowed here. Model initialization is timed separately.
    """
    model_name = "BAAI/bge-small-zh-v1.5"
    dim = 512
    fallback_reason = None

    def __init__(self, cache_dir: Path, allow_download: bool = False):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=self.model_name, cache_dir=str(cache_dir),
                                   local_files_only=not allow_download, threads=2)

    def encode(self, texts):
        from backend.rag.embedder import EmbeddingResult, SparseVector, _tokenize
        values = list(texts)
        dense = [v.tolist() for v in self.model.embed(values)]
        if len(dense) != len(values) or any(len(v) != self.dim for v in dense):
            raise ValueError("真实向量模型输出数量或维度不正确")
        sparse = []
        for text in values:
            counts = Counter(_tokenize(text))
            scale = max(counts.values(), default=1)
            sparse.append(SparseVector({token: count / scale for token, count in counts.items()}))
        return EmbeddingResult(dense=dense, sparse=sparse, model_name=self.model_name, dim=self.dim)

    def encode_single(self, text):
        result = self.encode([text])
        return result.dense[0], result.sparse[0]

    async def dense_async(self, texts, **_kwargs):
        import numpy as np
        # Both engines use the same embedding instruction (none) and no query cache.
        result = await asyncio.to_thread(self.encode, texts)
        return np.asarray(result.dense, dtype=np.float32)


class LegacyAdapter:
    def __init__(self, embedder, collection: str):
        from backend.rag.hybrid_service import HybridRAGService
        # Explicit memory backend: no production DSN, singleton, hooks or cleanup jobs.
        self.service = HybridRAGService(backend="memory", vector_dim=embedder.dim,
                                       rrf_k=60, embedder=embedder)
        self.collection = collection

    async def insert(self, documents: list[dict]):
        from backend.rag.hybrid_service import RAGDocument
        return await asyncio.to_thread(self.service.ingest_documents, [
            RAGDocument(collection=self.collection, scope="persistent", source_id=d["id"],
                        content=d["text"], source="synthetic_eval", layer="kb") for d in documents])

    async def retrieve(self, query: str, k: int) -> dict:
        hits = await asyncio.to_thread(self.service.hybrid_search, query,
                                       collection=self.collection, top_k=k)
        return {"sources": [h["source_id"] for h in hits], "graph": {}, "unknown_sources": 0}


class LightRAGAdapter:
    def __init__(self, folder: Path, embedder: SharedEmbedding, llm: ChatClient,
                 documents: list[dict], context_tokens: int, *,
                 graph_storage: str = "NetworkXStorage", workspace: str = "finsight_experiment",
                 stable_graph_order: bool = False):
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc
        if graph_storage not in {"NetworkXStorage", "Neo4JStorage"}:
            raise ValueError("Unsupported experiment graph storage")
        if graph_storage == "Neo4JStorage":
            from experiments.lightrag.neo4j_config import configure
            configure(workspace)
        self.documents = {d["id"]: d["text"] for d in documents}
        self.context_tokens = context_tokens
        # LightRAG serializes its dataclass with asdict/deepcopy. Plain closures
        # avoid deep-copying HTTP clients and ONNX sessions through bound methods.
        async def complete(prompt, **kwargs):
            return await llm.complete(prompt, **kwargs)

        async def embed(texts, **kwargs):
            return await embedder.dense_async(texts, **kwargs)

        self.rag = LightRAG(
            working_dir=str(folder), workspace=workspace,
            kv_storage="JsonKVStorage", vector_storage="NanoVectorDBStorage",
            graph_storage=graph_storage, doc_status_storage="JsonDocStatusStorage",
            llm_model_func=complete, llm_model_name=llm.config["model"],
            llm_model_max_async=2, embedding_func_max_async=1,
            embedding_func=EmbeddingFunc(embedding_dim=embedder.dim, max_token_size=512,
                                         func=embed, model_name=embedder.model_name),
            tiktoken_model_name="gpt-4", default_llm_timeout=int(llm.timeout),
            chunk_token_size=512, chunk_overlap_token_size=0,
            entity_extract_max_gleaning=0, max_parallel_insert=2,
            enable_llm_cache=True,
            addon_params={"language": "Chinese"},
        )
        if stable_graph_order:
            stabilize_graph_order(self.rag.chunk_entity_relation_graph)

    async def initialize(self):
        await self.rag.initialize_storages()

    async def insert(self, documents: list[dict]):
        await self.rag.ainsert([d["text"] for d in documents],
                               ids=[d["id"] for d in documents],
                               file_paths=[d["id"] for d in documents])
        counts = await self.rag.get_processing_status()
        if counts.get("processed", 0) != len(documents):
            raise RuntimeError(f"LightRAG 入库不完整：{counts}")
        return counts

    async def retrieve(self, query: str, k: int, mode: str = "mix") -> dict:
        from lightrag import QueryParam
        result = await self.rag.aquery_data(query, QueryParam(
            mode=mode, top_k=20, chunk_top_k=k, enable_rerank=False,
            max_entity_tokens=1000, max_relation_tokens=1000,
            max_total_tokens=self.context_tokens + 3000,
        ))
        if (result.get("metadata") or {}).get("failure_reason") == "no_results":
            return {"sources": [], "graph": {}, "unknown_sources": 0, "raw": result}
        if result.get("status") not in ("success",):
            raise RuntimeError("LightRAG 检索失败；不使用空结果冒充成功")
        data = result.get("data") or {}
        sources, unknown = [], 0
        for chunk in data.get("chunks", []):
            source = chunk.get("file_path")
            if source not in self.documents:
                unknown += 1
                continue
            if source not in sources:
                sources.append(source)
        # Graph descriptions are model-derived context, never treated as gold evidence.
        graph = {"entities": data.get("entities", []), "relationships": data.get("relationships", [])}
        return {"sources": sources[:k], "graph": graph, "unknown_sources": unknown,
                "raw": result}

    async def close(self):
        await self.rag.finalize_storages()
