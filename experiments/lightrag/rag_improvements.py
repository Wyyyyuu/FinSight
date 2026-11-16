"""Isolated before/after retrieval benchmark; no API calls or production DB access.

Run with experiments/lightrag/.venv/Scripts/python.exe experiments/lightrag/rag_improvements.py
The baseline snapshot is saved before edits, and embedded in the output for replay.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.lightrag.adapters import SharedEmbedding
from experiments.lightrag.metrics import retrieval_metrics, summarize
from backend.rag import hybrid_service as current
from backend.rag.embedder import EmbeddingService
from backend.rag.reranker import RerankerService


def main():
    baseline_path = HERE / "runs/rag-improvements-baseline/hybrid_service.py"
    spec = importlib.util.spec_from_file_location("rag_baseline_snapshot", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    out = HERE / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-rag-improvements")
    out.mkdir(parents=True, exist_ok=False)
    fixture = json.loads((HERE / "fixtures.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    encoder = SharedEmbedding(ROOT / ".annual-models")
    encoder.encode(["模型预热"])
    embedding_load_ms = (time.perf_counter() - started) * 1000
    os.environ["RAG_RERANKER_CACHE_DIR"] = str(HERE / "models")
    reranker = RerankerService(force_backend="bge-reranker-onnx")
    started = time.perf_counter()
    warm = reranker.rerank("收入", [{"content": "公司营业收入增长"}], top_n=1)
    assert warm and "rerank_score" in warm[0], "Real reranker unavailable; refusing an unmeasured arm"
    reranker_load_ms = (time.perf_counter() - started) * 1000
    arms = ["baseline", "relevance_only", "cjk", "cjk_rerank"]
    services = {}
    index_ms = {}
    for arm in arms:
        module = baseline if arm == "baseline" else current
        os.environ["RAG_LEXICAL_MODE"] = "cjk" if arm.startswith("cjk") else "legacy"
        service = module.HybridRAGService(backend="memory", vector_dim=512, rrf_k=60, embedder=encoder)
        started = time.perf_counter()
        service.ingest_documents([module.RAGDocument(collection="kb:synthetic", scope="persistent",
                                 source_id=d["id"], content=d["text"]) for d in fixture["documents"]])
        index_ms[arm] = (time.perf_counter() - started) * 1000
        services[arm] = service
    rows = []
    for case_index, case in enumerate(fixture["cases"]):
        # Rotate arm order to reduce systematic warmup/order bias.
        order = arms[case_index % len(arms):] + arms[:case_index % len(arms)]
        for arm in order:
            elapsed = []
            for repeat in range(3):
                started = time.perf_counter()
                hits = services[arm].hybrid_search(case["question"], collection="kb:synthetic",
                                                  top_k=12 if arm.endswith("rerank") else 4)
                candidates = [h["source_id"] for h in hits]
                if arm.endswith("rerank"):
                    hits = reranker.rerank(case["question"], hits, top_n=4)
                    assert len(hits) == 4 and all("rerank_score" in h for h in hits)
                elapsed.append((time.perf_counter() - started) * 1000)
            ids = [h["source_id"] for h in hits]
            rows.append(dict(arm=arm, case_id=case["id"], category=case["category"],
                             retrieved=ids, candidates=candidates, required=case["required_sources"],
                             retrieval_ms=elapsed, **retrieval_metrics(ids, case["required_sources"], 4)))
        print("completed", case["id"], flush=True)
    # Separate adversarial diagnostics, not pooled into the financial benchmark.
    probes = []
    for label, query, gold, noise in [
        ("scope", "apple revenue growth", "apple revenue growth", "weather ocean rainfall"),
        ("chinese", "营业收入", "公司营业收入同比增长", "这是今天的天气预报"),
    ]:
        for arm in arms[:3]:
            module = baseline if arm == "baseline" else current
            os.environ["RAG_LEXICAL_MODE"] = "cjk" if arm == "cjk" else "legacy"
            service = module.HybridRAGService(backend="memory", vector_dim=0, rrf_k=60,
                                              embedder=EmbeddingService(force_backend="hash"))
            service.ingest_documents([
                module.RAGDocument(collection="ws:probe", scope="persistent", source_id="noise", content=noise),
                module.RAGDocument(collection="ws:probe", scope="ephemeral", source_id="gold", content=gold)])
            result = service.hybrid_search(query, collection="ws:probe", top_k=1)
            probes.append(dict(probe=label, arm=arm, top1=result[0]["source_id"]))
    report = dict(status="complete", fixture_version=fixture["version"], k=4, rerank_candidates=12,
                  embedding="BAAI/bge-small-zh-v1.5", reranker="BAAI/bge-reranker-base ONNX",
                  embedding_load_ms=embedding_load_ms, reranker_load_ms=reranker_load_ms,
                  index_ms=index_ms, summary=summarize(rows), rows=rows, diagnostic_probes=probes,
                  baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                  baseline_source=baseline_path.read_text(encoding="utf-8"),
                  limitations=["Synthetic 16-document corpus, 10 answerable and 2 unscored abstention questions",
                               "Memory backend; no live PostgreSQL plan or latency measurement",
                               "No generated answers or LLM judge; evidence recall is not answer quality",
                               "Rerank latency includes retrieval; 3 repeats, models warmed, CPU threads=2",
                               "Graph retrieval and production embedding unchanged; zero LLM API calls"])
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {k: v for k, v in report.items() if k not in {"rows", "baseline_source"}}
    (HERE / "rag_improvements_assessment.json").write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out / "report.json"), flush=True)
    for arm, metrics in report["summary"].items():
        print(arm, "recall", metrics["recall_at_k"]["mean"], "ndcg", metrics["ndcg_at_k"]["mean"],
              "warm_p50_ms", metrics["warm_query_ms"]["p50"], flush=True)


if __name__ == "__main__":
    main()
