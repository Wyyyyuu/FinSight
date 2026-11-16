"""Ranking relevance, vector identity, CJK recall and PostgreSQL query contracts."""
import json
from contextlib import contextmanager

import pytest

from backend.rag import embedder as emb
from backend.rag.hybrid_service import HybridRAGService, RAGDocument, _PostgresHybridStore
from backend.rag.lexical import lexical_tokens, lexical_query


def doc(sid, content, scope="ephemeral"):
    return RAGDocument(collection="ws:test", source_id=sid, content=content, scope=scope)


def test_lifetime_never_outvotes_relevance():
    service = HybridRAGService.for_testing()
    service.ingest_documents([doc("old", "weather ocean rainfall", "persistent"),
                              doc("current", "apple revenue growth")])
    hits = service.hybrid_search("apple revenue growth", collection="ws:test")
    assert hits[0]["source_id"] == "current"
    assert hits[0]["rrf_score"] <= 2 / (service.rrf_k + 1)


def test_cjk_partial_phrase_gets_lexical_evidence(monkeypatch):
    monkeypatch.setenv("RAG_LEXICAL_MODE", "cjk")
    service = HybridRAGService.for_testing()
    service.ingest_documents([doc("noise", "这是今天的天气预报"), doc("gold", "公司营业收入同比增长")])
    hits = service.hybrid_search("营业收入", collection="ws:test", top_k=1)
    assert hits[0]["source_id"] == "gold"
    assert hits[0]["sparse_score"] > 0
    assert lexical_tokens("ＡＡＰＬ营业收入") == ["aapl", "营业", "业收", "收入"]
    assert "!" not in lexical_query("营业收入 | !恶意:* '输入'")


class SwitchingEmbedder:
    model_name = "model-a"
    dim = 16
    fallback_reason = None

    def encode(self, texts):
        values = list(texts)
        return emb.EmbeddingResult([[1.] + [0.] * 15 for _ in values],
                                   [emb.SparseVector({"123": 1}) for _ in values],
                                   self.model_name, self.dim)


@pytest.mark.parametrize("unknown", [False, True])
def test_unknown_or_different_models_are_lexical_only(unknown):
    encoder = SwitchingEmbedder()
    service = HybridRAGService(backend="memory", vector_dim=16, rrf_k=60, embedder=encoder)
    service.ingest_documents([doc("relevant", "apple revenue"), doc("noise", "weather")])
    if unknown:
        for row in service._store._docs.values():
            row["metadata"].pop("_embedding_profile")
    else:
        encoder.model_name = "model-b"
    hits = service.hybrid_search("apple", collection="ws:test")
    assert [h["source_id"] for h in hits] == ["relevant"]
    assert hits[0]["dense_rank"] is None
    assert hits[0]["dense_compatible"] is False


@pytest.mark.parametrize("unknown", [False, True])
def test_model_switch_cannot_overwrite_existing_vectors(unknown):
    encoder = SwitchingEmbedder()
    service = HybridRAGService(backend="memory", vector_dim=16, rrf_k=60, embedder=encoder)
    service.ingest_documents([doc("original", "apple")])
    if unknown:
        service._store._docs[("ws:test", "original")]["metadata"].pop("_embedding_profile")
    else:
        encoder.model_name = "model-b"
    with pytest.raises(ValueError, match="profile changed"):
        service.ingest_documents([doc("new", "weather"), doc("original", "replacement")])
    assert service.count_documents() == 1
    assert service._store._docs[("ws:test", "original")]["content"] == "apple"


def test_missing_bge_reports_actual_dimension_and_strict_mode(monkeypatch):
    monkeypatch.setattr(emb.EmbeddingService, "_check_bge", lambda self: False)
    encoder = emb.EmbeddingService(force_backend="bge-m3")
    result = encoder.encode(["中文"])
    assert encoder.dim == result.dim == len(result.dense[0]) == 1024
    assert result.profile == "hash:1024:v1"
    monkeypatch.setenv("RAG_REQUIRE_SEMANTIC_EMBEDDING", "1")
    with pytest.raises(RuntimeError, match="hash fallback refused"):
        encoder.encode(["中文"])


@pytest.mark.parametrize("dense", [[[float("nan")] * 16], [[1.] * 15], []])
def test_invalid_actual_vectors_rejected(dense):
    with pytest.raises(ValueError):
        emb.validate_embedding_result(emb.EmbeddingResult(dense, [emb.SparseVector()], "real", 16), 1)


class CaptureEngine:
    def __init__(self, rowcount=1):
        self.calls = []
        self.rowcount = rowcount

    @contextmanager
    def connect(self):
        yield self

    begin = connect

    def execute(self, sql, params):
        self.calls.append((str(sql), params))
        return self

    def mappings(self):
        return []


def postgres_stub(rowcount=1):
    store = object.__new__(_PostgresHybridStore)
    store._schema_ready = True
    store._embedder = SwitchingEmbedder()
    store._vector_dim = 16
    store._rrf_k = 60
    store._lexical_mode = "cjk"
    store._engine = CaptureEngine(rowcount)
    return store


def test_postgres_write_and_query_bind_actual_profile_and_legacy_lexical():
    store = postgres_stub()
    store.ingest_documents([doc("one", "营业收入增长")])
    insert, params = store._engine.calls[-1]
    metadata = json.loads(params["metadata"])
    assert metadata["_embedding_profile"] == "model-a:16:v1"
    assert metadata["_lexical_version"] == "cjk-bigram-v1"
    assert "营业 业收 收入" in params["search_content"]
    assert "metadata = EXCLUDED.metadata" in insert
    assert "WHERE rag_documents_v2.metadata->>'_embedding_profile' = EXCLUDED.metadata" in insert
    assert "IS NULL" not in insert.split("DO UPDATE SET", 1)[1]
    store.hybrid_search("营业收入", collection="ws:test", top_k=4)
    sql, query = store._engine.calls[-1]
    dense, sparse = sql.split("dense AS (", 1)[1].split("sparse AS (", 1)
    assert "metadata->>'_embedding_profile' = :embedding_profile" in dense
    assert "_embedding_profile" not in sparse  # raw-text lexical survives unknown vectors
    assert query["embedding_profile"] == metadata["_embedding_profile"]
    assert query["lexical_query"] == "营业 | 业收 | 收入"
    assert "scope_boost" not in sql
    assert "ELSE plainto_tsquery" in sql


def test_postgres_conflicting_upsert_is_reported():
    with pytest.raises(ValueError, match="profile changed"):
        postgres_stub(rowcount=0).ingest_documents([doc("one", "apple")])
