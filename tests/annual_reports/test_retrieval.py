from __future__ import annotations

import pytest

from backend.annual_reports.documents import AnnualReportError, AnnualReportStore
from backend.annual_reports.retrieval import (
    SemanticEmbedder,
    reciprocal_rank_fusion,
    tokenize,
)


@pytest.fixture
def library(tmp_path):
    store = AnnualReportStore(tmp_path)
    a2023 = store.ingest_bytes(
        "经营活动现金流量净额为 80 亿元，应收账款为 50 亿元。".encode(),
        "a2023.txt",
        "甲公司",
        2023,
    )
    a2024 = store.ingest_bytes(
        "经营活动现金流量净额为 40 亿元，应收账款为 100 亿元。".encode(),
        "a2024.txt",
        "甲公司",
        2024,
    )
    b2024 = store.ingest_bytes(
        "经营活动现金流量净额为 900 亿元，应收账款为 2 亿元。".encode(),
        "b2024.txt",
        "乙公司",
        2024,
    )
    return store, a2023, a2024, b2024


def test_cross_company_and_year_filters_are_applied_before_ranking(library):
    store, a2023, a2024, b2024 = library
    hits = store.search(
        "经营活动现金流量", [a2023["id"], a2024["id"]], years=[2024], mode="bm25"
    )
    assert len(hits) == 1
    assert hits[0]["document_id"] == a2024["id"]
    assert hits[0]["company"] == "甲公司" and hits[0]["year"] == 2024
    assert "900" not in hits[0]["text"] and "80" not in hits[0]["text"]
    assert (
        store.search("经营活动现金流量", [a2023["id"]], years=[2024], mode="bm25") == []
    )
    assert (
        store.search("经营活动现金流量", [b2024["id"]], years=[2024], mode="bm25")[0][
            "company"
        ]
        == "乙公司"
    )


def test_missing_scope_fails_closed(library):
    store, a2023, a2024, _ = library
    for ids, years in [
        ([], None),
        (["unknown"], None),
        ([a2024["id"]], []),
        ([a2023["id"]], [2024]),
    ]:
        assert store.search("经营活动现金流量", ids, years=years) == []
    assert store.search("", [a2024["id"]]) == []
    assert store.search("   ", [a2024["id"]]) == []
    assert store.search("unfindableenglishword", [a2024["id"]], mode="bm25") == []


def test_explicit_document_scope_is_not_sql_interpolation(library):
    store, _, a2024, _ = library
    assert store.search("经营活动现金流量", ["' OR 1=1 --"], mode="bm25") == []
    hits = store.search("现金流量", [a2024["id"], "' OR 1=1 --"], mode="bm25")
    assert {hit["document_id"] for hit in hits} == {a2024["id"]}


def test_chinese_bm25_prioritizes_relevant_paragraph(tmp_path):
    store = AnnualReportStore(tmp_path)
    text = "# 公司治理\n董事会完成了年度会议和组织调整。\f# 经营分析\n应收账款增加，客户回款放缓，导致经营现金流下降。\f# 员工情况\n员工总人数增长，培训活动有序开展。"
    document = store.ingest_bytes(text.encode(), "年报.md", "甲公司", 2024)
    hit = store.search("应收账款和回款", [document["id"]], mode="bm25")[0]
    assert hit["page"] == 2
    assert hit["section"] == "经营分析"
    assert hit["retrieval_mode"] == "bm25"
    assert {"应收", "收账", "账款"} <= set(tokenize("应收账款"))


def test_unconfigured_semantics_is_explicit_bm25_fallback(library, monkeypatch):
    monkeypatch.delenv("ANNUAL_REPORT_EMBEDDING_PROVIDER", raising=False)
    old_store, _, a2024, _ = library
    store = AnnualReportStore(old_store.data_dir)
    for mode in ["hybrid", "semantic", "vector"]:
        hit = store.search("现金流量", [a2024["id"]], mode=mode)[0]
        assert hit["retrieval_mode"] == "bm25_fallback"
        assert "未启用语义模型" in hit["retrieval_warning"]
    assert store.retrieval_status()["loaded"] is False


class ControlledEmbedder:
    """Deterministic test double; production never substitutes these vectors."""

    cache_key = "controlled-semantic-test-v1"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [
            [1.0, 0.0] if "资金回笼" in text or "收到款项" in text else [0.0, 1.0]
            for text in texts
        ]


def test_injected_semantics_and_cache_survive_restart(tmp_path):
    embedder = ControlledEmbedder()
    store = AnnualReportStore(tmp_path, embedder=embedder)
    doc = store.ingest_bytes(
        "# 业务\n经营活动收到款项明显提升。\f# 人员\n员工培训全面开展。".encode(),
        "年报.md",
        "甲公司",
        2024,
    )
    assert store.search("资金回笼", [doc["id"]], mode="bm25") == []
    semantic = store.search("资金回笼", [doc["id"]], top_k=1, mode="semantic")
    assert semantic[0]["page"] == 1 and semantic[0]["retrieval_mode"] == "semantic"
    assert len(embedder.calls) == 2  # One query, one batch of source chunks.
    hybrid = store.search("资金回笼", [doc["id"]], top_k=1, mode="hybrid")
    assert hybrid[0]["page"] == 1 and hybrid[0]["retrieval_mode"] == "hybrid"
    assert len(embedder.calls) == 3  # Cached document vectors are reused.
    restarted_embedder = ControlledEmbedder()
    restarted = AnnualReportStore(tmp_path, embedder=restarted_embedder)
    assert restarted.search("资金回笼", [doc["id"]], top_k=1)[0]["page"] == 1
    assert restarted_embedder.calls == [["资金回笼"]]


def test_semantic_candidates_cannot_escape_document_scope(tmp_path):
    embedder = ControlledEmbedder()
    store = AnnualReportStore(tmp_path, embedder=embedder)
    selected = store.ingest_bytes(
        "员工培训全面开展。".encode(), "甲.txt", "甲公司", 2024
    )
    excluded = store.ingest_bytes(
        "经营活动收到款项明显提升。".encode(), "乙.txt", "乙公司", 2024
    )
    hits = store.search("资金回笼", [selected["id"]], mode="hybrid")
    assert {hit["document_id"] for hit in hits} == {selected["id"]}
    assert all(excluded["id"] not in str(call) for call in embedder.calls)
    assert all("收到款项" not in text for call in embedder.calls for text in call)


@pytest.mark.parametrize(
    "invalid_vectors", [[[0.0, 0.0]], [[float("nan"), 1.0]], [], [[1.0], [2.0]]]
)
def test_invalid_semantics_falls_back_without_claiming_hybrid(
    tmp_path, invalid_vectors
):
    store = AnnualReportStore(tmp_path, embedder=lambda texts: invalid_vectors)
    doc = store.ingest_bytes("经营现金流下降。".encode(), "a.txt", "甲公司", 2024)
    hits = store.search("现金流", [doc["id"]])
    assert hits[0]["retrieval_mode"] == "bm25_fallback"
    assert "语义计算失败" in hits[0]["retrieval_warning"]


def test_model_failure_does_not_prevent_keyword_search(tmp_path):
    class FailingEmbedder:
        def embed(self, texts):
            raise RuntimeError("model unavailable")

    store = AnnualReportStore(tmp_path, embedder=FailingEmbedder())
    doc = store.ingest_bytes("经营现金流下降。".encode(), "a.txt", "甲公司", 2024)
    hit = store.search("现金流", [doc["id"]], mode="hybrid")[0]
    assert hit["retrieval_mode"] == "bm25_fallback"
    assert (
        store.search("现金流", [doc["id"]], mode="bm25")[0]["retrieval_mode"] == "bm25"
    )


def test_rrf_rewards_evidence_retrieved_by_both_routes():
    fused = reciprocal_rank_fusion(
        [("lexical", 100), ("both", 20)], [("semantic", 0.9), ("both", 0.7)]
    )
    assert fused[0][0] == "both"
    assert fused[0][1] == pytest.approx(2 / 62)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "madeup"},
        {"top_k": 0},
        {"top_k": 51},
        {"top_k": True},
        {"years": ["2024"]},
        {"document_ids": "id"},
    ],
)
def test_bad_search_arguments_are_rejected(library, kwargs):
    store, _, doc, _ = library
    arguments = {"query": "现金流", "document_ids": [doc["id"]], **kwargs}
    with pytest.raises(AnnualReportError):
        store.search(**arguments)


def test_fastembed_load_is_lazy_offline_and_failure_is_memoized(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    constructor_calls = []

    def unavailable_model(**kwargs):
        constructor_calls.append(kwargs)
        raise FileNotFoundError("uncached model")

    monkeypatch.setitem(
        sys.modules, "fastembed", SimpleNamespace(TextEmbedding=unavailable_model)
    )
    monkeypatch.setenv("ANNUAL_REPORT_EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("ANNUAL_REPORT_MODEL_CACHE", str(tmp_path / "models"))
    monkeypatch.delenv("ANNUAL_REPORT_ALLOW_MODEL_DOWNLOAD", raising=False)
    store = AnnualReportStore(tmp_path / "library")
    doc = store.ingest_bytes("现金回款正常。".encode(), "report.txt", "甲公司", 2024)
    assert constructor_calls == []
    assert store.search("现金", [doc["id"]], mode="bm25")[0]["retrieval_mode"] == "bm25"
    assert constructor_calls == []
    for _ in range(2):
        assert store.search("现金", [doc["id"]])[0]["retrieval_mode"] == "bm25_fallback"
    assert len(constructor_calls) == 1
    assert constructor_calls[0]["local_files_only"] is True
    assert constructor_calls[0]["cache_dir"] == str(tmp_path / "models")


def test_corrupt_persisted_embedding_is_recomputed(tmp_path):
    import sqlite3

    embedder = ControlledEmbedder()
    store = AnnualReportStore(tmp_path, embedder=embedder)
    doc = store.ingest_bytes(
        "经营活动收到款项明显提升。".encode(), "a.txt", "甲公司", 2024
    )
    hit = store.search("资金回笼", [doc["id"]], mode="semantic")[0]
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE embeddings SET vector = ? WHERE chunk_id = ?",
            ("[NaN, 0]", hit["id"]),
        )
    recovered = store.search("资金回笼", [doc["id"]], mode="semantic")[0]
    assert recovered["score"] == pytest.approx(1.0)
    assert len(embedder.calls) == 4  # Both searches encoded query plus source.


def test_long_chinese_evidence_tail_is_embedded_instead_of_truncated():
    captured = []

    class WindowEmbedder:
        def embed(self, texts):
            captured.extend(texts)
            return [
                [0.0, 1.0] if "现金流下降" in text else [1.0, 0.0] for text in texts
            ]

    semantic = SemanticEmbedder(WindowEmbedder())
    vector = semantic.encode(["销售情况总体稳定。" * 90 + "现金流下降"])[0]
    assert len(captured) > 1
    assert all(len(window) <= 400 for window in captured)
    assert "现金流下降" in captured[-1]
    assert vector[0] > 0 and vector[1] > 0
