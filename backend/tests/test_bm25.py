import pytest

from backend.rag.bm25 import BM25Index, reciprocal_rank_fusion


def test_rare_terms_and_document_length_matter():
    index = BM25Index(["revenue supplier", "revenue " * 80, "revenue growth", "revenue decline"])
    assert index.scores("supplier revenue")[0] == max(index.scores("supplier revenue"))
    assert index.scores("supplier")[1] == 0
    assert BM25Index(["营收增长", "今天下雨"]).scores("营收")[0] > 0


def test_empty_and_repeated_query_terms():
    assert BM25Index([]).scores("query") == []
    index = BM25Index(["", "apple"])
    assert index.scores("apple apple") == index.scores("apple")
    assert index.scores("missing") == [0, 0]


def test_collection_statistics_are_isolated():
    before = BM25Index(["private supplier", "revenue"])
    BM25Index(["supplier"] * 100)
    assert before.scores("supplier") == BM25Index(["private supplier", "revenue"]).scores("supplier")


def test_fusion_zero_lexical_has_no_rank_bonus():
    assert reciprocal_rank_fusion([0.2, 0.9], [0, 0]) == [1, 0]
    assert reciprocal_rank_fusion([0.2, 0.9], [5, 0], sparse_weight=2) == [0, 1]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([1], [float("nan")])
