# -*- coding: utf-8 -*-
"""Tests for backend.rag.reranker — RerankerService with fallback."""
from __future__ import annotations

from backend.rag.reranker import RerankerService
import pytest


def test_reranker_disabled_returns_original_order():
    """When reranker is disabled, documents are returned in original order."""
    svc = RerankerService(force_backend="none")
    assert not svc.is_enabled

    docs = [
        {"content": "Apple revenue guidance", "source_id": "a"},
        {"content": "Microsoft Azure growth", "source_id": "b"},
        {"content": "Tesla delivery numbers", "source_id": "c"},
    ]
    result = svc.rerank("Apple earnings", docs, top_n=2)
    assert len(result) == 2
    assert result[0]["source_id"] == "a"
    assert result[1]["source_id"] == "b"


def test_reranker_disabled_empty_query():
    svc = RerankerService(force_backend="none")
    docs = [{"content": "test", "source_id": "x"}]
    result = svc.rerank("", docs, top_n=5)
    assert len(result) == 1


def test_reranker_disabled_empty_docs():
    svc = RerankerService(force_backend="none")
    result = svc.rerank("test query", [], top_n=5)
    assert result == []


def test_reranker_top_n_clipping():
    """Top-N should limit the output."""
    svc = RerankerService(force_backend="none")
    docs = [{"content": f"doc {i}", "source_id": str(i)} for i in range(10)]
    result = svc.rerank("query", docs, top_n=3)
    assert len(result) == 3


@pytest.mark.parametrize("scores", [[1.0], [float("nan"), 1.0]])
def test_invalid_reranker_output_preserves_candidates(monkeypatch, scores):
    svc = RerankerService(force_backend="bge-reranker-onnx")
    monkeypatch.setattr(svc, "_check_available", lambda: True)
    monkeypatch.setattr(svc, "_predict_onnx", lambda query, documents: scores)
    docs = [{"content": "first", "source_id": "a"}, {"content": "second", "source_id": "b"}]
    assert svc.rerank("query", docs, top_n=2) == docs


def test_onnx_reranker_sorts_without_losing_source_metadata(monkeypatch):
    svc = RerankerService(force_backend="bge-reranker-onnx")
    monkeypatch.setattr(svc, "_check_available", lambda: True)
    monkeypatch.setattr(svc, "_predict_onnx", lambda query, documents: [-2.0, 3.0])
    docs = [{"content": "noise", "source_id": "a"},
            {"content": "evidence", "source_id": "b", "metadata": {"url": "https://example.org"}}]
    ranked = svc.rerank("query", docs, top_n=1)
    assert ranked[0]["source_id"] == "b"
    assert ranked[0]["metadata"] == docs[1]["metadata"]
    assert "rerank_score" not in docs[1]
