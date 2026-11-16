"""Offline unit checks prepared for the next verification round; no API/model calls."""
import json
import asyncio

import pytest

from experiments.lightrag.metrics import citation_metrics, retrieval_metrics, summarize
from experiments.lightrag.run import load_dataset, pack_context, paired_deltas, parse_judge
from experiments.lightrag.adapters import CallBudget, ChatClient


def test_retrieval_deduplicates_and_penalizes_missing_hits():
    value = retrieval_metrics(["S01", "S01", "S99"], ["S01", "S02"], 4)
    assert value["recall_at_k"] == .5
    assert value["precision_at_k"] == .25
    assert value["all_evidence"] == 0


def test_abstention_has_no_artificial_recall_score():
    assert all(v is None for v in retrieval_metrics([], [], 4).values())


def test_citation_presence_is_not_faithfulness():
    assert citation_metrics("收入为100亿[S01]，利润为10亿[S99]。", ["S01"])["citation_id_validity"] == .5
    assert citation_metrics("没有证据。", ["S01"])["citation_id_validity"] is None


def test_judge_rejects_missing_and_nonfinite_scores():
    with pytest.raises(ValueError):
        parse_judge('{"faithfulness": 1}', False)
    with pytest.raises(ValueError):
        parse_judge('{"faithfulness": NaN,"answer_coverage":1,"citation_support":1}', False)


def test_partial_results_keep_missing_metrics_visible():
    values = summarize([
        {"arm": "a", "recall_at_k": 1, "retrieval_ms": [10, 2]},
        {"arm": "a", "error": "TimeoutError", "retrieval_ms": []},
    ])
    assert values["a"]["recall_at_k"] == {"mean": 1, "measured": 1, "total": 2}
    assert values["a"]["errors"] == 1


def test_context_budget_keeps_citation_units_whole():
    class CharacterTokenizer:
        def encode(self, text):
            return list(text)
    context, ids = pack_context({"sources": ["S01", "S02"], "graph": {}},
                                {"S01": "short", "S02": "x" * 100}, CharacterTokenizer(), 30)
    assert ids == ["S01"]
    assert len(context) <= 30


def test_graph_cannot_bypass_original_source_boundary():
    class CharacterTokenizer:
        def encode(self, text):
            return list(text)
    graph = {"entities": [
        {"file_path": "S01<SEP>S02", "description": "unsupported merged fact"},
        {"file_path": "S01", "description": "supported fact"},
        {"file_path": "S01", "description": "claim citing absent [S02]"},
    ]}
    context, ids = pack_context({"sources": ["S01"], "graph": graph},
                                {"S01": "original"}, CharacterTokenizer(), 2000)
    assert ids == ["S01"]
    assert "unsupported merged fact" not in context
    assert "supported fact" in context
    assert "absent" not in context


def test_pairing_never_compares_different_questions():
    assert paired_deltas([
        {"case_id": "x", "arm": "legacy_semantic", "recall_at_k": .5},
        {"case_id": "y", "arm": "lightrag_mix", "recall_at_k": 1},
    ]) == []


def test_dataset_rejects_unknown_gold_sources(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"documents": [{"id": "S01", "text": "text"}],
                               "cases": [{"id": "q", "required_sources": ["S99"],
                                          "unanswerable": False}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_deepseek_non_thinking_and_truncated_usage_are_recorded():
    import httpx

    async def scenario():
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"finish_reason": "length", "message": {"content": "partial"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 24}})
        client = ChatClient({"url": "https://api.deepseek.com/chat/completions",
                             "key": "test-only", "model": "test-model"}, CallBudget(1), 1)
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            with pytest.raises(RuntimeError):
                await client.complete("test")
            assert requests[0]["thinking"] == {"type": "disabled"}
            assert client.calls[0]["ok"] is False
            assert client.calls[0]["completion_tokens"] == 24
        finally:
            await client.close()
    asyncio.run(scenario())
