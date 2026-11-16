import json
from pathlib import Path

import pytest
from test_company_retrieval import RankedStore

from backend.annual_reports import intent, workflow


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(workflow, "_request_model_answer", lambda _: None)
    monkeypatch.delenv("ANNUAL_REPORT_AGENTIC_RAG_ENABLED", raising=False)
    store = RankedStore()
    store.add("甲公司", 2024, "2024年营业收入为120亿元。\n2023年营业收入为100亿元。")
    return store


def analyze(store, question):
    return workflow.analyze_reports(store, question, list(store.documents), mode="bm25", max_retries=0)


@pytest.mark.parametrize(("base_model", "source"), [
    (None, "tinybert"),
    ("hfl/minirbt-h256", "minirbt"),
    ("jhu-clsp/mmBERT-small", "mmbert"),
    ("Qwen/Qwen3-0.6B-Base", "qwen"),
])
def test_local_intent_uses_fiscal_years_and_existing_decimal_calculation(store, monkeypatch, tmp_path, base_model, source):
    if base_model is not None:
        (tmp_path / "metadata.json").write_text(json.dumps({"base_model": base_model}), encoding="utf-8")
    monkeypatch.setenv("ANNUAL_REPORT_INTENT_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("ANNUAL_REPORT_AGENTIC_RAG_ENABLED", "1")
    monkeypatch.setattr(workflow, "_annual_model_config", lambda: pytest.fail("No remote plan needed"))
    monkeypatch.setattr(intent, "predict_intent", lambda q, path: "yoy")
    result = analyze(store, "2024年营收跟上一年比多了多少？")
    assert result["metrics"]["intent_source"] == source
    assert result["metrics"]["requested_years"] == [2023, 2024]
    assert result["status"] == "complete"
    assert result["calculations"][0]["change_pct"] == 20
    assert result["calculations"][0]["company"] == "甲公司"


@pytest.mark.parametrize(("question", "operation", "years"), [
    ("甲公司的营业收入在2024年有多少呢？", "lookup", [2024]),
    ("能否告诉我2024年营收的实际数额？", "lookup", [2024]),
    ("和上年同期相比，2024年营收多出多少钱？", "yoy", [2023, 2024]),
    ("营业收入在2024年与上年同期相比涨了几个点？", "yoy", [2023, 2024]),
    ("2024年营收跟上一财年一比少了多少万元？", "yoy", [2023, 2024]),
    ("与上年同期相比，2024年营收高出多少亿元？", "yoy", [2023, 2024]),
    ("跟上年度对照，2024年营收差额有多少？", "yoy", [2023, 2024]),
])
def test_local_intent_accepts_alternate_word_order(store, monkeypatch, question, operation, years):
    monkeypatch.setenv("ANNUAL_REPORT_INTENT_MODEL_DIR", "test-model")
    monkeypatch.setattr(intent, "predict_intent", lambda *_: operation)
    result = analyze(store, question)
    assert result["status"] == "complete"
    assert result["metrics"]["requested_years"] == years
    assert bool(result["calculations"]) is (operation == "yoy")


@pytest.mark.parametrize("question", [
    "2024年海外业务收入比上一年多了多少？",
    "2024年扣非归母净利润比上一年高多少？",
    "2024年扣非后的归母净利润跟上一个财年相比多多少？",
    "2024年扣除非经常性损益后的归母净利润是多少？",
    "2024年营收同比是否可持续？",
    "乙公司2024年营收跟上一年比多了多少？",
    "2024年营收跟上一年比多了多少，再查2023年研发？",
    "营业收入在2024年第四季度有多少？",
    "和上年同期相比，2024年营收为何多出这么多？",
    "乙公司的营业收入在2024年有多少？",
    "2024年度经营现金流和两年前的差额有多大？",
    "跟上年度对照，2024年营收按固定汇率高出多少？",
    "2024年营收跟上一个财年一比高出多少，剔除新增业务后还有多少？",
    "跟前两年对照，2024年营收差额有多少？",
])
def test_candidate_cannot_bypass_scope_and_operation_guards(store, monkeypatch, question):
    monkeypatch.setenv("ANNUAL_REPORT_INTENT_MODEL_DIR", "test-model")
    monkeypatch.setattr(intent, "predict_intent", lambda *args: pytest.fail("Unsafe subject reached classifier"))
    result = analyze(store, question)
    assert result["status"] == "insufficient_evidence"
    assert result["metrics"]["intent_source"] == "unresolved"
    if any(term in question for term in ("海外", "扣非", "扣除非经常性损益", "季度")):
        assert result["metrics"]["fact_count"] == 0


def test_qwen_abstention_routes_to_existing_llm_plan(store, monkeypatch, tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"base_model": "Qwen/Qwen3-0.6B-Base"}), encoding="utf-8")
    monkeypatch.setenv("ANNUAL_REPORT_INTENT_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("ANNUAL_REPORT_AGENTIC_RAG_ENABLED", "1")
    monkeypatch.setattr(intent, "predict_intent", lambda *_: None)
    monkeypatch.setattr(workflow, "_annual_model_config", lambda: ("example", "key", "model"))

    def complete(_config, instruction, payload, **kwargs):
        if "reports" in payload:
            return '{"checks":[{"need":"两年营收金额","query":"营业收入 2024 2023"}]}'
        return '{"excerpts":[[{"source":"S1","quote":"2024年营业收入为120亿元。"}]],"question_covered":false,"follow_up_query":""}'

    monkeypatch.setattr(workflow, "_request_annual_model", complete)
    result = analyze(store, "2024年营收跟上一年比多了多少？")
    assert result["metrics"]["intent_source"] == "llm"
    assert result["metrics"]["retrieval_strategy"] == "agentic_planned"
    assert result["status"] == "insufficient_evidence"


def test_llm_can_resolve_the_same_validated_intent_after_local_abstention(store, monkeypatch):
    monkeypatch.setenv("ANNUAL_REPORT_INTENT_MODEL_DIR", "test-model")
    monkeypatch.setenv("ANNUAL_REPORT_AGENTIC_RAG_ENABLED", "1")
    monkeypatch.setattr(intent, "predict_intent", lambda *_: None)
    monkeypatch.setattr(workflow, "_annual_model_config", lambda: ("example", "key", "model"))
    monkeypatch.setattr(workflow, "_request_annual_model", lambda *_a, **_kw:
                        '{"operation":"yoy","checks":[{"need":"两年营业收入","query":"营业收入 2024 2023"}]}')
    result = analyze(store, "2024年营收跟上一年比多了多少？")
    assert result["metrics"]["intent_source"] == "llm"
    assert result["status"] == "complete"
    assert result["calculations"][0]["change_pct"] == 20


def test_missing_local_artifact_is_safe(tmp_path):
    assert intent.predict_intent("2024年营收大约多少", Path(tmp_path) / "missing") is None


def test_previous_year_wording_uses_same_base_as_yoy(store, monkeypatch):
    monkeypatch.setenv("ANNUAL_REPORT_INTENT_MODEL_DIR", "test-model")
    monkeypatch.setattr(intent, "predict_intent", lambda *_: pytest.fail("Rules should bypass Qwen"))
    result = analyze(store, "2024年营业收入较上年变化？")
    assert result["metrics"]["intent_source"] == "rules"
    assert result["calculations"][0]["change_pct"] == 20
