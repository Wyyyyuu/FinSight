"""Behavioral tests for document boundaries, graph routes, and financial arithmetic."""
import json
import re

import pytest

from backend.annual_reports import workflow


class FakeStore:
    def __init__(self, texts=None, *, search_override=None, bind_year=True):
        self.documents = {}
        self.hits = []
        self.calls = []
        self.search_override = search_override
        for year, text in (texts or {}).items():
            if bind_year:
                # Most fixtures are prose statements; make their source years explicit.
                text = "\n".join(f"{year}年{line}" if re.match(r"(?:营业收入|归母净利润|经营活动产生的现金流量净额)", line)
                                 and not re.search(r"(?:19|20)\d{2}年", line) else line for line in text.splitlines())
            key = str(year)
            self.documents[key] = {"id": key, "filename": f"示例公司{year}年报.pdf", "company": "示例公司",
                                   "year": year, "page_count": 30, "chunk_count": 1, "warnings": []}
            self.hits.append({"id": f"{key}:1", "document_id": key, "filename": "untrusted.pdf",
                              "company": "untrusted company", "year": year, "page": 8,
                              "section": "主要财务数据", "text": text, "score": .8,
                              "retrieval_mode": "hybrid"})

    def get_document(self, document_id):
        return self.documents.get(document_id)

    def search(self, query, document_ids, years=None, top_k=8, mode="hybrid"):
        self.calls.append({"query": query, "ids": document_ids, "years": years, "mode": mode})
        if self.search_override:
            return self.search_override(self, query, document_ids, years)
        return [hit for hit in self.hits if hit["document_id"] in document_ids and (not years or hit["year"] in years)]


@pytest.fixture(autouse=True)
def no_model_credentials(monkeypatch):
    for key in ("ANNUAL_REPORT_LLM_BASE_URL", "ANNUAL_REPORT_LLM_API_KEY", "ANNUAL_REPORT_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)


def run(store, question="比较2022和2023年营业收入", **kwargs):
    return workflow.analyze_reports(store, question, list(store.documents), **kwargs)


def test_real_graph_runs_all_nodes_without_supplement_when_sufficient():
    store = FakeStore({2022: "营业收入：100亿元", 2023: "营业收入：120亿元"})
    result = run(store)
    assert result["status"] == "complete"
    assert [event["node"] for event in result["trace"]] == ["plan", "retrieve", "assess", "calculate", "answer"]
    assert result["metrics"]["answer_mode"] == "extractive"
    assert result["calculations"][0]["change_pct"] == 20.0
    assert "[S1]" in result["answer"] and "[S2]" in result["answer"]
    graph = workflow.build_analysis_graph(store).get_graph()
    assert {"plan", "retrieve", "assess", "supplement", "calculate", "answer"} <= set(graph.nodes)


def test_supplement_can_find_missing_year_then_stops():
    def search(store, query, ids, years):
        return [hit for hit in store.hits if (not years or hit["year"] in years) and (hit["year"] == 2022 or query.startswith("营业收入"))]
    result = run(FakeStore({2022: "营业收入：1亿元", 2023: "营业收入：2亿元"}, search_override=search))
    assert result["status"] == "complete"
    assert result["metrics"]["retries"] == 1
    assert [e["node"] for e in result["trace"]].count("assess") == 2
    assert result["calculations"][0]["change_pct"] == 100


def test_empty_results_refuse_and_retry_count_is_bounded():
    store = FakeStore({2022: "", 2023: ""}, search_override=lambda *args: [])
    result = run(store, max_retries=3)
    assert result["status"] == "insufficient_evidence"
    assert result["citations"] == [] and result["calculations"] == []
    assert result["metrics"]["retries"] == 3
    assert len(store.calls) == 5
    assert "没有检索到" in result["answer"]


@pytest.mark.parametrize("selection", [[], ["missing"], ["2023", "missing"]])
def test_unavailable_selection_fails_closed_without_search(selection):
    store = FakeStore({2023: "营业收入：1亿元"})
    with pytest.raises(ValueError):
        workflow.analyze_reports(store, "营业收入", selection)
    assert store.calls == []


def test_api_year_filter_remains_hard_boundary_and_question_year_is_not_invented():
    store = FakeStore({2021: "营业收入：10亿元", 2023: "营业收入：30亿元"})
    result = run(store, years=[2021])
    assert result["status"] == "insufficient_evidence"
    assert result["metrics"]["missing_years"] == [2022, 2023]
    assert {tuple(call["years"]) for call in store.calls} == {(2021,)}
    assert {hit["year"] for hit in result["citations"]} == {2021}
    assert result["calculations"] == []


def test_year_range_and_yoy_require_prior_period():
    store = FakeStore({2021: "营业收入：10亿元", 2022: "营业收入：20亿元", 2023: "营业收入：30亿元"})
    assert run(store, "对比2021年至2023年营业收入")["metrics"]["requested_years"] == [2021, 2022, 2023]
    result = run(store, "2023年营业收入同比是多少")
    assert result["metrics"]["requested_years"] == [2022, 2023]
    assert result["calculations"][0]["change_pct"] == 50


def test_citations_reject_unselected_document_and_use_canonical_metadata():
    def search(store, query, ids, years):
        foreign = {**store.hits[0], "id": "foreign:1", "document_id": "private", "text": "营业收入：999亿元"}
        invalid_page = {**store.hits[0], "id": "bad-page", "page": 99}
        return [foreign, invalid_page, store.hits[0]]
    store = FakeStore({2023: "营业收入：1亿元"}, search_override=search)
    result = run(store, "2023年营业收入是多少")
    assert len(result["citations"]) == 1
    source = result["citations"][0]
    assert source["filename"] == "示例公司2023年报.pdf"
    assert source["company"] == "示例公司"
    assert "999" not in result["answer"]


@pytest.mark.parametrize("before,after,expected", [
    ("营业收入：1亿元", "营业收入：12,500万元", 25),
    ("单位：万元\n营业收入：10,000", "单位：万元\n营业收入：12,000", 20),
    ("营业收入（万元）：10,000", "营业收入（万元）：15,000", 50),
    ("2022年营业收入：100百万元", "2023年营业收入：120百万元", 20),
])
def test_units_are_normalized_before_computing_yoy(before, after, expected):
    result = run(FakeStore({2022: before, 2023: after}))
    assert result["status"] == "complete", result["metrics"]["evidence_gaps"]
    calc = result["calculations"][0]
    assert calc["change_pct"] == expected
    assert calc["unit"] == "元"
    assert len(calc["operands"]) == 2
    assert all(item["citation"] in calc["source_labels"] for item in calc["operands"])


@pytest.mark.parametrize("negative", ["-100万元", "−100万元", "（100万元）", "(100)万元"])
def test_negative_profit_does_not_report_misleading_growth_rate(negative):
    result = run(FakeStore({2022: f"归母净利润：{negative}", 2023: "归母净利润：50万元"}), "比较2022和2023年归母净利润")
    assert result["status"] == "complete", result["metrics"]["evidence_gaps"]
    calc = result["calculations"][0]
    assert calc["from_value"] == "-1000000"
    assert calc["delta"] == "1500000"
    assert calc["change_pct"] is None


def test_zero_baseline_is_not_divided_by_zero():
    result = run(FakeStore({2022: "营业收入：0元", 2023: "营业收入：1亿元"}))
    assert result["calculations"][0]["change_pct"] is None


def test_simple_year_table_is_bound_to_explicit_columns():
    table = "单位：万元\n项目 | 2023年 | 2022年\n营业收入 | 12,000 | 10,000"
    result = run(FakeStore({2022: "营业收入：10000万元", 2023: table}))
    assert result["status"] == "complete", result["metrics"]["evidence_gaps"]
    assert result["calculations"][0]["change_pct"] == 20


@pytest.mark.parametrize("ambiguous", [
    "营业收入：100 120",
    "单位：万元\n项目 | 2023年 | 2022年\n调整后 | 调整前\n营业收入 | 120 | 100",
    "营业收入同比增长率：20%",
    "营业收入：100",
])
def test_ambiguous_values_are_not_computed(ambiguous):
    result = run(FakeStore({2022: "营业收入：100万元", 2023: ambiguous}))
    assert result["status"] == "insufficient_evidence"
    assert result["calculations"] == []


def test_conflicting_values_do_not_arbitrarily_select_one():
    store = FakeStore({2022: "营业收入：100万元", 2023: "营业收入：120万元\n营业收入：140万元"})
    result = run(store)
    assert result["status"] == "insufficient_evidence"
    assert result["calculations"] == []
    assert any("冲突" in gap for gap in result["metrics"]["evidence_gaps"])


def test_numbers_do_not_invent_causation():
    texts = {2022: "营业收入：100亿元\n经营活动产生的现金流量净额：20亿元",
             2023: "营业收入：120亿元\n经营活动产生的现金流量净额：10亿元"}
    result = run(FakeStore(texts), "为什么2022至2023年营业收入增长但经营现金流下降")
    assert result["status"] == "insufficient_evidence"
    assert len(result["calculations"]) == 2
    assert any("因果" in gap for gap in result["metrics"]["evidence_gaps"])


def test_model_failure_falls_back_without_leaking_provider_error(monkeypatch):
    def fail(state):
        raise RuntimeError("secret_token_in_provider_error")
    monkeypatch.setattr(workflow, "_request_model_answer", fail)
    result = run(FakeStore({2023: "营业收入：1亿元"}), "2023年营业收入")
    assert result["status"] == "complete"
    assert result["metrics"]["answer_mode"] == "extractive"
    assert result["trace"][-1]["status"] == "fallback"
    assert "secret_token" not in json.dumps(result)


@pytest.mark.parametrize("claim", [
    {"text": "营业收入：1亿元", "citations": ["S999"]},
    {"text": "营业收入增长来自市场份额提升", "citations": ["S1"]},
    {"text": "营业收入：999亿元", "citations": ["S1"]},
])
def test_model_citations_cannot_launder_unsupported_claims(monkeypatch, claim):
    monkeypatch.setattr(workflow, "_request_model_answer", lambda state: json.dumps({"claims": [claim]}, ensure_ascii=False))
    result = run(FakeStore({2023: "营业收入：1亿元"}), "2023年营业收入")
    assert result["metrics"]["answer_mode"] == "extractive"
    assert result["trace"][-1]["status"] == "fallback"


def test_verified_model_excerpt_is_accepted(monkeypatch):
    raw = json.dumps({"claims": [{"text": "2023年营业收入：1亿元", "citations": ["S1"]}]}, ensure_ascii=False)
    monkeypatch.setattr(workflow, "_request_model_answer", lambda state: raw)
    result = run(FakeStore({2023: "营业收入：1亿元"}), "2023年营业收入")
    assert result["metrics"]["answer_mode"] == "llm"
    assert "2023年营业收入：1亿元 [S1]" in result["answer"]


def test_insufficient_evidence_never_calls_generation(monkeypatch):
    monkeypatch.setattr(workflow, "_request_model_answer", lambda state: pytest.fail("must not generate missing evidence"))
    result = run(FakeStore({2023: "营业收入：1亿元"}))
    assert result["status"] == "insufficient_evidence"


def test_search_error_is_bounded_and_refuses():
    def fail(*args):
        raise RuntimeError("index unavailable")
    result = run(FakeStore({2023: "营业收入：1亿元"}, search_override=fail), "2023年营业收入", max_retries=1)
    assert result["status"] == "insufficient_evidence"
    assert result["metrics"]["retrieval_calls"] == 2
    assert result["trace"][1]["status"] == "partial"


def test_demo_sentence_format_and_direct_causes():
    store = FakeStore({2022: "2022 年营业收入为 100 亿元。\n2022 年经营活动产生的现金流量净额为 18 亿元。",
                       2023: "2023 年营业收入为 120 亿元。\n2023 年经营活动产生的现金流量净额为 15 亿元。\n营业收入增长主要系销量增加。经营现金流下降主要系应收账款增加。"})
    result = run(store, "为什么2022到2023年营业收入增长但经营现金流下降")
    assert result["status"] == "complete", result["metrics"]["evidence_gaps"]
    assert len(result["calculations"]) == 2


def test_unrelated_reason_does_not_explain_requested_metric():
    store = FakeStore({2022: "营业收入：100亿元", 2023: "营业收入：120亿元\n投资收益下降主要系股票价格下降。"})
    result = run(store, "2022至2023年营业收入增长原因是什么")
    assert result["status"] == "insufficient_evidence"
    assert any("直接解释营业收入" in gap for gap in result["metrics"]["evidence_gaps"])


def test_generic_question_requires_topic_evidence():
    store = FakeStore({2023: "营业收入：100亿元"})
    result = run(store, "2023年董事薪酬情况")
    assert result["status"] == "insufficient_evidence"
    assert any("薪酬" in gap for gap in result["metrics"]["evidence_gaps"])
    assert run(store, "总结这份年报")["status"] == "complete"


def test_nonstandard_topic_cannot_be_answered_from_random_hit():
    store = FakeStore({2023: "营业收入：100亿元"})
    assert run(store, "2023年员工福利情况")["status"] == "insufficient_evidence"
    store.hits[0]["text"] += "\n员工福利包括补充医疗保险。"
    assert run(store, "2023年员工福利情况")["status"] == "complete"


def test_foreign_currency_is_not_labeled_as_rmb():
    result = run(FakeStore({2022: "营业收入：100万元", 2023: "币种：美元\n营业收入：120万元"}))
    assert result["status"] == "insufficient_evidence"
    assert result["calculations"] == []


def test_segment_revenue_is_not_company_revenue():
    result = run(FakeStore({2022: "营业收入：100亿元", 2023: "其中境外营业收入：120亿元"}))
    assert result["status"] == "insufficient_evidence"
    assert result["calculations"] == []


def test_explicit_dated_values_and_conflicts_are_respected():
    table = "营业收入：2023年120亿元；2022年100亿元"
    result = run(FakeStore({2022: "营业收入：100亿元", 2023: table}))
    assert result["status"] == "complete"
    assert result["calculations"][0]["change_pct"] == 20


def test_no_year_inference_across_different_companies():
    store = FakeStore({2022: "营业收入：100亿元", 2023: "营业收入：120亿元"})
    store.documents["2023"]["company"] = "另一公司"
    result = run(store)
    assert result["status"] == "insufficient_evidence"
    assert result["calculations"] == []


def test_model_client_uses_only_opt_in_config_and_validates_response(monkeypatch):
    import httpx

    for key, value in {"ANNUAL_REPORT_LLM_BASE_URL": "https://model.example/v1",
                       "ANNUAL_REPORT_LLM_API_KEY": "test-key",
                       "ANNUAL_REPORT_LLM_MODEL": "test-model"}.items():
        monkeypatch.setenv(key, value)
    original_client = httpx.Client
    requests = []
    def handler(request):
        requests.append(request)
        content = json.dumps({"claims": [{"text": "2023年营业收入：1亿元", "citations": ["S1"]}]}, ensure_ascii=False)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    result = run(FakeStore({2023: "营业收入：1亿元"}), "2023年营业收入")
    assert result["metrics"]["answer_mode"] == "llm"
    assert str(requests[0].url) == "https://model.example/v1/chat/completions"
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content)["model"] == "test-model"


@pytest.mark.parametrize("kwargs", [{"mode": "unknown"}, {"max_retries": 4}, {"years": [99]}])
def test_invalid_options_do_not_start_retrieval(kwargs):
    store = FakeStore({2023: "营业收入：1亿元"})
    with pytest.raises(ValueError):
        run(store, "营业收入", **kwargs)
    assert store.calls == []


MIDEA_FLAT_TABLE = """六、主要会计数据和财务指标
公司是否需追溯调整或重述以前年度会计数据
□ 是 √ 否
2024年 2023年 本年比上年增减 2022年
营业收入（千元） 407,149,600 372,037,280 9.44% 343,917,531
归属于上市公司股东的净利润（千元） 38,537,237 33,719,935 14.29% 29,553,507
经营活动产生的现金流量净额（千元） 60,511,572 57,902,611 4.51% 34,657,828
"""


def test_real_flat_annual_table_skips_ratio_column_and_covers_prior_factual_years():
    store = FakeStore({2024: MIDEA_FLAT_TABLE}, bind_year=False)
    result = run(store, "比较2022至2024年营业收入、归母净利润和经营现金流")
    assert result["status"] == "complete", result["metrics"]["evidence_gaps"]
    assert result["metrics"]["missing_years"] == []
    assert len(result["calculations"]) == 6
    latest_revenue = next(c for c in result["calculations"] if c["metric"] == "营业收入" and c["to_year"] == 2024)
    assert latest_revenue["to_value"] == "407149600000"
    assert latest_revenue["change_pct"] == 9.44
    assert result["citations"][0]["year"] == 2024
    assert all(op["document_id"] == "2024" for op in latest_revenue["operands"])


def test_prior_year_question_can_supplement_from_newer_selected_report():
    store = FakeStore({2024: MIDEA_FLAT_TABLE}, bind_year=False)
    result = run(store, "2023年营业收入是多少")
    assert result["status"] == "complete"
    assert result["metrics"]["retries"] == 1
    assert store.calls[0]["years"] == [2023]
    assert store.calls[1]["years"] is None
    assert result["metrics"]["requested_years"] == [2023]
    assert "3,720.3728亿元" in result["answer"]
    assert result["citations"][0]["year"] == 2024


def test_explicit_document_filter_cannot_be_relaxed_by_supplement():
    store = FakeStore({2024: MIDEA_FLAT_TABLE}, bind_year=False)
    result = run(store, "2023年营业收入是多少", years=[2023])
    assert result["status"] == "insufficient_evidence"
    assert all(call["years"] == [2023] for call in store.calls)
    assert result["citations"] == []


def test_unlabeled_year_cannot_be_inferred_from_report_metadata():
    result = run(FakeStore({2023: "营业收入：1亿元"}, bind_year=False), "2023年营业收入")
    assert result["status"] == "insufficient_evidence"
    assert result["metrics"]["fact_count"] == 0


def test_wrong_directional_premise_is_explicitly_corrected():
    store = FakeStore({2024: MIDEA_FLAT_TABLE + "2024年经营现金流增加主要系回款增加。"}, bind_year=False)
    result = run(store, "为什么2024年经营现金流下降")
    assert result["status"] == "insufficient_evidence"
    assert result["calculations"][0]["change_pct"] == 4.51
    assert any("问题前提与证据不符" in gap and "实际增加" in gap for gap in result["metrics"]["evidence_gaps"])
    assert result["trace"][-2]["status"] == "contradiction"


def test_reasons_from_newer_year_cannot_explain_older_year():
    store = FakeStore({2024: MIDEA_FLAT_TABLE + "2024年经营现金流增加主要系回款增加。"}, bind_year=False)
    result = run(store, "为什么2023年经营现金流增长")
    assert result["status"] == "insufficient_evidence"
    assert any("2023年原文" in gap for gap in result["metrics"]["evidence_gaps"])


@pytest.mark.parametrize("row", [
    "2023年营业收入：100万元（调整前）；2023年120万元（调整后）",
    "营业收入（万元）：2023年9.44%；2022年10%",
    "2023年家电营业收入：100万元",
    "项目 | 2023年 | 2022年 | 同比增减\n营业收入（万元） | 120 | 100 | 20",
])
def test_ambiguous_adjustments_ratios_and_segments_are_not_amounts(row):
    result = run(FakeStore({2023: row}, bind_year=False), "2023年营业收入是多少")
    assert result["status"] == "insufficient_evidence"
    assert result["metrics"]["fact_count"] == 0


@pytest.mark.parametrize("mode,canonical", [("bm25", "bm25"), ("keyword", "bm25"), ("vector", "semantic"), ("semantic", "semantic")])
def test_retrieval_mode_names_match_store_contract(mode, canonical):
    store = FakeStore({2023: "2023年营业收入：100万元"})
    result = run(store, "2023年营业收入", mode=mode)
    assert result["status"] == "complete"
    assert all(call["mode"] == canonical for call in store.calls)


def test_supplement_targets_primary_financial_statement_when_generic_search_is_noisy():
    def search(store, query, ids, years):
        if "主要会计数据" in query:
            return store.hits
        return []
    store = FakeStore({2024: MIDEA_FLAT_TABLE}, bind_year=False, search_override=search)
    result = run(store, "2023年营业收入是多少")
    assert result["status"] == "complete"
    assert result["metrics"]["retries"] == 1
    assert result["metrics"]["fact_count"] == 1


def test_ambiguous_extra_hit_does_not_invalidate_a_complete_unambiguous_fact_set():
    store = FakeStore({2022: "2022年营业收入：100万元", 2023: "2023年营业收入：120万元\n营业收入：300 400"}, bind_year=False)
    result = run(store)
    assert result["status"] == "complete"
    assert result["metrics"]["evidence_gaps"] == []
    assert result["metrics"]["extraction_warnings"]
    assert result["calculations"][0]["change_pct"] == 20


def test_conflicts_outside_question_years_do_not_block_requested_fact():
    store = FakeStore({2024: "2024年营业收入：120万元\n2022年营业收入：100万元\n2022年营业收入：110万元"}, bind_year=False)
    result = run(store, "2024年营业收入是多少")
    assert result["status"] == "complete"
    assert result["metrics"]["fact_count"] == 1


def test_model_must_quote_the_requested_year_not_just_the_correct_document(monkeypatch):
    raw = json.dumps({"claims": [{"text": "2024年营业收入：120万元", "citations": ["S1"]}]}, ensure_ascii=False)
    monkeypatch.setattr(workflow, "_request_model_answer", lambda state: raw)
    store = FakeStore({2024: "2024年营业收入：120万元\n2023年营业收入：100万元"}, bind_year=False)
    result = run(store, "2023年营业收入是多少")
    assert result["status"] == "complete"
    assert result["metrics"]["answer_mode"] == "extractive"
    assert result["trace"][-1]["status"] == "fallback"
