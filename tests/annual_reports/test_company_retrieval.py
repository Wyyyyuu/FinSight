"""Company/report-year coverage through the public LangGraph analysis entry point."""
import json
from collections import Counter

import httpx
import pytest

from backend.annual_reports import workflow


class RankedStore:
    """Global ranking intentionally places every long-company chunk before peers."""

    def __init__(self):
        self.documents = {}
        self.hits = []
        self.calls = []
        self.override = None

    def add(self, company, year, text, *, chunks=1, suffix=""):
        document_id = f"{company}-{year}{suffix}"
        self.documents[document_id] = {
            "id": document_id, "company": company, "year": year,
            "filename": f"{document_id}.pdf", "page_count": 300,
        }
        for index in range(chunks):
            self.hits.append({"id": f"{document_id}:{index}", "document_id": document_id,
                              "year": year, "page": index + 1, "text": text,
                              "score": 100 - len(self.hits), "retrieval_mode": "bm25"})
        return document_id

    def get_document(self, document_id):
        return self.documents.get(document_id)

    def search(self, query, document_ids, years=None, top_k=8, mode="hybrid"):
        self.calls.append({"query": query, "ids": list(document_ids), "years": years, "top_k": top_k})
        if self.override:
            return self.override(query, document_ids, years, top_k)
        return [hit for hit in self.hits if hit["document_id"] in document_ids
                and (not years or hit["year"] in years)][:top_k]


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.delenv("ANNUAL_REPORT_AGENTIC_RAG_ENABLED", raising=False)
    monkeypatch.setattr(workflow, "_request_model_answer", lambda state: None)


def analyze(store, question, **kwargs):
    return workflow.analyze_reports(store, question, list(store.documents), mode="bm25", **kwargs)


@pytest.fixture
def agentic_model(monkeypatch):
    monkeypatch.setenv("ANNUAL_REPORT_AGENTIC_RAG_ENABLED", "1")
    monkeypatch.setenv("ANNUAL_REPORT_LLM_BASE_URL", "https://annual-model.example/v1")
    monkeypatch.setenv("ANNUAL_REPORT_LLM_API_KEY", "test-key")
    monkeypatch.setenv("ANNUAL_REPORT_LLM_MODEL", "test-model")


def test_open_question_plans_searches_and_rechecks_missing_evidence(agentic_model, monkeypatch):
    store = RankedStore()
    profit = "2024年非经常性损益主要为资产处置收益。"
    cash = "2024年销售商品收到的现金增加，销售回款率较上年提升。"
    chosen = store.add("甲公司", 2024, profit)
    older = store.add("甲公司", 2023, "上一报告年度披露。")
    foreign = store.add("未选公司", 2024, "其他公司的现金流证据。")
    model_calls = []

    def model(config, instruction, payload, **kwargs):
        model_calls.append(payload)
        if "reports" in payload:
            return json.dumps({"checks": [
                {"need": "一次性收益来源", "query": "非经常性损益 资产处置"},
                {"need": "销售回款表现", "query": "利润现金含量"},
            ]})
        by_text = {item["text"]: item["source"] for item in payload["evidence"]}
        rows = [[{"source": by_text[profit], "quote": profit}], []]
        if cash in by_text:
            rows[1].append({"source": by_text[cash], "quote": cash})
        return json.dumps({"excerpts": rows, "question_covered": cash in by_text,
                           "follow_up_query": "销售商品收到的现金 回款率" if cash not in by_text else ""})

    def search(query, ids, years, top_k):
        if query == "利润现金含量":
            return [store.hits[2]]
        if query == "销售商品收到的现金 回款率":
            return [{**store.hits[0], "id": "cash", "page": 5, "text": cash}, store.hits[1], store.hits[2]]
        return [store.hits[0]]

    monkeypatch.setattr(workflow, "_request_annual_model", model)
    store.override = search
    result = workflow.analyze_reports(store, "2024年这家公司的盈利质量怎么样", [chosen, older], years=[2024], mode="bm25")

    assert result["status"] == "complete", result["metrics"]["evidence_gaps"]
    assert result["metrics"]["retrieval_strategy"] == "agentic_planned"
    assert [call["query"] for call in store.calls] == ["非经常性损益 资产处置", "利润现金含量", "销售商品收到的现金 回款率"]
    assert len(model_calls) == 3 and result["metrics"]["retries"] == 1
    assert model_calls[0]["reports"] == [{"company": "甲公司", "report_year": 2024}]
    assert all(call["ids"] == [chosen] and call["years"] == [2024] for call in store.calls)
    assert {item["document_id"] for item in result["citations"]} == {chosen}
    assert foreign not in str(result)
    assert "一次性收益来源" in result["answer"] and profit in result["answer"] and cash in result["answer"]
    assert any(event["node"] == "assess" and "1/2" in event["detail"] for event in result["trace"])


@pytest.mark.parametrize("excerpts", [[[]], [[{"source": "S1", "quote": "管理层确认盈利质量改善。"}]]])
def test_open_question_cannot_be_complete_without_reviewed_original_evidence(agentic_model, monkeypatch, excerpts):
    store = RankedStore()
    store.add("甲公司", 2024, "本报告披露盈利质量相关风险。")

    def model(config, instruction, payload, **kwargs):
        if "reports" in payload:
            return json.dumps({"checks": [{"need": "盈利质量依据", "query": "非经常性损益 利润现金含量"}]})
        return json.dumps({"excerpts": excerpts, "question_covered": True, "follow_up_query": ""})

    monkeypatch.setattr(workflow, "_request_annual_model", model)
    result = analyze(store, "如何评价2024年盈利质量")
    assert result["status"] == "insufficient_evidence"
    assert "未核实取证要点：盈利质量依据" in result["metrics"]["evidence_gaps"]
    assert "管理层确认盈利质量改善" not in result["answer"]
    assert result["citations"][0]["text"] == "本报告披露盈利质量相关风险。"


def test_open_planning_failure_uses_original_search_for_mixed_metric_question(agentic_model, monkeypatch):
    store = RankedStore()
    store.add("甲公司", 2024, "2024年营业收入：120亿元\n2023年营业收入：100亿元")

    def unavailable(*args, **kwargs):
        raise httpx.ReadTimeout("private-provider-error")

    monkeypatch.setattr(workflow, "_request_annual_model", unavailable)
    question = "2024年营业收入增长是否可持续"
    result = analyze(store, question)
    assert [call["query"] for call in store.calls] == [question]
    assert result["trace"][0]["status"] == "fallback"
    assert result["status"] == "insufficient_evidence" and result["metrics"]["fact_count"] == 2
    assert "private-provider-error" not in str(result)


def test_agentic_supplement_targets_only_the_missing_company_and_year(agentic_model, monkeypatch):
    store = RankedStore()
    values = "单位：亿元\n项目 2024年 2023年\n经营活动产生的现金流量净额 20 10"
    midea = store.add("美的", 2024, values + "\n2024年经营现金流增加主要系回款增加。", chunks=8)
    haier = store.add("海尔", 2024, values)
    older = store.add("海尔", 2023, "2023年经营现金流变动主要系支付减少。")
    foreign = store.add("未选公司", 2024, "2024年经营现金流变动主要系收到补贴。")
    sources = {hit["document_id"]: hit for hit in store.hits}
    reason = "2024年经营现金流增加主要系销售商品收到的现金增加。"
    calls = []

    def respond(request):
        calls.append(json.loads(json.loads(request.content)["messages"][1]["content"]))
        return httpx.Response(200, json={"choices": [{"message": {"content":
            json.dumps({"target": 0, "query": "管理层讨论 现金流变化"}, ensure_ascii=False)
        }}]})

    original_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs))

    def search(query, ids, years, top_k):
        if "管理层讨论" in query:
            return [{**sources[haier], "id": "cause", "page": 10, "text": reason}, sources[midea], sources[foreign]]
        return [hit for hit in store.hits if hit["document_id"] in ids]

    store.override = search
    result = workflow.analyze_reports(
        store, "对比美的和海尔2024年经营现金流的同比变化，并解释各自原因",
        [midea, haier, older], years=[2024], mode="bm25",
    )

    assert result["status"] == "complete", result["metrics"]["evidence_gaps"]
    assert result["metrics"]["retrieval_strategy"] == "agentic_supplement"
    assert len(calls) == 1
    assert set(calls[0]) == {"question", "targets"}
    assert [(target["company"], target["report_year"], target["kind"]) for target in calls[0]["targets"]] == [
        ("海尔", 2024, "cause"),
    ]
    assert [call["ids"] for call in store.calls] == [[midea], [haier], [haier]]
    assert all(call["years"] == [2024] for call in store.calls)
    assert {hit["document_id"] for hit in result["citations"]} == {midea, haier}
    assert foreign not in str(result)
    citation = next(hit for hit in result["citations"] if hit["id"] == "cause")
    assert citation["page"] == 10 and reason in result["answer"]
    assert any(f"定向补查新增 1 条证据 [{citation['label']}]" in event["detail"] for event in result["trace"])
    assert {row["company"]: row["change_pct"] for row in result["calculations"]} == {"美的": 100, "海尔": 100}
    assert any("工作流选择 Agent 补查" in event["detail"] and "海尔" in event["detail"] for event in result["trace"])


@pytest.mark.parametrize(("question", "source", "retries", "strategy"), [
    ("2024年营业收入是多少", "2024年营业收入：120亿元", 1, "standard"),
    ("2024年营业收入是多少", "主要业务是制造。", 1, "rule_supplement"),
    ("对比美的和海尔2024年营业收入", "2024年营业收入：120亿元", 1, "rule_supplement"),
    ("为什么2024年经营现金流下降", "2024年经营现金流：20亿元\n2023年经营现金流：10亿元", 1, "rule_supplement"),
    ("为什么2023年经营现金流增长", "2023年经营现金流：20亿元\n2022年经营现金流：10亿元", 1, "rule_supplement"),
    ("为什么2024年经营现金流下降", "主要业务是制造。", 0, "standard"),
])
def test_workflow_routes_without_model_when_agent_is_ineligible(agentic_model, monkeypatch, question, source, retries, strategy):
    monkeypatch.setattr(workflow, "_request_annual_model", lambda *args, **kwargs: pytest.fail("Routing must stay in code"))
    store = RankedStore()
    store.add("美的", 2024, source)
    result = analyze(store, question, max_retries=retries)
    assert result["metrics"]["retrieval_strategy"] == strategy


@pytest.mark.parametrize("invalid_plan", [
    '{"target": 99, "query": "任意年报"}',
    '{"target": 0, "query": "主要系", "document_id": "未选年报"}',
    None,
])
def test_agentic_plan_failure_uses_rule_supplement_once(agentic_model, monkeypatch, invalid_plan):
    store = RankedStore()
    store.add("甲公司", 2024, "2024年营业收入：120亿元\n2023年营业收入：100亿元")
    model_calls = []

    def respond(*args, **kwargs):
        model_calls.append(1)
        if invalid_plan is None:
            raise httpx.ReadTimeout("private-provider-error")
        return invalid_plan

    monkeypatch.setattr(workflow, "_request_annual_model", respond)
    store.override = lambda query, *args: [*store.hits, {**store.hits[0], "id": "cause",
        "text": "2024年营业收入增长主要系产品销量增加。"}] if "变动原因" in query else store.hits
    result = analyze(store, "为什么2024年营业收入增长", max_retries=2)
    assert len(model_calls) == 1
    assert result["status"] == "complete", result["metrics"]["evidence_gaps"]
    assert result["metrics"]["retrieval_strategy"] == "rule_supplement"
    assert result["metrics"]["retries"] == 1
    assert "private-provider-error" not in str(result)
    assert any("已回退" in event["detail"] for event in result["trace"])


def test_long_company_cannot_monopolize_same_year_comparison_or_preview():
    store = RankedStore()
    for index, company in enumerate(("美的", "格力", "海尔", "海信")):
        store.add(company, 2024, f"2024年营业收入：{100 + index}亿元", chunks=16 if index == 0 else 1)
    result = analyze(store, "比较2024年各公司的营业收入")
    assert result["status"] == "complete", result["metrics"]["evidence_gaps"]
    assert result["metrics"]["fact_count"] == 4
    assert result["calculations"] == []  # Same-year facts are not mislabeled as YoY.
    assert len(store.calls) == 4
    assert {hit["company"] for hit in result["citations"][:4]} == {"美的", "格力", "海尔", "海信"}
    assert all(len(call["ids"]) == 1 and call["years"] == [2024] for call in store.calls)


def test_each_company_and_report_year_retains_its_own_recall_quota():
    store = RankedStore()
    for company in ("长报告公司", "短报告公司"):
        for year, amount in ((2023, 100), (2024, 120)):
            store.add(company, year, f"{year}年营业收入：{amount}亿元", chunks=12 if company == "长报告公司" else 1)
    result = analyze(store, "比较2023和2024年营业收入")
    assert result["status"] == "complete"
    assert len(result["calculations"]) == 2
    assert {calc["company"] for calc in result["calculations"]} == {"长报告公司", "短报告公司"}
    assert all(calc["change_pct"] == 20 for calc in result["calculations"])
    assert {(store.documents[call["ids"][0]]["company"], call["years"][0]) for call in store.calls} == {
        (company, year) for company in ("长报告公司", "短报告公司") for year in (2023, 2024)
    }


def test_metric_supplement_also_partitions_companies_and_reads_prior_year_columns():
    store = RankedStore()
    for company in ("美的", "格力", "海尔", "海信"):
        store.add(company, 2024, "单位：亿元\n项目 2024年 2023年\n营业收入 120 100")

    def search(query, ids, years, top_k):
        source = next(hit for hit in store.hits if hit["document_id"] in ids)
        if query.startswith("经营活动"):
            return [{**source, "id": source["id"] + "cash",
                     "text": "单位：亿元\n项目 2024年 2023年\n经营活动产生的现金流量净额 20 10"}]
        return [source]

    store.override = search
    result = analyze(store, "比较2023至2024年营业收入和经营现金流")
    assert result["status"] == "complete", result["metrics"]["evidence_gaps"]
    assert result["metrics"]["retries"] == 1
    assert result["metrics"]["fact_count"] == 16
    assert len(result["calculations"]) == 8
    assert len(store.calls) == 20
    assert all(len(call["ids"]) == 1 and call["years"] == [2024] for call in store.calls)
    assert Counter(call["ids"][0] for call in store.calls) == Counter({doc_id: 5 for doc_id in store.documents})


def test_selection_year_and_current_partition_reject_provider_leaks():
    store = RankedStore()
    selected = store.add("甲", 2024, "2024年营业收入：100亿元")
    excluded_year = store.add("乙", 2023, "2023年营业收入：900亿元")
    unselected = store.add("丙", 2024, "2024年营业收入：999亿元")
    store.override = lambda *args: [store.hits[2], store.hits[1], store.hits[0]]
    result = workflow.analyze_reports(store, "2024年营业收入", [selected, excluded_year], years=[2024])
    assert result["status"] == "complete"
    assert {hit["document_id"] for hit in result["citations"]} == {selected}
    assert store.calls[0]["ids"] == [selected]
    assert store.calls[0]["years"] == [2024]
    assert unselected not in str(store.calls)
    assert "999" not in result["answer"] and "900" not in result["answer"]


def test_multiple_selected_documents_share_only_their_company_year_partition():
    store = RankedStore()
    original = store.add("甲", 2024, "2024年营业收入：100亿元")
    duplicate = store.add("甲", 2024, "2024年营业收入：100亿元", suffix="-duplicate")
    peer = store.add("乙", 2024, "2024年营业收入：200亿元")
    result = analyze(store, "比较2024年营业收入")
    assert result["status"] == "complete"
    assert [call["ids"] for call in store.calls] == [[original, duplicate], [peer]]
    assert result["metrics"]["fact_count"] == 2


def test_other_company_evidence_cannot_hide_a_missing_company():
    store = RankedStore()
    store.add("甲", 2024, "2024年营业收入：100亿元")
    store.add("乙", 2024, "本公司业务包括制造", chunks=0)
    result = analyze(store, "比较2024年营业收入")
    assert result["status"] == "insufficient_evidence"
    assert any("乙 2024年营业收入" in gap for gap in result["metrics"]["evidence_gaps"])
    assert Counter(call["ids"][0] for call in store.calls) == Counter({doc_id: 4 for doc_id in store.documents})
    generic = analyze(store, "比较2024年各公司年报")
    assert generic["status"] == "insufficient_evidence"
    assert any("乙 2024年可引用" in gap for gap in generic["metrics"]["evidence_gaps"])


def test_large_selection_stops_at_call_budget_and_reports_remaining_gaps():
    store = RankedStore()
    for index in range(30):
        store.add(f"公司{index}", 2024, "", chunks=0)
    result = analyze(store, "2024年营业收入、归母净利润和经营现金流的变动原因", max_retries=3)
    assert result["status"] == "insufficient_evidence"
    assert result["metrics"]["retrieval_limited"] is True
    assert len(store.calls) == workflow.MAX_RETRIEVAL_CALLS
    assert Counter(call["ids"][0] for call in store.calls) == Counter({doc_id: 4 for doc_id in store.documents})
    assert any("检索预算" in gap for gap in result["metrics"]["evidence_gaps"])


def test_large_unique_result_sets_are_bounded_without_starving_last_company():
    store = RankedStore()
    for index in range(30):
        store.add(f"公司{index}", 2024, "与所问指标无关的文本", chunks=8)

    def fresh_chunks(query, ids, years, top_k):
        return [{**hit, "id": f"{hit['id']}:{len(store.calls)}"}
                for hit in store.hits if hit["document_id"] in ids][:top_k]

    store.override = fresh_chunks
    result = analyze(store, "比较2024年营业收入和经营现金流", max_retries=3)
    assert result["status"] == "insufficient_evidence"
    assert result["metrics"]["retrieval_limited"] is True
    assert len(result["citations"]) == workflow.MAX_EVIDENCE_HITS
    assert set(Counter(hit["company"] for hit in result["citations"]).values()) == {16}
    assert len(store.calls) <= workflow.MAX_RETRIEVAL_CALLS


def test_more_than_thirty_documents_fails_before_loading_or_searching():
    store = RankedStore()
    with pytest.raises(ValueError, match="30"):
        workflow.analyze_reports(store, "营业收入", [str(index) for index in range(31)])
    assert store.calls == []
