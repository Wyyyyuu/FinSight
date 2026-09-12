"""Company/report-year coverage through the public LangGraph analysis entry point."""
from collections import Counter

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
    monkeypatch.setattr(workflow, "_request_model_answer", lambda state: None)


def analyze(store, question, **kwargs):
    return workflow.analyze_reports(store, question, list(store.documents), mode="bm25", **kwargs)


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
