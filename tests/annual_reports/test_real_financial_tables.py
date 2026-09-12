"""Regression excerpts from 2024 CNINFO annual-report PDFs; no inferred amounts.

Source announcement IDs: Gree 1223330631 (physical p7), Haier 1222926246
(p10/p41), Hisense 1222945858 (p10/p30/p116). Whitespace is shortened here;
broken metric names, merged number cells, and actual column labels are retained.
"""
import sqlite3

import pytest

from backend.annual_reports.documents import AnnualReportStore, _same_page_table_context
from backend.annual_reports.workflow import _extract_facts

GREE_TABLE = """项目  2024 年  2023 年  本年比上年增减  2022 年
营业收入（元）  189,163,654,064.64  203,979,266,387.09  -7.26%  188,988,382,706.68
归属于上市公司股东  32,184,570,372.28  29,017,387,604.18  10.91%  24,506,623,782.46
的净利润（元）
经营活动产生的现金  29,369,250,570.66  56,398,426,354.17  -47.93%  28,668,435,921.27
流量净额（元）"""

HAIER_ADJUSTED = """单位： 元 币种：人民币
                   2023年                           本期比
主要会计数据 2024年                                   上年同           2022年
             调整后             调整前               期增减
                                                     (%)
营业收入 285,981,225,203.93 274,204,520,847.97 261,427,783,050.10 4.29 243,578,924,958.47
经营活动产生的 26,543,081,911.96 26,535,780,568.36 25,262,376,228.30 0.03 20,256,557,145.86
现金流量净额"""

HAIER_PAGE41 = """海尔智家股份有限公司 2024 年年度报告
五、报告期内主要经营情况
1、利润表及现金流量表相关科目变动分析表
单位： 元 币种：人民币
科目  本期数  上年同期数  变动比例（ %）
营业收入  285,981,225,203.93  274,204,520,847.97  4.29
研发费用  10,740,112,353.47  10,380,219,803.05  3.47
经营活动产生的现金流量净额  26,543,081,911.96  26,535,780,568.36  0.03"""

HISENSE_GLUE = """项目  2024年  2023年  本年比上年增减  2022年
（ %）
营业收入（元） 92,745,611,109.5285,600,189,224.06 8.35 74,115,151,039.29"""

HISENSE_PAGE30 = """海信家电集团股份有限公司 2024年年度报告全文
5、现金流
单位：元
项目  2024年  2023年  同比增减（%）
经营活动现金流入小计  81,425,983,334.68  76,815,643,741.00  6.00
经营活动产生的现金流量净额  5,132,164,941.24  10,611,857,591.35  -51.64"""

HISENSE_PAGE116 = """海信家电集团股份有限公司 2024年年度报告全文
3、合并利润表
单位：元
项目  2024年度  2023 年度
一、营业总收入  92,745,611,109.52  85,600,189,224.06
其中：营业收入  92,745,611,109.52  85,600,189,224.06"""

REVENUE = "营业收入"
CASH = "经营活动产生的现金流量净额"


def extract(text, *, context="", metrics=None):
    hit = {"id": "source", "label": "S1", "company": "测试公司", "document_id": "doc",
           "year": 2024, "page": 1, "section": "正文", "text": text, "source_context": context}
    return _extract_facts([hit], metrics or [REVENUE, CASH], [2023, 2024])[0]


def values(facts):
    return {(fact["metric"], fact["year"]): fact["value"] for fact in facts}


def test_gree_exact_wrapped_metric_and_unit_are_joined():
    facts = values(extract(GREE_TABLE, metrics=[REVENUE, CASH, "归母净利润"]))
    assert facts[(CASH, 2023)] == "56398426354.17"
    assert facts[(CASH, 2024)] == "29369250570.66"
    assert facts[("归母净利润", 2024)] == "32184570372.28"
    assert facts[(REVENUE, 2024)] == "189163654064.64"


@pytest.mark.parametrize("suffix", ["流量总额（元）", "流量净额（元） 100", "\n流量净额（元）", "流量净额（美元）"])
def test_wrapped_metrics_require_exact_adjacent_suffix(suffix):
    assert not extract(GREE_TABLE.replace("流量净额（元）", suffix), metrics=[CASH])


def test_adjustment_subcolumns_remain_rejected():
    assert not extract(HAIER_ADJUSTED)


def test_merged_decimal_cells_are_not_guessed_or_split_by_precision():
    assert not extract(HISENSE_GLUE, metrics=[REVENUE])


def test_percent_unit_in_header_covers_bare_ratio_cells():
    assert values(extract(HISENSE_PAGE30)) == {
        (CASH, 2024): "5132164941.24", (CASH, 2023): "10611857591.35",
    }
    assert not extract(HISENSE_PAGE30.replace("（%）", ""))


def test_same_page_context_recovers_relative_period_and_unit_for_existing_tail_chunk():
    tail = HAIER_PAGE41.splitlines()[-1]
    context = _same_page_table_context(HAIER_PAGE41, tail)
    assert "本期数" in context and "单位： 元" in context and "2024 年年度报告" in context
    assert values(extract(tail, context=context)) == {
        (CASH, 2024): "26543081911.96", (CASH, 2023): "26535780568.36",
    }
    assert values(extract(HAIER_PAGE41))[(REVENUE, 2023)] == "274204520847.97"


@pytest.mark.parametrize("alteration", [
    lambda text: text.replace("2024 年年度报告", "年度报告"),
    lambda text: text.replace("2024 年年度报告", "2023 年年度报告"),
    lambda text: text.replace("五、报告期内主要经营情况", "五、半年度主要经营情况"),
    lambda text: text.replace("本期数  上年同期数", "上年同期数  本期数"),
    lambda text: text.replace("单位： 元", "计量单位未注明"),
])
def test_relative_period_requires_explicit_matching_annual_title_and_units(alteration):
    assert not extract(alteration(HAIER_PAGE41))


def test_income_inside_explicit_consolidated_statement_is_company_income():
    chunk = "\n".join(HISENSE_PAGE116.splitlines()[-3:])
    context = _same_page_table_context(HISENSE_PAGE116, chunk)
    assert "合并利润表" in context
    assert values(extract(chunk, context=context, metrics=[REVENUE])) == {
        (REVENUE, 2024): "92745611109.52", (REVENUE, 2023): "85600189224.06",
    }
    assert not extract(chunk, context=context.replace("合并利润表", "母公司利润表"), metrics=[REVENUE])
    assert not extract(chunk.replace("其中：营业收入", "其中境外营业收入"), context=context)


def test_context_never_supplies_a_unit_from_a_different_table_or_page():
    chunk = HISENSE_PAGE30.splitlines()[-1]
    changed = HISENSE_PAGE30.replace("单位：元", "单位：万元\n六、另一个无单位的表")
    context = _same_page_table_context(changed, chunk)
    assert "万元" not in context
    assert not extract(chunk, context=context)
    assert _same_page_table_context("单位：元\n项目 2024年 2023年", chunk) == ""


def test_search_recovers_literal_context_without_rewriting_chunks_or_embeddings(tmp_path):
    store = AnnualReportStore(tmp_path)
    document = store.ingest_bytes(HISENSE_PAGE30.encode(), "source.txt", "海信家电", 2024)
    with sqlite3.connect(store.db_path) as connection:
        before = connection.execute("SELECT id,text FROM chunks ORDER BY id").fetchall()
        vectors_before = connection.execute("SELECT * FROM embeddings").fetchall()
    hits = store.search(CASH, [document["id"]], mode="bm25")
    assert any("单位：元" in hit.get("source_context", "") for hit in hits)
    for hit in hits:
        for line in hit.get("source_context", "").splitlines():
            assert line in HISENSE_PAGE30
    with sqlite3.connect(store.db_path) as connection:
        assert connection.execute("SELECT id,text FROM chunks ORDER BY id").fetchall() == before
        assert connection.execute("SELECT * FROM embeddings").fetchall() == vectors_before
