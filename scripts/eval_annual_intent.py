"""Compare rule and local classifier routes on held-out questions and real BM25 execution."""

import argparse
import json
import re
import statistics
import tempfile
from pathlib import Path
from unittest.mock import patch

from backend.annual_reports.documents import AnnualReportStore
from backend.annual_reports.intent import (
    _PREVIOUS,
    LOCAL_INTENT_NAMES,
    _load_classifier,
    local_intent_source,
    normalize_intent_question,
    predict_intent,
)
from backend.annual_reports.workflow import analyze_reports

# Manually written evaluation questions. Training and threshold calibration do not read these.
CASES = (
    ("2024年营业收入是多少？", "lookup", "营业收入"),
    ("2024年归母净利润是多少？", "lookup", "归母净利润"),
    ("2024年经营现金流同比变化幅度是多少？", "yoy", "经营活动产生的现金流量净额"),
    ("2024年营业收入较上年变化？", "yoy", "营业收入"),
    ("2024年归母净利润同比是多少？", "yoy", "归母净利润"),
    ("查一下2024年营收是多少钱呀？", "lookup", "营业收入"),
    ("2024年营业收入的实际金额有多少呢？", "lookup", "营业收入"),
    ("能否告知2024年营收金额？", "lookup", "营业收入"),
    ("2024年经营现金流金额到底是多少啊？", "lookup", "经营活动产生的现金流量净额"),
    ("2024年的归母净利润大约是什么水平？", "lookup", "归母净利润"),
    ("想知道2024年归母净利润的数是多少？", "lookup", "归母净利润"),
    ("请查下2024年经营性现金流大概为多少？", "lookup", "经营活动产生的现金流量净额"),
    ("远山制造2024年营业收入的规模是多少？", "lookup", "营业收入"),
    ("营收在2024年达到了什么规模？", "lookup", "营业收入"),
    ("2024年营收跟上一年比多了多少？", "yoy", "营业收入"),
    ("2024年营业收入和上一财年相比高多少？", "yoy", "营业收入"),
    ("2024年的归母净利润与去年相比多了多少钱？", "yoy", "归母净利润"),
    ("2024年经营现金流比上一年度少了多少钱？", "yoy", "经营活动产生的现金流量净额"),
    ("2024年经营性现金流对照上个财年差多少？", "yoy", "经营活动产生的现金流量净额"),
    ("2024年归母净利润比前一个年度高几个点？", "yoy", "归母净利润"),
    ("远山制造2024年营收跟去年的变动幅度有多大？", "yoy", "营业收入"),
    ("2024年的经营活动产生的现金流量净额和上年相比少了多少？", "yoy", "经营活动产生的现金流量净额"),
    ("和上一年对照，2024年归母净利润多出多少？", "yoy", "归母净利润"),
    ("2024年营收与上一年相比到底差多少啊？", "yoy", "营业收入"),
    ("2024年海外业务收入比上一年多了多少？", "open", ""),
    ("2024年扣非归母净利润与上年相比高多少？", "open", ""),
    ("2024年境内营收总共多少？", "open", ""),
    ("2024年营收同比增长是否可持续？", "open", ""),
    ("2024年营收比上一年多了多少，主要由什么驱动？", "open", ""),
    ("2024年经营现金流比上年多了多少，存货有哪些变化？", "open", ""),
    ("2024年归母净利润比上一年变化的部分有多少来自政府补助？", "open", ""),
    ("2024年营收按季度环比多少？", "open", ""),
    ("2024年营收比另一家公司高多少？", "open", ""),
    ("2024年毛利率比去年高几个点？", "open", ""),
    ("2024年营业收入在行业里的占比是多少？", "open", ""),
    ("2024年营收在海外都靠哪些产品？", "open", ""),
    ("2024年经营现金流和利润的增长能反映盈利质量吗？", "open", ""),
    ("远山制造2024年营收是多少，海川制造2023年经营现金流为什么下滑？", "open", ""),
    ("2024年经营现金流是否足以覆盖未来资本开支？", "open", ""),
    ("2024年营业收入减去营业成本后还剩多少？", "open", ""),
)

# Written separately for the mmBERT comparison and kept out of all training and calibration.
FRESH_CASES = (
    ("看看2024年经营性现金流大概为多少？", "lookup", "经营活动产生的现金流量净额"),
    ("远山制造的2024年度归母净利润实际金额多少？", "lookup", "归母净利润"),
    ("2024年归母净利润到底达到什么水平？", "lookup", "归母净利润"),
    ("告诉我2024年营收总共是多少钱？", "lookup", "营业收入"),
    ("2024年经营现金流的数字是啥？", "lookup", "经营活动产生的现金流量净额"),
    ("年报里写的营业收入，2024年的那笔是多少？", "lookup", "营业收入"),
    ("2024年营业收入到底有多少钱呢？", "lookup", "营业收入"),
    ("2024年营收跟去年同期比高多少啊？", "yoy", "营业收入"),
    ("想了解2024年归母净利润对照上一财年多多少？", "yoy", "归母净利润"),
    ("2024年度经营性现金流跟上一个财年相比低了多少？", "yoy", "经营活动产生的现金流量净额"),
    ("2024年经营现金流比上年少多少个点？", "yoy", "经营活动产生的现金流量净额"),
    ("2024年营业收入的同比变动百分比为多少？", "yoy", "营业收入"),
    ("2024年归母净利润相较于前一年度高多少呢？", "yoy", "归母净利润"),
    ("上年是基期，远山制造2024年的营收增幅多少？", "yoy", "营业收入"),
    ("2024年境外市场的营收跟去年同期比高多少？", "open", ""),
    ("2024年扣除非经常性损益后的归母净利润总共是多少钱？", "open", ""),
    ("2024年经营现金流和上年相比低了多少，为什么？", "open", ""),
    ("2024年归母净利润的增加主要靠卖资产吗？", "open", ""),
    ("2024年营收比上年多多少，扣除汇率影响后呢？", "open", ""),
    ("比较远山制造和海川制造2024年的海外业务收入", "open", ""),
    ("2024年营收比上年多多少，能覆盖下一年的设备支出吗？", "open", ""),
    ("2024年经营现金流减去购建固定资产支出后是多少钱？", "open", ""),
)

# Held out by wording and kept out of the v2 training and threshold selection.
V2_CASES = (
    ("麻烦查一下远山制造2024年经营现金流总共有多少钱？", "lookup", "经营活动产生的现金流量净额"),
    ("远山制造的营业收入在2024年有多少呢？", "lookup", "营业收入"),
    ("能否告诉我2024年归母净利润的实际数额？", "lookup", "归母净利润"),
    ("能给我2024年营收的金额吗？", "lookup", "营业收入"),
    ("经营性现金流于2024年度达到什么水平？", "lookup", "经营活动产生的现金流量净额"),
    ("2024年归母净利润总数是多少呀？", "lookup", "归母净利润"),
    ("请问2024年度归母净利润大约为多少亿？", "lookup", "归母净利润"),
    ("远山制造2024年经营性现金流到底是什么数？", "lookup", "经营活动产生的现金流量净额"),
    ("和上年同期相比，2024年营收多出多少钱？", "yoy", "营业收入"),
    ("远山制造的归母净利润在2024年跟上一年相比高多少？", "yoy", "归母净利润"),
    ("2024年度经营现金流和去年同期的差额有多大？", "yoy", "经营活动产生的现金流量净额"),
    ("比上一财年，2024年营业收入高多少呢？", "yoy", "营业收入"),
    ("2024年营收对照上个财年大约多出了多少？", "yoy", "营业收入"),
    ("帮我看看2024年归母净利润比前一个年度的差额？", "yoy", "归母净利润"),
    ("营业收入在2024年与上年同期相比涨了几个点？", "yoy", "营业收入"),
    ("对照去年，2024年经营性现金流少了多少呀？", "yoy", "经营活动产生的现金流量净额"),
    ("远山制造的海外营收在2024年有多少呢？", "open", ""),
    ("能给我2024年扣非归母净利润的金额吗？", "open", ""),
    ("和上年同期相比，2024年营收为何多出这么多？", "open", ""),
    ("2024年经营现金流和两年前的差额有多大？", "open", ""),
    ("营业收入在2024年第四季度有多少？", "open", ""),
    ("2024年营收和上年相比多出多少，能持续吗？", "open", ""),
    ("海川制造和远山制造2024年的营收相差多大？", "open", ""),
    ("对照去年，2024年经营性现金流剔除政府补助后少了多少？", "open", ""),
    ("2024年境内市场的营业收入有多少亿？", "open", ""),
    ("归母净利润于2024年和上年相比涨了几个点，主要由什么推动？", "open", ""),
)

# Independently authored for v3, with difficult cases beyond the supported grammar.
V3_CASES = (
    ("远山制造的营收于2024年度一共有多少钱呢？", "lookup", "营业收入"),
    ("告诉我2024年经营现金流具体是多少亿元呀？", "lookup", "经营活动产生的现金流量净额"),
    ("想问归母净利润在2024年度大约是多少万？", "lookup", "归母净利润"),
    ("2024年度营业收入的总数是多少呢？", "lookup", "营业收入"),
    ("请查下远山制造2024年归母净利润到底什么规模？", "lookup", "归母净利润"),
    ("能给我经营性现金流于2024年的实际金额吗？", "lookup", "经营活动产生的现金流量净额"),
    ("2024年归母净利润实际为多少万元啊？", "lookup", "归母净利润"),
    ("帮我看看远山制造的营业收入在2024年实际有多少？", "lookup", "营业收入"),
    ("能否告知2024年经营现金流合计是多少亿？", "lookup", "经营活动产生的现金流量净额"),
    ("2024年的营收实际数额是啥？", "lookup", "营业收入"),
    ("归母净利润在2024年究竟多少钱啊？", "lookup", "归母净利润"),
    ("2024年度经营现金流具体有多大？", "lookup", "经营活动产生的现金流量净额"),
    ("2024年营收上涨之后的总金额是多少？", "lookup", "营业收入"),
    ("2024年报里有没有写归母净利润最后是多少？", "lookup", "归母净利润"),
    ("我想了解2024年经营性现金流的实际数字？", "lookup", "经营活动产生的现金流量净额"),
    ("想知道远山制造的2024年营收合计有多少万？", "lookup", "营业收入"),
    ("请说下远山制造归母净利润于2024年具体有多大？", "lookup", "归母净利润"),
    ("远山制造2024年经营现金流总共有多少呢？", "lookup", "经营活动产生的现金流量净额"),
    ("与上个财年相比，2024年营业收入高出多少亿元？", "yoy", "营业收入"),
    ("2024年经营性现金流跟去年同期比较的差额有多大？", "yoy", "经营活动产生的现金流量净额"),
    ("远山制造的归母净利润于2024年对照上一年度高多少啊？", "yoy", "归母净利润"),
    ("2024年营业收入比上一个财年多了多少万元？", "yoy", "营业收入"),
    ("较前一年度，远山制造2024年度经营现金流低了多少？", "yoy", "经营活动产生的现金流量净额"),
    ("能给我2024年归母净利润和上年同期的差额吗？", "yoy", "归母净利润"),
    ("和去年比较，2024年度营收多出多少万呢？", "yoy", "营业收入"),
    ("经营性现金流在2024年跟上年同期比少了多少个点？", "yoy", "经营活动产生的现金流量净额"),
    ("2024年归母净利润与上一财年比变动的百分比是多少？", "yoy", "归母净利润"),
    ("跟上一年度比，远山制造2024年营业收入净增多少呀？", "yoy", "营业收入"),
    ("经营现金流于2024年与上一个财年对照差多少钱？", "yoy", "经营活动产生的现金流量净额"),
    ("归母净利润在2024年较去年同期涨了几个百分点？", "yoy", "归母净利润"),
    ("对照上年同期，2024年营收具体多出多少亿？", "yoy", "营业收入"),
    ("2024年经营现金流比上一年少掉的那部分金额有多大？", "yoy", "经营活动产生的现金流量净额"),
    ("请帮我把2024年的归母净利润和上一财年做个同比？", "yoy", "归母净利润"),
    ("远山制造2024年的营业收入比上年多多少个百分点？", "yoy", "营业收入"),
    ("2024年经营性现金流跟上一年比到底差多少万呢？", "yoy", "经营活动产生的现金流量净额"),
    ("2024年度归母净利润对照上年同期的变化幅度有多大？", "yoy", "归母净利润"),
    ("远山制造境外营收于2024年度一共有多少钱呢？", "open", ""),
    ("2024年扣非后的归母净利润跟上一个财年相比多多少？", "open", ""),
    ("2024年经营现金流具体是多少，扣掉资本支出后还有多少？", "open", ""),
    ("2024年营业收入比上一个财年多的部分由哪个地区贡献？", "open", ""),
    ("归母净利润于2024年第三季度究竟多少钱？", "open", ""),
    ("与上个财年相比，2024年营业收入为啥多出这么多？", "open", ""),
    ("2024年的营收占该行业的份额是多少？", "open", ""),
    ("远山制造与海川制造2024年营业收入相差多少？", "open", ""),
    ("2024年经营现金流少了多少，能说明偿债能力下降吗？", "open", ""),
    ("2024年全年归母净利润比前年的差额是多少？", "open", ""),
    ("2024年营收剔除新增并表后的实际数额？", "open", ""),
    ("2024年经营现金流按母公司报表口径多少钱？", "open", ""),
    ("帮我看2024年调整后的归母净利润和上年比怎么样？", "open", ""),
    ("2024年营业收入有多少来自境内市场？", "open", ""),
    ("2024年经营性现金流与过去四年中位数相差多少？", "open", ""),
    ("2024年净利润增长能持续到下一财年吗？", "open", ""),
    ("2024年营业收入先查合并口径，再把分产品的拆一下？", "open", ""),
    ("2024年度归母净利润和上年同期的差额里有多少是资产出售？", "open", ""),
)

# Authored independently of training templates. Each direct question precedes its boundary counterpart.
BOUNDARY_CASES = (
    ("远山制造2024年营收实际有多少亿？", "lookup", "营业收入"),
    ("远山制造2024年线上渠道营收实际有多少亿？", "open", ""),
    ("归母净利润在2024年合计多少钱呢？", "lookup", "归母净利润"),
    ("归母净利润在2024年下半年合计多少钱呢？", "open", ""),
    ("请说下2024年经营现金流一共有多少？", "lookup", "经营活动产生的现金流量净额"),
    ("请说下2024年未扣营运资金变动的经营现金流一共有多少？", "open", ""),
    ("2024年度营业收入的总数大约是多少亿元？", "lookup", "营业收入"),
    ("2024年度营业收入中采用总额法确认的大约是多少亿元？", "open", ""),
    ("远山制造的归母净利润在2024年度实际有多少？", "lookup", "归母净利润"),
    ("远山制造的归母净利润在2024年度实际有多少，附带说明公允价值收益的影响？", "open", ""),
    ("想问2024年经营性现金流的实际金额有多少？", "lookup", "经营活动产生的现金流量净额"),
    ("想问2024年经营性现金流只算持续经营部分的实际金额有多少？", "open", ""),
    ("2024年度营业收入跟上年度相比多出几亿？", "yoy", "营业收入"),
    ("2024年度营业收入跟疫情前基期相比多出几亿？", "open", ""),
    ("跟上个财年比，2024年归母净利润净增多少？", "yoy", "归母净利润"),
    ("跟上个财年比，2024年归母净利润净增多少来自少计减值？", "open", ""),
    ("经营现金流在2024年度较上一年度减少多少万元？", "yoy", "经营活动产生的现金流量净额"),
    ("经营现金流在2024年度较上一年度减少多少万元，收现和付现分别贡献多少？", "open", ""),
    ("2024年营收比去年同期多出多少钱呀？", "yoy", "营业收入"),
    ("2024年营收比去年同期多出多少钱呀，用不变价重算一下？", "open", ""),
    ("远山制造2024年归母净利润与上一年比多了多少？", "yoy", "归母净利润"),
    ("远山制造2024年归母净利润与海川制造上一年比多了多少？", "open", ""),
    ("2024年的经营性现金流跟上一个财年一比低了多少？", "yoy", "经营活动产生的现金流量净额"),
    ("2024年的经营性现金流跟上一个财年一比低了多少，排除经营租赁分类变动后呢？", "open", ""),
)

# Fresh contrasts for evaluating paired fine-tuning after inspecting the original suite.
BOUNDARY_V2_CASES = (
    ("2024年度的营业收入实际合计多少钱啊？", "lookup", "营业收入"),
    ("2024年度仅列报为代理业务的营业收入实际合计多少钱啊？", "open", ""),
    ("归母净利润于2024年具体多少亿呀？", "lookup", "归母净利润"),
    ("归母净利润于2024年把已处置业务剔除之后具体多少亿呀？", "open", ""),
    ("想了解经营现金流在2024年度到底多少钱？", "lookup", "经营活动产生的现金流量净额"),
    ("想了解经营现金流在2024年度按经营租赁重分类后的口径到底多少钱？", "open", ""),
    ("远山制造2024年的营收实际有多少万元呢？", "lookup", "营业收入"),
    ("远山制造2024年按终端销售而非经销出货确认的营收实际有多少万元呢？", "open", ""),
    ("2024年归母净利润一共多少钱呀？", "lookup", "归母净利润"),
    ("2024年头九个月的归母净利润一共多少钱呀？", "open", ""),
    ("经营性现金流在2024年到底有多少亿元呢？", "lookup", "经营活动产生的现金流量净额"),
    ("经营性现金流在2024年按未抵销集团往来的口径到底有多少亿元呢？", "open", ""),
    ("和上年同期比较，2024年营业收入净增多少万元呢？", "yoy", "营业收入"),
    ("和上次行业低谷比较，2024年营业收入净增多少万元呢？", "open", ""),
    ("2024年度经营现金流比上一个财年少了多少钱呀？", "yoy", "经营活动产生的现金流量净额"),
    ("2024年度经营现金流比上一个财年少了多少钱呀，剔除票据结算变动后还有多少？", "open", ""),
    ("较上一年度，2024年归母净利润高了几个百分点呀？", "yoy", "归母净利润"),
    ("较上一年度，2024年归母净利润高了几个百分点呀，分别由哪些子公司贡献？", "open", ""),
    ("远山制造的营收在2024年对照上一年度多了多少亿？", "yoy", "营业收入"),
    ("远山制造的营收在2024年对照海川制造的上一年度多了多少亿？", "open", ""),
    ("2024年经营性现金流和上年同期比到底差多少亿元？", "yoy", "经营活动产生的现金流量净额"),
    ("2024年经营性现金流和上年同期比到底差多少亿元，其中供应商账期变化影响多少？", "open", ""),
    ("归母净利润在2024年度跟上一个财年比低多少万元？", "yoy", "归母净利润"),
    ("归母净利润在2024年度跟年初承诺的目标比低多少万元？", "open", ""),
)

VALUES = {"营业收入": (100, 120), "归母净利润": (10, 12), "经营活动产生的现金流量净额": (20, 15)}
UNIT = 100_000_000


def run(store, documents, model_dir, cases=CASES):
    rows = []
    calls = 0

    def offline_model(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError("benchmark measures route dispatch without a live model")

    env = {"ANNUAL_REPORT_INTENT_MODEL_DIR": model_dir, "ANNUAL_REPORT_AGENTIC_RAG_ENABLED": "1"}
    config = ("offline", "offline", "offline")
    with (patch.dict("os.environ", env),
          patch("backend.annual_reports.workflow._annual_model_config", return_value=config),
          patch("backend.annual_reports.workflow._request_model_answer", return_value=None),
          patch("backend.annual_reports.workflow._request_annual_model", side_effect=offline_model)):
        for question, expected, metric in cases:
            before_calls = calls
            chosen = documents if "海川制造" in question else documents[:1]
            response = analyze_reports(store, question, chosen, mode="bm25", max_retries=0)
            metrics = response["metrics"]
            calculations = response["calculations"]
            expected_years = [2023, 2024] if expected == "yoy" else [2024]
            correct = (response["status"] == "complete" and metrics["requested_years"] == expected_years
                       and metrics["fact_count"] == len(expected_years)
                       and metrics["intent_source"] in {"rules", *LOCAL_INTENT_NAMES})
            if expected == "lookup":
                correct = correct and not calculations and metric in response["answer"] and f"{VALUES[metric][1]}亿元" in response["answer"]
            elif expected == "yoy":
                prior, latest = VALUES[metric]
                correct = (correct and len(calculations) == 1 and calculations[0]["metric"] == metric
                           and int(calculations[0]["from_value"]) == prior * UNIT
                           and int(calculations[0]["to_value"]) == latest * UNIT)
            rows.append({"question": question, "expected": expected, "route": metrics["intent_source"],
                         "remote_planned": calls > before_calls, "correct_local": bool(correct and expected != "open"),
                         "incorrect_open_completion": expected == "open" and response["status"] == "complete",
                         "elapsed_ms": metrics["elapsed_ms"]})
    return {"local_correct": sum(row["correct_local"] for row in rows),
            "local_expected": sum(row["expected"] != "open" for row in rows),
            "remote_plans": calls, "local_model_routed": sum(row["route"] in LOCAL_INTENT_NAMES for row in rows),
            "open_routed_to_llm": sum(row["expected"] == "open" and row["remote_planned"] for row in rows),
            "open_false_complete": sum(row["incorrect_open_completion"] for row in rows),
            "elapsed_median_ms": round(statistics.median(row["elapsed_ms"] for row in rows), 2), "rows": rows}


def classify_cases(directory, cases, *, paired=False):
    rows = []
    for question, expected, _ in cases:
        compact = normalize_intent_question(question)
        rows.append((expected, predict_intent(compact, directory)))
    scores = {"correct": sum(expected == predicted for expected, predicted in rows), "questions": len(rows),
              "abstained": sum(predicted is None for _, predicted in rows),
              "open_as_fixed": sum(expected == "open" and predicted in {"lookup", "yoy"} for expected, predicted in rows)}
    if paired:
        scores["pairs_both_correct"] = sum(a == b and c == d for (a, b), (c, d) in zip(rows[::2], rows[1::2], strict=True))
        scores["pairs"] = len(rows) // 2
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(".annual-models/annual-intent"))
    parser.add_argument("--compare", type=Path, action="append", default=[])
    suites = {"original": CASES, "fresh": FRESH_CASES, "v2": V2_CASES, "v3": V3_CASES,
              "boundary": BOUNDARY_CASES, "boundary-v2": BOUNDARY_V2_CASES}
    parser.add_argument("--suite", choices=suites, default="original")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    candidates = [directory.resolve() for directory in [args.model, *args.compare]]
    for directory in candidates:
        if not all((directory / name).is_file() for name in ("metadata.json", "model.onnx", "tokenizer.json")):
            parser.error(f"train and export the local classifier first: {directory}")
    names = [local_intent_source(directory) for directory in candidates]
    if len(set(names)) != len(names):
        names = [directory.name for directory in candidates]
    if len(set(names)) != len(names) or {"baseline", "guard_only", "cases", "suite", "method"}.intersection(names):
        parser.error("each candidate needs a distinct artifact directory name")
    cases = suites[args.suite]
    table = "2024年年度报告\n单位：亿元\n项目 2024年 2023年\n" + "\n".join(
        f"{metric} {latest} {prior}" for metric, (prior, latest) in VALUES.items())
    # Keep the SQLite fixture under the local model directory for Windows environments
    # where the shell cannot create subdirectories in the default TEMP location.
    with tempfile.TemporaryDirectory(dir=args.model.resolve().parent) as root:
        store = AnnualReportStore(root)
        ids = [store.ingest_bytes(table.encode(), f"{company}.txt", company, 2024)["id"]
               for company in ("远山制造", "海川制造")]
        results = {"baseline": run(store, ids, "", cases)}
        for name, directory in zip(names, candidates, strict=True):
            _load_classifier(directory)
            results[name] = run(store, ids, str(directory), cases)
            results[name]["classifier"] = classify_cases(directory, cases, paired=args.suite.startswith("boundary"))
        # Separate learned routing from the added scope guards and query normalization.
        with patch("backend.annual_reports.intent.predict_intent",
                   side_effect=lambda question, _: "yoy" if re.search(_PREVIOUS, question) else "lookup"):
            guard_only = run(store, ids, str(candidates[0]), cases)
        for row in guard_only["rows"]:
            if row["route"] in LOCAL_INTENT_NAMES:
                row["route"] = "extended_rules"
        guard_only["local_model_routed"] = 0
        results["guard_only"] = guard_only
    result = {"cases": len(cases), "suite": args.suite, **results,
              "method": "Synthetic reports, held-out questions, real ONNX/BM25, simulated LLM timeout"}
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {name: value for name, value in values.items() if name != "rows"}
                      for key, values in results.items()}, indent=2))
    if any(results[name]["open_false_complete"] for name in names):
        raise SystemExit("A local classifier completed an unsupported question")


if __name__ == "__main__":
    main()
