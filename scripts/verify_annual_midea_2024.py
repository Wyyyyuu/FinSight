"""Reproduce three factual smoke checks against a locally supplied Midea 2024 PDF.

This is a single-report smoke verification, NOT an estimate of business accuracy
or generalization. Parsing, retrieval, LangGraph routing, and arithmetic run for
real. Generative-model access is disabled, credentials are not read, and optional
embedding models must already exist locally. The PDF is never copied/downloaded.

Example:
    python scripts/verify_annual_midea_2024.py --pdf C:/reports/midea-2024.pdf
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DIRECTORY = ROOT / "data" / "annual_reports" / "verification"
REVENUE = "营业收入"
CASH_FLOW = "经营活动产生的现金流量净额"
EXPECTED_VALUES = {
    REVENUE: {2022: "343917531000", 2023: "372037280000", 2024: "407149600000"},
    CASH_FLOW: {2022: "34657828000", 2023: "57902611000", 2024: "60511572000"},
}
EXPECTED_GROWTH = {(REVENUE, 2022, 2023): 8.18, (REVENUE, 2023, 2024): 9.44,
                   (CASH_FLOW, 2022, 2023): 67.07, (CASH_FLOW, 2023, 2024): 4.51}


def _check(checks: list[dict[str, Any]], name: str, actual: Any, expected: Any) -> None:
    checks.append({"name": name, "passed": actual == expected, "expected": expected, "actual": actual})


def _decimal_equal(actual: Any, expected: Any) -> bool:
    try:
        return Decimal(str(actual)) == Decimal(str(expected))
    except ArithmeticError:
        return False


def _check_bound_citations(checks: list[dict[str, Any]], result: dict[str, Any], document_id: str) -> None:
    citations = result.get("citations", [])
    _check(checks, "has_citations", bool(citations), True)
    _check(checks, "citations_only_selected_2024_report",
           all(c.get("document_id") == document_id and c.get("year") == 2024 for c in citations), True)
    sources = {c.get("label"): c for c in citations}
    operands = [operand for calc in result.get("calculations", []) for operand in calc.get("operands", [])]
    _check(checks, "calculation_operands_bound_to_returned_citations",
           all(operand.get("citation") in sources
               and sources[operand["citation"]].get("page") == operand.get("page")
               and sources[operand["citation"]].get("document_id") == operand.get("document_id")
               for operand in operands), True)
    _check(checks, "no_generation_model", result.get("metrics", {}).get("answer_mode"), "extractive")


def _check_comparison(result: dict[str, Any], document_id: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    _check(checks, "status", result.get("status"), "complete")
    calculations = result.get("calculations", [])
    _check(checks, "four_comparisons", len(calculations), 4)
    by_period = {(c.get("metric"), c.get("from_year"), c.get("to_year")): c for c in calculations}
    for key, expected in EXPECTED_GROWTH.items():
        calc = by_period.get(key, {})
        name = f"{key[0]}_{key[1]}_{key[2]}"
        _check(checks, f"{name}_growth_pct", calc.get("change_pct"), expected)
        expected_delta = Decimal(EXPECTED_VALUES[key[0]][key[2]]) - Decimal(EXPECTED_VALUES[key[0]][key[1]])
        _check(checks, f"{name}_delta_yuan", str(calc.get("delta")), str(expected_delta))
    operands = [operand for calc in calculations for operand in calc.get("operands", [])]
    for metric, values in EXPECTED_VALUES.items():
        for year, expected in values.items():
            observed = [op for op in operands if op.get("metric") == metric and op.get("year") == year]
            _check(checks, f"{metric}_{year}_value_yuan",
                   bool(observed) and all(op.get("unit") == "元" and _decimal_equal(op.get("value"), expected) for op in observed), True)
    _check(checks, "missing_years", result.get("metrics", {}).get("missing_years"), [])
    _check_bound_citations(checks, result, document_id)
    return checks


def _check_single_fact(result: dict[str, Any], document_id: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    _check(checks, "status", result.get("status"), "complete")
    _check(checks, "single_factual_year", result.get("metrics", {}).get("requested_years"), [2023])
    _check(checks, "no_unrequested_comparison", result.get("calculations"), [])
    # Verify the public answer independently, not through the workflow's own numeric extractor.
    match = re.search(r"美的集团\s+2023年营业收入：([+\-]?[\d,.]+)(亿元|万元|元)\s+\[(S\d+)\]", result.get("answer", ""))
    amount = None
    label = None
    if match:
        amount = str(Decimal(match.group(1).replace(",", "")) * {"亿元": Decimal(100000000), "万元": Decimal(10000), "元": Decimal(1)}[match.group(2)])
        label = match.group(3)
    _check(checks, "2023_revenue_is_372037280000_yuan", _decimal_equal(amount, "372037280000"), True)
    sources = {citation.get("label"): citation for citation in result.get("citations", [])}
    source = sources.get(label, {})
    _check(checks, "2023_fact_from_2024_report_page_9", (source.get("year"), source.get("page")), (2024, 9))
    _check(checks, "raw_source_has_2023_revenue_thousands", "372,037,280" in source.get("text", ""), True)
    _check_bound_citations(checks, result, document_id)
    return checks


def _check_wrong_premise(result: dict[str, Any], document_id: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    _check(checks, "status", result.get("status"), "insufficient_evidence")
    calc = next((item for item in result.get("calculations", []) if item.get("metric") == CASH_FLOW
                 and item.get("from_year") == 2023 and item.get("to_year") == 2024), {})
    _check(checks, "cash_flow_actually_grew_4_51_pct", calc.get("change_pct"), 4.51)
    _check(checks, "2023_cash_flow_yuan", _decimal_equal(calc.get("from_value"), "57902611000"), True)
    _check(checks, "2024_cash_flow_yuan", _decimal_equal(calc.get("to_value"), "60511572000"), True)
    _check(checks, "positive_delta_yuan", _decimal_equal(calc.get("delta"), "2608961000"), True)
    gaps = result.get("metrics", {}).get("evidence_gaps", [])
    _check(checks, "wrong_decline_premise_explicitly_corrected",
           any("问题前提与证据不符" in gap and "实际增加" in gap and "下降" in gap for gap in gaps), True)
    _check(checks, "correction_visible_in_answer", "问题前提与证据不符" in result.get("answer", ""), True)
    _check_bound_citations(checks, result, document_id)
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True, type=Path, help="Local Midea 2024 annual report PDF (295 pages)")
    parser.add_argument("--mode", choices=("bm25", "hybrid"), default="bm25")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIRECTORY / "midea_2024_smoke_store")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY / "midea_2024_smoke_result.json")
    args = parser.parse_args(argv)
    pdf = args.pdf.resolve()
    output = args.output.resolve()
    if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
        parser.error("--pdf must name an existing local PDF file")
    if output == pdf:
        parser.error("--output cannot overwrite the source PDF")

    started = time.perf_counter()
    report: dict[str, Any] = {
        "verification_kind": "single_report_smoke",
        "scope_note": "单份美的2024年报的三个冒烟用例；不代表泛化业务准确率。",
        "created_at": datetime.now(timezone.utc).isoformat(), "requested_mode": args.mode,
        "pdf": str(pdf), "data_dir": str(args.data_dir.resolve()),
        "generation": "disabled; no model credentials read", "downloads": "disabled",
        "setup_assertions": [], "cases": [],
    }
    previous_download_setting = os.environ.get("ANNUAL_REPORT_ALLOW_MODEL_DOWNLOAD")
    previous_embedding_provider = os.environ.get("ANNUAL_REPORT_EMBEDDING_PROVIDER")
    os.environ["ANNUAL_REPORT_ALLOW_MODEL_DOWNLOAD"] = "0"
    if args.mode == "hybrid":
        os.environ["ANNUAL_REPORT_EMBEDDING_PROVIDER"] = "fastembed"
    try:
        from backend.annual_reports import documents, workflow

        content = pdf.read_bytes()
        report["pdf_sha256"] = hashlib.sha256(content).hexdigest()
        ingest_started = time.perf_counter()
        store = documents.AnnualReportStore(args.data_dir)
        document = store.ingest_bytes(content, pdf.name, "美的集团", 2024)
        report["ingest_elapsed_ms"] = round((time.perf_counter() - ingest_started) * 1000, 2)
        report["document"] = document
        _check(report["setup_assertions"], "expected_295_page_report", document.get("page_count"), 295)
        cases = [
            ("three_year_comparison", "比较2022至2024年营业收入和经营现金流", _check_comparison),
            ("2023_revenue_in_2024_report", "2023年营业收入是多少", _check_single_fact),
            ("incorrect_cash_flow_decline", "为什么2024年经营现金流下降", _check_wrong_premise),
        ]
        # Only the optional generator boundary is replaced. Real PDF, store, graph,
        # retrieval, citations and arithmetic remain active. No credential lookup occurs.
        with patch.object(workflow, "_request_model_answer", return_value=None):
            for case_id, question, checker in cases:
                case_started = time.perf_counter()
                case: dict[str, Any] = {"id": case_id, "question": question, "assertions": []}
                try:
                    result = workflow.analyze_reports(store, question, [document["id"]], mode=args.mode)
                    case.update({"status": result.get("status"), "retrieval_mode": result.get("retrieval_mode"),
                                 "assertions": checker(result, document["id"]), "result": result})
                except Exception as exc:  # noqa: BLE001 - preserve all case failures in the report and exit nonzero.
                    case.update({"status": "error", "retrieval_mode": None, "error_type": type(exc).__name__})
                    _check(case["assertions"], "case_completed_without_exception", False, True)
                _check(case["assertions"], "actual_retrieval_mode_matches_requested_mode", case.get("retrieval_mode"), args.mode)
                case["elapsed_ms"] = round((time.perf_counter() - case_started) * 1000, 2)
                case["passed"] = all(item["passed"] for item in case["assertions"])
                report["cases"].append(case)
                print(f"{'PASS' if case['passed'] else 'FAIL'} {case_id}: {case['status']}, "
                      f"retrieval={case['retrieval_mode']}, {case['elapsed_ms']:.2f} ms", flush=True)
                for assertion in case["assertions"]:
                    print(f"  {'PASS' if assertion['passed'] else 'FAIL'} {assertion['name']}", flush=True)
    except Exception as exc:  # noqa: BLE001 - setup failures must still produce an auditable JSON result.
        report["setup_error_type"] = type(exc).__name__
        _check(report["setup_assertions"], "setup_completed_without_exception", False, True)
    finally:
        if previous_download_setting is None:
            os.environ.pop("ANNUAL_REPORT_ALLOW_MODEL_DOWNLOAD", None)
        else:
            os.environ["ANNUAL_REPORT_ALLOW_MODEL_DOWNLOAD"] = previous_download_setting
        if args.mode == "hybrid":
            if previous_embedding_provider is None:
                os.environ.pop("ANNUAL_REPORT_EMBEDDING_PROVIDER", None)
            else:
                os.environ["ANNUAL_REPORT_EMBEDDING_PROVIDER"] = previous_embedding_provider

    assertions = [*report["setup_assertions"], *(a for case in report["cases"] for a in case["assertions"])]
    report["summary"] = {
        "cases_total": len(report["cases"]), "cases_passed": sum(case["passed"] for case in report["cases"]),
        "assertions_total": len(assertions), "assertions_passed": sum(item["passed"] for item in assertions),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "passed": len(report["cases"]) == 3 and all(item["passed"] for item in assertions),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(f"{'PASS' if summary['passed'] else 'FAIL'} single-report smoke: "
          f"{summary['cases_passed']}/{summary['cases_total']} cases, "
          f"{summary['assertions_passed']}/{summary['assertions_total']} assertions. JSON: {output}")
    print(report["scope_note"])
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
