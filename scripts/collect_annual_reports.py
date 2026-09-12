"""Discover, download and import complete Chinese A-share annual reports.

Example: python scripts/collect_annual_reports.py --codes 000651 600690 000921 --years 2024 --analyze
"""
from __future__ import annotations

import argparse
import sys
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.annual_reports.crawler import (
    CninfoCollector,
    CollectionError,
    utc_now,
    validate_security,
    write_json,
)


def save_comparison(path: Path, analyses: list[dict]) -> None:
    """Summarize only calculated values; attach each operand's own PDF page."""
    lines = ["# 年报数据对比", "", "以下金额单位为人民币亿元；同比由原文金额计算。上年数据采用所选年报中的比较口径，证据不足时保留缺口。", "",
             "| 公司 | 指标 | 年度 | 上年金额 | 本年金额 | 同比 | 年报来源 |",
             "|---|---|---|---:|---:|---:|---|"]
    for analysis in analyses:
        result = analysis["result"]
        sources = {source["document_id"]: source for source in analysis["sources"]}
        for calculation in result.get("calculations", []):
            links = []
            for operand in calculation.get("operands", []):
                source = sources.get(operand["document_id"])
                if source:
                    link = f"[{operand['year']}年·第{operand['page']}页]({source['pdf_url']}#page={operand['page']})"
                    if link not in links:
                        links.append(link)
            previous = Decimal(calculation["from_value"]) / Decimal(100000000)
            current = Decimal(calculation["to_value"]) / Decimal(100000000)
            percent = calculation.get("change_pct")
            growth = "不适用" if percent is None else f"{percent:.2f}%"
            lines.append(f"| {calculation['company']} | {calculation['metric']} | {calculation['from_year']}→{calculation['to_year']} | "
                         f"{previous:,.2f} | {current:,.2f} | {growth} | {' / '.join(links)} |")
    for analysis in analyses:
        result = analysis["result"]
        lines.extend(["", f"## {analysis['sources'][0]['company']}", "",
                      f"状态：{result['status']}；实际检索：{result['retrieval_mode']}。"])
        for gap in result.get("metrics", {}).get("evidence_gaps", []):
            lines.append(f"- 证据缺口：{gap}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", nargs="+", required=True, help="沪深 A 股六位股票代码")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8001", help="本机年报服务地址")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/annual_reports/collected")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--analyze", action="store_true", help="导入后逐公司运行 LangGraph 并保存 JSON/Markdown")
    parser.add_argument("--mode", choices=("hybrid", "bm25"), default="hybrid")
    args = parser.parse_args(argv)
    codes, years = list(dict.fromkeys(args.codes)), sorted(set(args.years))
    if len(codes) * len(years) > 12:
        parser.error("每批最多 12 份年报，请分批执行。")
    if args.download_only and args.analyze:
        parser.error("--analyze 需要导入服务，不能与 --download-only 同用。")
    base = args.api_base.rstrip("/")
    address = urlsplit(base)
    if (address.scheme != "http" or address.hostname not in {"127.0.0.1", "localhost", "::1"}
            or address.username or address.password or address.path or address.query or address.fragment):
        parser.error("--api-base 须为本机 HTTP 服务根地址，例如 http://127.0.0.1:8001。")
    try:
        for code in codes:
            for year in years:
                validate_security(code, year)
    except CollectionError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = utc_now().replace(":", "").replace(".", "-")
    manifest_path = args.output_dir / f"manifest-{run_id}.json"
    manifest = {"created_at": utc_now(), "source": "巨潮资讯公开公告", "reports": [], "errors": [], "analyses": []}
    completed_analyses = []
    # Never send local uploads/API requests through ambient corporate proxies.
    with httpx.Client(timeout=httpx.Timeout(1800, connect=5), trust_env=False) as api, CninfoCollector() as crawler:
        if not args.download_only:
            try:
                api.get(base + "/api/annual-reports/health", timeout=10).raise_for_status()
            except httpx.HTTPError:
                print("年报服务未就绪，请先运行 scripts/start-annual-reports.ps1 -Semantic。", file=sys.stderr)
                return 1
        for code in codes:
            for year in years:
                entry = {"stock_code": code, "report_year": year}
                try:
                    report = crawler.discover(code, year)
                    print(f"发现 {report.company} {year}：{report.title}", flush=True)
                    entry = crawler.download(report, args.output_dir)
                    manifest["reports"].append(entry)
                    write_json(manifest_path, manifest)
                    print(f"下载核验通过：{entry['page_count']} 页；SHA256 {entry['content_sha256'][:16]}", flush=True)
                    if not args.download_only:
                        with Path(entry["local_path"]).open("rb") as pdf:
                            response = api.post(base + "/api/annual-reports/documents",
                                                data={"company": report.company, "year": str(year)},
                                                files={"file": (f"{report.company}_{year}年年度报告.pdf", pdf, "application/pdf")})
                        response.raise_for_status()
                        document = response.json()
                        entry.update(document_id=document["id"], imported_at=utc_now(),
                                     document=document, import_status="complete")
                        print(f"已导入 {report.company}：{document['chunk_count']} 个文档块", flush=True)
                except (CollectionError, httpx.HTTPError, OSError, ValueError, KeyError) as exc:
                    # Public URLs contain no credentials; response bodies from APIs are not dumped.
                    error = {"stock_code": code, "report_year": year, "type": type(exc).__name__, "message": str(exc)}
                    manifest["errors"].append(error)
                    print(f"失败 {code}/{year}: {exc}", file=sys.stderr, flush=True)
                write_json(manifest_path, manifest)
        if args.analyze:
            for code in codes:
                reports = [r for r in manifest["reports"] if r["stock_code"] == code and r.get("document_id")]
                if not reports:
                    continue
                company = reports[0]["company"]
                first, last = min(r["report_year"] for r in reports) - 1, max(r["report_year"] for r in reports)
                question = f"比较{company}{first}年至{last}年的营业收入和经营活动产生的现金流量净额，计算同比变化并引用年报原文。"
                try:
                    print(f"分析 {company}（首次语义索引可能需要数分钟）…", flush=True)
                    response = api.post(base + "/api/annual-reports/analyze", json={
                        "question": question, "document_ids": [r["document_id"] for r in reports],
                        "mode": args.mode, "max_retries": 2,
                    })
                    response.raise_for_status()
                    result = response.json()
                    analysis_path = args.output_dir / f"analysis-{code}-{run_id}.json"
                    full_analysis = {"question": question, "sources": reports, "result": result}
                    write_json(analysis_path, full_analysis)
                    completed_analyses.append(full_analysis)
                    analysis_path.with_suffix(".md").write_text(
                        f"# {company}年报分析\n\n{question}\n\n"
                        + "\n".join(f"- 来源：[{r['title']}]({r['pdf_url']})" for r in reports)
                        + f"\n\n状态：{result['status']}；检索：{result['retrieval_mode']}\n\n{result['answer']}\n",
                        encoding="utf-8",
                    )
                    manifest["analyses"].append({"stock_code": code, "company": company,
                        "status": result["status"], "retrieval_mode": result["retrieval_mode"],
                        "output": str(analysis_path.resolve()), "evidence_gaps": result.get("metrics", {}).get("evidence_gaps", [])})
                    print(f"{company}：{result['status']}；{len(result.get('calculations', []))} 项计算", flush=True)
                except (httpx.HTTPError, OSError, ValueError, KeyError) as exc:
                    manifest["errors"].append({"stock_code": code, "stage": "analysis", "message": str(exc)})
                    print(f"分析失败 {company}: {exc}", file=sys.stderr, flush=True)
                write_json(manifest_path, manifest)
        if completed_analyses:
            comparison_path = args.output_dir / f"comparison-{run_id}.md"
            save_comparison(comparison_path, completed_analyses)
            (args.output_dir / "comparison-latest.md").write_text(
                comparison_path.read_text(encoding="utf-8"), encoding="utf-8",
            )
            manifest["comparison_path"] = str(comparison_path.resolve())
            write_json(manifest_path, manifest)
            print(f"对比报告：{comparison_path.resolve()}", flush=True)
    print(f"抓取记录：{manifest_path.resolve()}", flush=True)
    return 1 if manifest["errors"] or any(a["status"] != "complete" or a["retrieval_mode"] != args.mode
                                          for a in manifest["analyses"]) else 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
