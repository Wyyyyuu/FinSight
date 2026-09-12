"""Source identity, bounded network behavior and real PDF collector regression."""
from __future__ import annotations

import io
from dataclasses import replace

import httpx
import pytest
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

from backend.annual_reports.crawler import (
    CninfoCollector,
    CollectionError,
    is_full_report,
    pdf_url,
    select_announcement,
    validate_security,
    verify_pdf,
)


def row(**overrides):
    return {"secCode": "000651", "secName": "格力电器", "announcementTitle": "2024年年度报告",
            "announcementId": "1223330631", "announcementTime": 1745769600000,
            "adjunctUrl": "finalpage/2025-04-28/1223330631.PDF", "adjunctType": "PDF", **overrides}


@pytest.fixture
def report():
    return select_announcement([row()], "000651", 2024)


@pytest.fixture
def real_pdf():
    stream = io.BytesIO()
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    pdf = canvas.Canvas(stream)
    for page in range(20):
        pdf.setFont("STSong-Light", 12)
        pdf.drawString(40, 760, "000651 格力电器 2024年年度报告" if page == 0 else f"第{page + 1}页 财务报告")
        pdf.showPage()
    pdf.save()
    return stream.getvalue()


@pytest.mark.parametrize("title", ["2024年年度报告", "格力电器：2024 年年度报告全文", "2024年年度报告（修订版）"])
def test_complete_chinese_reports(title):
    assert is_full_report(title, 2024)


@pytest.mark.parametrize("title", ["2024年年度报告摘要", "2024年年度报告（英文版）",
    "关于2024年年度报告的更正公告", "2023年年度报告", "2024年半年度报告", "2024年年度报告取消公告",
    "2024年年度报告审计意见", "2024年年度报告（修订说明）"])
def test_reject_summaries_other_periods_and_announcements(title):
    assert not is_full_report(title, 2024)


def test_exact_security_and_latest_full_replacement_selected():
    chosen = select_announcement([
        row(secCode="000333", announcementTime=1745969600000),
        row(announcementTitle="2024年年度报告摘要", announcementTime=1745969600000),
        row(), row(announcementTitle="<em>2024年</em>年度报告（修订版）", announcementTime=1745869600000),
    ], "000651", 2024)
    assert chosen.title == "2024年年度报告（修订版）"
    assert chosen.stock_code == "000651"
    assert chosen.published_at.endswith("+00:00")


@pytest.mark.parametrize("path", ["https://evil.example/r.pdf", "//evil.example/r.pdf", "../r.pdf",
    "finalpage/2025-04-28/1.PDF?url=http://localhost", "finalpage/2025-04-28/../1.PDF"])
def test_attachment_cannot_escape_official_domain(path):
    with pytest.raises(CollectionError):
        pdf_url(path)


@pytest.mark.parametrize("code,year", [("../651", 2024), ("000651", 3000), ("000651", True), ("A00651", 2024)])
def test_security_arguments(code, year):
    with pytest.raises(CollectionError):
        validate_security(code, year)


def test_pdf_cover_verifies_actual_year_and_company(report, real_pdf):
    checked = verify_pdf(real_pdf, report)
    assert checked["page_count"] == 20
    assert len(checked["content_sha256"]) == 64
    for other in (replace(report, stock_code="600690", company="海尔智家"), replace(report, report_year=2023)):
        with pytest.raises(CollectionError):
            verify_pdf(real_pdf, other)
    with pytest.raises(CollectionError):
        verify_pdf(b"<html>verification required</html>", report)


def collector(handler, monkeypatch, **options):
    instance = CninfoCollector(httpx.Client(transport=httpx.MockTransport(handler)), **options)
    monkeypatch.setattr(instance, "_pace", lambda: None)
    return instance


def test_paginated_discovery_and_not_accepting_incomplete_results(monkeypatch):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"announcements": [row()] if len(requests) == 2 else [],
                                         "hasMore": len(requests) == 1})
    crawler = collector(handler, monkeypatch)
    assert crawler.discover("000651", 2024).company == "格力电器"
    assert len(requests) == 2 and b"pageNum=2" in requests[1].content
    capped = collector(lambda req: httpx.Response(200, json={"announcements": [row()], "hasMore": True}),
                       monkeypatch, max_pages=1)
    with pytest.raises(CollectionError, match="上限"):
        capped.discover("000651", 2024)


@pytest.mark.parametrize("status", [401, 403, 429])
def test_access_limits_stop_without_bypass_or_repeated_requests(monkeypatch, status):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status)
    with pytest.raises(CollectionError):
        collector(handler, monkeypatch).discover("000651", 2024)
    assert len(calls) == 1


def test_download_cache_hash_and_no_duplicate_http(tmp_path, monkeypatch, report, real_pdf):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=real_pdf)
    crawler = collector(handler, monkeypatch)
    first = crawler.download(report, tmp_path)
    second = crawler.download(report, tmp_path)
    assert len(calls) == 1
    assert first["content_sha256"] == second["content_sha256"]
    assert second["download_cached"] and not first["download_cached"]
    assert second["fetched_at"] == first["fetched_at"]
    assert second["verified_at"] >= first["verified_at"]
    assert second["pdf_url"].startswith("https://static.cninfo.com.cn/")


@pytest.mark.parametrize("response", [httpx.Response(302, headers={"location": "http://127.0.0.1/private"}),
    httpx.Response(200, headers={"content-length": "21000000"}), httpx.Response(200, content=b"bad pdf")])
def test_redirect_oversize_invalid_pdf_does_not_create_file(tmp_path, monkeypatch, report, response):
    crawler = collector(lambda request: response, monkeypatch)
    with pytest.raises(CollectionError):
        crawler.download(report, tmp_path)
    assert list(tmp_path.rglob("*.pdf")) == []


def test_forged_attachment_dataclass_is_rejected_before_network(tmp_path, monkeypatch, report):
    crawler = collector(lambda request: pytest.fail("network must not run"), monkeypatch)
    for url in ("http://127.0.0.1/file.pdf", report.pdf_url + "?next=bad"):
        with pytest.raises(CollectionError):
            crawler.download(replace(report, pdf_url=url), tmp_path)


def test_custom_client_cannot_enable_cross_domain_redirects(tmp_path, monkeypatch, report):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://example.invalid/leak"})
    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    crawler = CninfoCollector(client)
    monkeypatch.setattr(crawler, "_pace", lambda: None)
    with pytest.raises(CollectionError):
        crawler.download(report, tmp_path)
    with pytest.raises(CollectionError):
        crawler.discover("000651", 2024)
    assert len(calls) == 2
    assert all("example.invalid" not in url for url in calls)


def test_invalid_source_payload_fails_closed(monkeypatch):
    crawler = collector(lambda req: httpx.Response(200, content=b"<html>challenge</html>"), monkeypatch)
    with pytest.raises(CollectionError, match="JSON"):
        crawler.discover("000651", 2024)
    with pytest.raises(CollectionError, match="未找到"):
        select_announcement([row(secCode="000333")], "000651", 2024)


def test_ingest_downloaded_pdf_retains_real_pages(tmp_path, monkeypatch, report, real_pdf):
    from backend.annual_reports.documents import AnnualReportStore
    crawler = collector(lambda req: httpx.Response(200, content=real_pdf), monkeypatch)
    entry = crawler.download(report, tmp_path / "downloads")
    store = AnnualReportStore(tmp_path / "local")
    doc = store.ingest_bytes(real_pdf, "格力电器2024.pdf", report.company, report.report_year)
    assert doc["page_count"] == entry["page_count"] == 20
    assert "000651" in store.get_page(doc["id"], 1)["text"]
    assert store.ingest_bytes(real_pdf, "格力电器2024.pdf", report.company, report.report_year)["deduplicated"]
