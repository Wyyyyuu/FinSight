"""Collector CLI checks using an actual HTTP transport boundary and local files."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from scripts import collect_annual_reports as cli


class FakeCollector:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def discover(self, code, year):
        from backend.annual_reports.crawler import Announcement
        return Announcement(code, "格力电器", year, "2024年年度报告", "123", "2025-04-28T00:00:00+00:00",
                            "https://static.cninfo.com.cn/finalpage/2025-04-28/123.PDF")

    def download(self, report, directory):
        path = directory / "123.pdf"
        path.write_bytes(b"%PDF-test fixture")
        return {"stock_code": report.stock_code, "company": report.company, "report_year": report.report_year,
                "title": report.title, "pdf_url": report.pdf_url, "local_path": str(path),
                "page_count": 20, "content_sha256": "0" * 64}


def configure(monkeypatch, *, analysis_status="complete", retrieval="hybrid", health_status=200):
    requests = []
    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/health"):
            return httpx.Response(health_status, json={"status": "ok"})
        if request.url.path.endswith("/documents"):
            assert b"%PDF-test fixture" in request.content
            assert "格力电器".encode() in request.content
            return httpx.Response(201, json={"id": "doc-123", "chunk_count": 22})
        assert request.url.path.endswith("/analyze")
        assert json.loads(request.content)["document_ids"] == ["doc-123"]
        return httpx.Response(200, json={"status": analysis_status, "retrieval_mode": retrieval,
            "answer": "来源可核验", "calculations": [],
            "metrics": {"evidence_gaps": [] if analysis_status == "complete" else ["缺少明确年度证据"]}})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(cli.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(cli, "CninfoCollector", FakeCollector)
    return requests


def test_cli_import_analysis_and_manifest_are_connected(tmp_path, monkeypatch):
    requests = configure(monkeypatch)
    assert cli.main(["--codes", "000651", "--years", "2024", "--analyze", "--output-dir", str(tmp_path)]) == 0
    manifest = json.loads(next(tmp_path.glob("manifest-*.json")).read_text(encoding="utf-8"))
    assert manifest["reports"][0]["document_id"] == "doc-123"
    assert len(manifest["analyses"]) == 1
    assert Path(manifest["comparison_path"]).is_file()
    assert len(requests) == 3


@pytest.mark.parametrize("status,retrieval", [("insufficient_evidence", "hybrid"), ("complete", "bm25")])
def test_cli_does_not_claim_success_for_gaps_or_retrieval_fallback(tmp_path, monkeypatch, status, retrieval):
    configure(monkeypatch, analysis_status=status, retrieval=retrieval)
    assert cli.main(["--codes", "000651", "--years", "2024", "--analyze", "--output-dir", str(tmp_path)]) == 1
    manifest = json.loads(next(tmp_path.glob("manifest-*.json")).read_text(encoding="utf-8"))
    assert manifest["analyses"][0]["status"] == status
    assert manifest["analyses"][0]["retrieval_mode"] == retrieval


def test_service_unavailable_does_not_start_collection(tmp_path, monkeypatch):
    requests = configure(monkeypatch, health_status=503)
    assert cli.main(["--codes", "000651", "--years", "2024", "--output-dir", str(tmp_path)]) == 1
    assert len(requests) == 1
    assert list(tmp_path.glob("*.pdf")) == []


def test_download_only_skips_local_api(tmp_path, monkeypatch):
    requests = configure(monkeypatch)
    assert cli.main(["--codes", "000651", "--years", "2024", "--download-only", "--output-dir", str(tmp_path)]) == 0
    assert requests == []
    manifest = json.loads(next(tmp_path.glob("manifest-*.json")).read_text(encoding="utf-8"))
    assert "document_id" not in manifest["reports"][0]


def test_comparison_values_keep_pdf_page_and_decimal_precision(tmp_path):
    target = tmp_path / "comparison.md"
    cli.save_comparison(target, [{"sources": [{"document_id": "a", "company": "测试公司", "pdf_url": "https://example.org/report.pdf"}],
        "result": {"status": "complete", "retrieval_mode": "hybrid", "metrics": {}, "calculations": [{
            "company": "测试公司", "metric": "营业收入", "from_year": 2023, "to_year": 2024,
            "from_value": "10000000000.00", "to_value": "12000000000.00", "change_pct": 20.0,
            "operands": [{"document_id": "a", "year": 2023, "page": 9}, {"document_id": "a", "year": 2024, "page": 9}],
        }]}}])
    text = target.read_text(encoding="utf-8")
    assert "100.00 | 120.00 | 20.00%" in text
    assert "report.pdf#page=9" in text


@pytest.mark.parametrize("base", ["https://example.org", "http://127.0.0.1/api", "http://user:secret@localhost:8001"])
def test_local_upload_cannot_be_redirected_by_cli_target(base, tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["--codes", "000651", "--years", "2024", "--api-base", base, "--output-dir", str(tmp_path)])
