from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.annual_reports.app import create_app
from backend.annual_reports.documents import AnnualReportStorageError, AnnualReportStore
from backend.api.annual_report_router import _store_at


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNUAL_REPORT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANNUAL_REPORT_EMBEDDING_PROVIDER", "bm25")
    _store_at.cache_clear()
    with TestClient(
        create_app(), base_url="http://127.0.0.1", client=("127.0.0.1", 51000)
    ) as test_client:
        yield test_client
    _store_at.cache_clear()


def upload(
    client,
    company="甲公司",
    year=2024,
    text="2024年营业收入为 150 亿元。经营活动产生的现金流量净额为 9 亿元。",
):
    return client.post(
        "/api/annual-reports/documents",
        data={"company": company, "year": year},
        files={"file": ("年报.md", text.encode(), "text/markdown")},
    )


def test_upload_list_page_and_duplicate(client):
    first = upload(client)
    assert first.status_code == 201, first.text
    doc = first.json()
    assert upload(client).json()["id"] == doc["id"]
    assert len(client.get("/api/annual-reports/documents").json()["documents"]) == 1
    page = client.get(f"/api/annual-reports/documents/{doc['id']}/pages/1")
    assert page.status_code == 200
    assert "150" in page.json()["text"]
    assert page.json()["company"] == "甲公司"
    assert (
        client.get(f"/api/annual-reports/documents/{doc['id']}/pages/9999").status_code
        == 404
    )


def test_end_to_end_demo_analysis_and_citations(client):
    demo = client.post("/api/annual-reports/demo")
    assert demo.status_code == 201
    docs = demo.json()["documents"]
    ids = [doc["id"] for doc in docs]
    response = client.post(
        "/api/annual-reports/analyze",
        json={
            "question": "比较2023年和2024年营业收入与经营活动产生的现金流量净额变化",
            "document_ids": ids,
            "mode": "bm25",
        },
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["trace"]
    assert result["citations"]
    assert all(c["document_id"] in ids and c["page"] >= 1 for c in result["citations"])
    assert all(c["year"] in {2023, 2024} for c in result["citations"])
    assert result["calculations"]


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "收入多少", "document_ids": []},
        {"question": "   ", "document_ids": ["x"]},
        {"question": "收入多少", "document_ids": ["x"], "max_retries": 100},
        {"question": "收入多少", "document_ids": ["x"], "years": []},
        {"question": "收入多少", "document_ids": ["x"], "mode": "unbounded"},
    ],
)
def test_analysis_validation(client, payload):
    assert client.post("/api/annual-reports/analyze", json=payload).status_code == 422


def test_unknown_document_fails_closed(client):
    assert (
        client.post(
            "/api/annual-reports/analyze",
            json={"question": "收入多少", "document_ids": ["missing"]},
        ).status_code
        == 404
    )
    assert (
        client.get("/api/annual-reports/documents/missing/pages/1").status_code == 404
    )


def test_upload_rejections(client):
    assert upload(client, company="  ").status_code == 422
    assert upload(client, year=1800).status_code == 422
    malformed = client.post(
        "/api/annual-reports/documents",
        data={"company": "甲", "year": 2024},
        files={"file": ("broken.pdf", b"%PDF-1.4 invalid", "application/pdf")},
    )
    assert malformed.status_code == 422


def test_demo_is_idempotent_and_marked_synthetic(client):
    first = client.post("/api/annual-reports/demo").json()
    second = client.post("/api/annual-reports/demo").json()
    assert {d["id"] for d in first["documents"]} == {
        d["id"] for d in second["documents"]
    }
    assert len(client.get("/api/annual-reports/documents").json()["documents"]) == 3
    assert all("虚构" in d["company"] for d in first["documents"])


@pytest.mark.parametrize(
    "header", ["forwarded", "x-forwarded-for", "x-real-ip", "cf-connecting-ip"]
)
def test_local_only_and_proxy_headers(client, header):
    assert (
        client.get(
            "/api/annual-reports/documents", headers={header: "8.8.8.8"}
        ).status_code
        == 401
    )
    with TestClient(create_app(), client=("203.0.113.1", 12345)) as remote:
        assert remote.get("/api/annual-reports/documents").status_code == 401


def test_authenticated_workspaces_are_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNUAL_REPORT_DATA_DIR", str(tmp_path))
    app = create_app()

    @app.middleware("http")
    async def test_auth(request, call_next):
        # This test-only authenticator models the verified identity from main.py.
        request.state.rag_authenticated_user = {
            "user_id": request.headers.get("test-user", "alice")
        }
        return await call_next(request)

    with TestClient(app) as client:
        doc = upload(client).json()
        assert client.get("/api/annual-reports/documents").json()["documents"]
        bob = {"test-user": "bob"}
        assert (
            client.get("/api/annual-reports/documents", headers=bob).json()["documents"]
            == []
        )
        assert (
            client.get(
                f"/api/annual-reports/documents/{doc['id']}/pages/1", headers=bob
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/annual-reports/analyze",
                headers=bob,
                json={"question": "收入多少", "document_ids": [doc["id"]]},
            ).status_code
            == 404
        )


def test_cors_and_unknown_api_path(client):
    result = client.options(
        "/api/annual-reports/documents",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-session-id,content-type",
        },
    )
    assert result.status_code == 200
    assert result.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert client.get("/api/no-such-endpoint").status_code == 404
    assert client.get("/health").json()["status"] == "ok"


def test_storage_failure_returns_retryable_error_without_internal_paths(
    client, monkeypatch
):
    def broken_list(self):
        raise AnnualReportStorageError("secret/internal/database.sqlite")

    monkeypatch.setattr(AnnualReportStore, "list_documents", broken_list)
    response = client.get("/api/annual-reports/documents")
    assert response.status_code == 503
    assert "secret" not in response.text


def test_health_reports_actual_semantic_availability(client):
    state = client.get("/api/annual-reports/health").json()
    assert state["retrieval"]["loaded"] is False
    assert state["retrieval"]["provider"] == "bm25"


@pytest.mark.parametrize(
    "headers",
    [
        {"host": "untrusted.example"},
        {"origin": "https://untrusted.example"},
        {"origin": "null"},
    ],
)
def test_anonymous_local_reports_reject_untrusted_browser_origins(client, headers):
    assert client.post("/api/annual-reports/demo", headers=headers).status_code == 401
