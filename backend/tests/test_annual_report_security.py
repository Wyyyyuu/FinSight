"""Exercise the shared annual-report router behind the real FinSight middleware."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def main_client(tmp_path, monkeypatch):
    from backend.api import main
    from backend.api.annual_report_router import _store_at

    for key in ("SUPABASE_URL", "VITE_SUPABASE_URL", "API_AUTH_KEY", "API_AUTH_KEYS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("API_AUTH_ENABLED", "false")
    monkeypatch.setenv("RAG_OBSERVABILITY_DEV_AUTH_ENABLED", "true")
    monkeypatch.setenv("RAG_OBSERVABILITY_DEV_ACCESS_TOKEN", "annual-test-token")
    monkeypatch.setenv("RAG_OBSERVABILITY_DEV_USER_ID", "alice")
    monkeypatch.setenv("ANNUAL_REPORT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ANNUAL_REPORT_EMBEDDING_PROVIDER", "bm25")
    monkeypatch.setattr(
        main, "_rate_limiter", main.SimpleRateLimiter(100, 60, enabled=False)
    )
    _store_at.cache_clear()
    # No lifespan context: the API boundary needs no finance scheduler or network service.
    client = TestClient(
        main.app, base_url="http://127.0.0.1", client=("127.0.0.1", 51000)
    )
    yield client
    client.close()
    _store_at.cache_clear()


def test_configured_auth_rejects_anonymous_local_and_forged_identity(main_client):
    for headers in (
        {},
        {"X-User-Id": "alice", "X-Session-Id": "alice"},
        {"Authorization": "Bearer wrong"},
    ):
        assert (
            main_client.get(
                "/api/annual-reports/documents", headers=headers
            ).status_code
            == 401
        )


def test_verified_users_cannot_read_or_analyze_another_workspace(
    main_client, monkeypatch
):
    headers = {"Authorization": "Bearer annual-test-token", "X-Session-Id": "bob"}
    response = main_client.post("/api/annual-reports/demo", headers=headers)
    assert response.status_code == 201
    document = response.json()["documents"][0]
    monkeypatch.setenv("RAG_OBSERVABILITY_DEV_USER_ID", "bob")
    headers["X-Session-Id"] = "alice"
    assert main_client.get("/api/annual-reports/documents", headers=headers).json() == {
        "documents": []
    }
    assert (
        main_client.get(
            f"/api/annual-reports/documents/{document['id']}/pages/1", headers=headers
        ).status_code
        == 404
    )
    result = main_client.post(
        "/api/annual-reports/analyze",
        headers=headers,
        json={"question": "营业收入多少", "document_ids": [document["id"]]},
    )
    assert result.status_code == 404


def test_api_key_gate_preserves_verified_bearer_access(main_client, monkeypatch):
    monkeypatch.setenv("API_AUTH_ENABLED", "true")
    monkeypatch.setenv("API_AUTH_KEYS", "annual-test-internal-key")
    response = main_client.get(
        "/api/annual-reports/documents",
        headers={"Authorization": "Bearer annual-test-token"},
    )
    assert response.status_code == 200
    response = main_client.get(
        "/api/annual-reports/documents",
        headers={"X-API-Key": "annual-test-internal-key"},
    )
    assert response.status_code == 200


def test_unconfigured_main_allows_only_direct_loopback(main_client, monkeypatch):
    monkeypatch.setenv("RAG_OBSERVABILITY_DEV_AUTH_ENABLED", "false")
    assert main_client.get("/api/annual-reports/documents").status_code == 200
    assert (
        main_client.get(
            "/api/annual-reports/documents", headers={"X-Real-IP": "203.0.113.1"}
        ).status_code
        == 401
    )
