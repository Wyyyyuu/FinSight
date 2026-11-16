# -*- coding: utf-8 -*-
import json
import os
import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_dashboard_understanding_keeps_snapshot_symbols_as_context():
    from backend.dashboard.agent_bridge import DashboardDeepDiveRequest, build_dashboard_deep_dive_execution
    from backend.graph.nodes.understand_request import understand_request

    execution = build_dashboard_deep_dive_execution(DashboardDeepDiveRequest(
        symbol='NVDA', tab='overview',
        dashboard_snapshot={'drawdown': -0.12, 'window': 30, 'benchmark': '^DJI', 'peer': 'MSFT'},
    ))
    result = asyncio.run(understand_request({
        'query': execution.query, 'ui_context': execution.ui_context,
        'output_mode': execution.output_mode,
    }))
    tasks = result['tasks']
    company_tasks = [task for task in tasks if task.get('tickers')]
    assert company_tasks
    assert all(task['tickers'] == ['NVDA'] for task in company_tasks)
    assert all(task['operation']['name'] != 'compare' for task in company_tasks)
    for frame in result.get('request_frames', []):
        assert set(frame.get('subject', {}).get('tickers', [])) <= {'NVDA'}


def test_dashboard_understanding_keeps_explicit_peer_scope():
    from backend.dashboard.agent_bridge import DashboardDeepDiveRequest, build_dashboard_deep_dive_execution
    from backend.graph.nodes.understand_request import understand_request

    execution = build_dashboard_deep_dive_execution(DashboardDeepDiveRequest(
        symbol='NVDA', tab='peers', dashboard_snapshot={'peers': [{'symbol': 'AMD'}]},
    ))
    result = asyncio.run(understand_request({
        'query': execution.query, 'ui_context': execution.ui_context,
        'output_mode': execution.output_mode,
    }))
    tasks = result['tasks']
    assert any(task.get('tickers') == ['NVDA', 'AMD'] and task['operation']['name'] == 'compare' for task in tasks)


def test_dashboard_deep_dive_builds_technical_execute_request():
    from backend.dashboard.agent_bridge import (
        DashboardDeepDiveRequest,
        build_dashboard_deep_dive_execution,
    )

    request = DashboardDeepDiveRequest(
        symbol="nvda",
        tab="technical",
        metric="RSI",
        dashboard_snapshot={"rsi": 72.4, "trend": "above MA20"},
        user_question="这是不是过热了？",
        session_id="public:anonymous:dash-bridge",
        run_id="run-technical",
    )

    execution = build_dashboard_deep_dive_execution(request)

    assert execution.tickers == ["NVDA"]
    assert execution.agents == ["technical_agent", "price_agent"]
    assert execution.output_mode == "investment_report"
    assert execution.confirmation_mode == "skip"
    assert execution.source == "dashboard_deep_dive_technical"
    assert "NVDA" in execution.query
    assert "RSI" in execution.query
    assert "这是不是过热了？" in execution.query
    assert execution.ui_context["active_symbol"] == "NVDA"
    assert execution.ui_context["dashboard_tab"] == "technical"
    assert execution.ui_context["dashboard_metric"] == "RSI"
    assert execution.ui_context["dashboard_snapshot"] == {"rsi": 72.4, "trend": "above MA20"}
    assert execution.ui_context["agents_override"] == ["technical_agent", "price_agent"]
    assert execution.ui_context["tickers_override"] == ["NVDA"]


def test_dashboard_deep_dive_peers_adds_peer_tickers_from_snapshot():
    from backend.dashboard.agent_bridge import (
        DashboardDeepDiveRequest,
        build_dashboard_deep_dive_execution,
    )

    request = DashboardDeepDiveRequest(
        symbol="AAPL",
        tab="peers",
        dashboard_snapshot={
            "subject_symbol": "AAPL",
            "peers": [
                {"symbol": "MSFT", "forward_pe": 31.2},
                {"symbol": "GOOGL", "forward_pe": 22.8},
            ],
        },
    )

    execution = build_dashboard_deep_dive_execution(request)

    assert execution.tickers == ["AAPL", "MSFT", "GOOGL"]
    assert execution.agents == ["price_agent", "fundamental_agent", "risk_agent"]
    assert "对比" in execution.query
    assert "MSFT" in execution.query
    assert execution.ui_context["dashboard_tab"] == "peers"


def test_dashboard_deep_dive_endpoint_reuses_execution_pipeline(monkeypatch):
    local_app_data = Path(os.environ.get("LOCALAPPDATA", "."))
    monkeypatch.setenv(
        "ALERT_LOG_DIR",
        str(local_app_data / "Temp" / "claude" / "finsight-alert-test-logs"),
    )

    from backend.api import execution_router as execution_router_module
    from backend.api.execution_router import ExecutionRouterDeps, create_execution_router

    captured = {}

    async def fake_run_graph_pipeline(**kwargs):
        captured.update(kwargs)
        yield {"type": "done", "response": "ok"}

    monkeypatch.setattr(execution_router_module, "run_graph_pipeline", fake_run_graph_pipeline)

    async def fake_get_graph_runner():
        return object()

    app = FastAPI()
    app.include_router(
        create_execution_router(
            ExecutionRouterDeps(
                get_graph_runner=fake_get_graph_runner,
                resolve_thread_id=lambda session_id: f"thread:{session_id or 'generated'}",
                schedule_report_index=lambda **_kwargs: None,
                update_session_context=lambda **_kwargs: None,
                redact_sensitive_payload=lambda payload: payload,
                is_raw_trace_event=lambda _payload: False,
                contract_info=lambda: {"contract": "test"},
                sse_event_schema_version="sse.test",
            )
        )
    )

    response = TestClient(app).post(
        "/api/dashboard/deep-dive",
        json={
            "symbol": "AAPL",
            "tab": "news",
            "metric": "Reuters headline",
            "dashboard_snapshot": {"headline": "Apple supplier update"},
            "user_question": "这条新闻会影响下季度预期吗？",
            "session_id": "public:anonymous:dash-bridge",
            "run_id": "run-news",
            "trace_raw": False,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    payloads = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert payloads == [{"type": "done", "response": "ok"}]

    assert captured["thread_id"] == "thread:public:anonymous:dash-bridge"
    assert captured["run_id"] == "run-news"
    assert captured["output_mode"] == "investment_report"
    assert captured["confirmation_mode"] == "skip"
    assert captured["source"] == "dashboard_deep_dive_news"
    assert captured["trace_raw_enabled"] is False
    assert captured["ui_context"]["agents_override"] == ["news_agent"]
    assert captured["ui_context"]["tickers_override"] == ["AAPL"]
    assert captured["ui_context"]["dashboard_snapshot"] == {"headline": "Apple supplier update"}
