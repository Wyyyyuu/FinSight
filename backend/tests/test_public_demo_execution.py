import asyncio
import json
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

from backend.api.public_demo import PublicDemoGateway


@pytest.mark.asyncio
async def test_deep_dive_stream_and_report_replay_are_visitor_scoped(monkeypatch, tmp_path):
    from backend.api import execution_router
    from backend.api.execution_router import ExecutionRouterDeps, create_execution_router
    from backend.api.report_router import ReportRouterDeps, create_report_router
    from backend.services.report_index import ReportIndexStore

    monkeypatch.setenv('REPORT_INDEX_SQLITE_PATH', str(tmp_path / 'reports.sqlite'))
    store = ReportIndexStore()
    calls = []

    async def pipeline(**kwargs):
        calls.append(kwargs)
        report = {'report_id': str(uuid4()), 'ticker': 'NVDA', 'summary': kwargs['query'], 'citations': []}
        store.upsert_report(session_id=kwargs['thread_id'], report=report)
        yield {'type': 'token', 'content': 'Visitor-owned research'}
        yield {'type': 'done', 'run_id': kwargs['run_id'], 'session_id': kwargs['thread_id'], 'report': report}

    async def get_runner():
        return object()

    monkeypatch.setattr(execution_router, 'run_graph_pipeline', pipeline)
    api = FastAPI()
    api.include_router(create_execution_router(ExecutionRouterDeps(
        get_graph_runner=get_runner, resolve_thread_id=lambda sid: sid,
        schedule_report_index=lambda **kwargs: None, update_session_context=lambda **kwargs: None,
        redact_sensitive_payload=lambda data: data, is_raw_trace_event=lambda data: False,
        contract_info=lambda: {}, sse_event_schema_version='test',
    )))
    api.include_router(create_report_router(ReportRouterDeps(
        resolve_thread_id=lambda sid: sid, get_report_index_store=lambda: store,
    )))
    gateway = PublicDemoGateway(api, b'a' * 32, 'https://demo.test')
    async with httpx.AsyncClient(transport=httpx.ASGITransport(gateway), base_url='https://demo.test') as a, httpx.AsyncClient(transport=httpx.ASGITransport(gateway), base_url='https://demo.test') as b:
        for client in (a, b):
            await client.get('/public/session')
        response = await a.post('/api/dashboard/deep-dive', json={
            'symbol': 'NVDA', 'tab': 'overview', 'user_question': 'my private question',
            'session_id': 'browser-thread', 'run_id': 'browser-run', 'trace_raw': True,
        })
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('text/event-stream')
        assert gateway.active == 0
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
        done = events[-1]
        assert done['type'] == 'done' and done['run_id'] == 'browser-run'
        assert calls[0]['trace_raw_enabled'] is False
        assert calls[0]['ui_context']['dashboard_tab'] == 'overview'
        assert 'my private question' in done['report']['summary']
        report_id = done['report']['report_id']
        for sid in ('browser-thread', done['session_id']):
            index = await a.get('/api/reports/index', params={'session_id': sid})
            assert index.status_code == 200 and index.json()['count'] == 1
            replay = await a.get(f'/api/reports/replay/{report_id}', params={'session_id': sid})
            assert replay.status_code == 200
            assert replay.json()['report']['summary'] == done['report']['summary']
            foreign_index = await b.get('/api/reports/index', params={'session_id': sid})
            assert foreign_index.json()['count'] == 0
            foreign_replay = await b.get(f'/api/reports/replay/{report_id}', params={'session_id': sid})
            assert foreign_replay.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize('second_is_resume', [False, True])
async def test_concurrent_execution_traces_never_cross_sessions(monkeypatch, second_is_resume):
    from backend.services import execution_service
    from backend.graph import runner as runner_module
    from backend.orchestration.trace_emitter import get_trace_emitter

    both_running = asyncio.Event()
    emit_now = asyncio.Event()
    finished_emitting = asyncio.Event()
    started = 0
    emitted = 0

    async def emit_private_trace(thread):
        nonlocal started, emitted
        started += 1
        if started == 2:
            both_running.set()
        await emit_now.wait()
        get_trace_emitter().emit_agent_start('test_agent', query=f'private-{thread}', ticker='NVDA')
        emitted += 1
        if emitted == 2:
            finished_emitting.set()
        await finished_emitting.wait()
        await asyncio.sleep(0)  # let queued trace callbacks reach each SSE queue
        raise asyncio.CancelledError()

    async def fake_graph(_runner, *, thread_id, **kwargs):
        await emit_private_trace(thread_id)

    class Runner:
        async def resume(self, *, thread_id, resume_value):
            await emit_private_trace(thread_id)
            yield {}  # async generator contract

    async def get_runner():
        return Runner()

    monkeypatch.setattr(runner_module, 'run_graph_traced', fake_graph)
    deps = execution_service.ExecutionDeps(
        get_graph_runner=get_runner, schedule_report_index=lambda **kwargs: None,
        update_session_context=lambda **kwargs: None, redact_sensitive_payload=lambda data: data,
        is_raw_trace_event=lambda data: False, contract_info=lambda: {}, sse_event_schema_version='test',
    )

    async def collect(stream):
        return [event async for event in stream]

    first = asyncio.create_task(collect(execution_service.run_graph_pipeline(
        deps=deps, query='A', thread_id='visitor-A', run_id='run-A', trace_raw_enabled=False,
    )))
    second_stream = execution_service.resume_graph_pipeline(
        deps=deps, thread_id='visitor-B', resume_value='yes', run_id='run-B', trace_raw_enabled=False,
    ) if second_is_resume else execution_service.run_graph_pipeline(
        deps=deps, query='B', thread_id='visitor-B', run_id='run-B', trace_raw_enabled=False,
    )
    second = asyncio.create_task(collect(second_stream))
    await asyncio.wait_for(both_running.wait(), timeout=5)
    # An unowned event must not be relabelled as either visitor's progress.
    get_trace_emitter().emit_agent_start('test_agent', query='unowned-private-event')
    emit_now.set()
    first_events, second_events = await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
    for own, other, events in [('visitor-A', 'visitor-B', first_events), ('visitor-B', 'visitor-A', second_events)]:
        traces = [event for event in events if event.get('type') == 'agent_start']
        assert len(traces) == 1, events
        text = json.dumps(traces)
        assert f'private-{own}' in text
        assert f'private-{other}' not in text
        assert 'unowned-private-event' not in text
        assert traces[0]['session_id'] == own
