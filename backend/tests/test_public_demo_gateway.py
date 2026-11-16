import asyncio
import json
from urllib.parse import parse_qs

import httpx
import pytest
from starlette.responses import JSONResponse, StreamingResponse

from backend.api.public_demo import PublicDemoGateway, COOKIE


@pytest.fixture
def app():
    async def echo(scope, receive, send):
        if scope['path'].endswith('/stream'):
            async def chunks():
                yield b'data: first\n\n'
                await asyncio.sleep(0)
                yield b'data: done\n\n'
            return await StreamingResponse(chunks(), media_type='text/event-stream')(scope, receive, send)
        data = b''
        if scope['method'] == 'POST':
            data = (await receive())['body']
        await JSONResponse({'identity': scope['state']['rag_authenticated_user'],
                            'query': parse_qs(scope.get('query_string', b'').decode()),
                            'body': json.loads(data) if data else None,
                            'headers': dict((k.decode(), v.decode()) for k,v in scope['headers'])})(scope, receive, send)
    return PublicDemoGateway(echo, b'a' * 32, 'https://finsight-research.pages.dev')


@pytest.mark.asyncio
async def test_visitor_isolation_and_forged_identity(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='https://demo.test') as a, httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='https://demo.test') as b:
        assert (await a.post('/chat/supervisor', json={'query':'hi'})).status_code == 401
        for client in (a,b):
            response=await client.get('/public/session')
            assert 'HttpOnly' in response.headers['set-cookie'] and 'Secure' in response.headers['set-cookie']
        first=(await a.post('/chat/supervisor',json={'session_id':'private:owner:thread','query':'hi'},headers={'X-API-Key':'forged','Authorization':'Bearer forged'})).json()
        second=(await b.post('/chat/supervisor',json={'session_id':first['body']['session_id'],'query':'hi'})).json()
        assert first['identity'] != second['identity']
        assert first['body']['session_id'] != second['body']['session_id']
        assert 'authorization' not in first['headers'] and 'x-api-key' not in first['headers']
        assert first['body']['options']['trace_raw_override'] == 'off'
        repeat=(await a.post('/chat/supervisor',json={'session_id':first['body']['session_id'],'query':'again'})).json()
        assert repeat['body']['session_id'] == first['body']['session_id']
        a.cookies.clear()
        a.cookies.set(COOKIE,'0'*32+'.1.forged')
        assert (await a.get('/api/annual-reports/documents')).status_code == 401


@pytest.mark.asyncio
async def test_private_routes_origin_limits_and_stream(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='https://demo.test') as client:
        await client.get('/public/session')
        for path in ('/api/config','/diagnostics/rag/chunks','/health','/api/conversations','/api/reports/compare','/docs','/api/user/profile'):
            assert (await client.get(path)).status_code == 403
        assert (await client.post('/chat/supervisor',json={'query':'hi'},headers={'Origin':'https://evil.test'})).status_code == 403
        assert (await client.post('/chat/supervisor',content=b'x'*131073)).status_code == 413
        response=await client.post('/chat/supervisor/stream',json={'query':'hi'})
        assert response.text == 'data: first\n\ndata: done\n\n'
        assert response.headers['cache-control'] == 'no-store'
        assert app.active == 0
        # Rate limit applies across newly-created visitors, not just one cookie.
        app.generations.extend([__import__('time').monotonic()]*6)
        assert (await client.post('/chat/supervisor',json={'query':'hi'})).status_code == 429


@pytest.mark.asyncio
async def test_dashboard_requires_visitor_and_trusted_origin(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='https://demo.test') as client:
        paths = ('/api/dashboard?symbol=NVDA', '/api/dashboard/insights?symbol=NVDA&force=true')
        for path in paths:
            assert (await client.get(path)).status_code == 401
        await client.get('/public/session')
        for path in paths:
            assert (await client.get(path, headers={'Origin': 'https://evil.test'})).status_code == 403
            response = await client.get(path, headers={'Origin': 'https://finsight-research.pages.dev'})
            assert response.status_code == 200
            assert response.json()['identity']['role'] == 'guest'
            assert response.headers['cache-control'] == 'no-store'
        for path in ('/api/dashboard/health', '/api/dashboard/private', '/api/reports/compare'):
            assert (await client.get(path)).status_code == 403
        assert (await client.post('/api/reports/any/favorite', json={})).status_code == 403
        assert (await client.post('/api/dashboard', json={})).status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['fund-flow', 'northbound', 'limit-board', 'lhb', 'concept'])
async def test_cn_market_reads_require_session_and_origin_but_no_generation_budget(app, name):
    path = f'/api/cn/market/{name}'
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='https://demo.test') as client:
        assert (await client.get(path)).status_code == 401
        await client.get('/public/session')
        assert (await client.get(path, headers={'Origin': 'https://evil.test'})).status_code == 403
        app.active = 2
        app.generations.extend([__import__('time').monotonic()] * 6)
        response = await client.get(path, params={'limit': 20}, headers={'Origin': 'https://finsight-research.pages.dev'})
        assert response.status_code == 200
        assert response.json()['query'] == {'limit': ['20']}
        assert response.json()['identity']['role'] == 'guest'
        assert response.headers['cache-control'] == 'no-store'
        assert (await client.post(path, json={})).status_code == 403
        assert (await client.get('/api/cn/market/private')).status_code == 403


@pytest.mark.asyncio
async def test_dashboard_insights_share_generation_limits(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='https://demo.test') as client:
        await client.get('/public/session')
        for _ in range(6):
            assert (await client.get('/api/dashboard/insights?symbol=NVDA&force=true')).status_code == 200
            assert app.active == 0
        assert (await client.get('/api/dashboard/insights?symbol=NVDA')).status_code == 429
        assert (await client.post('/chat/supervisor', json={'query': 'hi'})).status_code == 429
        assert (await client.get('/api/dashboard?symbol=NVDA')).status_code == 200


@pytest.mark.asyncio
async def test_dashboard_insights_release_concurrency_on_failure(app):
    async def fail(scope, receive, send):
        assert app.active == 1
        raise RuntimeError('upstream unavailable')

    app.app = fail
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='https://demo.test') as client:
        await client.get('/public/session')
        app.active = 2
        assert (await client.get('/api/dashboard/insights?symbol=NVDA')).status_code == 429
        app.active = 0
        with pytest.raises(RuntimeError, match='upstream unavailable'):
            await client.get('/api/dashboard/insights?symbol=NVDA')
        assert app.active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['/api/dashboard/deep-dive', '/api/execute', '/api/execute/resume'])
async def test_execution_identity_and_limits(app, path):
    payload = {'symbol': 'NVDA', 'tab': 'overview', 'query': 'Analyze NVDA',
               'session_id': 'private:owner:thread', 'thread_id': 'private:owner:thread',
               'run_id': 'client-correlation', 'trace_raw': True, 'resume_value': 'continue'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='https://demo.test') as a, httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='https://demo.test') as b:
        assert (await a.post(path, json=payload)).status_code == 401
        for client in (a, b):
            await client.get('/public/session')
        assert (await a.post(path, json=payload, headers={'Origin': 'https://evil.test'})).status_code == 403
        first = (await a.post(path, json=payload)).json()['body']
        second = (await b.post(path, json={**payload, 'session_id': first['session_id'], 'thread_id': first['session_id']})).json()['body']
        assert first['session_id'] != second['session_id']
        assert first['run_id'] == second['run_id'] == 'client-correlation'
        assert first['trace_raw'] is second['trace_raw'] is False
        if path.endswith('/resume'):
            assert first['thread_id'] == first['session_id']
            assert second['thread_id'] == second['session_id']
        repeat = (await a.post(path, json={**payload, 'session_id': first['session_id'], 'thread_id': first['session_id']})).json()['body']
        assert repeat['session_id'] == first['session_id']
        app.active = 2
        assert (await a.post(path, json=payload)).status_code == 429
        app.active = 0
        app.generations.extend([__import__('time').monotonic()] * 6)
        assert (await a.post(path, json=payload)).status_code == 429


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['/api/reports/index', '/api/reports/replay/report-123', '/api/reports/citations'])
async def test_report_query_uses_execution_owner(app, path):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='https://demo.test') as a, httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='https://demo.test') as b:
        assert (await a.get(path, params={'session_id': 'browser-thread'})).status_code == 401
        for client in (a, b):
            await client.get('/public/session')
        execution = (await a.post('/api/execute', json={'query': 'NVDA', 'session_id': 'browser-thread'})).json()['body']
        for sid in ('browser-thread', execution['session_id']):
            own = await a.get(path, params={'session_id': sid, 'ticker': 'NVDA'})
            assert own.status_code == 200
            assert own.json()['query']['session_id'] == [execution['session_id']]
        foreign = await b.get(path, params={'session_id': execution['session_id']})
        assert foreign.json()['query']['session_id'] != [execution['session_id']]
        assert (await a.get(path)).status_code == 422
        assert (await a.get(path, params=[('session_id', 'one'), ('session_id', 'two')])).status_code == 422
        assert (await a.get(path, params={'session_id': 'browser-thread'}, headers={'Origin': 'https://evil.test'})).status_code == 403


@pytest.mark.asyncio
async def test_resume_uses_interrupted_thread_not_current_browser_thread(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='https://demo.test') as client:
        await client.get('/public/session')
        started = (await client.post('/api/execute', json={'query': 'NVDA', 'session_id': 'original'})).json()['body']
        resumed = (await client.post('/api/execute/resume', json={
            'thread_id': started['session_id'], 'session_id': 'new-browser-thread', 'resume_value': 'yes',
        })).json()['body']
        assert resumed['session_id'] == resumed['thread_id'] == started['session_id']
        assert (await client.post('/api/execute/resume', json={'session_id': 'original'})).status_code == 422
