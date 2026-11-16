"""Restricted, isolated anonymous demo entry point. Never expose main:app directly.

Browser identity is a server-signed HttpOnly cookie, not a client-selected user ID.
The launcher runs this app from a separate working/data directory.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import time
from collections import deque
from http.cookies import CookieError, SimpleCookie
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

from starlette.responses import JSONResponse

COOKIE = '__Host-finsight_demo'
MAX_BODY = 21 * 1024 * 1024
CHAT = {'/chat/supervisor', '/chat/supervisor/stream'}
EXECUTION = {'/api/execute', '/api/dashboard/deep-dive', '/api/execute/resume'}
REPORT_READS = {'/api/reports/index', '/api/reports/citations'}
CN_MARKET_READS = {
    '/api/cn/market/fund-flow', '/api/cn/market/northbound',
    '/api/cn/market/limit-board', '/api/cn/market/lhb', '/api/cn/market/concept',
}
REPORT_REPLAY = re.compile(r'^/api/reports/replay/[A-Za-z0-9._-]{1,128}$')
EXACT = {
    'GET': REPORT_READS | CN_MARKET_READS | {'/api/tools/capabilities', '/api/agents', '/api/skills',
            '/api/dashboard', '/api/dashboard/insights',
            '/api/reports/price-drift',
            '/api/annual-reports/health', '/api/annual-reports/documents'},
    'POST': CHAT | EXECUTION | {'/api/chart/detect', '/api/chart/data',
                    '/api/annual-reports/documents', '/api/annual-reports/demo',
                    '/api/annual-reports/analyze'},
}
GET_PATHS = re.compile(
    r'^(/api/stock/(price|news|kline)/[^/]+|/api/financials/[^/]+(/summary)?'
    r'|/api/annual-reports/documents/[A-Za-z0-9_-]+/pages/[0-9]+)$'
)


def _visitor_session(visitor: str, client_session: object) -> str:
    """Use the same visitor-owned thread for execution, resume and report reads."""
    value = str(client_session or 'default')
    prefix = f'demo:{visitor}:'
    thread = value[len(prefix):] if value.startswith(prefix) else ''
    if not re.fullmatch(r'[a-f0-9]{32}', thread):
        thread = hashlib.sha256(value.encode()).hexdigest()[:32]
    return prefix + thread


class PublicDemoGateway:
    def __init__(self, app, secret: bytes, origin: str):
        if len(secret) < 32:
            raise ValueError('Public demo cookie key must contain at least 32 bytes.')
        self.app, self.secret, self.origin = app, secret, origin
        self.generations = deque()
        self.active = 0

    def identity(self, headers):
        cookie = SimpleCookie()
        try:
            cookie.load(headers.get('cookie', ''))
            raw = cookie[COOKIE].value
            identifier, stamp, signature = raw.split('.')
            message = f'{identifier}.{stamp}'
            expected = hmac.new(self.secret, message.encode(), hashlib.sha256).hexdigest()
            if (re.fullmatch(r'[a-f0-9]{32}', identifier)
                and hmac.compare_digest(expected, signature)
                and 0 <= time.time() - int(stamp) <= 30 * 86400):
                return identifier
        except (KeyError, ValueError, CookieError):
            pass
        return None

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        path, method = scope['path'], scope['method']
        headers = {k.decode('latin1').lower(): v.decode('latin1') for k, v in scope['headers']}

        async def reply(status, detail):
            await JSONResponse({'detail': detail}, status_code=status,
                               headers={'Cache-Control': 'no-store'})(scope, receive, send)

        if headers.get('origin') and headers['origin'] != self.origin:
            return await reply(403, '请求来源不受支持。')
        if path == '/public/health' and method == 'GET':
            return await JSONResponse({'status': 'ok', 'mode': 'local-demo'})(scope, receive, send)
        if path == '/public/session' and method == 'GET':
            response = JSONResponse({'mode': 'anonymous-demo'}, headers={'Cache-Control': 'no-store'})
            if not self.identity(headers):
                message = f'{secrets.token_hex(16)}.{int(time.time())}'
                signature = hmac.new(self.secret, message.encode(), hashlib.sha256).hexdigest()
                response.set_cookie(COOKIE, f'{message}.{signature}', max_age=30 * 86400,
                                    secure=True, httponly=True, samesite='lax', path='/')
            return await response(scope, receive, send)
        report_read = method == 'GET' and (path in REPORT_READS or REPORT_REPLAY.fullmatch(path))
        if path not in EXACT.get(method, set()) and not (method == 'GET' and (GET_PATHS.fullmatch(path) or report_read)):
            return await reply(403, '演示站暂未开放该功能。')
        visitor = self.identity(headers)
        if visitor is None:
            return await reply(401, '访客会话未建立或已过期，请刷新页面。')

        body = bytearray()
        if method == 'POST':
            limit = MAX_BODY if path == '/api/annual-reports/documents' else 128 * 1024
            try:
                if int(headers.get('content-length', '0')) > limit:
                    return await reply(413, '请求内容过大。')
            except ValueError:
                return await reply(400, '无效的请求长度。')
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                body.extend(message.get('body', b''))
                if len(body) > limit:
                    return await reply(413, '请求内容过大。')
                if not message.get('more_body'):
                    break
        if path in CHAT | EXECUTION:
            try:
                payload = json.loads(body)
                client_session = payload.get('session_id')
                if path == '/api/execute/resume':
                    client_session = payload.get('thread_id')
                    if not isinstance(client_session, str) or not client_session.strip():
                        return await reply(422, '缺少待恢复的任务会话。')
                    payload['thread_id'] = _visitor_session(visitor, client_session)
                payload['session_id'] = _visitor_session(visitor, client_session)
                if path in CHAT:
                    options = payload.get('options') or {}
                    options['trace_raw_override'] = 'off'
                    payload['options'] = options
                else:
                    # run_id is only the browser's event correlation ID. The
                    # signed visitor session owns graph state and report access.
                    payload['trace_raw'] = False
                body = json.dumps(payload).encode()
            except (ValueError, TypeError, AttributeError):
                return await reply(422, '无效的分析请求。')

        query_string = scope.get('query_string', b'')
        if report_read:
            query = parse_qsl(query_string.decode('utf-8', errors='replace'), keep_blank_values=True)
            sessions = [value for key, value in query if key == 'session_id']
            if len(sessions) != 1 or not sessions[0].strip():
                return await reply(422, '报告读取需要唯一的会话编号。')
            query = [(key, value) for key, value in query if key != 'session_id']
            query.append(('session_id', _visitor_session(visitor, sessions[0])))
            query_string = urlencode(query).encode()

        expensive = path in CHAT | EXECUTION or path in {'/api/annual-reports/analyze', '/api/annual-reports/demo', '/api/annual-reports/documents', '/api/dashboard/insights'}
        if expensive:
            now = time.monotonic()
            while self.generations and now - self.generations[0] > 60:
                self.generations.popleft()
            if self.active >= 2 or len(self.generations) >= 6:
                return await reply(429, '演示服务繁忙，请稍后重试。')
            self.generations.append(now)
            self.active += 1
        # Only this gateway can populate this state. User-supplied identity and
        # internal credentials are removed before the existing API sees them.
        inner = dict(scope)
        inner['query_string'] = query_string
        inner['state'] = dict(scope.get('state', {}))
        inner['state']['rag_authenticated_user'] = {'user_id': f'demo-{visitor}', 'role': 'guest'}
        removed = {b'authorization', b'x-api-key', b'x-session-id', b'content-length',
                   b'x-forwarded-for', b'x-real-ip', b'cf-connecting-ip', b'forwarded'}
        inner['headers'] = [(k, v) for k, v in scope['headers'] if k.lower() not in removed]
        if method == 'POST':
            inner['headers'].append((b'content-length', str(len(body)).encode()))
        sent = False

        async def replay_receive():
            nonlocal sent
            if method == 'POST' and not sent:
                sent = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            return await receive()

        async def no_cache_send(message):
            if message['type'] == 'http.response.start':
                message = dict(message)
                message['headers'] = [(k, v) for k, v in message.get('headers', []) if k.lower() != b'cache-control']
                message['headers'] += [(b'cache-control', b'no-store'), (b'x-content-type-options', b'nosniff')]
            await send(message)

        try:
            await self.app(inner, replay_receive, no_cache_send)
        finally:
            if expensive:
                self.active -= 1


def create_app():
    # Fail closed unless started with the separate demo launcher configuration.
    root = Path(os.environ['FINSIGHT_PUBLIC_DEMO_DIR']).resolve()
    if Path.cwd().resolve() != root:
        raise RuntimeError('Public demo must use its isolated working directory.')
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    os.environ['LANGGRAPH_CHECKPOINTER_BACKEND'] = 'sqlite'
    # This legacy cache is keyed by ticker/mode, not visitor or question.
    # A public execution must never replay another visitor's custom research.
    os.environ['REPORT_CACHE_TTL_HOURS'] = '0'
    for name in ('API_AUTH_ENABLED', 'RAG_OBSERVABILITY_DEV_AUTH_ENABLED'):
        os.environ[name] = 'false'
    for name in ('SUPABASE_URL', 'SUPABASE_PUBLISHABLE_KEY', 'VITE_SUPABASE_URL',
                 'VITE_SUPABASE_PUBLISHABLE_KEY', 'API_AUTH_KEY', 'API_AUTH_KEYS',
                 'LANGGRAPH_CHECKPOINT_POSTGRES_DSN', 'RAG_OBSERVABILITY_DSN',
                 'RAG_V2_POSTGRES_DSN'):
        os.environ[name] = ''
    from backend.api.main import app
    return PublicDemoGateway(app, (root / 'cookie.key').read_bytes(),
                             os.environ['PUBLIC_DEMO_ORIGIN'])
