import { afterEach, describe, expect, it, vi } from 'vitest';

import { apiClient, parseSSEStream, RATE_LIMIT_EVENT, rateLimitEvents } from './client';

function sseResponse(events: Array<Record<string, unknown>>): Response {
  const body = events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join('');
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(body));
      controller.close();
    },
  });
  return new Response(stream);
}

describe('public demo conversation identity', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it.each([true, false])('keeps browser conversation stable only in demo mode (%s)', async (demo) => {
    vi.stubEnv('VITE_PUBLIC_DEMO', String(demo));
    const clientSession = 'public:anonymous:browser-thread';
    const serverSession = 'demo:visitor:server-thread';
    const onDone = vi.fn();
    const onToken = vi.fn();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(sseResponse([
      { type: 'token', content: 'hello' },
      { type: 'done', session_id: serverSession },
    ])));
    await apiClient.sendMessageStream(
      'hello', onToken, undefined, undefined, onDone,
      undefined, undefined, undefined, undefined, undefined, undefined, clientSession,
    );
    expect(onToken).toHaveBeenCalledWith('hello');
    expect(onDone.mock.calls[0][2].session_id).toBe(demo ? clientSession : serverSession);
  });
});

describe('parseSSEStream', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('normalizes user-visible trace events into thinking callbacks', async () => {
    const thinking: any[] = [];
    const rawEvents: any[] = [];

    await parseSSEStream(
      sseResponse([
        {
          type: 'trace',
          stage: 'understanding',
          title: '已理解请求',
          summary: 'GOOGL:price；阻塞项:1',
          tasks: [{ id: 'task_1', operation: 'price' }],
        },
      ]),
      {
        onThinking: (step) => thinking.push(step),
        onRawEvent: (event) => rawEvents.push(event),
      },
      { traceRawEnabled: true },
    );

    expect(thinking).toHaveLength(1);
    expect(thinking[0]).toMatchObject({
      stage: 'understanding',
      message: 'GOOGL:price；阻塞项:1',
      eventType: 'trace',
    });
    expect(thinking[0].result.tasks).toHaveLength(1);
    expect(rawEvents[0].eventType).toBe('trace');
  });

  it('surfaces cancelled trace and pipeline events as thinking callbacks', async () => {
    const thinking: any[] = [];

    await parseSSEStream(
      sseResponse([
        {
          type: 'trace',
          stage: 'cancelled',
          status: 'cancelled',
          summary: '已停止生成，保留已完成的结果。',
        },
        {
          type: 'pipeline_stage',
          stage: 'cancelled',
          status: 'cancelled',
          message: 'Generation cancelled by client',
        },
      ]),
      {
        onThinking: (step) => thinking.push(step),
      },
      { traceRawEnabled: true },
    );

    expect(thinking).toHaveLength(2);
    expect(thinking[0]).toMatchObject({
      stage: 'cancelled',
      message: '已停止生成，保留已完成的结果。',
      eventType: 'trace',
    });
    expect(thinking[1]).toMatchObject({
      stage: 'cancelled',
      message: 'Generation cancelled by client',
      eventType: 'pipeline_stage',
    });
  });

  it('does not report missing done when execute stream was aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    const onError = vi.fn();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(sseResponse([{ type: 'token', content: 'partial' }])));

    await apiClient.executeAgent(
      { query: 'AAPL', session_id: 'public:user:thread' },
      { onError },
      { signal: controller.signal },
    );

    expect(onError).not.toHaveBeenCalled();
  });

  it('does not report missing done when resume stream was aborted', async () => {
    const controller = new AbortController();
    controller.abort();
    const onError = vi.fn();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(sseResponse([{ type: 'token', content: 'partial' }])));

    await apiClient.resumeExecution(
      { thread_id: 'public:user:thread', resume_value: 'continue' },
      { onError },
      { signal: controller.signal },
    );

    expect(onError).not.toHaveBeenCalled();
  });
});

describe('execution gateway responses', () => {
  afterEach(() => vi.unstubAllGlobals());

  it.each([
    [{ detail: '演示站暂未开放该功能。' }, '演示站暂未开放该功能。'],
    [{ detail: { message: '会话已过期，请刷新页面。' } }, '会话已过期，请刷新页面。'],
    [{}, '请求失败（HTTP 403），请稍后重试'],
  ])('shows the actual denied execution reason', async (payload, message) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(payload), { status: 403 })));
    await expect(apiClient.executeAgent({ query: 'NVDA' }, {}, { endpoint: '/api/dashboard/deep-dive' }))
      .rejects.toThrow(message);
  });

  it('reports a denied resume to callback consumers', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: '访客会话已过期' }), { status: 401 })));
    await expect(apiClient.resumeExecution({ thread_id: 'thread', resume_value: 'yes' }, {}))
      .rejects.toThrow('访客会话已过期');
  });

  it('preserves the run correlation and report from deep-dive completion', async () => {
    const report = { report_id: 'report-1', ticker: 'NVDA', summary: 'Research result' };
    const onDone = vi.fn();
    const onThinking = vi.fn();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(sseResponse([
      { type: 'pipeline_stage', run_id: 'browser-run', stage: 'planning', status: 'start' },
      { type: 'done', run_id: 'browser-run', session_id: 'demo:visitor:thread', report },
    ])));
    await apiClient.executeAgent({ query: 'NVDA', run_id: 'browser-run' }, { onDone, onThinking }, { endpoint: '/api/dashboard/deep-dive' });
    expect(onThinking).toHaveBeenCalled();
    expect(onDone).toHaveBeenCalledTimes(1);
    expect(onDone.mock.calls[0][0]).toEqual(report);
    expect(onDone.mock.calls[0][2].run_id).toBe('browser-run');
  });
});

describe('SSE read timeout (P1-2)', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('reports a connection error when no data arrives within the read timeout', async () => {
    vi.useFakeTimers();
    // 一个永远不发数据的流，模拟连接挂死（网络断/Tunnel 超时）
    const cancel = vi.fn();
    const stream = new ReadableStream({ start() { /* never enqueue */ }, cancel });
    const response = new Response(stream);
    const onError = vi.fn();
    const onDone = vi.fn();

    const promise = parseSSEStream(response, { onError, onDone }, { readTimeoutMs: 1000 });
    await vi.advanceTimersByTimeAsync(1100);
    await promise;

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0]).toContain('连接中断');
    expect(onDone).not.toHaveBeenCalled();
    expect(cancel).toHaveBeenCalledTimes(1);
  });

  it('does not time out while frames keep arriving (heartbeats reset the timer)', async () => {
    vi.useFakeTimers();
    const encoder = new TextEncoder();
    let pushCount = 0;
    let intervalId: ReturnType<typeof setInterval> | null = null;

    // 每 500ms 发一个心跳帧，发 5 个之后正常结束（总时长 2500ms > 单次超时 1000ms）
    const stream = new ReadableStream({
      start(controller) {
        intervalId = setInterval(() => {
          pushCount += 1;
          controller.enqueue(encoder.encode('data: {"type":"heartbeat"}\n\n'));
          if (pushCount >= 5) {
            if (intervalId) clearInterval(intervalId);
            controller.close();
          }
        }, 500);
      },
    });
    const response = new Response(stream);
    const onError = vi.fn();

    const promise = parseSSEStream(response, { onError }, { readTimeoutMs: 1000 });
    await vi.advanceTimersByTimeAsync(3000);
    await promise;

    expect(onError).not.toHaveBeenCalled();
  });

  it('does not report timeout error when stream was aborted', async () => {
    vi.useFakeTimers();
    const controller = new AbortController();
    const stream = new ReadableStream({ start() { /* never enqueue */ } });
    const response = new Response(stream);
    const onError = vi.fn();

    const promise = parseSSEStream(
      response,
      { onError },
      { readTimeoutMs: 1000, signal: controller.signal },
    );
    controller.abort();
    await vi.advanceTimersByTimeAsync(1100);
    await promise;

    expect(onError).not.toHaveBeenCalled();
  });
});

describe('429 rate limit event (P1-8)', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('dispatches a rate-limit event when stream endpoint returns 429', async () => {
    const listener = vi.fn();
    rateLimitEvents.addEventListener(RATE_LIMIT_EVENT, listener);

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response('rate limited', {
          status: 429,
          headers: { 'Retry-After': '30' },
        }),
      ),
    );

    await expect(
      apiClient.sendMessageStream('AAPL 分析', vi.fn()),
    ).rejects.toThrow();

    expect(listener).toHaveBeenCalledTimes(1);
    const detail = (listener.mock.calls[0][0] as CustomEvent).detail;
    expect(detail.retryAfterSeconds).toBe(30);

    rateLimitEvents.removeEventListener(RATE_LIMIT_EVENT, listener);
  });

  it('dispatches a rate-limit event with null retry-after when header missing', async () => {
    const listener = vi.fn();
    rateLimitEvents.addEventListener(RATE_LIMIT_EVENT, listener);

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response('rate limited', { status: 429 })),
    );

    await expect(
      apiClient.executeAgent(
        { query: 'AAPL', session_id: 'public:user:thread' },
        {},
      ),
    ).rejects.toThrow();

    expect(listener).toHaveBeenCalledTimes(1);
    const detail = (listener.mock.calls[0][0] as CustomEvent).detail;
    expect(detail.retryAfterSeconds).toBeNull();

    rateLimitEvents.removeEventListener(RATE_LIMIT_EVENT, listener);
  });
});
