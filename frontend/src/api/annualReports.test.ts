import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { annualReportsApi } from './annualReports';

const auth = vi.hoisted(() => ({ getSession: vi.fn(), devToken: vi.fn() }));
vi.mock('./supabaseClient', () => ({ getSupabaseClient: () => ({ auth: { getSession: auth.getSession } }) }));
vi.mock('../auth/devAuth', () => ({ getRagInspectorDevAccessToken: auth.devToken }));
vi.mock('../store/useStore', () => ({ useStore: { getState: () => ({ sessionId: 'public:anonymous:annual-test' }) } }));

describe('annualReportsApi', () => {
  beforeEach(() => {
    auth.getSession.mockResolvedValue({ data: { session: { access_token: 'session-token' } } });
    auth.devToken.mockReturnValue(null);
  });
  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.unstubAllEnvs(); });

  it('sends the exact selected document scope, session authentication and cancellation signal', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: 'complete' })));
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();
    await annualReportsApi.analyze({ question: '对比营业收入', document_ids: ['only-selected-id'], max_retries: 1, mode: 'hybrid' }, controller.signal);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toContain('/api/annual-reports/analyze');
    expect(init.signal).toBe(controller.signal);
    expect(JSON.parse(init.body).document_ids).toEqual(['only-selected-id']);
    expect(init.headers.get('Authorization')).toBe('Bearer session-token');
    expect(init.headers.get('X-Session-Id')).toBe('public:anonymous:annual-test');
  });

  it('uploads multipart metadata without breaking the browser-generated content boundary', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ id: 'uploaded' })));
    vi.stubGlobal('fetch', fetchMock);
    const file = new File(['营业收入：100 万元'], 'annual.txt', { type: 'text/plain' });
    await annualReportsApi.upload(file, ' 测试公司 ', 2024);
    const init = fetchMock.mock.calls[0][1];
    expect(init.body.get('file').name).toBe('annual.txt');
    expect(init.body.get('company')).toBe('测试公司');
    expect(init.body.get('year')).toBe('2024');
    expect(init.headers.has('Content-Type')).toBe(false);
  });

  it('falls back to the configured dev token when session lookup fails', async () => {
    auth.getSession.mockRejectedValue(new Error('session unavailable'));
    auth.devToken.mockReturnValue('dev-token');
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ documents: [] })));
    vi.stubGlobal('fetch', fetchMock);
    await annualReportsApi.list();
    expect(fetchMock.mock.calls[0][1].headers.get('Authorization')).toBe('Bearer dev-token');
  });

  it('preserves actionable backend errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: '该 PDF 没有可提取的文字，请先进行 OCR。' }), { status: 422 })));
    await expect(annualReportsApi.list()).rejects.toThrow('该 PDF 没有可提取的文字，请先进行 OCR。');
  });

  it('keeps an aborted request distinguishable from a network failure', async () => {
    const controller = new AbortController();
    controller.abort();
    const abort = new DOMException('Aborted', 'AbortError');
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(abort));
    await expect(annualReportsApi.list(controller.signal)).rejects.toBe(abort);
  });

  it('supports a same-origin standalone deployment', async () => {
    vi.resetModules();
    vi.stubEnv('VITE_ANNUAL_REPORT_API_BASE_URL', '.');
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ documents: [] })));
    vi.stubGlobal('fetch', fetchMock);
    const { annualReportsApi: api } = await import('./annualReports');
    await api.list();
    expect(fetchMock.mock.calls[0][0]).toBe('/api/annual-reports/documents');
  });

  it('supports a separately configured annual report API without changing existing app routes', async () => {
    vi.resetModules();
    vi.stubEnv('VITE_ANNUAL_REPORT_API_BASE_URL', 'https://annual.example.test/');
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ documents: [] })));
    vi.stubGlobal('fetch', fetchMock);
    const { annualReportsApi: api } = await import('./annualReports');
    await api.list();
    expect(fetchMock.mock.calls[0][0]).toBe('https://annual.example.test/api/annual-reports/documents');
  });
});
