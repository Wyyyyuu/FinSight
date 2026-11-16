import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

describe('public demo session bootstrap', () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubEnv('VITE_PUBLIC_DEMO', 'true');
    vi.stubEnv('VITE_API_BASE_URL', '/backend');
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it('starts one session before sending concurrent protected requests', async () => {
    let finishSession!: (response: Response) => void;
    const session = new Promise<Response>((resolve) => { finishSession = resolve; });
    const fetchMock = vi.fn().mockReturnValueOnce(session).mockResolvedValue(new Response(null));
    vi.stubGlobal('fetch', fetchMock);
    const { ensurePublicDemoSession, fetchWithPublicDemoSession } = await import('./publicDemoSession');

    void ensurePublicDemoSession();
    const quotes = fetchWithPublicDemoSession('/backend/api/quotes');
    const reports = fetchWithPublicDemoSession('/backend/api/annual-reports/documents');

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe('/backend/public/session');
    finishSession(new Response(null));
    await Promise.all([quotes, reports]);
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      '/backend/public/session',
      '/backend/api/quotes',
      '/backend/api/annual-reports/documents',
    ]);
  });

  it('cancels a waiting API request while keeping the shared session available', async () => {
    let finishSession!: (response: Response) => void;
    const session = new Promise<Response>((resolve) => { finishSession = resolve; });
    const fetchMock = vi.fn().mockReturnValueOnce(session).mockResolvedValue(new Response(null));
    vi.stubGlobal('fetch', fetchMock);
    const { fetchWithPublicDemoSession } = await import('./publicDemoSession');
    const controller = new AbortController();

    const canceled = fetchWithPublicDemoSession('/backend/api/execute', { signal: controller.signal });
    const otherRequest = fetchWithPublicDemoSession('/backend/health');
    controller.abort();
    await expect(canceled).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetchMock).toHaveBeenCalledTimes(1);

    finishSession(new Response(null));
    await otherRequest;
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual(['/backend/public/session', '/backend/health']);
  });

  it.each([() => new Response(null, { status: 503 }), () => Promise.reject(new TypeError('network'))])(
    'allows a fresh bootstrap after a session failure',
    async (failSession) => {
      const fetchMock = vi.fn().mockImplementationOnce(async () => failSession()).mockResolvedValue(new Response(null));
      vi.stubGlobal('fetch', fetchMock);
      const { ensurePublicDemoSession } = await import('./publicDemoSession');

      await ensurePublicDemoSession();
      await ensurePublicDemoSession();
      await ensurePublicDemoSession();

      expect(fetchMock).toHaveBeenCalledTimes(2);
    },
  );
});
