import { afterEach, describe, expect, it, vi } from 'vitest';
import { useDashboardData } from './useDashboardData';

const state = vi.hoisted(() => ({
  setActiveAsset: vi.fn(),
  setCapabilities: vi.fn(),
  setDashboardData: vi.fn(),
  setLoading: vi.fn(),
  setError: vi.fn(),
}));

vi.mock('../store/dashboardStore', () => ({ useDashboardStore: () => state }));
vi.mock('react', () => ({
  useRef: (current: unknown) => ({ current }),
  useCallback: (callback: unknown) => callback,
  useEffect: vi.fn(),
}));

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe('dashboard loading failures', () => {
  it.each([
    [{ detail: '演示站暂未开放该功能。' }, '演示站暂未开放该功能。'],
    [{ detail: { message: '标的不存在' } }, '标的不存在'],
    [{ message: '服务暂时不可用' }, '服务暂时不可用'],
    [{}, 'HTTP 403'],
  ])('preserves the API error message for %j', async (body, message) => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status: 403 })));
    await useDashboardData(null).refetch('NVDA');
    expect(state.setError).toHaveBeenLastCalledWith(message);
    expect(state.setLoading).toHaveBeenLastCalledWith(false);
    expect(state.setDashboardData).not.toHaveBeenCalled();
  });

  it('falls back to HTTP status for non-JSON failures', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>Forbidden</html>', { status: 403 })));
    await useDashboardData(null).refetch('NVDA');
    expect(state.setError).toHaveBeenLastCalledWith('HTTP 403');
  });

  it('discards an old rejection when the user switches symbols', async () => {
    let rejectOld!: (error: Error) => void;
    const data = { snapshot: { price: 200 } };
    const fetchMock = vi.fn()
      .mockImplementationOnce(() => new Promise((_resolve, reject) => { rejectOld = reject; }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        state: { active_asset: { symbol: 'MSFT' }, capabilities: {} }, data,
      })));
    vi.stubGlobal('fetch', fetchMock);
    const { refetch } = useDashboardData(null);
    const oldRequest = refetch('NVDA');
    await refetch('MSFT');
    rejectOld(new Error('Old request denied'));
    await oldRequest;
    expect(state.setError).toHaveBeenLastCalledWith(null);
    expect(state.setDashboardData).toHaveBeenLastCalledWith(data);
    expect(fetchMock.mock.calls[0][1].signal.aborted).toBe(true);
  });
});
