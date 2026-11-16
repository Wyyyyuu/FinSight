import { buildApiUrl } from '../config/runtime';

let sessionRequest: Promise<void> | undefined;

export function ensurePublicDemoSession(): Promise<void> | undefined {
  if (import.meta.env.VITE_PUBLIC_DEMO !== 'true') return;

  // All concurrent requests must share the same signed visitor cookie.
  sessionRequest ??= fetch(buildApiUrl('/public/session'), {
    credentials: 'same-origin',
    signal: AbortSignal.timeout(12000),
  }).then((response) => {
    if (!response.ok) sessionRequest = undefined;
  }, () => {
    sessionRequest = undefined;
  });
  return sessionRequest;
}

export function waitForPublicDemoSession(signal?: AbortSignal | null): Promise<void> | undefined {
  if (import.meta.env.VITE_PUBLIC_DEMO !== 'true') return;

  signal?.throwIfAborted();
  const pending = ensurePublicDemoSession();
  if (!pending || !signal) return pending;
  return new Promise<void>((resolve, reject) => {
    const abort = () => reject(signal.reason);
    signal.addEventListener('abort', abort, { once: true });
    void pending.then(() => {
      signal.removeEventListener('abort', abort);
      resolve();
    });
  });
}

export async function fetchWithPublicDemoSession(url: string, init?: RequestInit): Promise<Response> {
  await waitForPublicDemoSession(init?.signal);
  return fetch(url, init);
}
