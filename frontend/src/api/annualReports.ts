import { buildApiUrl } from '../config/runtime';
import { getRagInspectorDevAccessToken } from '../auth/devAuth';
import { getSupabaseClient } from './supabaseClient';
import { useStore } from '../store/useStore';

export interface AnnualDocument {
  id: string;
  filename: string;
  company: string;
  year: number;
  page_count: number;
  chunk_count: number;
  created_at: string;
  warnings: string[];
}

export interface AnnualCitation {
  id: string;
  document_id: string;
  filename: string;
  company: string;
  year: number;
  page: number;
  section: string;
  text: string;
  score: number;
  label: string;
}

export interface AnnualPage {
  document_id: string;
  page: number;
  text: string;
  company: string;
  year: number;
  filename: string;
}

export interface AnnualAnalysis {
  answer: string;
  status: 'complete' | 'insufficient_evidence';
  citations: AnnualCitation[];
  trace: { node: string; status: string; detail: string }[];
  metrics: Record<string, unknown>;
  retrieval_mode: string;
  calculations: Record<string, unknown>[];
}

export interface AnnualAnalysisRequest {
  question: string;
  document_ids: string[];
  years?: number[];
  max_retries?: 1;
  mode?: 'hybrid' | 'bm25';
}

// A standalone deployment can set this to "." to use the serving origin.
// Otherwise preserve the existing application's configured API address.
const annualApiBase = String(import.meta.env.VITE_ANNUAL_REPORT_API_BASE_URL ?? '').trim();
const annualUrl = (path: string) => {
  const endpoint = `/api/annual-reports${path}`;
  if (annualApiBase === '.') return endpoint;
  return annualApiBase ? `${annualApiBase.replace(/\/+$/, '')}${endpoint}` : buildApiUrl(endpoint);
};

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const client = getSupabaseClient();
  let token: string | null = null;
  if (client) {
    try {
      const { data } = await client.auth.getSession();
      token = data.session?.access_token ?? null;
    } catch {
      // Match api/client's best-effort session lookup and local dev fallback.
    }
  }
  token ||= getRagInspectorDevAccessToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const sessionId = useStore.getState().sessionId;
  if (sessionId) headers.set('X-Session-Id', sessionId);
  if (init.body && !(init.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  const response = await fetch(annualUrl(path), { ...init, headers }).catch((error: unknown) => {
    if (init.signal?.aborted) throw error;
    throw new Error('暂时无法连接年报服务，请稍后重试。');
  });
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : null;
    throw new Error(typeof detail === 'string' ? detail : `请求未完成（${response.status}），请稍后重试。`);
  }
  return response.json() as Promise<T>;
}

export const annualReportsApi = {
  list: (signal?: AbortSignal) => request<{ documents: AnnualDocument[] }>('/documents', { signal }),
  upload: (file: File, company: string, year: number, signal?: AbortSignal) => {
    const body = new FormData();
    body.append('file', file);
    body.append('company', company.trim());
    body.append('year', String(year));
    return request<AnnualDocument>('/documents', { method: 'POST', body, signal });
  },
  demo: (signal?: AbortSignal) => request<{ documents: AnnualDocument[] }>('/demo', { method: 'POST', signal }),
  page: (documentId: string, page: number, signal?: AbortSignal) =>
    request<AnnualPage>(`/documents/${encodeURIComponent(documentId)}/pages/${page}`, { signal }),
  analyze: (body: AnnualAnalysisRequest, signal?: AbortSignal) =>
    request<AnnualAnalysis>('/analyze', { method: 'POST', body: JSON.stringify(body), signal }),
};
