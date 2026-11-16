import React, { useState, useCallback, useEffect, useRef } from 'react';
import { RefreshCw, TrendingUp, Globe, BarChart3, Layers, ArrowUpDown } from 'lucide-react';
import axios from 'axios';
import type { CNMarketListResponse } from '../../api/client';
import { apiClient } from '../../api/client';

type TabKey = 'fund_flow' | 'northbound' | 'limit_board' | 'lhb' | 'concept';
type RowData = Record<string, unknown>;

interface TabConfig {
  key: TabKey;
  label: string;
  icon: React.ReactNode;
}

const TABS: TabConfig[] = [
  { key: 'fund_flow',   label: '资金流向', icon: <TrendingUp size={13} /> },
  { key: 'northbound',  label: '北向成交', icon: <Globe size={13} /> },
  { key: 'limit_board', label: '涨跌停',   icon: <ArrowUpDown size={13} /> },
  { key: 'lhb',         label: '龙虎榜',   icon: <BarChart3 size={13} /> },
  { key: 'concept',     label: '概念板块', icon: <Layers size={13} /> },
];

// --- Formatters ---

function fmtFlow(val: unknown): string {
  const n = val == null || val === '' ? NaN : Number(val);
  if (!Number.isFinite(n)) return '-';
  const abs = Math.abs(n);
  if (abs >= 1e8) return `${(n / 1e8).toFixed(2)}亿`;
  if (abs >= 1e4) return `${(n / 1e4).toFixed(2)}万`;
  return n.toFixed(2);
}

function fmtPct(val: unknown): string {
  const n = val == null || val === '' ? NaN : Number(val);
  if (!Number.isFinite(n)) return '-';
  return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`;
}

function pctColor(val: unknown): string {
  const n = val == null || val === '' ? NaN : Number(val);
  if (!Number.isFinite(n)) return 'text-fin-muted';
  return n > 0 ? 'text-fin-success' : n < 0 ? 'text-fin-danger' : 'text-fin-muted';
}

function flowColor(val: unknown): string {
  const n = val == null || val === '' ? NaN : Number(val);
  if (!Number.isFinite(n)) return 'text-fin-muted';
  return n > 0 ? 'text-fin-success' : n < 0 ? 'text-fin-danger' : 'text-fin-muted';
}

// --- Sub-tables ---

export function FundFlowTable({ rows }: { rows: RowData[] }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-fin-muted border-b border-fin-border">
          <th className="pb-1.5 text-left font-medium whitespace-nowrap">代码</th>
          <th className="pb-1.5 text-left font-medium whitespace-nowrap">名称</th>
          <th className="pb-1.5 text-right font-medium whitespace-nowrap">涨跌幅</th>
          <th className="pb-1.5 text-right font-medium whitespace-nowrap">主力净流入</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-fin-border/30">
        {rows.map((row, i) => (
          <tr key={`${String(row.symbol ?? i)}:${i}`} className="hover:bg-fin-hover/30 transition-colors">
            <td className="py-1.5 font-mono text-fin-muted whitespace-nowrap">{String(row.symbol ?? '-')}</td>
            <td className="py-1.5 text-fin-text max-w-[80px] truncate whitespace-nowrap">{String(row.name ?? '-')}</td>
            <td className={`py-1.5 text-right whitespace-nowrap tabular-nums ${pctColor(row.change_percent)}`}>{fmtPct(row.change_percent)}</td>
            <td className={`py-1.5 text-right whitespace-nowrap tabular-nums ${flowColor(row.main_net_inflow)}`}>{fmtFlow(row.main_net_inflow)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function NorthboundTable({ rows }: { rows: RowData[] }) {
  return (
    <table className="w-full text-xs">
      <thead><tr className="text-fin-muted border-b border-fin-border">
        <th className="pb-1.5 text-left font-medium whitespace-nowrap">交易日期</th>
        <th className="pb-1.5 text-right font-medium whitespace-nowrap">沪股通成交额</th>
        <th className="pb-1.5 text-right font-medium whitespace-nowrap">深股通成交额</th>
        <th className="pb-1.5 text-right font-medium whitespace-nowrap">合计成交额</th>
      </tr></thead>
      <tbody className="divide-y divide-fin-border/30">{rows.map((row, i) => (
        <tr key={String(row.trade_date ?? i)} className="hover:bg-fin-hover/30 transition-colors">
          <td className="py-1.5 text-fin-muted whitespace-nowrap">{String(row.trade_date ?? '-')}</td>
          <td className="py-1.5 text-right text-fin-text whitespace-nowrap tabular-nums">{fmtFlow(row.sh_turnover)}</td>
          <td className="py-1.5 text-right text-fin-text whitespace-nowrap tabular-nums">{fmtFlow(row.sz_turnover)}</td>
          <td className="py-1.5 text-right text-fin-text whitespace-nowrap tabular-nums">{fmtFlow(row.total_turnover)}</td>
        </tr>
      ))}</tbody>
    </table>
  );
}

export function LimitBoardTable({ rows }: { rows: RowData[] }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-fin-muted border-b border-fin-border">
          <th className="pb-1.5 text-left font-medium whitespace-nowrap">代码</th>
          <th className="pb-1.5 text-left font-medium whitespace-nowrap">名称</th>
          <th className="pb-1.5 text-right font-medium whitespace-nowrap">涨跌幅</th>
          <th className="pb-1.5 text-right font-medium whitespace-nowrap">换手率</th>
          <th className="pb-1.5 text-right font-medium whitespace-nowrap">状态</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-fin-border/30">
        {rows.map((row, i) => (
          <tr key={`${String(row.symbol ?? i)}:${i}`} className="hover:bg-fin-hover/30 transition-colors">
            <td className="py-1.5 font-mono text-fin-muted whitespace-nowrap">{String(row.symbol ?? '-')}</td>
            <td className="py-1.5 text-fin-text max-w-[80px] truncate whitespace-nowrap">{String(row.name ?? '-')}</td>
            <td className={`py-1.5 text-right whitespace-nowrap tabular-nums ${pctColor(row.change_percent)}`}>{fmtPct(row.change_percent)}</td>
            <td className="py-1.5 text-right text-fin-text whitespace-nowrap tabular-nums">{String(row.turnover_rate ?? '-')}</td>
            <td className="py-1.5 text-right text-fin-text whitespace-nowrap tabular-nums">{row.limit_side === 'up' ? '涨停' : row.limit_side === 'down' ? '跌停' : '-'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function LhbTable({ rows }: { rows: RowData[] }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-fin-muted border-b border-fin-border">
          <th className="pb-1.5 text-left font-medium whitespace-nowrap">代码</th>
          <th className="pb-1.5 text-left font-medium whitespace-nowrap">名称</th>
          <th className="pb-1.5 text-left font-medium whitespace-nowrap">上榜日期</th>
          <th className="pb-1.5 text-right font-medium whitespace-nowrap">涨跌幅</th>
          <th className="pb-1.5 text-right font-medium whitespace-nowrap">净买入</th>
          <th className="pb-1.5 text-left font-medium pl-2 whitespace-nowrap">上榜原因</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-fin-border/30">
        {rows.map((row, i) => (
          <tr key={`${String(row.symbol ?? i)}:${i}`} className="hover:bg-fin-hover/30 transition-colors">
            <td className="py-1.5 font-mono text-fin-muted whitespace-nowrap">{String(row.symbol ?? '-')}</td>
            <td className="py-1.5 text-fin-text max-w-[70px] truncate whitespace-nowrap">{String(row.name ?? '-')}</td>
            <td className="py-1.5 text-fin-muted whitespace-nowrap">{String(row.trade_date ?? '-').slice(0, 10)}</td>
            <td className={`py-1.5 text-right whitespace-nowrap tabular-nums ${pctColor(row.change_percent)}`}>{fmtPct(row.change_percent)}</td>
            <td className={`py-1.5 text-right whitespace-nowrap tabular-nums ${flowColor(row.net_buy)}`}>{fmtFlow(row.net_buy)}</td>
            <td className="py-1.5 pl-2 text-fin-muted max-w-[100px] truncate whitespace-nowrap" title={String(row.reason ?? '')}>
              {String(row.reason ?? '-')}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ConceptTable({ rows }: { rows: RowData[] }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-fin-muted border-b border-fin-border">
          <th className="pb-1.5 text-left font-medium whitespace-nowrap">板块名称</th>
          <th className="pb-1.5 text-right font-medium whitespace-nowrap">涨跌幅</th>
          <th className="pb-1.5 text-right font-medium whitespace-nowrap">主力净流入</th>
          <th className="pb-1.5 text-right font-medium whitespace-nowrap">涨/跌家数</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-fin-border/30">
        {rows.map((row, i) => (
          <tr key={String(row.concept_code ?? i)} className="hover:bg-fin-hover/30 transition-colors">
            <td className="py-1.5 text-fin-text max-w-[110px] truncate whitespace-nowrap">{String(row.concept_name ?? '-')}</td>
            <td className={`py-1.5 text-right whitespace-nowrap tabular-nums ${pctColor(row.change_percent)}`}>{fmtPct(row.change_percent)}</td>
            <td className={`py-1.5 text-right whitespace-nowrap tabular-nums ${flowColor(row.main_net_inflow)}`}>{fmtFlow(row.main_net_inflow)}</td>
            <td className="py-1.5 text-right whitespace-nowrap tabular-nums">
              <span className="text-fin-success">{String(row.up_count ?? '-')}</span>
              <span className="text-fin-muted mx-0.5">/</span>
              <span className="text-fin-danger">{String(row.down_count ?? '-')}</span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// --- Main Panel ---

const makeTabState = <T,>(init: T): Record<TabKey, T> => ({
  fund_flow: init, northbound: init, limit_board: init, lhb: init, concept: init,
});

export async function loadCNMarketTab(tab: TabKey): Promise<CNMarketListResponse> {
  const requests = {
    fund_flow: () => apiClient.getCNFundFlow(20),
    northbound: () => apiClient.getCNNorthbound(20),
    limit_board: () => apiClient.getCNLimitBoard(20),
    lhb: () => apiClient.getCNLhb(20),
    concept: () => apiClient.getCNConcept({ limit: 20 }),
  };
  try {
    const result = await requests[tab]();
    if (!result.success || !Array.isArray(result.items)) {
      throw new Error(result.error || '行情源未返回有效数据，请稍后重试。');
    }
    return result;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const detail = error.response?.data?.detail;
      if (typeof detail === 'string') throw new Error(detail);
      if (error.code === 'ECONNABORTED') throw new Error('行情请求超时，请稍后重试。');
    }
    throw error;
  }
}

export const CNMarketPanel: React.FC = () => {
  const inFlight = useRef(new Set<TabKey>());
  const [responses, setResponses] = useState<Record<TabKey, CNMarketListResponse | null>>(makeTabState(null));
  const [activeTab, setActiveTab] = useState<TabKey>('fund_flow');
  const [tabData, setTabData]     = useState<Record<TabKey, RowData[]>>(makeTabState([]));
  const [loading, setLoading]     = useState<Record<TabKey, boolean>>(makeTabState(false));
  const [errors, setErrors]       = useState<Record<TabKey, string | null>>(makeTabState(null));

  const loadTab = useCallback(async (tab: TabKey) => {
    if (inFlight.current.has(tab)) return;
    inFlight.current.add(tab);
    setLoading(prev => ({ ...prev, [tab]: true }));
    setErrors(prev => ({ ...prev, [tab]: null }));
    try {
      const result = await loadCNMarketTab(tab);
      setTabData(prev => ({ ...prev, [tab]: result.items }));
      setResponses(prev => ({ ...prev, [tab]: result }));
    } catch (e) {
      setErrors(prev => ({ ...prev, [tab]: e instanceof Error ? e.message : '加载失败' }));
    } finally {
      inFlight.current.delete(tab);
      setLoading(prev => ({ ...prev, [tab]: false }));
    }
  }, []);

  useEffect(() => { void loadTab('fund_flow'); }, [loadTab]);

  const handleTabClick = (tab: TabKey) => {
    setActiveTab(tab);
    if (!responses[tab] && !loading[tab]) {
      loadTab(tab);
    }
  };

  const response = responses[activeTab];
  const rows      = tabData[activeTab];
  const isLoading = loading[activeTab];
  const error     = errors[activeTab];

  return (
    <section aria-busy={isLoading} className="rounded-xl border border-fin-border bg-fin-card p-4">
      {/* Header */}
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-fin-text">A股市场</h3>
        <button
          type="button"
          onClick={() => loadTab(activeTab)}
          disabled={isLoading}
          className="p-1.5 rounded-lg text-fin-muted hover:text-fin-text hover:bg-fin-hover transition-colors disabled:opacity-40"
          title="刷新"
          aria-label="刷新行情"
        >
          <RefreshCw size={13} className={isLoading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 mb-3 border-b border-fin-border pb-2 overflow-x-auto">
        {TABS.map(({ key, label, icon }) => (
          <button
            key={key}
            type="button"
            onClick={() => handleTabClick(key)}
            aria-pressed={activeTab === key}
            className={[
              'flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium whitespace-nowrap transition-colors',
              activeTab === key
                ? 'bg-fin-primary/15 text-fin-primary'
                : 'text-fin-muted hover:text-fin-text hover:bg-fin-hover',
            ].join(' ')}
          >
            {icon}
            {label}
          </button>
        ))}
      </div>

      {/* Error */}
      {error && <div role="alert" className="mb-3 text-xs text-fin-danger">
        <p>{error}{rows.length > 0 ? ' 当前保留上次成功加载的数据。' : ''}</p>
        <button type="button" onClick={() => loadTab(activeTab)} disabled={isLoading}
          className="mt-2 rounded px-2 py-1 text-fin-primary hover:bg-fin-hover disabled:opacity-40">重试</button>
      </div>}
      {response && <div className="mb-3 space-y-1 text-xs text-fin-muted">
        <p>来源：东方财富{response.as_of ? ` · 数据日期：${response.as_of}` : ''}
          {response.retrieved_at ? ` · 获取时间：${new Date(response.retrieved_at).toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai', hour12: false })}（北京时间）` : ''}
        </p>
        {response.note && <p>{response.note}</p>}
        {response.warning && <p role="status" className="text-fin-warning">{response.warning}</p>}
      </div>}

      {/* Loading skeleton */}
      {isLoading && rows.length === 0 && (
        <div role="status" aria-label="正在加载行情" className="space-y-1.5">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-7 rounded bg-fin-border/30 animate-pulse" />
          ))}
        </div>
      )}

      {/* Empty prompt */}
      {!isLoading && !error && rows.length === 0 && (
        <div className="flex flex-col items-center justify-center py-8 text-fin-muted gap-2">
          <p className="text-xs">{response ? '当前数据源暂无记录' : '等待加载行情'}</p>
          <button
            type="button"
            data-testid="cn-market-load-btn"
            onClick={() => loadTab(activeTab)}
            className="text-xs text-fin-primary hover:underline"
          >
            重新加载
          </button>
        </div>
      )}

      {/* Table */}
      {rows.length > 0 && (
        <div className="overflow-x-auto max-h-72 overflow-y-auto">
          {activeTab === 'fund_flow' && <FundFlowTable rows={rows} />}
          {activeTab === 'northbound' && <NorthboundTable rows={rows} />}
          {activeTab === 'limit_board' && <LimitBoardTable rows={rows} />}
          {activeTab === 'lhb' && <LhbTable rows={rows} />}
          {activeTab === 'concept' && <ConceptTable rows={rows} />}
        </div>
      )}
    </section>
  );
};

export default CNMarketPanel;
