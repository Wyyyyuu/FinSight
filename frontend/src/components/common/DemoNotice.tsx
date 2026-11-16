import { useState } from 'react';
import { X } from 'lucide-react';

export function DemoNotice({ mode }: { mode: 'demo' | 'preview' }) {
  const storageKey = `finsight-${mode}-notice-dismissed-v1`;
  const [dismissed, setDismissed] = useState(() => {
    try {
      return window.localStorage.getItem(storageKey) === '1';
    } catch {
      return false;
    }
  });

  const dismiss = () => {
    setDismissed(true);
    try {
      window.localStorage.setItem(storageKey, '1');
    } catch {
      // Dismiss for this visit even when browser storage is unavailable.
    }
  };

  if (dismissed) return null;

  return (
    <aside
      role="note"
      aria-label={mode === 'demo' ? '本地演示说明' : '公开预览说明'}
      className="fixed bottom-3 right-3 z-[100] max-w-[min(24rem,calc(100vw-1.5rem))] rounded-lg border border-fin-border bg-fin-card py-3 pl-4 pr-12 text-xs text-fin-text shadow-lg"
    >
      <button
        type="button"
        aria-label="关闭演示提示"
        onClick={dismiss}
        className="absolute right-0 top-0 flex h-11 w-11 items-center justify-center rounded-lg text-fin-muted transition-colors hover:bg-fin-hover hover:text-fin-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fin-primary"
      >
        <X size={16} aria-hidden="true" />
      </button>
      <strong className="mb-1 block">
        {mode === 'demo' ? '在线演示 · 本地后端' : '界面预览 · 在线分析尚未开放'}
      </strong>
      {mode === 'demo'
        ? '后端电脑在线时可分析；资料按浏览器隔离。登录、历史同步和管理功能暂未开放。'
        : '展示数据仅为示例。聊天、登录与年报分析将在接入后端后开放。'}
    </aside>
  );
}
