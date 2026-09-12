import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ArrowLeft, ArrowRight, BookOpen, Check, CheckCircle2, ChevronLeft, ChevronRight, FileText, FlaskConical, GitBranch, Loader2, Moon, RefreshCw, Search, ShieldCheck, Sun, Upload, X } from 'lucide-react';
import { annualReportsApi, type AnnualAnalysis, type AnnualCitation, type AnnualDocument, type AnnualPage } from '../api/annualReports';
import { Button, Card, Input } from '../components/ui';
import { useStore } from '../store/useStore';

const EXAMPLES = [
  '营业收入增长，但经营活动现金流下降，年报中有哪些原因？',
  '对比所选年度的营业收入和净利润，计算同比变化。',
  '公司披露了哪些主要经营风险？请逐条给出原文依据。',
];
const NODE_LABELS: Record<string, string> = {
  scope: '锁定资料范围', plan: '拆解问题', retrieve: '检索年报', assess: '检查证据',
  retry: '补充检索', refine: '改写查询', calculate: '核对计算', answer: '整理回答',
  verify: '检查证据', synthesize: '整理回答', prepare: '准备问题', evidence_check: '检查证据',
  supplement: '补充检索',
};
const message = (error: unknown) => error instanceof Error ? error.message : '请求未完成，请稍后重试。';
const displayValue = (value: unknown): string => value === null || value === undefined ? '—'
  : typeof value === 'object' ? JSON.stringify(value) : String(value);
const amount = (value: unknown) => {
  const raw = displayValue(value);
  if (!/^-?\d+(\.\d+)?$/.test(raw)) return raw;
  const [integer, fraction] = raw.split('.');
  return integer.replace(/\B(?=(\d{3})+(?!\d))/g, ',') + (fraction === undefined ? '' : `.${fraction}`);
};
const focusClass = 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fin-primary focus-visible:ring-offset-2';

function SourcePageDialog({ citation, document, onClose }: {
  citation: AnnualCitation | null;
  document?: AnnualDocument;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [pageNumber, setPageNumber] = useState(citation?.page ?? 1);
  const [page, setPage] = useState<AnnualPage | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const element = dialog.current;
    const opener = window.document.activeElement instanceof HTMLElement ? window.document.activeElement : null;
    if (citation) element?.showModal();
    else element?.close();
    return () => {
      element?.close();
      opener?.focus({ preventScroll: true });
    };
  }, [citation]);
  useEffect(() => {
    if (!citation) return;
    const controller = new AbortController();
    setLoading(true);
    setError('');
    setPage(null);
    annualReportsApi.page(citation.document_id, pageNumber, controller.signal)
      .then((data) => { if (!controller.signal.aborted) setPage(data); })
      .catch((error: unknown) => { if (!controller.signal.aborted) setError(message(error)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [citation, pageNumber, retry]);

  return (
    <dialog ref={dialog} onCancel={(event) => { event.preventDefault(); onClose(); }} onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}
      aria-labelledby="annual-source-title"
      className="m-auto max-h-[90dvh] w-[calc(100%_-_2rem)] max-w-3xl overflow-hidden rounded-2xl border border-fin-border bg-fin-card p-0 text-fin-text shadow-2xl backdrop:bg-black/50">
      <div className="flex max-h-[90dvh] flex-col" onClick={(event) => event.stopPropagation()}>
        <header className="flex items-start justify-between gap-3 border-b border-fin-border p-5">
          <div className="min-w-0">
            <div className="text-xs font-medium text-fin-primary">来源原文 · 第 {pageNumber} 页</div>
            <h2 id="annual-source-title" className="mt-1 break-words font-semibold">{citation?.filename}</h2>
            <p className="mt-1 text-xs text-fin-text-secondary">{citation?.company} · {citation?.year} 年度 · 提取文本</p>
          </div>
          <Button aria-label="关闭原文" variant="ghost" onClick={onClose}><X size={18} /></Button>
        </header>
        <div className="min-h-40 overflow-y-auto p-5 sm:p-7" aria-live="polite" aria-busy={loading}>
          {loading ? <p className="flex items-center gap-2 text-sm text-fin-text-secondary"><Loader2 size={16} className="animate-spin motion-reduce:animate-none" />正在读取原文…</p>
            : error ? <div role="alert"><p className="text-sm text-fin-danger">{error}</p><Button className="mt-3" onClick={() => setRetry((value) => value + 1)}>重试读取</Button></div>
            : <p className="whitespace-pre-wrap break-words text-sm leading-7">{page?.text || '此页没有可提取的文本。'}</p>}
        </div>
        <footer className="flex items-center justify-between gap-3 border-t border-fin-border p-4 text-xs text-fin-text-secondary">
          <Button size="sm" disabled={pageNumber <= 1 || loading} onClick={() => setPageNumber((value) => value - 1)}><ChevronLeft size={15} />上一页</Button>
          <span>{pageNumber} / {document?.page_count ?? citation?.page}</span>
          <Button size="sm" disabled={pageNumber >= (document?.page_count ?? citation?.page ?? 1) || loading} onClick={() => setPageNumber((value) => value + 1)}>下一页<ChevronRight size={15} /></Button>
        </footer>
      </div>
    </dialog>
  );
}

function CalculationTable({ rows, citations, onOpenSource }: {
  rows: Record<string, unknown>[];
  citations: AnnualCitation[];
  onOpenSource: (citation: AnnualCitation) => void;
}) {
  if (!rows.length) return null;
  return (
    <section className="mt-6" aria-labelledby="annual-calculations">
      <h3 id="annual-calculations" className="mb-3 text-sm font-semibold">计算核对</h3>
      <div className="overflow-x-auto rounded-lg border border-fin-border" tabIndex={0} role="region" aria-label="计算核对表，可横向滚动">
        <table className="w-full text-left text-xs">
          <caption className="sr-only">基于所选年报数据的计算结果</caption>
          <thead className="bg-fin-bg-secondary"><tr>{['公司 / 指标', '对比年度', '基期值', '本期值', '变化额', '变化率'].map((label) => <th key={label} scope="col" className="whitespace-nowrap px-3 py-2.5 font-medium">{label}</th>)}</tr></thead>
          {rows.map((row, index) => <tbody key={index}>
            <tr className="border-t border-fin-border">
              <th scope="row" className="min-w-28 px-3 py-3 font-medium"><div>{displayValue(row.company)}</div><div className="mt-1 text-fin-text-secondary">{displayValue(row.metric)}</div></th>
              <td className="whitespace-nowrap px-3 py-3">{displayValue(row.from_year)} → {displayValue(row.to_year)}<div className="mt-1 text-fin-text-secondary">{row.comparison_type === 'year_over_year' ? '同比' : '跨期'}</div></td>
              <td className="whitespace-nowrap px-3 py-3 tabular-nums">{amount(row.from_value)}<span className="ml-1 text-fin-text-secondary">{displayValue(row.unit)}</span></td>
              <td className="whitespace-nowrap px-3 py-3 tabular-nums">{amount(row.to_value)}<span className="ml-1 text-fin-text-secondary">{displayValue(row.unit)}</span></td>
              <td className="whitespace-nowrap px-3 py-3 tabular-nums">{amount(row.delta)}<span className="ml-1 text-fin-text-secondary">{displayValue(row.unit)}</span></td>
              <td className="min-w-28 px-3 py-3 tabular-nums">{row.change_pct === null ? <span className="text-fin-text-secondary">不计算（基期非正）</span> : typeof row.change_pct === 'number' ? `${row.change_pct.toFixed(2)}%` : '—'}</td>
            </tr>
            <tr><td colSpan={6} className="px-3 pb-3"><details className="rounded-lg bg-fin-bg p-3"><summary className={`w-fit cursor-pointer rounded text-fin-text-secondary ${focusClass}`}>核对公式与来源</summary><p className="mt-3 whitespace-pre-wrap leading-6">{displayValue(row.formula)}</p><div className="mt-2 flex flex-wrap gap-2">{Array.isArray(row.source_labels) && row.source_labels.map((label, sourceIndex) => {
              const citation = citations.find((hit) => hit.label.replaceAll('[', '').replaceAll(']', '') === String(label).replaceAll('[', '').replaceAll(']', ''));
              return citation ? <button type="button" key={sourceIndex} onClick={() => onOpenSource(citation)} className={`rounded border border-fin-border px-2 py-1 text-fin-primary hover:bg-fin-hover ${focusClass}`}>{String(label)} · {citation.year} 年第 {citation.page} 页</button> : <span key={sourceIndex}>{String(label)}</span>;
            })}</div></details></td></tr>
          </tbody>)}
        </table>
      </div>
    </section>
  );
}

export function AnnualReportsPage() {
  const { theme, setTheme } = useStore();
  const [documents, setDocuments] = useState<AnnualDocument[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError] = useState('');
  const [company, setCompany] = useState('');
  const [year, setYear] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [demoLoading, setDemoLoading] = useState(false);
  const [notice, setNotice] = useState('');
  const [question, setQuestion] = useState('');
  const [analysis, setAnalysis] = useState<AnnualAnalysis | null>(null);
  const [analyzedQuestion, setAnalyzedQuestion] = useState('');
  const [analyzedIds, setAnalyzedIds] = useState<string[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState('');
  const [source, setSource] = useState<AnnualCitation | null>(null);
  const listController = useRef<AbortController | null>(null);
  const uploadController = useRef<AbortController | null>(null);
  const analyzeController = useRef<AbortController | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLElement>(null);
  const selectedDocuments = useMemo(() => documents.filter((doc) => selectedIds.includes(doc.id)), [documents, selectedIds]);
  const scopeChanged = analysis && (selectedIds.length !== analyzedIds.length || selectedIds.some((id) => !analyzedIds.includes(id)));
  const evidenceGaps = Array.isArray(analysis?.metrics.evidence_gaps) ? analysis.metrics.evidence_gaps.filter((gap): gap is string => typeof gap === 'string') : [];

  const refresh = useCallback(async () => {
    listController.current?.abort();
    const controller = new AbortController();
    listController.current = controller;
    setListLoading(true);
    setListError('');
    try {
      const data = await annualReportsApi.list(controller.signal);
      if (controller.signal.aborted) return;
      setDocuments(data.documents);
      setSelectedIds((current) => current.filter((id) => data.documents.some((doc) => doc.id === id)));
    } catch (error) {
      if (!controller.signal.aborted) setListError(message(error));
    } finally {
      if (!controller.signal.aborted) setListLoading(false);
    }
  }, []);
  useEffect(() => {
    void refresh();
    return () => {
      listController.current?.abort();
      uploadController.current?.abort();
      analyzeController.current?.abort();
    };
  }, [refresh]);

  const upload = async (event: FormEvent) => {
    event.preventDefault();
    setUploadError('');
    setNotice('');
    if (!company.trim()) { setUploadError('请填写年报所属公司。'); return; }
    const parsedYear = Number(year);
    if (!year.trim() || !Number.isInteger(parsedYear) || parsedYear < 1900 || parsedYear > new Date().getFullYear() + 1) {
      setUploadError('请填写有效的四位报告年度。'); return;
    }
    if (!file) { setUploadError('请选择 PDF、TXT 或 Markdown 文件。'); return; }
    if (!/\.(pdf|txt|md)$/i.test(file.name)) { setUploadError('仅支持 PDF、TXT 和 Markdown（.md）文件。'); return; }
    if (!file.size) { setUploadError('文件为空，请重新选择。'); return; }
    const controller = new AbortController();
    uploadController.current = controller;
    setUploading(true);
    try {
      const doc = await annualReportsApi.upload(file, company, parsedYear, controller.signal);
      if (controller.signal.aborted) return;
      setDocuments((current) => [doc, ...current.filter((item) => item.id !== doc.id)]);
      setSelectedIds((current) => [...new Set([...current, doc.id])]);
      setFile(null);
      if (fileInput.current) fileInput.current.value = '';
      setNotice(`已保存并选中 ${doc.company} ${doc.year} 年报。`);
    } catch (error) {
      if (!controller.signal.aborted) setUploadError(message(error));
    } finally {
      if (!controller.signal.aborted) setUploading(false);
    }
  };

  const loadDemo = async () => {
    const controller = new AbortController();
    uploadController.current = controller;
    setDemoLoading(true);
    setUploadError('');
    setNotice('');
    try {
      const data = await annualReportsApi.demo(controller.signal);
      if (controller.signal.aborted) return;
      setDocuments((current) => [...data.documents, ...current.filter((item) => !data.documents.some((doc) => doc.id === item.id))]);
      setSelectedIds(data.documents.map((doc) => doc.id));
      setQuestion(EXAMPLES[0]);
      setNotice('已载入人工合成示例，仅供体验流程，不代表真实公司年报。');
    } catch (error) {
      if (!controller.signal.aborted) setUploadError(message(error));
    } finally {
      if (!controller.signal.aborted) setDemoLoading(false);
    }
  };

  const analyze = async (event: FormEvent) => {
    event.preventDefault();
    if (!question.trim() || !selectedDocuments.length || analyzing) return;
    const controller = new AbortController();
    analyzeController.current = controller;
    const ids = selectedDocuments.map((doc) => doc.id);
    setAnalyzing(true);
    setAnalysisError('');
    setAnalysis(null);
    setAnalyzedQuestion(question.trim());
    setAnalyzedIds(ids);
    try {
      const data = await annualReportsApi.analyze({ question: question.trim(), document_ids: ids, max_retries: 1, mode: 'hybrid' }, controller.signal);
      if (controller.signal.aborted) return;
      setAnalysis(data);
      requestAnimationFrame(() => resultRef.current?.focus({ preventScroll: true }));
    } catch (error) {
      if (!controller.signal.aborted) setAnalysisError(message(error));
    } finally {
      if (!controller.signal.aborted) setAnalyzing(false);
    }
  };

  return (
    <main id="main-content" className="h-dvh overflow-y-auto bg-fin-bg font-sans text-fin-text">
      <header className="border-b border-fin-border bg-fin-card px-4 sm:px-8">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-3 py-4">
          <Link to="/chat" className={`inline-flex min-h-10 items-center gap-2 rounded-lg text-sm font-semibold ${focusClass}`}><ArrowLeft size={17} /><span>FinSight <span className="hidden font-normal text-fin-text-secondary sm:inline">/ 研究工作区</span></span></Link>
          <Button variant="ghost" aria-label={theme === 'dark' ? '切换浅色主题' : '切换深色主题'} onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}</Button>
        </div>
      </header>
      <div className="mx-auto max-w-[1440px] px-4 py-6 sm:px-8 sm:py-8">
        <div className="mb-7 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="mb-2 flex items-center gap-2 text-xs font-semibold tracking-wide text-fin-primary"><BookOpen size={15} />ANNUAL REPORT RESEARCH</p>
            <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">中文年报分析</h1>
            <p className="mt-2 text-sm leading-6 text-fin-text-secondary">先选资料，再核对证据。让每个结论都能回到年报原文。</p>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-fin-border bg-fin-card px-3 py-2 text-xs text-fin-text-secondary"><ShieldCheck size={15} className="text-fin-primary" />仅检索本次选中的资料</div>
        </div>
        <div className="grid items-start gap-5 lg:grid-cols-[330px_minmax(0,1fr)] xl:gap-7">
          <aside className="min-w-0 space-y-4" aria-label="年报资料管理">
            <Card className="p-5">
              <h2 className="flex items-center gap-2 text-base font-semibold"><span className="inline-flex h-6 w-6 items-center justify-center rounded-md bg-fin-primary/10 text-xs text-fin-primary">1</span>准备年报资料</h2>
              <p className="mb-4 mt-2 text-xs leading-5 text-fin-text-secondary">填写报告所属公司和年度，上传后自动保存到资料库。</p>
              <form onSubmit={(event) => void upload(event)} noValidate className="space-y-3" aria-busy={uploading}>
                <Input id="annual-company" label="公司名称 *" placeholder="例如：星海制造" value={company} onChange={(event) => setCompany(event.target.value)} required maxLength={160} disabled={uploading || demoLoading} />
                <Input id="annual-year" label="报告年度 *" placeholder="例如：2024" inputMode="numeric" value={year} onChange={(event) => setYear(event.target.value)} required maxLength={4} disabled={uploading || demoLoading} />
                <div>
                  <label htmlFor="annual-file" className="mb-1 block text-xs font-medium text-fin-text-secondary">年报文件 *</label>
                  <input ref={fileInput} id="annual-file" type="file" accept=".pdf,.txt,.md" onChange={(event) => setFile(event.target.files?.[0] ?? null)} disabled={uploading || demoLoading}
                    className={`block min-h-11 w-full min-w-0 rounded-lg border border-dashed border-fin-border bg-fin-bg p-2 text-xs text-fin-text-secondary file:mr-2 file:rounded file:border-0 file:bg-fin-primary/10 file:px-2 file:py-1 file:text-fin-primary ${focusClass}`} aria-describedby="annual-file-help" />
                  <p id="annual-file-help" className="mt-1.5 text-xs leading-5 text-fin-text-secondary">支持 PDF、TXT、MD。扫描件请先识别文字；页码以提取后的文档页序为准。</p>
                </div>
                <Button type="submit" variant="primary" className="w-full" disabled={uploading || demoLoading}>{uploading ? <Loader2 size={16} className="animate-spin motion-reduce:animate-none" /> : <Upload size={16} />}{uploading ? '正在解析并保存…' : '上传并保存'}</Button>
              </form>
              {uploadError && <p role="alert" className="mt-3 break-words text-sm text-fin-danger">{uploadError}</p>}
              {notice && <p role="status" className="mt-3 text-xs leading-5 text-fin-text-secondary">{notice}</p>}
              <div className="mt-4 border-t border-fin-border pt-4">
                <Button variant="ghost" className="w-full" disabled={uploading || demoLoading || analyzing} onClick={() => void loadDemo()}>{demoLoading ? <Loader2 size={15} className="animate-spin motion-reduce:animate-none" /> : <FlaskConical size={15} />}载入示例资料</Button>
                <p className="mt-1 text-center text-xs text-fin-text-secondary">人工合成数据，仅供体验分析流程</p>
              </div>
            </Card>
            <Card className="p-5">
              <div className="flex items-center justify-between gap-2">
                <h2 className="text-sm font-semibold">资料库 <span className="font-normal text-fin-text-secondary">{documents.length}</span></h2>
                <Button size="sm" variant="ghost" disabled={listLoading || uploading || demoLoading || analyzing} onClick={() => void refresh()} aria-label="刷新资料列表"><RefreshCw size={14} className={listLoading ? 'animate-spin motion-reduce:animate-none' : ''} />刷新</Button>
              </div>
              <div className="mb-3 mt-1 flex items-center justify-between text-xs text-fin-text-secondary"><span>已选 {selectedDocuments.length} 份</span><Button size="sm" variant="ghost" disabled={!selectedIds.length || analyzing} onClick={() => setSelectedIds([])}>清空选择</Button></div>
              {listError && <p role="alert" className="mb-3 text-sm leading-5 text-fin-danger">{listError}</p>}
              {listLoading && !documents.length ? <p role="status" className="py-5 text-center text-sm text-fin-text-secondary">正在加载资料…</p> : !documents.length ? <div className="rounded-lg border border-dashed border-fin-border px-4 py-6 text-center"><FileText size={25} className="mx-auto mb-2 text-fin-text-secondary" /><p className="text-sm">还没有年报资料</p><p className="mt-1 text-xs leading-5 text-fin-text-secondary">上传第一份年报，或载入示例开始体验。</p></div> : <div className="max-h-[460px] space-y-2 overflow-y-auto pr-1">
                {documents.map((doc) => <label key={doc.id} className={`block cursor-pointer rounded-xl border p-3 transition-colors ${selectedIds.includes(doc.id) ? 'border-fin-primary/50 bg-fin-primary/5' : 'border-fin-border hover:bg-fin-hover'} ${analyzing ? 'opacity-60' : ''}`}>
                  <div className="flex items-start gap-2.5"><input type="checkbox" className="mt-1 h-4 w-4 shrink-0 accent-[rgb(var(--fin-primary))]" checked={selectedIds.includes(doc.id)} disabled={analyzing} onChange={(event) => setSelectedIds((current) => event.target.checked ? [...current, doc.id] : current.filter((id) => id !== doc.id))} aria-label={`选择 ${doc.company} ${doc.year} 年报 ${doc.filename}`} /><div className="min-w-0"><div className="text-sm font-medium">{doc.company} <span className="text-fin-primary">{doc.year}</span></div><p className="mt-1 break-words text-xs text-fin-text-secondary">{doc.filename}</p><p className="mt-1.5 text-xs text-fin-text-secondary">{doc.page_count} 页 · {doc.chunk_count} 个段落</p></div></div>
                  {doc.warnings?.length > 0 && <ul className="mt-2 space-y-1 border-t border-fin-border pt-2 text-xs leading-5 text-fin-text-secondary">{doc.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul>}
                </label>)}
              </div>}
            </Card>
          </aside>
          <div className="min-w-0 space-y-5">
            <Card className="p-5 sm:p-6">
              <h2 className="flex items-center gap-2 text-base font-semibold"><span className="inline-flex h-6 w-6 items-center justify-center rounded-md bg-fin-primary/10 text-xs text-fin-primary">2</span>围绕所选年报提问</h2>
              <div className="my-4 rounded-lg bg-fin-bg px-3 py-2.5 text-xs leading-5 text-fin-text-secondary" aria-live="polite">{selectedDocuments.length ? <><span className="font-medium text-fin-text">本次范围：</span>{selectedDocuments.map((doc) => `${doc.company} ${doc.year}`).join('、')}</> : '请先在资料库中勾选年报，支持跨年度同时选择。'}</div>
              <form onSubmit={(event) => void analyze(event)}>
                <label htmlFor="annual-question" className="mb-2 block text-sm font-medium">你想核对什么？</label>
                <textarea id="annual-question" rows={4} maxLength={2000} value={question} onChange={(event) => setQuestion(event.target.value)} disabled={analyzing}
                  placeholder="例如：2024 年营业收入增长，经营活动现金流为什么下降？请结合年报解释。"
                  className={`w-full resize-y rounded-xl border border-fin-border bg-fin-bg px-4 py-3 text-sm leading-6 placeholder:text-fin-text-secondary disabled:opacity-60 ${focusClass}`} />
                <div className="mt-3 flex flex-wrap items-center justify-between gap-3"><p className="text-xs text-fin-text-secondary">回答附带来源页码，证据不足时会明确说明。</p>{analyzing ? <Button key="cancel" type="button" onClick={(event) => { event.preventDefault(); analyzeController.current?.abort(); setAnalyzing(false); setAnalysisError('已取消本次分析，可以修改问题后重新开始。'); }}><X size={16} />取消分析</Button> : <Button key="analyze" type="submit" variant="primary" disabled={!question.trim() || !selectedDocuments.length || uploading || demoLoading}><Search size={16} />分析所选资料<ArrowRight size={15} /></Button>}</div>
              </form>
              <div className="mt-5 border-t border-fin-border pt-4"><p className="mb-2 text-xs font-medium text-fin-text-secondary">试试这样问</p><div className="flex flex-wrap gap-2">{EXAMPLES.map((example, index) => <button key={example} type="button" disabled={analyzing} onClick={() => setQuestion(example)} className={`rounded-lg border border-fin-border px-3 py-2 text-left text-xs leading-5 hover:border-fin-primary/40 hover:bg-fin-hover disabled:opacity-50 ${focusClass}`}>{['收入与现金流', '跨年指标对比', '经营风险核查'][index]}</button>)}</div></div>
            </Card>
            {analysisError && <div role="alert" className="rounded-xl border border-fin-warning/40 bg-fin-card p-4 text-sm leading-6">{analysisError}</div>}
            {analyzing && <Card className="p-7" role="status" aria-live="polite"><div className="flex items-center gap-3"><Loader2 size={20} className="animate-spin text-fin-primary motion-reduce:animate-none" /><div><p className="font-medium">正在检索并核对年报证据</p><p className="mt-1 text-sm text-fin-text-secondary">完成后将展示回答、引用原文和实际执行步骤。</p></div></div></Card>}
            {!analysis && !analyzing && !analysisError && <div className="rounded-xl border border-dashed border-fin-border px-6 py-12 text-center"><BookOpen size={31} className="mx-auto mb-4 text-fin-primary/70" /><h2 className="font-medium">把结论建立在可核对的证据上</h2><p className="mx-auto mt-2 max-w-md text-sm leading-6 text-fin-text-secondary">选中资料并提交问题后，这里会展示分析结果。点击来源即可查看原文，核对公司、年度与页码。</p><div className="mt-5 flex flex-wrap justify-center gap-4 text-xs text-fin-text-secondary"><span className="flex items-center gap-1"><Check size={13} />限定资料范围</span><span className="flex items-center gap-1"><Check size={13} />原文可追溯</span><span className="flex items-center gap-1"><Check size={13} />计算可核对</span></div></div>}
            {analysis && <section ref={resultRef} tabIndex={-1} className="space-y-5 outline-none" aria-label="年报分析结果">
              {scopeChanged && <p role="status" className="rounded-lg border border-fin-warning/40 bg-fin-card px-4 py-3 text-sm leading-6">资料选择已变化。下方仍是上一次分析的结果，请重新提交问题以更新。</p>}
              <Card className="p-5 sm:p-6">
                <div className="flex flex-wrap items-center justify-between gap-3"><h2 className="text-base font-semibold">分析回答</h2><span className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium text-fin-text ${analysis.status === 'complete' ? 'border-fin-success/40' : 'border-fin-warning/50'}`}>{analysis.status === 'complete' ? <CheckCircle2 size={14} className="text-fin-success" /> : <Search size={14} className="text-fin-warning" />}{analysis.status === 'complete' ? '证据充分' : '证据不足'}</span></div>
                <p className="mt-3 border-l-2 border-fin-primary/40 pl-3 text-sm leading-6 text-fin-text-secondary">{analyzedQuestion}</p>
                {typeof analysis.metrics.answer_mode === 'string' && <p className="mt-3 text-xs text-fin-text-secondary">{analysis.metrics.answer_mode === 'llm' ? '模型辅助摘录' : '证据摘录'} · {analysis.retrieval_mode.toLowerCase().includes('bm25') ? '关键词检索' : '混合检索'}</p>}
                {analysis.status === 'insufficient_evidence' && <p className="mt-4 rounded-lg bg-fin-bg px-3 py-2 text-sm leading-6 text-fin-text-secondary">当前资料尚不能完整支持结论。请根据下方说明补充对应公司、年度或主题的年报。</p>}
                {evidenceGaps.length > 0 && <div className="mt-3 rounded-lg border border-fin-border px-3 py-3 text-sm"><h3 className="font-medium">待补充的证据</h3><ul className="mt-2 list-disc space-y-1 pl-5 leading-6 text-fin-text-secondary">{evidenceGaps.map((gap, index) => <li key={index}>{gap}</li>)}</ul></div>}
                <div className="prose prose-sm mt-5 max-w-none break-words text-fin-text prose-headings:text-fin-text prose-p:leading-7 prose-a:text-fin-primary prose-strong:text-fin-text prose-code:text-fin-text prose-th:text-fin-text prose-td:text-fin-text dark:prose-invert"><ReactMarkdown remarkPlugins={[remarkGfm]}>{analysis.answer}</ReactMarkdown></div>
                <CalculationTable rows={analysis.calculations ?? []} citations={analysis.citations} onOpenSource={setSource} />
              </Card>
              <Card className="p-5 sm:p-6">
                <div className="flex flex-wrap items-center justify-between gap-2"><h2 className="text-base font-semibold">来源证据 <span className="text-sm font-normal text-fin-text-secondary">{analysis.citations.length}</span></h2><span className="text-xs text-fin-text-secondary">点击卡片，查看原文页</span></div>
                {analysis.citations.length ? <div className="mt-4 grid gap-3 xl:grid-cols-2">{analysis.citations.map((citation, index) => <button key={`${citation.id}-${index}`} type="button" onClick={() => setSource(citation)} className={`min-w-0 rounded-xl border border-fin-border p-4 text-left transition-colors hover:border-fin-primary/50 hover:bg-fin-hover ${focusClass}`} aria-label={`查看来源 ${citation.label || index + 1}：${citation.company} ${citation.year}，第 ${citation.page} 页`}><div className="flex items-start justify-between gap-2"><span className="rounded bg-fin-primary/10 px-2 py-1 text-xs font-semibold text-fin-primary">{citation.label || `[${index + 1}]`}</span><span className="flex shrink-0 items-center gap-1 text-xs text-fin-text-secondary">第 {citation.page} 页<ChevronRight size={13} /></span></div><p className="mt-2 text-sm font-medium">{citation.company} · {citation.year}</p><p className="mt-1 break-words text-xs text-fin-text-secondary">{citation.section || citation.filename}</p><p className="mt-3 line-clamp-3 whitespace-pre-wrap break-words text-xs leading-6 text-fin-text-secondary">{citation.text}</p></button>)}</div> : <p className="mt-4 rounded-lg bg-fin-bg p-4 text-sm text-fin-text-secondary">没有找到足够相关的原文。请检查资料范围，或将问题聚焦到年报中明确披露的事项。</p>}
              </Card>
              <Card className="p-5 sm:p-6"><details open><summary className={`cursor-pointer rounded text-sm font-semibold ${focusClass}`}><span className="ml-1 inline-flex items-center gap-2"><GitBranch size={16} className="text-fin-primary" />分析过程 <span className="font-normal text-fin-text-secondary">LangGraph</span></span></summary><ol className="mt-4 space-y-3">{analysis.trace.map((step, index) => <li key={`${step.node}-${index}`} className="flex items-start gap-3"><span className="mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-fin-bg-secondary text-[10px] text-fin-text-secondary">{index + 1}</span><div className="min-w-0"><div className="text-xs font-medium">{NODE_LABELS[step.node] ?? step.node}</div><p className="mt-1 break-words text-xs leading-5 text-fin-text-secondary">{step.detail}</p></div></li>)}</ol></details></Card>
            </section>}
            <p className="px-1 text-xs leading-5 text-fin-text-secondary">分析结果用于资料研究，请结合引用原文核实数据和统计口径。</p>
          </div>
        </div>
      </div>
      {source && <SourcePageDialog key={`${source.document_id}-${source.page}`} citation={source} document={documents.find((doc) => doc.id === source.document_id)} onClose={() => setSource(null)} />}
    </main>
  );
}
