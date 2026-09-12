import { expect, test, type Page } from '@playwright/test';

const docs = [
  { id: 'report-2024', filename: '星海制造2024.txt', company: '星海制造', year: 2024, page_count: 2, chunk_count: 4, created_at: '2026-09-12', warnings: [] },
  { id: 'report-2023', filename: '星海制造2023.txt', company: '星海制造', year: 2023, page_count: 2, chunk_count: 3, created_at: '2026-09-12', warnings: [] },
];
const answer = {
  answer: '营业收入为 120 亿元，较上一年度增加 20%。[1]', status: 'complete',
  citations: [{ id: 'hit-1', document_id: 'report-2024', filename: docs[0].filename, company: docs[0].company, year: 2024, page: 1, section: '财务摘要', text: '2024 年营业收入 120 亿元。', score: 0.9, label: '[1]' }],
  trace: [{ node: 'retrieve', status: 'complete', detail: '在选中的 1 份年报中找到 2 条证据。' }, { node: 'verify', status: 'complete', detail: '证据包含公司、年度和原文页码。' }],
  metrics: { answer_mode: 'extractive' }, retrieval_mode: 'hybrid', calculations: [{ company: '星海制造', metric: '营业收入', from_year: 2023, to_year: 2024, from_value: '10000000000', to_value: '12000000000', delta: '2000000000', change_pct: 20, formula: '(120 - 100) / 100 × 100', unit: '元', comparison_type: 'year_over_year', source_labels: ['1'], operands: [] }],
};

async function setup(page: Page) {
  await page.route('**/api/annual-reports/documents', (route) => route.fulfill({ json: { documents: docs } }));
  await page.route('**/api/annual-reports/documents/*/pages/*', (route) => route.fulfill({ json: {
    document_id: 'report-2024', page: Number(route.request().url().split('/').at(-1)),
    text: '原文核验：2024 年营业收入为 120 亿元。经营活动现金流量净额为 8 亿元。',
    company: '星海制造', year: 2024, filename: docs[0].filename,
  } }));
}

test('scopes analysis to selected reports, shows evidence and opens a keyboard accessible source page', async ({ page }) => {
  await setup(page);
  let submitted: unknown;
  await page.route('**/api/annual-reports/analyze', (route) => {
    submitted = route.request().postDataJSON();
    return route.fulfill({ json: answer });
  });
  await page.goto('/annual-reports');
  await expect(page.getByRole('heading', { name: '中文年报分析' })).toBeVisible();
  await page.screenshot({ path: 'test-results/annual-reports-entry.png' });
  await expect(page.getByRole('button', { name: '分析所选资料' })).toBeDisabled();
  await page.getByRole('checkbox', { name: /选择 星海制造 2024/ }).check();
  await page.getByLabel('你想核对什么？').fill('2024 年营业收入是多少？');
  await page.getByRole('button', { name: '分析所选资料' }).click();
  await expect(page.getByText('证据充分', { exact: true })).toBeVisible();
  expect(submitted).toMatchObject({ document_ids: ['report-2024'], question: '2024 年营业收入是多少？' });
  await page.getByText('核对公式与来源', { exact: true }).click();
  await expect(page.getByRole('table')).toContainText('(120 - 100) / 100 × 100');
  await expect(page.getByText('在选中的 1 份年报中找到 2 条证据。')).toBeVisible();
  await page.getByRole('button', { name: /查看来源/ }).click();
  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText('原文核验：2024 年营业收入为 120 亿元。');
  await page.screenshot({ path: 'test-results/annual-reports-source.png' });
  await page.getByRole('button', { name: '下一页' }).click();
  await expect(dialog).toContainText('来源原文 · 第 2 页');
  await expect(page.getByRole('button', { name: '下一页' })).toBeDisabled();
  await page.keyboard.press('Escape');
  await expect(dialog).not.toBeVisible();
  await expect(page.getByRole('button', { name: /查看来源/ })).toBeFocused();
  await page.getByRole('checkbox', { name: /选择 星海制造 2023/ }).check();
  await expect(page.getByText(/资料选择已变化/)).toBeVisible();
  await page.screenshot({ path: 'test-results/annual-reports-desktop.png', fullPage: true });
});

test('validates upload metadata and reloads the persisted document list', async ({ page }) => {
  await setup(page);
  let uploaded = false;
  await page.route('**/api/annual-reports/documents', async (route) => {
    if (route.request().method() === 'POST') {
      uploaded = true;
      return route.fulfill({ json: docs[0] });
    }
    return route.fulfill({ json: { documents: uploaded ? [docs[0]] : [] } });
  });
  await page.goto('/annual-reports');
  await page.getByRole('button', { name: '上传并保存' }).click();
  await expect(page.getByRole('alert')).toContainText('请填写年报所属公司');
  await page.getByLabel('公司名称 *').fill('星海制造');
  await page.getByLabel('报告年度 *').fill('0000');
  await page.getByRole('button', { name: '上传并保存' }).click();
  await expect(page.getByRole('alert')).toContainText('有效的四位报告年度');
  await page.getByLabel('报告年度 *').fill('2024');
  await page.getByLabel('年报文件 *').setInputFiles({ name: 'annual.txt', mimeType: 'text/plain', buffer: Buffer.from('测试年报：2024 年营业收入 120 亿元。') });
  await page.getByRole('button', { name: '上传并保存' }).click();
  await expect(page.getByRole('checkbox', { name: /选择 星海制造 2024/ })).toBeChecked();
  await page.reload();
  await expect(page.getByRole('checkbox', { name: /选择 星海制造 2024/ })).toBeVisible();
  await expect(page.getByRole('checkbox', { name: /选择 星海制造 2024/ })).not.toBeChecked();
});

test('renders insufficient evidence and request errors without manufacturing citations', async ({ page }) => {
  await setup(page);
  let count = 0;
  await page.route('**/api/annual-reports/analyze', (route) => {
    count += 1;
    return count === 1 ? route.fulfill({ json: { ...answer, status: 'insufficient_evidence', answer: '缺少对应年度的现金流数据。', citations: [], calculations: [] } })
      : route.fulfill({ status: 503, json: { detail: '年报检索暂时不可用，请重试。' } });
  });
  await page.goto('/annual-reports');
  await page.getByRole('checkbox', { name: /选择 星海制造 2024/ }).check();
  await page.getByRole('button', { name: '收入与现金流', exact: true }).click();
  await page.getByRole('button', { name: '分析所选资料' }).click();
  await expect(page.getByText('证据不足', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: /查看来源/ })).toHaveCount(0);
  await page.getByRole('button', { name: '分析所选资料' }).click();
  await expect(page.getByRole('alert')).toContainText('年报检索暂时不可用，请重试。');
});

test('cancels an in-flight analysis and works at a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await setup(page);
  let release!: () => void;
  const pending = new Promise<void>((resolve) => { release = resolve; });
  await page.route('**/api/annual-reports/analyze', async (route) => {
    await pending;
    await route.fulfill({ json: answer }).catch(() => undefined);
  });
  await page.goto('/annual-reports');
  await page.getByRole('checkbox', { name: /选择 星海制造 2024/ }).check();
  await page.getByRole('button', { name: '收入与现金流', exact: true }).click();
  await page.getByRole('button', { name: '分析所选资料' }).click();
  await expect(page.getByText('正在检索并核对年报证据')).toBeVisible();
  await page.getByRole('button', { name: '取消分析' }).click();
  release();
  await expect(page.getByRole('alert')).toContainText('已取消本次分析');
  await expect(page.getByText('证据充分', { exact: true })).toHaveCount(0);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  expect(overflow).toBe(false);
  await page.screenshot({ path: 'test-results/annual-reports-mobile.png', fullPage: true });
});

test('labels synthetic data and preserves precise values and undefined growth rates on mobile', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await setup(page);
  await page.route('**/api/annual-reports/demo', (route) => route.fulfill({ json: { documents: docs } }));
  await page.route('**/api/annual-reports/analyze', (route) => route.fulfill({ json: {
    ...answer,
    calculations: [{ ...answer.calculations[0], from_value: '0', to_value: '-0.123456789', delta: '-0.123456789', change_pct: null, formula: '(-0.123456789 - 0)' }],
  } }));
  await page.goto('/annual-reports');
  await page.getByRole('button', { name: '载入示例资料' }).click();
  await expect(page.getByRole('status')).toContainText('人工合成示例');
  await expect(page.getByRole('checkbox', { checked: true })).toHaveCount(2);
  await page.getByRole('button', { name: '分析所选资料' }).click();
  await expect(page.getByRole('table')).toContainText('-0.123456789');
  await expect(page.getByRole('table')).toContainText('不计算（基期非正）');
  await page.getByRole('button', { name: '切换深色主题' }).click();
  await page.getByRole('region', { name: '年报分析结果' }).scrollIntoViewIfNeeded();
  const overflow = await page.getByRole('main').evaluate((element) => element.scrollWidth > element.clientWidth);
  expect(overflow).toBe(false);
  await page.screenshot({ path: 'test-results/annual-reports-mobile-result-dark.png' });
  await page.getByRole('button', { name: /查看来源/ }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.screenshot({ path: 'test-results/annual-reports-mobile-source-dark.png' });
});
