'use strict';

// 仅使用独立书库和配置，验证成本设置与统计显示，不调用模型。
const {chromium} = require('../.cache/browser/node_modules/playwright');
const assert = require('node:assert/strict');
const base = process.env.READER_TEST_URL || 'http://127.0.0.1:8765';

(async () => {
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(base + '/#settings');
    await page.getByLabel('每次问答新评分段落上限').fill('64');
    await page.getByLabel('每次问答原文输入字数上限').fill('64000');
    await page.getByLabel('重排方式', {exact: true}).selectOption('none');
    const saved = page.waitForResponse(response => response.url().endsWith('/api/settings') && response.request().method() === 'PUT');
    await page.getByRole('button', {name: '保存配置 →'}).click();
    assert.equal((await saved).status(), 200);
    await page.reload();
    await page.getByLabel('每次问答新评分段落上限').waitFor();
    assert.equal(await page.getByLabel('每次问答新评分段落上限').inputValue(), '64');
    assert.equal(await page.getByLabel('每次问答原文输入字数上限').inputValue(), '64000');
    assert.equal(await page.getByLabel('重排方式', {exact: true}).inputValue(), 'none');
    await page.screenshot({path: 'artifacts/rerank-cost-settings.png', fullPage: true});
    const name = '重排费用验证-' + Date.now();
    const uploaded = await page.request.post(base + '/api/books', {multipart: {title: name,
      file: {name: name + '.txt', mimeType: 'text/plain', buffer: Buffer.from('第一章\n林舟用铜钥匙打开了铁门。')}}});
    let job = await uploaded.json();
    for (let i = 0; i < 100 && job.status !== 'completed'; i++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      job = await (await page.request.get(base + '/api/jobs/' + job.id)).json();
    }
    assert.equal(job.status, 'completed');
    const session = await (await page.request.post(base + '/api/books/' + job.book_id + '/conversations', {data: {title: '统计显示'}})).json();
    const fixture = {id: 'rerank-cost-fixture', kind: 'ask', status: 'completed', payload: {question: '铜钥匙的用途？'},
      result: {status: 'unclear', claims: [], notice: '统计显示测试', coverage: {mode: 'retrieved', selected_chunks: 1, total_chunks: 1,
        rerank: {enabled: true, documents: 64, document_chars: 63000, requests: 2, cache_hits: 30, unscored: 4}}}};
    await page.route('**/api/jobs?**', async route => {
      const url = new URL(route.request().url());
      if (url.searchParams.get('conversation_id') === session.id) await route.fulfill({json: [fixture]});
      else await route.continue();
    });
    await page.goto(base + '/#book/' + job.book_id + '/chat/' + session.id);
    await page.getByText(/本次重排：新评分 64 段/).waitFor();
    assert(await page.getByText(/评分批次 2 次 · 复用评分 30 次/).isVisible());
    assert(await page.getByText(/部分候选未进行重排/).isVisible());
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert.deepEqual(errors, []);
    console.log('重排预算保存、关闭开关、使用量与缓存统计及移动端布局验证通过，零模型调用。');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
