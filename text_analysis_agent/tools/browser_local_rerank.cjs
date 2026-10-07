'use strict';

// 下载状态使用模拟数据，启用动作只写独立配置；不会启动模型下载或问答 API。
const {chromium} = require('../.cache/browser/node_modules/playwright');
const assert = require('node:assert/strict');
const base = process.env.READER_TEST_URL || 'http://127.0.0.1:8765';
(async () => {
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const model = {id: 'qwen3-reranker-0.6b', model: 'Qwen/Qwen3-Reranker-0.6B', status: 'not_downloaded',
    stage: '尚未下载', percent: 0, downloaded_bytes: 0, total_bytes: 1207472032, dependencies_ready: true};
  try {
    await page.request.put(base + '/api/settings', {data: {values: {RERANK_PROVIDER: 'api'}}});
    await page.route('**/api/models', route => route.fulfill({json: [model]}));
    await page.route('**/api/models/*/download', async route => {
      Object.assign(model, {status: 'downloading', stage: '下载 model.safetensors', percent: 37,
        downloaded_bytes: 446764652, bytes_per_second: 5000000, eta_seconds: 152});
      await route.fulfill({status: 202, json: model});
    });
    await page.goto(base + '/#settings');
    await page.getByRole('button', {name: '下载模型', exact: true}).click();
    const progress = page.getByRole('progressbar', {name: '模型下载进度'});
    await progress.waitFor();
    assert.equal(await progress.getAttribute('aria-valuenow'), '37');
    assert(await page.getByText(/预计剩余/).isVisible());
    Object.assign(model, {status: 'ready', stage: '已下载，可在本机重排', percent: 100, downloaded_bytes: 1207472032});
    await page.getByRole('button', {name: '已下载', exact: true}).waitFor();
    const saved = page.waitForResponse(response => response.url().endsWith('/api/settings') && response.request().method() === 'PUT');
    await page.getByRole('button', {name: '启用本地重排', exact: true}).click();
    assert.equal((await saved).status(), 200);
    assert.equal(await page.getByLabel('重排方式', {exact: true}).inputValue(), 'local');
    await page.reload();
    await page.getByLabel('重排方式', {exact: true}).waitFor();
    assert.equal(await page.getByLabel('重排方式', {exact: true}).inputValue(), 'local');
    assert.equal(await page.getByLabel('本地重排模型', {exact: true}).inputValue(), model.model);
    await page.screenshot({path: 'artifacts/local-rerank-settings.png', fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert.deepEqual(errors, []);
    console.log('本地模型下载进度、预计时间、启用与配置保存及移动布局验证通过，零模型 API 调用。');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
