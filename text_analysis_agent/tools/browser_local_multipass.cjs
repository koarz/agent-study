'use strict';

// 只读取设置并模拟任务结果，不修改配置，也不调用问答或模型接口。
const {chromium} = require('../.cache/browser/node_modules/playwright');
const assert = require('node:assert/strict');
const base = process.env.READER_TEST_URL || 'http://127.0.0.1:8000';
(async () => {
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const now = Date.now();
  const progressJob = {id: 'local-multipass-demo', kind: 'ask', status: 'running',
    payload: {question: '演示问题'}, result: null, error: null, created_at: new Date(now).toISOString(),
    progress: '本地重排 · 第 2/2 遍 · 批次 18/36 · 窗口 280/560',
    progress_detail: {current: 280, total: 560, reused: 0, unit: '窗口', stage: '本地分批分窗重排',
      percent: 50, elapsed_seconds: 600, remaining_seconds: 600,
      started_at: new Date(now - 600000).toISOString(), eta_at: new Date(now + 600000).toISOString()}};
  try {
    await page.goto(base + '/#settings');
    await page.getByLabel('本地重排遍数', {exact: true}).waitFor();
    assert.equal(await page.getByLabel('本地重排遍数', {exact: true}).inputValue(), '2');
    assert.equal(await page.getByLabel('本地每批段落数', {exact: true}).inputValue(), '8');
    assert.equal(await page.getByLabel('本地原文窗口字数', {exact: true}).inputValue(), '1200');
    await page.route('**/api/jobs**', route => route.fulfill({json:
      new URL(route.request().url()).pathname === '/api/jobs' ? [progressJob] : progressJob}));
    await page.goto(base + '/#tasks');
    await page.getByRole('progressbar').waitFor();
    assert.equal(await page.getByRole('progressbar').getAttribute('aria-valuenow'), '50');
    assert.match(await page.locator('#task-list').innerText(), /280 \/ 560 窗口/);
    assert.match(await page.locator('#task-list').innerText(), /第 2\/2 遍/);
    assert.match(await page.locator('#task-list').innerText(), /预计剩余/);
    assert.match(await page.locator('#task-list').innerText(), /预计完成/);
    await page.screenshot({path: 'artifacts/local-multipass-progress.png', fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert.deepEqual(errors, []);
    console.log('本地多遍设置、窗口进度、剩余时间、预计完成时间和手机布局通过，零模型 API 调用。');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
