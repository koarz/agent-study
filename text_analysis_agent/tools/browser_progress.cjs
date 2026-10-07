'use strict';

// 使用模拟任务状态核对进度布局及时间估算，不调用模型或修改真实任务。
const {chromium} = require('../.cache/browser/node_modules/playwright');
const assert = require('node:assert/strict');
const path = require('node:path');

async function main() {
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const now = Date.now();
  const tasks = [
    {id: 'progress-semantic-demo', kind: 'index', percent: 64, current: 3200, total: 5000, unit: '段', reused: 1000, stage: '建立语义索引', elapsed: 600, remaining: 480},
    {id: 'progress-graph-demo', kind: 'graph', percent: 25, current: 250, total: 1000, unit: '批', reused: 0, stage: '提取与核对原文', elapsed: 1200, remaining: 3600},
  ].map(item => ({...item, status: 'running', payload: {}, result: null, error: null, progress: '正在处理并保存原文证据', created_at: new Date(now).toISOString(),
    progress_detail: {...item, started_at: new Date(now - item.elapsed * 1000).toISOString(), elapsed_seconds: item.elapsed,
      remaining_seconds: item.remaining, eta_at: new Date(now + item.remaining * 1000).toISOString()}}));
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/jobs**', async route => {
    const pathname = new URL(route.request().url()).pathname;
    await route.fulfill({json: pathname === '/api/jobs' ? tasks : tasks.find(task => pathname.endsWith('/' + task.id))});
  });
  try {
    await page.goto('http://127.0.0.1:8000/#tasks');
    await page.getByRole('progressbar').nth(1).waitFor();
    assert.equal(await page.getByRole('progressbar').first().getAttribute('aria-valuenow'), '64');
    assert.match(await page.locator('#task-list').innerText(), /3,200 \/ 5,000 段/);
    assert.match(await page.locator('#task-list').innerText(), /预计剩余/);
    assert.match(await page.locator('#task-list').innerText(), /预计完成/);
    await page.screenshot({path: path.resolve(__dirname, '../artifacts/progress-desktop.png'), fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path: path.resolve(__dirname, '../artifacts/progress-mobile.png'), fullPage: true});
    assert.deepEqual(errors, []);
    console.log('进度条、数量、耗时、预计结束时间及手机布局验证通过（模拟任务，无模型调用）。');
  } finally {await browser.close();}
}
main().catch(error => {console.error(error); process.exitCode = 1;});
