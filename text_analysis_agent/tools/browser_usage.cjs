'use strict';

// 独立书库中预置数值记录，验证分类、复核汇总、时间筛选和移动布局，不调用模型。
const {chromium} = require('../.cache/browser/node_modules/playwright');
const assert = require('node:assert/strict');
const base = process.env.READER_TEST_URL || 'http://127.0.0.1:8765';

(async () => {
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(base + '/#library');
    await page.getByRole('link', {name: /用量统计/}).click();
    const total = page.getByRole('region', {name: '用量汇总'});
    await total.getByRole('heading', {name: '101,775（已知）', exact: true}).waitFor();
    assert(await page.getByRole('region', {name: '回答模型用量'}).getByText('175', {exact: true}).isVisible());
    assert(await page.getByRole('region', {name: '嵌入模型用量'}).getByText('400（已知）', {exact: true}).first().isVisible());
    assert(await page.getByText('回答模型 · 结论复核', {exact: true}).isVisible());
    const filtered = page.waitForResponse(response => response.url().endsWith('/api/usage?days=7'));
    await page.getByLabel('统计时间范围').selectOption('7');
    assert.equal((await filtered).status(), 200);
    await total.getByRole('heading', {name: '1,775（已知）', exact: true}).waitFor();
    assert(await page.getByRole('region', {name: '重排模型用量'}).getByText('1,200（已知）', {exact: true}).first().isVisible());
    const refresh = page.waitForResponse(response => response.url().endsWith('/api/usage?days=7'));
    await page.getByRole('button', {name: '刷新用量', exact: true}).click();
    assert.equal((await refresh).status(), 200);
    await page.screenshot({path: 'artifacts/usage-desktop.png', fullPage: true});
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert(await page.getByRole('link', {name: /用量统计/}).isVisible());
    await page.screenshot({path: 'artifacts/usage-mobile.png', fullPage: true});
    await page.goto(base + '/#settings');
    await page.getByRole('heading', {name: '回答模型', exact: true}).waitFor();
    assert.deepEqual(errors, []);
    console.log('三类模型、复核明细、总用量、时间筛选、刷新和移动布局验证通过，零模型调用。');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
