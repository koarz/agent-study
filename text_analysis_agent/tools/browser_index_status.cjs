'use strict';

// 模拟索引状态核对书籍页面，不提交任务或调用模型。
const {chromium} = require('../.cache/browser/node_modules/playwright');
const assert = require('node:assert/strict');
const path = require('node:path');

async function main() {
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const book = {id: 'index-state-demo', title: '索引状态示例', name: '示例.txt', archived: false,
    chars: 2000, chunks: 2, vector_chunks: 2, graph_facts: 0, index_status: {index: 'ready', graph: 'ready'}};
  let submissions = 0;
  await page.route('**/api/**', async route => {
    const request = route.request();
    if (request.method() !== 'GET') submissions += 1;
    const url = new URL(request.url());
    const response = url.pathname === '/api/health' ? {model_configured: true} :
      url.pathname === '/api/books' ? [book] : url.pathname === '/api/books/' + book.id ? book : [];
    await route.fulfill({json: response});
  });
  try {
    await page.goto('http://127.0.0.1:8000/#book/' + book.id + '/chat');
    for (const name of ['✓ 已建立语义索引', '✓ 已建立关系索引']) {
      await page.getByRole('button', {name, exact: true}).waitFor();
      assert(await page.getByRole('button', {name, exact: true}).isDisabled());
    }
    await page.screenshot({path: path.resolve(__dirname, '../artifacts/book-index-ready.png'), fullPage: true});
    await page.getByRole('button', {name: '人物与关系', exact: true}).click();
    await page.getByRole('heading', {name: '关系索引已建立。', exact: true}).waitFor();
    await page.getByRole('textbox', {name: '查询实体名称'}).fill('人物甲');
    await page.getByRole('button', {name: '查询关系', exact: true}).click();
    await page.getByRole('heading', {name: '暂未找到该实体的关系记录'}).waitFor();
    assert(!(await page.locator('#book-body').innerText()).includes('请先建立关系索引'));
    for (const status of ['building', 'partial', 'missing']) {
      book.index_status = {index: status, graph: status};
      await page.goto('http://127.0.0.1:8000/#book/' + book.id + '/chat');
      await page.reload();
      for (const kind of ['语义索引', '关系索引']) {
        const name = status === 'building' ? kind + '建立中…' : status === 'partial' ? '继续建立' + kind + ' ↗' : '建立' + kind + ' ↗';
        const control = page.getByRole('button', {name, exact: true});
        await control.waitFor();
        assert.equal(await control.isDisabled(), status === 'building');
      }
    }
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert.equal(submissions, 0);
    assert.deepEqual(errors, []);
    console.log('已建立、建立中、未完成与未建立状态，以及空关系结果提示验证通过。');
  } finally {await browser.close();}
}
main().catch(error => {console.error(error); process.exitCode = 1;});
