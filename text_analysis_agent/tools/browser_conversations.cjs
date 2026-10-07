'use strict';

// 在独立测试书库中验证会话隔离、重命名、草稿保留及精确引文高亮，不调用模型。
const {chromium} = require('../.cache/browser/node_modules/playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const base = process.env.READER_TEST_URL || 'http://127.0.0.1:8765';

async function main() {
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    const text = '😀引子。\r\n第一章 钥匙\r\n林舟用铜钥匙打开了铁门。\r\n林舟用铜钥匙打开了铁门。\r\n第二章 街道\r\n阿遥拿着银钥匙沿街行走。\r\n测试编号：' + Date.now();
    const uploaded = await page.request.post(base + '/api/books', {multipart: {title: '会话与高亮验证-' + Date.now(),
      file: {name: '会话与高亮验证-' + Date.now() + '.txt', mimeType: 'text/plain', buffer: Buffer.from(text)}}});
    let job = await uploaded.json();
    for (let i = 0; i < 60 && job.status !== 'completed'; i++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      job = await (await page.request.get(base + '/api/jobs/' + job.id)).json();
    }
    assert.equal(job.status, 'completed');
    const bookId = job.book_id;
    await page.goto(base + '/#book/' + bookId + '/chat');
    await page.getByLabel('问答检索方式').selectOption('extractive');
    await page.getByLabel('输入关于原文的问题').fill('铜钥匙');
    await page.getByLabel('输入关于原文的问题').press('Enter');
    await page.locator('.citation').first().waitFor();
    const first = (await (await page.request.get(base + '/api/books/' + bookId + '/conversations')).json())[0];
    await page.getByLabel('输入关于原文的问题').fill('还没发送的草稿');
    await page.getByRole('button', {name: '＋ 新会话', exact: true}).click();
    await page.waitForFunction(() => document.querySelectorAll('.conversation-item').length === 2);
    await page.waitForFunction(() => document.querySelectorAll('.user-message').length === 0);
    assert.equal(await page.locator('.user-message').count(), 0);
    assert.equal(await page.getByLabel('输入关于原文的问题').inputValue(), '');
    page.once('dialog', dialog => dialog.accept('阿遥线索'));
    await page.getByRole('button', {name: '重命名会话', exact: true}).click();
    await page.getByRole('button', {name: '阿遥线索', exact: true}).waitFor();
    await page.getByLabel('问答检索方式').selectOption('extractive');
    await page.getByLabel('输入关于原文的问题').fill('银钥匙');
    await page.getByLabel('输入关于原文的问题').press('Enter');
    await page.locator('.citation').first().waitFor();
    assert.deepEqual(await page.locator('.user-message').allTextContents(), ['银钥匙']);
    await page.getByRole('button', {name: first.title, exact: true}).click();
    await page.waitForFunction(() => document.querySelector('.user-message')?.textContent === '铜钥匙');
    assert.deepEqual(await page.locator('.user-message').allTextContents(), ['铜钥匙']);
    assert.equal(await page.getByLabel('输入关于原文的问题').inputValue(), '还没发送的草稿');

    // 为重复引文构造已知位置，使用真实原文接口检验高亮不定位到第一次出现。
    const list = await (await page.request.get(base + '/api/books/' + bookId + '/chunks?limit=5')).json();
    const chunk = list.items.find(c => c.text.includes('铜钥匙'));
    const quote = '林舟用铜钥匙打开了铁门。';
    const utf16Position = chunk.text.lastIndexOf(quote);
    const start = chunk.start + Array.from(chunk.text.slice(0, utf16Position)).length;
    const fake = {id: 'highlight-fixture', kind: 'ask', conversation_id: first.id, status: 'completed',
      payload: {question: '高亮验证'}, result: {status: 'answered', notice: '测试引文位置',
        claims: [{text: '原文引用示例', citations: [{chunk_id: chunk.id, source: chunk.source, chapter: chunk.chapter,
          quote, start, end: start + Array.from(quote).length, line_start: chunk.line_start, line_end: chunk.line_end}]}]}};
    await page.route('**/api/jobs?**', async route => {
      const url = new URL(route.request().url());
      if (url.searchParams.get('conversation_id') === first.id) await route.fulfill({json: [fake]});
      else await route.continue();
    });
    await page.reload();
    await page.getByRole('button', {name: '查看原文上下文 ↗'}).click();
    await page.locator('#quote-dialog[open] #quote-target').waitFor();
    assert.equal(await page.locator('#quote-target').innerText(), quote);
    const marker = await page.locator('#quote-target').evaluate(node => ({prefix: node.previousSibling.textContent, text: node.parentElement.textContent,
      visible: node.getBoundingClientRect().top >= 0 && node.getBoundingClientRect().bottom <= innerHeight}));
    assert.equal(marker.prefix.length, utf16Position);
    assert.equal(marker.text, chunk.text);
    assert(marker.visible);
    await page.screenshot({path: path.resolve(__dirname, '../artifacts/quote-highlight.png'), fullPage: true});
    await page.locator('#close-quote').click();
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.screenshot({path: path.resolve(__dirname, '../artifacts/conversations-mobile.png'), fullPage: true});
    assert.deepEqual(errors, []);
    console.log('会话记录隔离、重命名、草稿切换、Unicode 偏移、重复引文定位与高亮自动滚动验证通过（零模型调用）。');
  } finally {await browser.close();}
}
main().catch(error => {console.error(error); process.exitCode = 1;});
