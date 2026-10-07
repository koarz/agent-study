'use strict';

// 验证真实页面操作；测试书库应使用独立目录，不修改用户模型配置。
const {chromium} = require('../.cache/browser/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

async function main() {
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const base = process.env.READER_TEST_URL || 'http://127.0.0.1:8765';
  const output = path.resolve(__dirname, '../artifacts');
  fs.mkdirSync(output, {recursive: true});
  try {
    await page.goto(base);
    await page.getByRole('heading', {name: '把故事读清楚。'}).waitFor();
    await page.getByRole('button', {name: '＋ 导入原文', exact: true}).click();
    await page.locator('#file-input').setInputFiles(path.resolve(__dirname, '../examples/雾港来信.txt'));
    await page.locator('#upload-title').fill('雾港来信');
    await page.locator('#upload-submit').click();
    await page.waitForFunction(async () => {
      const jobs = await (await fetch('/api/jobs')).json();
      return jobs.some(job => job.kind === 'ingest' && job.status === 'completed');
    });
    await page.getByRole('link', {name: /我的书库/}).click();
    await page.getByRole('link', {name: '打开 雾港来信'}).waitFor();
    // 增加两份不同原文，检查书库布局和来源隔离。
    for (const [title, text] of [['旧城档案', '第一章 旧街\n顾青用银钥匙打开了木门。'], ['山间手记', '第一章 雨后\n阿遥沿着山路返回村庄。']]) {
      await page.request.post(base + '/api/books', {multipart: {title, file: {name: title + '.txt', mimeType: 'text/plain', buffer: Buffer.from(text)}}});
    }
    await page.waitForFunction(async () => (await (await fetch('/api/books')).json()).length === 3);
    await page.reload();
    await page.locator('.book-card').nth(2).waitFor();
    await page.screenshot({path: path.join(output, 'library-desktop.png'), fullPage: true});
    await page.getByRole('link', {name: '打开 雾港来信'}).click();
    await page.getByLabel('问答检索方式').selectOption('extractive');
    await page.getByLabel('输入关于原文的问题').fill('铜钥匙 铁门');
    await page.getByLabel('输入关于原文的问题').press('Enter');
    await page.locator('.citation').first().waitFor({timeout: 15000});
    assert.match(await page.locator('.chat-feed').innerText(), /铜钥匙/);
    assert.doesNotMatch(await page.locator('.chat-feed').innerText(), /银钥匙/);
    await page.screenshot({path: path.join(output, 'answer-desktop.png'), fullPage: true});
    await page.getByRole('button', {name: '查看原文上下文 ↗'}).first().click();
    await page.locator('#quote-dialog[open]').waitFor();
    assert.match(await page.locator('#quote-content').innerText(), /铜钥匙/);
    await page.locator('#close-quote').click();
    await page.getByRole('button', {name: '浏览原文', exact: true}).click();
    await page.locator('.original-text').first().waitFor();
    await page.getByRole('button', {name: '归档', exact: true}).click();
    await page.getByRole('button', {name: '已归档', exact: true}).click();
    await page.getByRole('link', {name: '打开 雾港来信'}).click();
    await page.getByRole('button', {name: '恢复书籍', exact: true}).click();
    await page.getByRole('button', {name: /全部书籍/}).click();
    await page.getByRole('link', {name: /模型设置/}).click();
    await page.locator('input[name="LLM_API_KEY"]').waitFor();
    assert.equal(await page.locator('input[name="LLM_API_KEY"]').inputValue(), '');
    await page.setViewportSize({width: 390, height: 844});
    await page.getByRole('link', {name: /我的书库/}).click();
    await page.locator('.book-card').nth(2).waitFor();
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    await page.screenshot({path: path.join(output, 'library-mobile.png'), fullPage: true});
    await page.getByRole('link', {name: '打开 雾港来信'}).click();
    await page.locator('.citation').first().waitFor();
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    await page.screenshot({path: path.join(output, 'answer-mobile.png'), fullPage: true});
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({passed: true, checks: ['前端上传', '三本书管理', '离线原文检索', '来源隔离', '引用上下文', '原文浏览', '归档与恢复', '密钥不回显', '桌面与手机布局'], model_calls: 0, screenshots: output}, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch(error => {console.error(error); process.exitCode = 1;});
