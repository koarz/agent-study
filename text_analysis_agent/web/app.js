'use strict';

// 新功能上线后提示刷新，保留尚未保存的设置和输入。
const loadedScriptVersion = new URL(document.currentScript.src).searchParams.get('v');
async function checkFrontendVersion() {
  if (!loadedScriptVersion || document.hidden || document.getElementById('refresh-page-version')) return;
  try {
    const response = await fetch('/', {cache: 'no-store'});
    if (!response.ok) return;
    const html = await response.text();
    const latest = html.match(/\/assets\/app\.js\?v=([a-z0-9]+)/)?.[1];
    if (latest && latest !== loadedScriptVersion) {
      const refresh = button('新版页面可用 · 刷新', () => location.reload(), 'secondary');
      refresh.id = 'refresh-page-version';
      document.querySelector('.topbar-note').replaceWith(refresh);
    }
  } catch (_) {}
}
setInterval(checkFrontendVersion, 60000);
document.addEventListener('visibilitychange', checkFrontendVersion);


const $ = (selector, root = document) => root.querySelector(selector);
const content = $('#content');
const state = {books: [], health: {}, archived: false, search: '', routeVersion: 0, watched: new Set(), mode: 'hybrid', conversations: {}, drafts: {}};
const kindLabels = {ingest: '导入原文', index: '建立语义索引', graph: '建立关系索引', scan: '全文扫描', ask: '原文问答'};
const statusLabels = {queued: '等待执行', running: '进行中', completed: '已完成', failed: '执行失败', interrupted: '已中断', retried: '已重试', paused: '已暂停'};
const answerLabels = {answered: '有原文依据', unclear: '信息不明确 / 依据不足', not_found: '未找到回答依据', extractive: '原文检索'};

// 所有文件名、原文和模型回答都通过文本节点插入，避免将内容当成 HTML 执行。
function h(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null) continue;
    if (key === 'class') node.className = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2).toLowerCase(), value);
    else if (['value', 'checked', 'disabled', 'hidden'].includes(key)) node[key] = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child !== null && child !== undefined) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(result.detail) ? result.detail.map(item => item.msg).join('；') : result.detail;
    throw new Error(detail || '请求暂时失败，请稍后重试');
  }
  return result;
}

function jsonRequest(method, body) {
  return {method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
}

let toastTimer;
function toast(message, error = false) {
  const node = $('#toast');
  node.textContent = message;
  node.className = 'toast' + (error ? ' error' : '');
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {node.hidden = true;}, error ? 6000 : 3500);
}

const chars = number => number >= 10000 ? (number / 10000).toFixed(number < 100000 ? 1 : 0) + ' 万字' : number.toLocaleString() + ' 字';
const date = value => new Date(value).toLocaleString('zh-CN', {month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'});
const button = (text, click, className = 'secondary', props = {}) => h('button', {type: 'button', class: 'button ' + className, onclick: click, ...props}, text);
const pill = (text, type = '') => h('span', {class: 'pill ' + type}, text);

function openUpload() {
  $('#upload-error').textContent = '';
  $('#upload-dialog').showModal();
}

function intro(title, description, action) {
  return h('div', {class: 'intro'}, h('div', {}, h('span', {class: 'eyebrow'}, 'READ WITH EVIDENCE'),
    h('h1', {}, title), h('p', {}, description)), action);
}

function empty(title, description, action) {
  return h('div', {class: 'empty-state'}, h('div', {class: 'empty-symbol'}, '▤'),
    h('h3', {}, title), h('p', {}, description), action);
}

async function renderLibrary(version) {
  const [books, health] = await Promise.all([api('/api/books?archived=' + state.archived), api('/api/health')]);
  if (version !== state.routeVersion) return;
  state.books = books;
  state.health = health;
  const grid = h('div', {class: 'books-grid'});
  const drawBooks = () => {
    const filtered = books.filter(book => book.title.toLowerCase().includes(state.search.toLowerCase()));
    grid.replaceChildren(...filtered.map(bookCard));
    if (!filtered.length) {
      grid.append(empty(state.search ? '没有找到这本书' : state.archived ? '归档里还没有书籍' : '给你的阅读，留一个位置。',
        state.search ? '试试其他书名。' : state.archived ? '书籍归档后仍保留原文和分析记录，可以随时恢复。' : '导入一本小说或一份文本，从提问开始，每个结论都附上原文出处。',
        !state.archived && !state.search ? button('＋ 导入第一本书', openUpload, 'primary') : null));
    } else if (!state.archived) {
      grid.append(h('button', {type: 'button', class: 'add-card', onclick: openUpload},
        h('span', {class: 'plus'}, '+'), h('strong', {}, '添加新的阅读'), h('small', {}, 'TXT / Markdown')));
    }
  };
  const search = h('input', {type: 'search', placeholder: '搜索书名', 'aria-label': '搜索书名', value: state.search,
    oninput: event => {state.search = event.target.value; drawBooks();}});
  const tabs = h('div', {class: 'tabs'}, ...[[false, '全部书籍'], [true, '已归档']].map(([archived, label]) =>
    h('button', {class: state.archived === archived ? 'active' : '', onclick: () => {state.archived = archived; route();}},
      label, state.archived === archived ? h('span', {class: 'count'}, books.length) : null)));
  content.replaceChildren(...[intro('把故事读清楚。', '从一段原文开始，让每个回答都有出处。', button('＋ 导入原文', openUpload, 'primary')),
    !health.model_configured ? h('div', {class: 'settings-notice'}, h('span', {}, '原文已准备好，模型配置后即可开始问答。导入与原文浏览无需模型。'),
      h('a', {class: 'text-link', href: '#settings'}, '配置模型 →')) : null,
    h('div', {class: 'library-toolbar'}, tabs, h('div', {class: 'search'}, h('span', {}, '⌕'), search)), grid].filter(Boolean));
  drawBooks();
  if (!state.archived) $('#book-count').textContent = books.length;
}

function bookCard(book) {
  const indexed = book.index_status?.index === 'ready';
  const card = h('article', {class: 'book-card', tabindex: '0', role: 'link', 'aria-label': '打开 ' + book.title,
    onclick: () => {location.hash = '#book/' + book.id + '/chat';},
    onkeydown: event => {if (event.key === 'Enter') location.hash = '#book/' + book.id + '/chat';}},
    h('div', {class: 'book-card-head'}, h('span', {class: 'book-symbol'}, '文'),
      pill(book.archived ? '已归档' : indexed ? '已建语义索引' : '原文已导入', indexed ? '' : 'neutral')),
    h('h3', {}, book.title), h('div', {class: 'book-meta'}, chars(book.chars) + ' · ' + book.chunks.toLocaleString() + ' 个原文段落'),
    h('div', {class: 'book-bottom'}, h('span', {}, book.graph_facts ? book.graph_facts.toLocaleString() + ' 条关系证据' : '打开阅读工作台'), h('span', {}, '↗')));
  return card;
}

// 完成后显示确认状态；正在执行的任务不能再次从书籍页面提交。
function indexControl(book, kind) {
  const status = book.index_status?.[kind] || 'missing';
  const name = kind === 'index' ? '语义索引' : '关系索引';
  const label = status === 'ready' ? '✓ 已建立' + name : status === 'building' ? name + '建立中…' :
    status === 'partial' ? '继续建立' + name + ' ↗' : '建立' + name + ' ↗';
  return button(label, () => queueAction(book, kind), status === 'ready' ? 'secondary index-complete' : 'secondary', {
    disabled: Boolean(book.archived) || ['ready', 'building'].includes(status),
    title: status === 'ready' ? name + '已建立完成' : status === 'building' ? '可在分析任务中查看进度' : '从原文建立' + name});
}

async function submitJob(book, kind, payload = {}) {
  const job = await api('/api/books/' + book.id + '/jobs', jsonRequest('POST', {kind, ...payload}));
  watchJob(job.id);
  toast(job.reused_job ? '该任务已在执行，正在显示当前进度' : kindLabels[kind] + '已加入后台任务');
  return job;
}

// 只追踪未完成任务，避免反复加载书库、打断输入或无止境刷新页面。
function watchJob(id) {
  if (state.watched.has(id)) return;
  state.watched.add(id);
  const poll = async () => {
    if (!state.watched.has(id)) return;
    try {
      const job = await api('/api/jobs/' + id);
      if (!state.watched.has(id)) return;
      if (['completed', 'failed', 'interrupted', 'retried', 'paused'].includes(job.status)) {
        state.watched.delete(id);
        toast(job.status === 'completed' ? kindLabels[job.kind] + '已完成' : job.status === 'retried' ? job.progress : job.error || job.progress, ['failed', 'interrupted'].includes(job.status));
        if (job.status === 'retried' && job.progress_detail?.replacement_id) watchJob(job.progress_detail.replacement_id);
        const hash = location.hash || '#library';
        if (hash === '#library' || hash === '#tasks' || hash.startsWith('#book/')) route();
        refreshActivity();
        return;
      }
      if (location.hash === '#tasks' || location.hash.endsWith('/tasks')) updateTaskRows();
      const progress = document.getElementById('progress-' + id);
      if (progress) progress.textContent = job.progress;
      setTimeout(poll, 2000);
    } catch (error) {
      if (!state.watched.has(id)) return;
      state.watched.delete(id);
      toast('任务状态暂时不可用，请到后台任务查看。', true);
    }
  };
  setTimeout(poll, 800);
}

async function refreshActivity() {
  try {
    const jobs = await api('/api/jobs?limit=100');
    const active = jobs.filter(job => ['queued', 'running'].includes(job.status));
    $('#task-dot').hidden = !active.length;
    active.forEach(job => watchJob(job.id));
  } catch (_) {}
}

async function renderBook(bookId, tab, version, conversationId) {
  const book = await api('/api/books/' + bookId);
  if (version !== state.routeVersion) return;
  $('#breadcrumb').textContent = book.title;
  const body = h('div', {id: 'book-body'});
  const archive = async () => {
    try {
      await api('/api/books/' + bookId, jsonRequest('PATCH', {archived: !book.archived}));
      toast(book.archived ? '书籍已恢复' : '书籍已归档，原文和记录均已保留');
      location.hash = '#library';
    } catch (error) {toast(error.message, true);}
  };
  const rename = async () => {
    const title = prompt('修改显示书名（原文引用名称保持导入时的名称）', book.title);
    if (title && title.trim() !== book.title) {
      try {await api('/api/books/' + bookId, jsonRequest('PATCH', {title: title.trim()})); route();}
      catch (error) {toast(error.message, true);}
    }
  };
  content.replaceChildren(h('div', {class: 'book-heading'}, h('div', {}, h('span', {class: 'eyebrow'}, '阅读工作台'),
    h('h1', {}, book.title), h('p', {class: 'book-meta'}, chars(book.chars) + ' · ' + book.chunks.toLocaleString() + ' 个段落 · 原文快照已保留')),
    h('div', {class: 'book-actions'}, button('修改书名', rename, 'quiet'), button(book.archived ? '恢复书籍' : '归档', archive, 'quiet'))),
    h('div', {class: 'tabs workspace-tabs'}, ...[['chat', '原文问答'], ['original', '浏览原文'], ['graph', '人物与关系'], ['tasks', '分析任务']].map(([value, label]) =>
      h('button', {class: tab === value ? 'active' : '', onclick: () => {location.hash = '#book/' + bookId + '/' + value;}}, label))), body);
  if (tab === 'original') await renderOriginal(book, body, version);
  else if (tab === 'graph') renderGraph(book, body);
  else if (tab === 'tasks') await renderTasks(body, bookId, version);
  else await renderChat(book, body, version, conversationId);
}

async function renderChat(book, body, version, requestedConversation) {
  const conversations = await api('/api/books/' + book.id + '/conversations');
  if (version !== state.routeVersion) return;
  let conversationId = requestedConversation || state.conversations[book.id] || conversations[0]?.id;
  if (conversationId && !conversations.some(item => item.id === conversationId)) {
    if (requestedConversation) throw new Error('这个会话不属于当前书籍或已不存在。');
    conversationId = conversations[0]?.id;
  }
  if (conversationId) {
    state.conversations[book.id] = conversationId;
    history.replaceState(null, '', '#book/' + book.id + '/chat/' + conversationId);
  }
  const jobs = conversationId ? await api('/api/jobs?kind=ask&book_id=' + book.id + '&conversation_id=' + conversationId + '&limit=200') : [];
  if (version !== state.routeVersion) return;
  const draftKey = book.id + '/' + (conversationId || 'new');
  const currentConversation = conversations.find(item => item.id === conversationId);
  const feed = h('div', {class: 'chat-feed'});
  const textarea = h('textarea', {rows: '2', value: state.drafts[draftKey] || '', placeholder: '写下你的问题，例如：这段情节发生在什么地方？', maxlength: '4000', 'aria-label': '输入关于原文的问题',
    oninput: event => {state.drafts[draftKey] = event.target.value;}});
  const submit = h('button', {type: 'submit', class: 'send-button', 'aria-label': '发送问题'}, '↑');
  const form = h('form', {class: 'chat-form', onsubmit: async event => {
    event.preventDefault();
    const question = textarea.value.trim();
    if (!question) return;
    submit.disabled = true;
    try {
      if (!conversationId) {
        const created = await api('/api/books/' + book.id + '/conversations', jsonRequest('POST', {}));
        conversationId = created.id;
        state.conversations[book.id] = conversationId;
        history.replaceState(null, '', '#book/' + book.id + '/chat/' + conversationId);
      }
      await submitJob(book, 'ask', {question, conversation_id: conversationId, retrieval: state.mode === 'extractive' ? 'lexical' : state.mode, extractive: state.mode === 'extractive'});
      delete state.drafts[draftKey];
      textarea.value = '';
      route();
    } catch (error) {toast(error.message, true); submit.disabled = false;}
  }}, h('div', {class: 'composer'}, textarea, submit),
    h('div', {class: 'composer-caption'}, h('span', {}, '回答附原文出处，缺少依据时会明确说明。'), h('span', {}, 'Enter 发送 · Shift + Enter 换行')));
  textarea.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {event.preventDefault(); form.requestSubmit();}
  });
  if (!jobs.length) {
    feed.append(h('div', {class: 'chat-empty'}, h('div', {class: 'empty-symbol'}, '“'), h('h3', {}, '有疑问，就回到原文。'),
      h('p', {}, '人物、情节、时间与关系，都从已有内容里寻找依据。', h('br'), '信息不明确的地方，会如实保留。'),
      h('div', {class: 'suggestions'}, ...['这本书开篇发生了什么？', '主角第一次出现时在做什么？', '原文有哪些明确的人物关系？'].map(text =>
        h('button', {onclick: () => {textarea.value = text; textarea.focus();}}, text + '  ↗')))));
  } else {
    for (const job of [...jobs].reverse()) {
      feed.append(h('div', {class: 'message user-message'}, job.payload.question));
      const answer = h('div', {class: 'message'}, h('div', {class: 'message-label'}, h('span', {class: 'avatar'}, '文'), '原文助手'));
      if (job.status === 'completed') {
        answer.append(answerContent(job.result, book));
        if (job.result?.validation_error) answer.append(button('重新核对', async () => {
          try {const next = await api('/api/jobs/' + job.id + '/retry', {method: 'POST'}); watchJob(next.id); route();}
          catch (error) {toast(error.message, true);}
        }, 'quiet'), h('p', {class: 'helper'}, '复用本次检索原文，不再次重排；回答和复核仍会调用模型。'));
      }
      else if (job.status === 'retried') answer.append(h('p', {class: 'muted'}, job.progress));
      else if (['failed', 'interrupted'].includes(job.status)) answer.append(h('p', {class: 'form-error'}, job.error || job.progress),
        button('继续执行', async () => {try {const next = await api('/api/jobs/' + job.id + '/retry', {method: 'POST'}); watchJob(next.id); route();} catch (error) {toast(error.message, true);}}, 'quiet'));
      else {answer.append(h('div', {class: 'pending-box'}, h('span', {class: 'spinner'}), h('span', {id: 'progress-' + job.id}, job.progress))); watchJob(job.id);}
      feed.append(answer);
    }
  }
  const select = h('select', {'aria-label': '问答检索方式', onchange: event => {state.mode = event.target.value;}},
    ...[['hybrid', '语义与关键词'], ['lexical', '关键词检索'], ['extractive', '仅查看原文']].map(([value, label]) => h('option', {value}, label)));
  select.value = state.mode;
  const newConversation = async () => {
    try {
      const created = await api('/api/books/' + book.id + '/conversations', jsonRequest('POST', {}));
      state.conversations[book.id] = created.id;
      location.hash = '#book/' + book.id + '/chat/' + created.id;
    } catch (error) {toast(error.message, true);}
  };
  const renameConversation = async () => {
    const title = prompt('会话名称', currentConversation.title);
    if (!title || !title.trim()) return;
    try {await api('/api/conversations/' + conversationId, jsonRequest('PATCH', {title: title.trim()})); route();}
    catch (error) {toast(error.message, true);}
  };
  const deleteConversation = async (item, control) => {
    if (!confirm('删除会话“' + item.title + '”及其中的全部问答记录？删除后无法恢复。')) return;
    control.disabled = true;
    try {
      const result = await api('/api/conversations/' + item.id, {method: 'DELETE'});
      // 清理当前页面的草稿和轮询；删除最后一个会话后保持可直接提问的空页面。
      delete state.drafts[book.id + '/' + item.id];
      for (const id of result.deleted_job_ids) state.watched.delete(id);
      if (state.conversations[book.id] === item.id) delete state.conversations[book.id];
      if (location.hash === '#book/' + book.id + '/chat/' + item.id) {
        history.replaceState(null, '', '#book/' + book.id + '/chat');
      }
      toast('会话已删除');
      if (location.hash.startsWith('#book/' + book.id + '/chat')) route();
      refreshActivity();
    } catch (error) {toast(error.message, true); control.disabled = false;}
  };
  const sessions = h('div', {class: 'side-card conversation-card'}, h('div', {class: 'conversation-heading'}, h('h3', {}, '会话'),
    button('＋ 新会话', newConversation, 'quiet', {disabled: Boolean(book.archived)})),
    h('p', {}, '会话只保存记录；每次问答独立，不携带历史消息。'),
    h('div', {class: 'conversation-list'}, ...conversations.map(item => h('div', {class: 'conversation-row'}, h('button', {
      class: 'conversation-item' + (item.id === conversationId ? ' active' : ''), title: item.title,
      'aria-current': item.id === conversationId ? 'true' : undefined,
      onclick: () => {state.conversations[book.id] = item.id; location.hash = '#book/' + book.id + '/chat/' + item.id;}}, item.title),
      button('删除', event => deleteConversation(item, event.currentTarget), 'quiet danger', {
        title: '删除此会话及问答记录', 'aria-label': '删除会话 ' + item.title})))),
    currentConversation ? button('重命名会话', renameConversation, 'quiet') : h('p', {class: 'muted'}, '发送问题后会自动创建会话。'));
  const sidebar = h('aside', {class: 'reader-aside'}, sessions, h('div', {class: 'side-card'}, h('h3', {}, '准备你的阅读'),
    h('p', {}, '语义索引用于寻找意思相近的原文；关系索引用于追踪跨章节人物与事件。'),
    indexControl(book, 'index'), indexControl(book, 'graph')),
    h('div', {class: 'side-card'}, h('h3', {}, '需要通读全文？'), h('p', {}, '人物全貌、事件盘点等问题，可以逐批扫描全文，得到带引用的证据报告。'),
      button('创建全文扫描 ↗', () => queueAction(book, 'scan'), 'secondary', {disabled: Boolean(book.archived)})),
    h('div', {class: 'side-card'}, h('h3', {}, '阅读范围'), h('div', {class: 'info-row'}, '当前书籍', '1 本'),
      h('div', {class: 'info-row'}, '原文字数', chars(book.chars)), h('p', {}, '检索未找到，不代表全文没有。模型的引用会经过程序检查和单独复核。')));
  body.replaceChildren(h('div', {class: 'reader-grid'}, h('section', {class: 'chat-panel'},
    h('div', {class: 'chat-head'}, h('span', {}, '◇ ' + (currentConversation?.title || '新会话') + ' · 依据原文回答'), select), feed, form), sidebar));
  requestAnimationFrame(() => {feed.scrollTop = feed.scrollHeight;});
  if (book.archived) {textarea.disabled = true; submit.disabled = true; textarea.placeholder = '请先恢复书籍再进行问答。';}
}

function answerContent(result, book) {
  const node = h('div', {}, pill(result.validation_error ? '回答未通过原文校验' : answerLabels[result.status] || result.status, result.status === 'answered' ? '' : 'warn'),
    h('p', {class: 'answer-notice'}, result.notice));
  if (result.validation_error) node.append(h('p', {class: 'helper'}, '校验原因：' + result.validation_error));
  if (result.validation_detail?.missing_mentions?.length) node.append(h('p', {class: 'helper'}, '引文缺少的专名：' + result.validation_detail.missing_mentions.join('、')));
  for (const claim of result.claims || []) {
    node.append(h('div', {class: 'claim-text'}, claim.text), ...(claim.citations || []).map(citation => citationView(citation, book)));
  }
  for (const chunk of result.evidence || []) {
    node.append(citationView({source: chunk.source, chapter: chunk.chapter, line_start: chunk.line_start, line_end: chunk.line_end,
      quote: chunk.text, chunk_id: chunk.id}, book));
  }
  const scope = result.coverage;
  if (scope) node.append(h('div', {class: 'coverage-note'},
    scope.mode === 'full_context' ? '本次已将完整原文纳入上下文。' : '本次依据检索到的片段回答，不能据此断言全文没有相关内容。',
    ' · ' + scope.selected_chunks + ' / ' + scope.total_chunks + ' 个段落'));
  const rerank = scope?.rerank;
  if (rerank?.enabled) {
    const local = rerank.provider === 'local';
    node.append(h('div', {class: 'coverage-note'},
      (local ? '本地重排（' + (rerank.device || 'cpu').toUpperCase() + '）：' + (rerank.scored_candidates ?? 0) + ' / ' + (rerank.candidates ?? 0) + ' 个召回候选 · 最多 ' + (rerank.passes ?? 1) + ' 遍 · 新评分 ' + (rerank.windows ?? 0) + ' 个窗口' : 'API 重排：新评分 ' + rerank.documents + ' 段') +
      ' · 原文输入 ' + chars(rerank.document_chars) + ' · 批次 ' + rerank.requests + ' 次 · 复用评分 ' + rerank.cache_hits + ' 次',
      rerank.unscored ? '。部分候选未评分，仍可通过召回排序提供依据。' : local ? '。已覆盖本次召回候选，未逐段扫描全文。' : '。'));
  }
  return node;
}

function citationView(citation, book) {
  return h('details', {class: 'citation', open: ''},
    h('summary', {}, citation.source + ' · ' + citation.chapter + ' · 第 ' + citation.line_start + '–' + citation.line_end + ' 行'),
    h('blockquote', {}, citation.quote),
    h('button', {class: 'text-link', onclick: () => openQuote(book.id, citation)}, '查看原文上下文 ↗'));
}

async function openQuote(bookId, citation) {
  try {
    const data = await api('/api/books/' + bookId + '/chunks/' + encodeURIComponent(citation.chunk_id));
    $('#quote-title').textContent = data.chunk.chapter;
    let highlighted = false;
    const sections = data.neighbors.map(chunk => {
      const original = h('div', {class: 'original-text'});
      const quote = citation.quote || '';
      let begin = -1;
      if (chunk.id === citation.chunk_id && quote) {
        // Python 偏移按 Unicode 字符计数；转换后再用于 JavaScript 切片，兼容表情等字符。
        if (Number.isInteger(citation.start)) {
          const offset = citation.start - chunk.start;
          if (offset >= 0 && offset <= Array.from(chunk.text).length) {
            const prefix = Array.from(chunk.text).slice(0, offset).join('');
            if (chunk.text.slice(prefix.length, prefix.length + quote.length) === quote) begin = prefix.length;
          }
        } else begin = chunk.text.indexOf(quote);
      }
      if (begin >= 0) {
        original.append(chunk.text.slice(0, begin), h('mark', {class: 'quote-highlight', id: 'quote-target', tabindex: '-1'}, quote), chunk.text.slice(begin + quote.length));
        highlighted = true;
      } else original.textContent = chunk.text;
      return h('section', {class: 'original-section'}, h('h3', {}, chunk.source + ' · 第 ' + chunk.line_start + '–' + chunk.line_end + ' 行'), original);
    });
    $('#quote-content').replaceChildren(h('p', {class: 'quote-location-note'}, highlighted ? '高亮部分为本次引用的原文。' : '未能定位本次引文，请核对原文位置。'), ...sections);
    $('#quote-dialog').showModal();
    if (highlighted) requestAnimationFrame(() => {
      const target = $('#quote-target');
      target.scrollIntoView({block: 'center', behavior: 'auto'});
      target.focus({preventScroll: true});
    });
  } catch (error) {toast(error.message, true);}
}

async function queueAction(book, kind) {
  try {
    let payload = {};
    if (kind === 'scan') {
      const question = prompt('本次全文扫描要寻找什么？', '按原文顺序整理主角的关键事件，逐条附上原文依据。');
      if (!question || !question.trim()) return;
      const plan = await api('/api/books/' + book.id + '/scan-plan');
      if (!confirm('将逐批扫描这本书，预计 ' + plan.total_batches + ' 批，约 ' + plan.model_calls_min + '–' + plan.model_calls_max +
        ' 次模型调用（已完成批次可复用）。结果会保存为证据报告。\n\n开始扫描？')) return;
      payload = {question: question.trim()};
    } else if (kind === 'graph') {
      const plan = await api('/api/books/' + book.id + '/scan-plan?batch_chars=6000');
      if (!confirm('将从原文提取并核对人物关系与事件。预计 ' + plan.total_batches + ' 批，最多 ' + plan.model_calls_max +
        ' 次模型调用，已有进度会复用。\n\n开始建立关系索引？')) return;
      payload.batch_chars = 6000;
    }
    await submitJob(book, kind, payload);
    location.hash = '#book/' + book.id + '/tasks';
  } catch (error) {toast(error.message, true);}
}

async function renderOriginal(book, body, version) {
  const chapters = await api('/api/books/' + book.id + '/chapters');
  if (version !== state.routeVersion) return;
  const panel = h('div', {class: 'original-panel'});
  let offset = 0;
  const load = async target => {
    try {
      const data = await api('/api/books/' + book.id + '/chunks?offset=' + target + '&limit=5');
      if (version !== state.routeVersion) return;
      offset = target;
      panel.replaceChildren(...data.items.map(chunk => h('section', {class: 'original-section'},
        h('h3', {}, chunk.chapter + ' · 第 ' + chunk.line_start + '–' + chunk.line_end + ' 行'),
        h('div', {class: 'original-text'}, chunk.text))),
        h('div', {class: 'pagination'}, button('← 上一页', () => load(Math.max(0, offset - 5)), 'quiet', {disabled: offset === 0}),
          h('span', {}, '段落 ' + (offset + 1) + '–' + Math.min(offset + 5, data.total) + ' / ' + data.total),
          button('下一页 →', () => load(offset + 5), 'quiet', {disabled: offset + 5 >= data.total})));
    } catch (error) {toast(error.message, true);}
  };
  const list = h('div', {class: 'chapter-list'}, h('h3', {}, '原文目录 · ' + chapters.length),
    ...chapters.map(chapter => h('button', {onclick: () => load(chapter.first_ordinal)}, chapter.chapter)));
  body.replaceChildren(h('div', {class: 'original-layout'}, list, panel));
  await load(0);
}

function renderGraph(book, body) {
  const input = h('input', {placeholder: '输入原文中的人物或实体名称', 'aria-label': '查询实体名称', maxlength: '120'});
  const results = h('div');
  const form = h('form', {class: 'graph-search', onsubmit: async event => {
    event.preventDefault();
    const entity = input.value.trim();
    if (!entity) return;
    try {
      const facts = await api('/api/books/' + book.id + '/graph?entity=' + encodeURIComponent(entity));
      results.replaceChildren(...facts.map(item => h('article', {class: 'fact-card'},
        h('h3', {}, item.fact.subject + '  —  ' + item.fact.predicate + '  —  ' + item.fact.object),
        h('div', {class: 'fact-qualifiers'}, pill(({narration: '叙述', dialogue: '人物台词', rumor: '传闻', dream: '梦境', hypothesis: '假设', unclear: '归属不明确'})[item.fact.attribution] || item.fact.attribution, 'neutral'),
          item.fact.certainty === 'unclear' ? pill('不确定', 'warn') : null, item.fact.time_text ? pill(item.fact.time_text, 'neutral') : null),
        ...item.citations.map(citation => citationView(citation, book)))));
      if (!facts.length) results.append(empty('暂未找到该实体的关系记录',
        book.index_status?.graph === 'ready' ? '关系索引已建立，请核对原文中的完整名称。索引未找到不代表全文没有。' :
          '请使用原文中的完整名称；关系索引尚未完成。索引未找到不代表全文没有。', indexControl(book, 'graph')));
    } catch (error) {toast(error.message, true);}
  }}, input, h('button', {type: 'submit', class: 'button primary'}, '查询关系'));
  body.replaceChildren(h('p', {class: 'task-notice'}, '关系记录保留出处、叙述归属与不确定性。人物台词、传闻和梦境不会直接当成确定事实。'), form, results);
  const ready = book.index_status?.graph === 'ready';
  results.append(empty(ready ? '关系索引已建立。' : '人物关系，从证据开始。',
    ready ? '输入原文中的人物或实体名称，查询已保存的关系与原文依据。' :
      '关系索引逐批整理有原文依据的关系与事件。服务重启后自动接续，临时接口故障会等待后恢复；可在分析任务中暂停。', indexControl(book, 'graph')));
}

async function renderTasks(target, bookId, version) {
  const jobs = await api('/api/jobs' + (bookId ? '?book_id=' + bookId : ''));
  if (version !== state.routeVersion) return;
  const list = h('div', {class: 'task-list', id: 'task-list', 'data-book-id': bookId || ''});
  if (!bookId) target.replaceChildren(intro('让分析慢慢发生。', '导入、索引和全文扫描在后台进行，过程和结果都留在这里。'),
    h('p', {class: 'task-notice'}, '关系索引重启后自动接续，临时接口故障会等待后恢复；可随时暂停。索引和扫描会复用已完成的检查点。'), list);
  else target.replaceChildren(h('p', {class: 'task-notice'}, '原文与已完成的证据会保留。全文扫描完成后，可下载带引用的报告。'), list);
  drawTasks(jobs, list);
}

function drawTasks(jobs, list) {
  list.replaceChildren(...jobs.map(job => {
    const actions = h('div', {class: 'task-buttons'});
    if (['failed', 'interrupted', 'paused'].includes(job.status)) actions.append(button('继续执行', async () => {
      try {const next = await api('/api/jobs/' + job.id + '/retry', {method: 'POST'}); watchJob(next.id); updateTaskRows();}
      catch (error) {toast(error.message, true);}
    }, 'quiet'));
    if (job.status === 'completed' && job.result?.validation_error) actions.append(button('重新核对', async () => {
      try {const next = await api('/api/jobs/' + job.id + '/retry', {method: 'POST'}); watchJob(next.id); updateTaskRows();}
      catch (error) {toast(error.message, true);}
    }, 'quiet'));
    if (job.kind === 'graph' && ['queued', 'running'].includes(job.status)) {
      actions.append(button('暂停', async () => {
        try {await api('/api/jobs/' + job.id + '/pause', {method: 'POST'}); updateTaskRows();}
        catch (error) {toast(error.message, true);}
      }, 'quiet'));
      if (job.auto_recovery?.waiting) actions.append(button('立即重试', async () => {
        try {const next = await api('/api/jobs/' + job.id + '/retry', {method: 'POST'}); watchJob(next.id); updateTaskRows();}
        catch (error) {toast(error.message, true);}
      }, 'quiet'));
    }
    if (job.result?.report) actions.append(h('a', {class: 'button quiet', href: '/api/jobs/' + job.id + '/download/report'}, '下载报告 ↗'));
    if (['queued', 'running'].includes(job.status)) watchJob(job.id);
    const title = kindLabels[job.kind] + (job.payload.question ? ' · ' + job.payload.question : job.payload.title ? ' · ' + job.payload.title : '');
    return h('article', {class: 'task-row'}, h('div', {class: 'task-symbol'}, job.kind === 'ingest' ? '↑' : '◷'),
      h('div', {class: 'task-body'}, h('h3', {}, title), h('p', {id: 'progress-' + job.id}, job.status === 'retried' ? job.progress : job.error || job.progress), taskProgress(job),
        h('small', {class: 'task-id', title: job.id}, '任务 ' + job.id.slice(0, 8))),
      h('span', {class: 'task-time'}, date(job.created_at)), h('div', {class: 'task-actions'},
        pill(job.auto_recovery?.waiting && job.status === 'queued' ? '等待自动恢复' : statusLabels[job.status], ['failed', 'interrupted'].includes(job.status) ? 'warn' : job.status === 'completed' ? '' : 'neutral'), actions));
  }));
  if (!jobs.length) list.append(empty('这里还没有后台任务', '导入书籍后，就可以在阅读工作台创建索引和扫描任务。', h('a', {href: '#library', class: 'button secondary'}, '返回书库')));
}

function duration(seconds) {
  seconds = Math.max(0, Math.round(seconds));
  if (seconds < 60) return seconds + ' 秒';
  if (seconds < 3600) return Math.ceil(seconds / 60) + ' 分钟';
  return Math.floor(seconds / 3600) + ' 小时 ' + Math.floor(seconds % 3600 / 60) + ' 分钟';
}

function taskProgress(job) {
  if (job.auto_recovery?.waiting && job.status === 'queued') {
    const recovery = job.auto_recovery;
    const remaining = Math.max(0, recovery.retry_at - Date.now() / 1000);
    return h('div', {class: 'task-progress'},
      h('div', {class: 'progress-caption'}, h('span', {}, '第 ' + recovery.attempt + ' 次自动恢复'),
        h('strong', {}, remaining > 0 ? '剩余等待 ' + duration(remaining) : '正在自动恢复…')),
      h('small', {class: 'estimate-note'}, '已完成的检查点保留。等待期间可暂停，或立即重试。'));
  }
  const detail = job.progress_detail;
  if (!detail?.total) return null;
  const active = ['queued', 'running'].includes(job.status);
  const stageFinished = active && detail.current >= detail.total;
  const estimating = active && !stageFinished;
  const percent = Math.max(0, Math.min(100, detail.percent || 0));
  const elapsed = estimating && detail.started_at ? (Date.now() - Date.parse(detail.started_at)) / 1000 : detail.elapsed_seconds;
  const remaining = detail.eta_at ? (Date.parse(detail.eta_at) - Date.now()) / 1000 : null;
  const remainingText = remaining === null ? '正在估算剩余时间…' : remaining <= 0 && percent < 100 ? '耗时超出估算，等待更新…' : '预计剩余 ' + duration(remaining);
  const eta = detail.eta_at ? new Date(detail.eta_at).toLocaleString('zh-CN', {month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'}) : '';
  return h('div', {class: 'task-progress'},
    h('div', {class: 'progress-caption'}, h('span', {}, detail.stage + ' · ' + detail.current.toLocaleString() + ' / ' + detail.total.toLocaleString() + ' ' + detail.unit), h('strong', {}, percent + '%')),
    h('div', {class: 'progress-track', role: 'progressbar', 'aria-label': kindLabels[job.kind] + '进度', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': percent},
      h('span', {class: 'progress-fill', style: 'width:' + percent + '%'})),
    h('div', {class: 'progress-estimate'}, h('span', {}, '已耗时 ' + duration(elapsed || 0)),
      estimating ? h('span', {}, remainingText) : null, estimating && eta && remaining > 0 ? h('span', {}, '预计完成 ' + eta) : null,
      detail.reused ? h('span', {}, '复用 ' + detail.reused.toLocaleString() + ' ' + detail.unit) : null),
    active ? h('small', {class: 'estimate-note'}, stageFinished ? '本阶段已完成，任务继续执行后续步骤。' : detail.stage?.startsWith('本地')
      ? '按本机实际推理速度估算，设备负载变化时会更新。' : '按实际处理速度估算，API 响应速度变化时会更新。') : null);
}

async function updateTaskRows() {
  const list = $('#task-list');
  if (!list) return;
  try {
    const bookId = list.getAttribute('data-book-id');
    const jobs = await api('/api/jobs' + (bookId ? '?book_id=' + bookId : ''));
    if (list.isConnected) drawTasks(jobs, list);
  } catch (_) {}
}

async function renderSettings(version) {
  const [settings, models] = await Promise.all([api('/api/settings'), api('/api/models')]);
  if (version !== state.routeVersion) return;
  const inputs = new Map();
  const field = (key, label, placeholder = '', options = {}) => {
    const secret = key.endsWith('API_KEY');
    const input = options.choices ? h('select', {name: key, 'aria-label': label}, ...options.choices.map(([value, text]) => h('option', {value}, text))) :
      h('input', {name: key, 'aria-label': label, type: secret ? 'password' : options.type || 'text', min: options.min, max: options.max, step: options.step,
        placeholder: secret && settings.secrets[key] ? '已配置 · 留空保留原密钥' : placeholder, autocomplete: 'off'});
    input.value = secret ? '' : settings.values[key] || options.default || '';
    inputs.set(key, input);
    return h('label', {class: 'field' + (options.wide ? ' wide' : '')}, label, input, options.help ? h('small', {}, options.help) : null);
  };
  const section = (title, description, ...fields) => h('section', {class: 'settings-section'}, h('h3', {}, title), h('p', {}, description), h('div', {class: 'form-grid'}, fields));
  const localPanel = h('div', {class: 'local-model-panel'});
  const modelOverview = h('section', {class: 'settings-model-overview', 'aria-label': '当前重排模型状态'});
  let localBusy = false;
  const drawLocal = model => {
    const active = model.status === 'downloading';
    const ready = model.status === 'ready';
    const enabled = settings.values.RERANK_PROVIDER === 'local' && settings.values.LOCAL_RERANK_MODEL_ID === model.model;
    const provider = settings.values.RERANK_PROVIDER || 'api';
    const selectedDevice = settings.values.LOCAL_DEVICE || 'auto';
    const runtime = selectedDevice === 'cpu' ? 'CPU' : model.cuda_available ? 'GPU · ' + model.gpu_name
      : selectedDevice === 'cuda' ? 'CUDA 不可用，请检查推理依赖' : 'CPU（当前环境没有可用 CUDA）';
    modelOverview.replaceChildren(h('div', {},
      h('strong', {}, enabled && ready ? '本地重排已启用' : provider === 'local' ? '本地重排等待就绪' : provider === 'none' ? '当前已关闭重排' : '当前使用 API 重排'),
      h('p', {}, enabled ? model.model + ' · ' + (ready ? '已下载 · ' + runtime : model.stage) : model.model + ' · ' + (ready ? '已下载，可以启用本地重排' : model.stage))),
      button('查看本地模型', () => localPanel.scrollIntoView({behavior: 'smooth', block: 'start'}), 'secondary'));
    const hardware = model.cuda_available ? '检测到 NVIDIA GPU：' + model.gpu_name + ' · CUDA 可用' : '当前环境没有可用的 CUDA，自动模式将使用 CPU';
    const deviceInput = inputs.get('LOCAL_DEVICE');
    const deviceHelp = deviceInput?.parentElement.querySelector('small');
    if (deviceHelp) deviceHelp.textContent = '可选：自动（优先 GPU）、NVIDIA GPU（CUDA）、CPU。' + hardware;
    const percent = Math.max(0, Math.min(100, model.percent || 0));
    const bytes = value => (Number(value || 0) / 1024 / 1024).toFixed(1) + ' MB';
    const download = button(ready ? '已下载' : active ? '正在下载…' : model.downloaded_bytes ? '继续下载' : '下载模型', async () => {
      download.disabled = true;
      try {drawLocal(await api('/api/models/' + model.id + '/download', {method: 'POST'})); watchDownload();}
      catch (error) {toast(error.message, true); download.disabled = false;}
    }, 'secondary');
    download.disabled = ready || active;
    const enable = button(enabled ? '本地重排已启用' : '启用本地重排', async () => {
      enable.disabled = true;
      try {
        await api('/api/settings', jsonRequest('PUT', {values: {RERANK_PROVIDER: 'local', LOCAL_RERANK_MODEL_ID: model.model, LOCAL_DEVICE: 'auto'}}));
        inputs.get('RERANK_PROVIDER').value = 'local'; inputs.get('LOCAL_RERANK_MODEL_ID').value = model.model;
        inputs.get('LOCAL_DEVICE').value = 'auto';
        settings.values.RERANK_PROVIDER = 'local'; settings.values.LOCAL_RERANK_MODEL_ID = model.model; settings.values.LOCAL_DEVICE = 'auto';
        drawLocal(model);
        toast('已启用本地重排，自动优先使用可用 GPU；首次加载需要等待。');
      } catch (error) {toast(error.message, true);} finally {enable.disabled = false;}
    }, 'primary');
    enable.disabled = enabled || !ready || !model.dependencies_ready;
    localPanel.replaceChildren(h('div', {class: 'local-model-heading'}, h('div', {}, h('strong', {}, model.model), h('p', {}, model.stage)),
      h('span', {class: 'pill'}, ready ? '已下载' : active ? '下载中' : '开源模型')),
      h('p', {class: 'helper'}, hardware),
      h('p', {class: 'helper'}, '约 1.2 GB 下载 · 原文在本机重排 · 不消耗重排 API token。CPU 速度取决于电脑和原文长度，仍可复用评分缓存。'),
      active || model.downloaded_bytes ? h('div', {class: 'task-progress'},
        h('div', {class: 'progress-caption'}, h('span', {}, bytes(model.downloaded_bytes) + ' / ' + bytes(model.total_bytes)), h('strong', {}, percent + '%')),
        h('div', {class: 'progress-track', role: 'progressbar', 'aria-label': '模型下载进度', 'aria-valuenow': percent, 'aria-valuemin': 0, 'aria-valuemax': 100},
          h('span', {class: 'progress-fill', style: 'width:' + percent + '%'})),
        active ? h('div', {class: 'progress-estimate'}, h('span', {}, model.bytes_per_second ? bytes(model.bytes_per_second) + '/秒' : '正在测量下载速度…'),
          h('span', {}, model.eta_seconds == null ? '正在估算剩余时间…' : '预计剩余 ' + duration(model.eta_seconds))) : null) : null,
      !model.dependencies_ready ? h('p', {class: 'helper'}, '本地推理组件尚未安装。在项目目录依次执行：',
        h('code', {}, '.venv/bin/python -m pip install "torch>=2.2,<3" --index-url https://download.pytorch.org/whl/cpu'),
        h('code', {}, '.venv/bin/python -m pip install -r requirements-rerank-local.txt')) : null,
      h('div', {class: 'local-model-actions'}, download, enable));
  };
  const updateDownload = async () => {
    if (localBusy) return;
    localBusy = true;
    try {
      const rows = await api('/api/models');
      if (version === state.routeVersion) {
        drawLocal(rows[0]);
        if (rows[0].status !== 'downloading') clearInterval(state.modelTimer);
      }
    } catch (_) {} finally {localBusy = false;}
  };
  const watchDownload = () => {clearInterval(state.modelTimer); state.modelTimer = setInterval(updateDownload, 1500);};
  const submit = h('button', {class: 'button primary', type: 'submit'}, '保存配置 →');
  const form = h('form', {class: 'settings-form', onsubmit: async event => {
    event.preventDefault(); submit.disabled = true;
    try {
      const values = Object.fromEntries([...inputs].filter(([key, input]) => !key.endsWith('API_KEY') || input.value.trim()).map(([key, input]) => [key, input.value]));
      const result = await api('/api/settings', jsonRequest('PUT', {values}));
      Object.assign(settings.values, values); updateDownload();
      toast(result.environment_overrides.length ? '已保存；部分设置由启动环境变量覆盖。' : '配置已保存，新任务会使用新配置。');
      for (const [key, input] of inputs) if (key.endsWith('API_KEY') && input.value) {input.value = ''; input.placeholder = '已配置 · 留空保留原密钥';}
    } catch (error) {toast(error.message, true);} finally {submit.disabled = false;}
  }},
    section('回答模型', '用于理解问题和生成回答。填写你使用的兼容模型服务；密钥保存在本项目配置文件中。',
      field('LLM_BASE_URL', '接口地址', 'https://你的服务商/v1', {wide: true}), field('LLM_MODEL_ID', '模型名称', '服务商提供的模型 ID'),
      field('LLM_API_KEY', 'API 密钥', '输入密钥'), field('LLM_TIMEOUT', '请求超时（秒）', '120', {default: '120'}),
      field('LLM_TEMPERATURE', '回答随机性（可选）', '支持时可填 0，不支持时留空', {help: '数值越低，回答通常越稳定。留空使用模型服务的默认设置；此参数不能保证事实准确。'})),
    section('原文检索', '长篇小说推荐混合检索：同时寻找词语一致和意思相近的原文。嵌入模型只需建立一次索引。',
      field('RETRIEVAL_MODE', '默认检索方式', '', {choices: [['hybrid', '语义与关键词（推荐）'], ['lexical', '关键词检索']], default: 'hybrid'}),
      field('EMBEDDING_MODEL_ID', '嵌入模型名称', '例如服务商提供的 Qwen3-Embedding 模型'),
      field('EMBEDDING_BASE_URL', '嵌入接口地址', 'https://你的服务商/v1', {wide: true}), field('EMBEDDING_API_KEY', '嵌入 API 密钥', '与回答服务相同时可留空'),
      field('EMBEDDING_DIMENSIONS', '向量维度（可选）', '使用服务默认值时留空'),
      field('EMBEDDING_QUERY_PREFIX', '查询指令前缀（可选）', '按嵌入模型的说明配置', {wide: true})),
    section('检索结果重排', '本地重排覆盖全部召回候选，分窗、多遍、分批评分并显示进度和预计时间；评分缓存 7 天。候选覆盖不代表扫描全文，CPU 耗时可能较长。API 使用独立费用上限。',
      field('RERANK_PROVIDER', '重排方式', '', {choices: [['api', 'API 重排（有费用）'], ['local', '本地开源模型（无 API 费用）'], ['none', '关闭重排']], default: 'api'}),
      field('LOCAL_RERANK_MODEL_ID', '本地重排模型', '', {choices: [['Qwen/Qwen3-Reranker-0.6B', 'Qwen3-Reranker-0.6B']], default: 'Qwen/Qwen3-Reranker-0.6B'}),
      field('LOCAL_DEVICE', '本地推理设备', '', {choices: [['auto', '自动（优先 GPU）'], ['cuda', 'NVIDIA GPU（CUDA）'], ['cpu', 'CPU']], default: 'auto',
        help: 'GPU 需要 CUDA 版 PyTorch 和可用的 NVIDIA 驱动。自动模式在 CUDA 不可用时使用 CPU。'}),
      field('LOCAL_RERANK_PASSES', '本地重排遍数', '2', {type: 'number', min: '1', max: '3', step: '1', default: '2',
        help: '首遍用当前问题，后续遍用当前问题加不同检索关键词。每遍覆盖相应候选；没有不同表述时不重复相同推理。'}),
      field('LOCAL_RERANK_BATCH_SIZE', '本地每批段落数', '8', {type: 'number', min: '1', max: '32', step: '1', default: '8',
        help: '每批保存评分，可复用已完成结果。CPU 逐窗口推理，GPU 自动使用小批次；此处控制段落结果的缓存批次。'}),
      field('LOCAL_RERANK_WINDOW_CHARS', '本地原文窗口字数', '1200', {type: 'number', min: '300', max: '1600', step: '1', default: '1200',
        help: '长段落拆成重叠窗口，完整覆盖到末尾；每段保留最高窗口分数。较小窗口会增加推理次数。'}),
      field('RERANK_MAX_DOCUMENTS', 'API 新评分段落上限', '128', {type: 'number', min: '0', max: '2048', step: '1', default: '128',
        help: '仅适用于 API。跨查询与检索轮次共享预算；0 表示只复用已有评分。'}),
      field('RERANK_MAX_CHARS', 'API 原文输入字数上限', '200000', {type: 'number', min: '0', max: '4000000', step: '1', default: '200000',
        help: '字数不等于 token 数。预算越低，检索效果可能下降；重试可能重复发送同一批输入。'}),
      field('RERANK_MODEL_ID', 'API 重排模型名称', '例如服务商提供的 Qwen3-Reranker 模型'),
      field('RERANK_URL', '重排完整接口地址', 'https://你的服务商/v1/rerank', {wide: true, help: '支持兼容重排与千问原生重排接口。千问官方根地址配合原生重排模型时，会自动使用专用路径。'}), field('RERANK_API_KEY', '重排 API 密钥', '与回答服务相同时可留空')),
    h('section', {class: 'settings-section'}, h('h3', {}, '下载本地重排模型'), h('p', {}, '下载到本项目，完成后可直接启用。服务重启或网络中断后可以继续下载。'), localPanel),
    section('独立复核与存储', '可以使用另一款模型检查回答。未单独填写时复用回答模型。普通个人书库可使用内置向量存储；较大书库可连接 Qdrant 服务。',
      field('VERIFY_MODEL_ID', '复核模型（可选）', '留空复用回答模型'), field('VERIFY_API_KEY', '复核密钥（可选）', '留空复用回答密钥'),
      field('VERIFY_BASE_URL', '复核接口地址（可选）', '留空复用回答接口', {wide: true}),
      field('QDRANT_URL', 'Qdrant 服务地址（可选）', 'http://localhost:6333'), field('QDRANT_API_KEY', 'Qdrant 密钥（可选）', '本地服务通常无需配置')),
    h('div', {class: 'settings-footer'}, submit, h('span', {}, '保存后对新任务生效 · 已保存密钥不回传到页面')));
  content.replaceChildren(intro('给阅读，配置一个助手。', '本机管理原文，回答与嵌入通过 API 调用，重排支持本地开源模型。'), modelOverview, form);
  drawLocal(models[0]);
  if (models[0].status === 'downloading') watchDownload();
}

// 只展示接口明确报告的 token；缓存命中属于输入的一部分，不重复加到汇总。
async function renderUsage(version) {
  const labels = {llm: '回答模型', embedding: '嵌入模型', rerank: '重排模型'};
  const descriptions = {llm: '包含问题分析、原文提取、回答与结论复核', embedding: '包含原文索引与查询向量', rerank: '包含 API 与本地评分，复用已有分数不计入'};
  const format = value => Number(value || 0).toLocaleString('zh-CN');
  const tokenText = (row, key) => {
    const records = row[key.replace('_tokens', '_records')];
    if (!records && row.requests) return '未返回';
    return format(row[key]) + (records < row.requests ? '（已知）' : '');
  };
  const period = h('select', {'aria-label': '统计时间范围'},
    ...[['0', '全部记录'], ['1', '最近 24 小时'], ['7', '最近 7 天'], ['30', '最近 30 天']].map(([value, label]) => h('option', {value}, label)));
  const panels = h('div', {id: 'usage-panels'});
  const updated = h('span', {class: 'helper'});
  let sequence = 0, busy = false;
  const refresh = button('刷新用量', () => load(), 'secondary');
  const load = async () => {
    const request = ++sequence;
    busy = true; refresh.disabled = true;
    try {
      const data = await api('/api/usage?days=' + period.value);
      if (version !== state.routeVersion || request !== sequence) return;
      const total = data.total;
      const metric = (label, value) => h('div', {class: 'usage-metric'}, h('span', {}, label), h('strong', {}, value));
      const summary = h('section', {class: 'usage-summary', 'aria-label': '用量汇总'},
        h('div', {}, h('span', {class: 'eyebrow'}, '三类模型汇总'), h('h2', {}, tokenText(total, 'total_tokens')), h('span', {class: 'helper'}, '总 token · 接口报告的已记录用量')),
        h('div', {class: 'usage-metrics'}, metric('输入 token', tokenText(total, 'input_tokens')), metric('输出 token', tokenText(total, 'output_tokens')),
          metric('调用记录', format(total.requests)), metric('失败调用', format(total.failed_requests))));
      const cards = h('div', {class: 'usage-cards'}, ...data.categories.map(row => {
        const models = [...new Set(data.models.filter(item => item.category === row.category).map(item => item.model))];
        return h('section', {class: 'usage-card', 'aria-label': labels[row.category] + '用量'},
          h('h3', {}, labels[row.category]), h('p', {}, descriptions[row.category]),
          h('div', {class: 'usage-number'}, tokenText(row, 'total_tokens')), h('span', {class: 'helper'}, '总 token'),
          h('div', {class: 'usage-metrics'}, metric('输入', tokenText(row, 'input_tokens')), metric('输出', tokenText(row, 'output_tokens')),
            metric('调用记录', format(row.requests)), metric('缺少总用量', format(row.missing_usage))),
          h('small', {class: 'usage-model'}, models.length ? models.join(' / ') : '暂无调用记录'),
          row.local_calls ? h('small', {class: 'usage-model'}, '其中本地重排 ' + format(row.local_calls) + ' 次 · API token 为 0') : null);
      }));
      const table = h('table', {class: 'usage-table'},
        h('thead', {}, h('tr', {}, ...['模型 / 用途', '调用记录', '输入 token', '输出 token', '总 token', '缺少总用量'].map(label => h('th', {scope: 'col'}, label)))),
        h('tbody', {}, ...data.models.map(row => h('tr', {},
          h('th', {scope: 'row'}, row.model, h('small', {}, labels[row.category] + ' · ' + row.operation)),
          h('td', {}, format(row.requests)), h('td', {}, tokenText(row, 'input_tokens')),
          h('td', {}, tokenText(row, 'output_tokens')), h('td', {}, tokenText(row, 'total_tokens')), h('td', {}, format(row.missing_usage))))));
      const details = h('section', {class: 'settings-section usage-details'}, h('h3', {}, '模型与用途明细'),
        h('p', {}, '同一模型按用途展开，结论复核计入回答模型，汇总只计一次。'),
        data.models.length ? h('div', {class: 'usage-table-scroll', tabindex: '0', 'aria-label': '模型用量明细表'}, table) : h('p', {class: 'helper'}, '此时间范围内还没有模型调用记录。'));
      const notes = h('div', {class: 'usage-notes'},
        h('p', {}, '统计保存在本机，页面打开时每 10 秒更新。记录的是本项目消耗，不代表服务商账户余额或剩余额度。'),
        h('p', {}, '调用记录包含实际 API 请求、本地模型调用及补入的旧日志。本地重排消耗的 API token 为 0。'),
        h('p', {}, '输入、输出和总 token 分别使用接口返回值；未返回的字段不估算，“已知”表示存在缺失记录。失败或连接中断的请求是否计费，以服务商账单为准。'),
        total.historical_calls ? h('p', {}, '当前范围补入 ' + format(total.historical_calls) + ' 条旧日志调用记录。旧日志按逻辑调用计数，无法还原所有重试；已轮转删除的日志无法补回。') : null,
        total.cached_records ? h('p', {}, '接口已报告缓存命中输入 ' + format(total.cached_input_tokens) + ' token，包含在输入中，不额外累加。') : null,
        data.first_record_at ? h('p', {}, '最早保留记录：' + new Date(data.first_record_at).toLocaleString('zh-CN')) : null);
      panels.replaceChildren(summary, cards, details, notes);
      updated.textContent = '更新于 ' + new Date(data.updated_at).toLocaleTimeString('zh-CN');
    } catch (error) {
      if (version === state.routeVersion && request === sequence) {
        if (!panels.children.length) panels.replaceChildren(empty('暂时无法读取用量', error.message));
        else toast(error.message, true);
      }
    } finally {
      if (request === sequence) {busy = false; refresh.disabled = false;}
    }
  };
  period.onchange = () => load();
  content.replaceChildren(intro('看清每一次模型消耗。', '回答、嵌入和重排分别统计，汇总以接口实际返回的用量为准。'),
    h('div', {class: 'usage-toolbar'}, h('label', {}, '统计范围', period), refresh, updated), panels);
  await load();
  if (version === state.routeVersion) state.usageTimer = setInterval(() => {if (!busy) load();}, 10000);
}

async function route() {
  checkFrontendVersion();
  clearInterval(state.usageTimer);
  clearInterval(state.modelTimer);
  const version = ++state.routeVersion;
  const parts = (location.hash || '#library').slice(1).split('/');
  const view = parts[0];
  document.querySelectorAll('[data-nav]').forEach(link => link.classList.toggle('active', link.dataset.nav === (view === 'book' ? 'library' : view)));
  $('#breadcrumb').textContent = ({library: '我的书库', tasks: '后台任务', settings: '模型设置', usage: '用量统计'})[view] || '阅读工作台';
  content.setAttribute('aria-busy', 'true');
  try {
    if (view === 'settings') await renderSettings(version);
    else if (view === 'usage') await renderUsage(version);
    else if (view === 'tasks') await renderTasks(content, null, version);
    else if (view === 'book' && parts[1]) await renderBook(parts[1], parts[2] || 'chat', version, parts[3]);
    else await renderLibrary(version);
  } catch (error) {
    if (version === state.routeVersion) content.replaceChildren(empty('暂时无法打开内容', error.message,
      button('重新加载', () => route(), 'secondary')));
  } finally {
    if (version === state.routeVersion) content.removeAttribute('aria-busy');
  }
}

// 上传只负责传文件；耗时的原文分块和检索索引由后台任务执行。
$('#upload-form').addEventListener('submit', async event => {
  event.preventDefault();
  const submit = $('#upload-submit');
  submit.disabled = true; submit.textContent = '正在上传…';
  $('#upload-error').textContent = '';
  try {
    const form = new FormData(event.target);
    const job = await api('/api/books', {method: 'POST', body: form});
    $('#upload-dialog').close(); event.target.reset(); $('#file-label').textContent = '点击选择，或将原文拖到这里';
    watchJob(job.id); toast('原文已上传，正在后台导入。');
    location.hash = '#tasks'; route();
  } catch (error) {$('#upload-error').textContent = error.message;}
  finally {submit.disabled = false; submit.textContent = '导入原文 ↗';}
});

function selectedFile() {
  const file = $('#file-input').files[0];
  $('#file-label').textContent = file ? file.name : '点击选择，或将原文拖到这里';
  if (file && !$('#upload-title').value) $('#upload-title').value = file.name.replace(/\.(txt|md|markdown)$/i, '');
}
$('#file-input').addEventListener('change', selectedFile);
for (const name of ['dragenter', 'dragover']) $('#drop-zone').addEventListener(name, event => {event.preventDefault(); $('#drop-zone').classList.add('dragover');});
for (const name of ['dragleave', 'drop']) $('#drop-zone').addEventListener(name, event => {event.preventDefault(); $('#drop-zone').classList.remove('dragover');});
$('#drop-zone').addEventListener('drop', event => {
  if (event.dataTransfer.files.length) {
    const transfer = new DataTransfer(); transfer.items.add(event.dataTransfer.files[0]);
    $('#file-input').files = transfer.files; selectedFile();
  }
});
$('#close-upload').onclick = $('#cancel-upload').onclick = () => $('#upload-dialog').close();
$('#close-quote').onclick = () => $('#quote-dialog').close();
window.addEventListener('hashchange', route);
api('/api/health').then(health => {state.health = health; state.mode = health.retrieval_mode || 'hybrid';}).catch(() => {});
route();
refreshActivity();
setInterval(refreshActivity, 12000);
// 任务页定时更新进度及剩余时间，页面刷新后也能恢复任务状态。
setInterval(() => {if ($('#task-list') && document.visibilityState === 'visible') updateTaskRows();}, 2000);
