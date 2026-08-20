/* ================= MiniLlama Chat 前端逻辑 ================= */
'use strict';

// ---------- 状态 ----------
const LS_CONVS = 'minillama.convs';
const LS_SETTINGS = 'minillama.settings';

const state = { convs: [], currentId: null };
let settings = {
  maxTokens: 32,
  temperature: 0,
  useContext: true,
  stream: true,
};
let sending = false;
let controller = null;      // AbortController，用于停止生成
let assistantEl = null;     // 当前正在生成的 assistant 气泡元素

// ---------- 元素 ----------
const el = (id) => document.getElementById(id);
const messagesBox = el('messages');
const inputBox = el('input');
const welcomeEl = el('welcome');

// ---------- 工具 ----------
function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* 极简 markdown 渲染：代码块 / 行内代码 / 标题 / 加粗 / 斜体 / 列表 / 引用 / 换行 */
function renderMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/```([\s\S]*?)```/g, function (m, code) {
    return '<pre><code>' + code.trim() + '</code></pre>';
  });
  html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');
  html = html.replace(/^### (.*)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.*)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.*)$/gm, '<h1>$1</h1>');
  html = html.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
  html = html.replace(/^[-*] (.*)$/gm, '<li>$1</li>');
  html = html.replace(/^&gt; (.*)$/gm, '<blockquote>$1</blockquote>');
  html = html.replace(/\n/g, '<br>');
  return html;
}

function scrollBottom() {
  messagesBox.scrollTop = messagesBox.scrollHeight;
}

// ---------- 持久化 ----------
function saveConvs() {
  try { localStorage.setItem(LS_CONVS, JSON.stringify(state.convs)); } catch (e) {}
}
function saveSettings() {
  try { localStorage.setItem(LS_SETTINGS, JSON.stringify(settings)); } catch (e) {}
}
function loadState() {
  try {
    const raw = localStorage.getItem(LS_CONVS);
    if (raw) {
      state.convs = JSON.parse(raw);
      if (Array.isArray(state.convs) && state.convs.length > 0) {
        state.currentId = state.convs[0].id;
      }
    }
  } catch (e) {}
  try {
    const raw = localStorage.getItem(LS_SETTINGS);
    if (raw) Object.assign(settings, JSON.parse(raw));
  } catch (e) {}
}

// ---------- 会话 ----------
function currentConv() {
  return state.convs.find((c) => c.id === state.currentId) || null;
}

function ensureConversation() {
  if (currentConv()) return currentConv();
  const conv = { id: uid(), title: '新对话', createdAt: Date.now(), messages: [] };
  state.convs.unshift(conv);
  state.currentId = conv.id;
  saveConvs();
  return conv;
}

function switchConversation(id) {
  if (sending) return;
  state.currentId = id;
  renderList();
  renderMessages();
  updateTitle();
}

function deleteConversation(id) {
  if (sending) return;
  const idx = state.convs.findIndex((c) => c.id === id);
  if (idx < 0) return;
  state.convs.splice(idx, 1);
  if (state.currentId === id) {
    state.currentId = state.convs.length ? state.convs[0].id : null;
  }
  saveConvs();
  renderList();
  renderMessages();
  updateTitle();
}

function updateTitle() {
  const conv = currentConv();
  el('chat-title').textContent = conv ? conv.title : '新对话';
}

// ---------- 渲染 ----------
function renderList() {
  const list = el('conv-list');
  list.innerHTML = '';
  for (const conv of state.convs) {
    const item = document.createElement('div');
    item.className = 'conv-item' + (conv.id === state.currentId ? ' active' : '');

    const title = document.createElement('span');
    title.className = 'conv-title';
    title.textContent = conv.title;
    title.title = conv.title;
    title.addEventListener('click', () => switchConversation(conv.id));

    const del = document.createElement('button');
    del.className = 'conv-del';
    del.textContent = '✕';
    del.title = '删除会话';
    del.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteConversation(conv.id);
    });

    item.appendChild(title);
    item.appendChild(del);
    list.appendChild(item);
  }
}

function makeMsgEl(role, html) {
  const msg = document.createElement('div');
  msg.className = 'msg ' + role;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '我' : '🦙';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = html;
  msg.appendChild(avatar);
  msg.appendChild(bubble);
  return msg;
}

function renderMessages() {
  const conv = currentConv();
  welcomeEl.style.display = conv && conv.messages.length ? 'none' : '';
  // 保留欢迎页之外的子元素，重建消息
  const keep = [welcomeEl];
  for (const child of Array.from(messagesBox.children)) {
    if (!keep.includes(child)) child.remove();
  }
  if (!conv) return;
  for (const m of conv.messages) {
    if (m.role === 'user') {
      messagesBox.appendChild(makeMsgEl('user', escapeHtml(m.content).replace(/\n/g, '<br>')));
    } else {
      messagesBox.appendChild(makeMsgEl('assistant', renderMarkdown(m.content)));
    }
  }
  scrollBottom();
}

function appendTyping() {
  const msg = document.createElement('div');
  msg.className = 'msg assistant';
  msg.innerHTML =
    '<div class="avatar">🦙</div>' +
    '<div class="bubble"><span class="typing"><span></span><span></span><span></span></span></div>';
  messagesBox.appendChild(msg);
  scrollBottom();
  return msg;
}

function setAssistantText(text) {
  if (!assistantEl) return;
  const bubble = assistantEl.querySelector('.bubble');
  bubble.innerHTML = renderMarkdown(text) + '<span class="cursor"></span>';
  scrollBottom();
}

function finalizeAssistant() {
  if (!assistantEl) return;
  const bubble = assistantEl.querySelector('.bubble');
  // 去掉光标
  bubble.innerHTML = bubble.innerHTML.replace(/<span class="cursor"><\/span>/, '');
  assistantEl = null;
}

function failAssistant(message) {
  if (assistantEl) {
    const bubble = assistantEl.querySelector('.bubble');
    bubble.className = 'bubble error';
    bubble.innerHTML = renderMarkdown(message);
    assistantEl = null;
    scrollBottom();
  }
}

// ---------- 请求 ----------
function buildPrompt() {
  const conv = currentConv();
  if (!conv) return '';
  const msgs = conv.messages;
  if (!settings.useContext) {
    const last = msgs.filter((m) => m.role === 'user').pop();
    return last ? last.content : '';
  }
  const parts = [];
  for (const m of msgs.slice(-10)) {
    parts.push((m.role === 'user' ? 'User: ' : 'Assistant: ') + m.content);
  }
  return parts.join('\n');
}

function describeError(err) {
  if (err && err.name === 'TypeError') {
    return (
      '❌ 无法连接到服务器。请确认：\n' +
      '1. serve 已启动（WSL 里运行 ./mini-llama serve models/tiny --port 8080）\n' +
      '2. 浏览器地址用的是 http://<WSL的IP>:8080 （Windows 的 localhost 可能连不上 WSL2）\n' +
      '3. 服务器端口与地址一致'
    );
  }
  return '❌ ' + (err && err.message ? err.message : String(err));
}

async function readSSE(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n\n')) >= 0) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      for (const line of raw.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        let data;
        try { data = JSON.parse(line.slice(6)); } catch (e) { continue; }
        if (data.error) throw new Error(data.error);
        if (typeof data.text === 'string') setAssistantText(data.text);
        if (data.done) return;
      }
    }
  }
}

async function requestCompletion(prompt) {
  const payload = {
    prompt: prompt,
    max_tokens: settings.maxTokens,
    temperature: settings.temperature,
    stream: !!settings.stream,
  };
  controller = new AbortController();
  const res = await fetch('/v1/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: controller.signal,
  });
  if (!res.ok) {
    let msg = 'HTTP ' + res.status;
    try {
      const j = await res.json();
      if (j && j.error) msg = j.error;
    } catch (e) {}
    throw new Error(msg);
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('text/event-stream') && res.body) {
    await readSSE(res);
  } else {
    const j = await res.json();
    if (j && j.error) throw new Error(j.error);
    setAssistantText((j && j.generated_text) || '');
  }
}

async function sendMessage() {
  if (sending) return;
  const text = inputBox.value.trim();
  if (!text) return;

  const conv = ensureConversation();
  // 会话标题：取第一条用户消息
  if (conv.title === '新对话') {
    conv.title = text.length > 20 ? text.slice(0, 20) + '…' : text;
    updateTitle();
  }
  conv.messages.push({ role: 'user', content: text });
  saveConvs();

  inputBox.value = '';
  autoResize();
  renderMessages();
  renderList();

  assistantEl = appendTyping();
  sending = true;
  el('send-btn').disabled = true;
  el('stop-wrap').classList.remove('hidden');

  try {
    const prompt = buildPrompt();
    await requestCompletion(prompt);
    finalizeAssistant();
  } catch (err) {
    if (err && err.name === 'AbortError') {
      finalizeAssistant(); // 用户主动停止，保留已生成内容
    } else {
      failAssistant(describeError(err));
    }
  } finally {
    sending = false;
    el('send-btn').disabled = false;
    el('stop-wrap').classList.add('hidden');
    saveConvs();
  }
}

// ---------- 输入框 ----------
function autoResize() {
  inputBox.style.height = 'auto';
  inputBox.style.height = Math.min(inputBox.scrollHeight, 160) + 'px';
}

// ---------- 健康检查 ----------
async function checkHealth() {
  const dot = el('status-dot');
  const name = el('model-name');
  try {
    const res = await fetch('/health');
    if (!res.ok) throw new Error('bad status');
    const j = await res.json();
    if (j && j.status === 'ok') {
      dot.className = 'dot online';
      if (j.model) {
        name.textContent = 'mini-llama · ' + j.model.n_layers + '层 · vocab ' + j.model.vocab_size;
      } else {
        name.textContent = 'mini-llama';
      }
    } else {
      dot.className = 'dot';
    }
  } catch (e) {
    dot.className = 'dot';
    name.textContent = 'mini-llama';
  }
}

// ---------- 设置弹窗 ----------
function openSettings() {
  el('max-tokens').value = settings.maxTokens;
  el('temperature').value = settings.temperature;
  el('use-context').checked = settings.useContext;
  el('stream').checked = settings.stream;
  el('max-tokens-val').textContent = settings.maxTokens;
  el('temperature-val').textContent = Number(settings.temperature).toFixed(2);
  el('settings-modal').classList.remove('hidden');
}
function closeSettings() {
  el('settings-modal').classList.add('hidden');
}

// ---------- 事件绑定 ----------
function init() {
  loadState();

  el('new-chat-btn').addEventListener('click', () => {
    if (sending) return;
    const conv = { id: uid(), title: '新对话', createdAt: Date.now(), messages: [] };
    state.convs.unshift(conv);
    state.currentId = conv.id;
    saveConvs();
    renderList();
    renderMessages();
    updateTitle();
  });

  el('send-btn').addEventListener('click', sendMessage);

  inputBox.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  inputBox.addEventListener('input', autoResize);

  el('stop-btn').addEventListener('click', () => {
    if (controller) controller.abort();
  });

  el('clear-btn').addEventListener('click', () => {
    if (sending) return;
    const conv = currentConv();
    if (conv && conv.messages.length) {
      conv.messages = [];
      saveConvs();
      renderMessages();
      updateTitle();
    }
  });

  el('sidebar-toggle').addEventListener('click', () => {
    el('sidebar').classList.toggle('collapsed');
  });

  el('settings-btn').addEventListener('click', openSettings);
  el('modal-close').addEventListener('click', closeSettings);
  el('settings-modal').addEventListener('click', (e) => {
    if (e.target === el('settings-modal')) closeSettings();
  });

  el('max-tokens').addEventListener('input', (e) => {
    settings.maxTokens = parseInt(e.target.value, 10) || 32;
    el('max-tokens-val').textContent = settings.maxTokens;
    saveSettings();
  });
  el('temperature').addEventListener('input', (e) => {
    settings.temperature = parseFloat(e.target.value) || 0;
    el('temperature-val').textContent = Number(settings.temperature).toFixed(2);
    saveSettings();
  });
  el('use-context').addEventListener('change', (e) => {
    settings.useContext = e.target.checked;
    saveSettings();
  });
  el('stream').addEventListener('change', (e) => {
    settings.stream = e.target.checked;
    saveSettings();
  });

  renderList();
  renderMessages();
  updateTitle();
  checkHealth();
  setInterval(checkHealth, 15000);
}

document.addEventListener('DOMContentLoaded', init);
