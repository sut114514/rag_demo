const $ = (selector) => document.querySelector(selector);
const history = [];
let busy = false;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function scrollToBottom() { $('#scroll-area').scrollTop = $('#scroll-area').scrollHeight; }
function addMessage(role, text) {
  const node = element('article', `message ${role}`);
  if (role === 'assistant') node.append(element('div', 'speaker', '知识库助手'));
  node.append(element('div', 'answer', text));
  $('#messages').append(node);
  scrollToBottom();
  return node;
}
function renderSources(node, sources) {
  if (!sources.length) return;
  const list = element('details', 'source-list');
  list.append(element('summary', '', `参考来源 · ${sources.length} 条配方（点击展开）`));
  for (const source of sources) {
    const detail = element('details', 'source');
    const summary = element('summary');
    summary.append(element('span', '', `[${source.citation}] ${source.title}`));
    summary.append(element('span', 'source-meta', `第 ${source.page} 页 · ${source.relation === 'ingredient' ? '材料关联' : '检索命中'}`));
    detail.append(summary, element('pre', '', source.text));
    const pdf = element('a', '', '查看 PDF 原文 ↗');
    pdf.href = `/source.pdf#page=${Number(source.page)}`;
    pdf.target = '_blank'; pdf.rel = 'noopener';
    detail.append(pdf);
    try {
      const url = new URL(source.url);
      if (url.protocol === 'https:' && url.hostname === 'terraria.wiki.gg') {
        const wiki = element('a', '', 'Wiki 来源 ↗');
        wiki.href = url.href; wiki.target = '_blank'; wiki.rel = 'noopener';
        detail.append(wiki);
      }
    } catch (_) { /* 非法来源 URL 不显示链接 */ }
    list.append(detail);
  }
  node.append(list);
}
async function ask(question) {
  question = question.trim();
  if (!question || busy) return;
  busy = true;
  $('#welcome').hidden = true;
  $('#send').disabled = true; $('#clear').disabled = true; $('#clear-mobile').disabled = true;
  $('#question').value = '';
  addMessage('user', question);
  const answerNode = addMessage('assistant', '正在检索配方并生成回答…');
  const answer = answerNode.querySelector('.answer'); answer.classList.add('pending');
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history: history.slice(-8), top_k: 4 }), signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请求无效，请检查问题长度后重试。');
    answer.textContent = data.answer; answer.classList.remove('pending');
    renderSources(answerNode, data.sources);
    answerNode.append(element('div', 'reply-meta', `${data.model || '本地检索'} · ${data.elapsed_seconds} 秒`));
    history.push({ role: 'user', content: question }, { role: 'assistant', content: data.answer.slice(0, 6000) });
    if (history.length > 8) history.splice(0, history.length - 8);
  } catch (error) {
    answer.className = 'answer error-message';
    answer.textContent = error.name === 'AbortError' ? '等待回答超时，请稍后重试。' : (error.message || '请求失败，请检查本地服务是否运行。');
    const retry = element('button', 'retry', '重试这个问题');
    retry.onclick = () => { if (!busy) ask(question); };
    answerNode.append(retry);
  } finally {
    clearTimeout(timeout); busy = false;
    $('#send').disabled = false; $('#clear').disabled = false; $('#clear-mobile').disabled = false;
    scrollToBottom(); $('#question').focus();
  }
}
$('#chat-form').addEventListener('submit', event => { event.preventDefault(); ask($('#question').value); });
$('#question').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault(); ask($('#question').value);
  }
});
document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => ask(button.dataset.question)));
function clearChat() {
  if (busy) return;
  history.length = 0; $('#messages').replaceChildren(); $('#welcome').hidden = false;
  $('#question').value = ''; $('#question').focus();
}
$('#clear').addEventListener('click', clearChat);
$('#clear-mobile').addEventListener('click', clearChat);
fetch('/api/health').then(response => { if (!response.ok) throw new Error(); return response.json(); }).then(data => {
  $('#kb-status').textContent = `${data.chunk_count} 条配方 · 本地向量检索`;
  $('#provider-status').textContent = data.api_key_configured ? 'API 密钥已配置' : '待配置 API 密钥';
  $('#model-label').textContent = data.model;
  $('#status-dot').className = `dot ${data.api_key_configured ? 'ready' : 'error'}`;
}).catch(() => {
  $('#kb-status').textContent = '知识库连接失败'; $('#provider-status').textContent = '服务未连接';
  $('#status-dot').className = 'dot error';
});
