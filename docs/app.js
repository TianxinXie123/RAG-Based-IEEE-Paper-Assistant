// app.js - the page's logic: find the backend (server.py), upload a paper, and chat about it.
"use strict";

// Where server.py listens by default; used when this page is not served by server.py itself
const DEFAULT_BACKEND = "http://localhost:8000";
const MAX_UPLOAD_MB = 50;     // same limit as server.py
const MAX_HISTORY_TURNS = 3;  // same as main.py: earlier turns sent along for follow-up questions
const RETRY_MS = 4000;        // how often to look for the backend while it is offline

// What the user's message says when a whole-paper task is asked with a button or command
const TASK_LABELS = {
  summary: "Summarize this paper",
  suggest: "Review this paper: strengths, weaknesses and suggestions",
};

const HELP_TEXT = `**Commands**

- \`/summary\` structured summary of the whole paper
- \`/suggest\` strengths, weaknesses and suggestions for the work
- \`/reset\` start a new chat
- \`/help\` show this list

Anything else is a question about the paper. Questions that look unrelated to the paper are not sent
to the model; start one with \`!\` to ask anyway.`;

const AVATAR_ICON = '<svg class="icon" viewBox="0 0 24 24"><path d="M2 4h6a4 4 0 0 1 4 4v13a3 3 0 0 0-3-3H2z"/><path d="M22 4h-6a4 4 0 0 0-4 4v13a3 3 0 0 1 3-3h7z"/></svg>';
const CHEVRON_ICON = '<svg class="icon" viewBox="0 0 24 24"><path d="m9 18 6-6-6-6"/></svg>';

// "[Source 2]", "[Source 1, Source 3]", "[Sources 1 and 2]"
const CITATION_RE = /\[Sources?\s+(\d+(?:\s*(?:,|and|&)\s*(?:Source\s+)?\d+)*)\]/g;

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status-button"),
  statusText: $("status-text"),
  upload: $("upload-section"),
  fileInput: $("file-input"),
  uploadError: $("upload-error"),
  indexingName: $("indexing-name"),
  indexingElapsed: $("indexing-elapsed"),
  paperName: $("paper-name"),
  paperMeta: $("paper-meta"),
  replace: $("replace-button"),
  empty: $("empty-state"),
  messages: $("messages"),
  newChat: $("new-chat-button"),
  composer: $("composer"),
  question: $("question"),
  send: $("send-button"),
  settings: $("settings-dialog"),
  backendInput: $("backend-input"),
  settingsReset: $("settings-reset"),
};

const state = {
  backend: null,     // base URL of the connected backend
  connecting: false,
  retryTimer: null,
  paper: null,       // {name, chunks, figures} of the loaded paper
  indexing: false,
  history: [],       // recent [question, answer] pairs
  controller: null,  // AbortController of the answer being streamed, if any
};

// localStorage can be unavailable (private windows, blocked storage); the page works without it
const storage = {
  get(key) {
    try { return localStorage.getItem(key); } catch { return null; }
  },
  set(key, value) {
    try {
      if (value) localStorage.setItem(key, value);
      else localStorage.removeItem(key);
    } catch { /* not saved, which only means it is asked again next time */ }
  },
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ---------- Backend connection ---------- */

function backendCandidates() {
  const saved = storage.get("backend");
  if (saved) return [saved];
  // Served by server.py: the same origin. Otherwise (GitHub Pages, a local file): the default address.
  const candidates = location.protocol.startsWith("http") ? [location.origin] : [];
  if (!candidates.includes(DEFAULT_BACKEND)) candidates.push(DEFAULT_BACKEND);
  return candidates;
}

async function connect() {
  if (state.connecting) return;
  state.connecting = true;
  clearTimeout(state.retryTimer);
  try {
    for (const base of backendCandidates()) {
      try {
        const res = await fetch(`${base}/api/health`, { signal: AbortSignal.timeout(3000) });
        const info = await res.json();
        if (info.app === "ieee-paper-assistant") return onConnected(base, info);
      } catch {
        // Nothing there (or not our backend): try the next address
      }
    }
    onDisconnected();
  } finally {
    state.connecting = false;
  }
}

function onConnected(base, info) {
  state.backend = base;
  setStatus("online", "Connected", `Backend: ${base}\nModel: ${info.model}`);
  if (!state.indexing) {
    if (info.paper) {
      showPaper(info.paper);
    } else {
      state.paper = null;  // the backend restarted, or no paper was uploaded yet
      setUploadState("empty");
    }
  }
  updateControls();
}

function onDisconnected() {
  state.backend = null;
  setStatus("offline", "Backend offline", "Click to change the backend address");
  if (!state.indexing) setUploadState("offline");
  updateControls();
  state.retryTimer = setTimeout(connect, RETRY_MS);
}

function setStatus(name, text, title) {
  els.status.dataset.state = name;
  els.statusText.textContent = text;
  els.status.title = title;
}

function openSettings() {
  els.backendInput.value = storage.get("backend") || state.backend || DEFAULT_BACKEND;
  els.settings.returnValue = "";
  els.settings.showModal();
}

/* ---------- Upload ---------- */

function setUploadState(name) {
  els.upload.dataset.state = name;
}

function showUploadError(message) {
  els.uploadError.textContent = message;
  setUploadState("error");
}

function showPaper(paper) {
  state.paper = paper;
  els.paperName.textContent = paper.name;
  els.paperName.title = paper.name;
  els.paperMeta.textContent = `Ready · indexed into ${paper.chunks} searchable passages`;
  setUploadState("ready");
  updateControls();
}

async function uploadPaper(file) {
  if (!file || !state.backend || state.indexing) return;
  if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") {
    return showUploadError(`"${file.name}" is not a PDF.`);
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    return showUploadError(`"${file.name}" is larger than ${MAX_UPLOAD_MB} MB.`);
  }

  state.indexing = true;
  els.indexingName.textContent = file.name;
  els.indexingElapsed.textContent = "0 s";
  setUploadState("indexing");
  updateControls();
  const started = Date.now();
  const timer = setInterval(() => {
    els.indexingElapsed.textContent = `${Math.round((Date.now() - started) / 1000)} s`;
  }, 1000);

  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${state.backend}/api/upload`, { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Upload failed (HTTP ${res.status}).`);

    showPaper(data.paper);
    // Earlier questions were about another paper
    state.history = [];
    if (els.messages.children.length) els.messages.append(el("div", "divider", `Now reading ${data.paper.name}`));
    els.question.focus({ preventScroll: true });
  } catch (err) {
    if (err instanceof TypeError) {  // fetch could not reach the server at all
      showUploadError("Could not reach the backend. Is server.py still running?");
      connect();
    } else {
      showUploadError(err.message);
    }
  } finally {
    clearInterval(timer);
    state.indexing = false;
    els.fileInput.value = "";  // so choosing the same file again still fires "change"
    updateControls();
  }
}

/* ---------- Chat ---------- */

function isNearBottom() {
  return window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160;
}

function scrollToBottom() {
  window.scrollTo({ top: document.documentElement.scrollHeight });
}

// Keep the newest text in view while it streams in, unless the reader has scrolled up
function keepInView(update) {
  const follow = isNearBottom();
  update();
  if (follow) scrollToBottom();
}

function renderMarkdown(container, text) {
  if (window.marked && window.DOMPurify) {
    container.innerHTML = DOMPurify.sanitize(marked.parse(text));
  } else {
    // The Markdown libraries did not load (offline?): show the plain text
    const plain = el("p", null, text);
    plain.style.whiteSpace = "pre-wrap";
    container.replaceChildren(plain);
  }
  linkCitations(container);
}

// Turn "[Source N]" in the rendered answer into buttons that open that source
function linkCitations(container) {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const textNodes = [];
  while (walker.nextNode()) {
    if (!walker.currentNode.parentElement.closest("code, pre")) textNodes.push(walker.currentNode);
  }
  for (const node of textNodes) {
    const text = node.nodeValue;
    const matches = [...text.matchAll(CITATION_RE)];
    if (!matches.length) continue;
    const parts = [];
    let last = 0;
    for (const match of matches) {
      parts.push(text.slice(last, match.index));
      for (const number of match[1].match(/\d+/g)) {
        const button = el("button", "cite", number);
        button.type = "button";
        button.dataset.source = number;
        button.title = `Show source ${number}`;
        parts.push(button);
      }
      last = match.index + match[0].length;
    }
    parts.push(text.slice(last));
    node.replaceWith(...parts);
  }
}

function showSource(cite) {
  const details = cite.closest(".msg-assistant").querySelector(".sources");
  const item = details?.querySelector(`.source[data-source="${cite.dataset.source}"]`);
  if (!item) return;
  details.open = true;
  item.classList.add("expanded", "highlight");
  item.scrollIntoView({ behavior: "smooth", block: "center" });
  setTimeout(() => item.classList.remove("highlight"), 1600);
}

function addUserMessage(text) {
  els.empty.hidden = true;
  els.messages.append(el("div", "msg-user", text));
  scrollToBottom();
}

// One answer of the assistant: a status line while waiting, then the streamed text, notes and sources
class AssistantMessage {
  constructor() {
    this.text = "";
    this.renderQueued = false;
    this.sources = null;

    this.root = el("div", "msg-assistant");
    const avatar = el("div", "avatar");
    avatar.innerHTML = AVATAR_ICON;
    this.body = el("div", "msg-body");
    this.statusLine = el("div", "status-line");
    this.statusLine.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    this.statusText = el("span");
    this.statusLine.append(this.statusText);
    this.answer = el("div", "markdown");
    this.body.append(this.statusLine, this.answer);
    this.root.append(avatar, this.body);
    keepInView(() => els.messages.append(this.root));
  }

  setStatus(text) {
    this.statusText.textContent = text;
  }

  appendText(piece) {
    this.text += piece;
    this.statusLine.hidden = true;
    if (this.renderQueued) return;
    // Render at most once per frame, however fast the pieces arrive
    this.renderQueued = true;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      keepInView(() => renderMarkdown(this.answer, this.text));
    });
  }

  setSources(sources) {
    const details = el("details", "sources");
    const summary = el("summary");
    summary.innerHTML = CHEVRON_ICON;
    summary.append(`${sources.length} ${sources.length === 1 ? "passage" : "passages"} from the paper`);
    const list = el("ol");
    sources.forEach((text, i) => {
      const item = el("li", "source");
      item.dataset.source = i + 1;
      item.title = "Click to show all of it";
      item.append(el("span", "source-label", `Source ${i + 1}`), el("p", "source-text", text));
      list.append(item);
    });
    details.append(summary, list);
    this.sources = details;
    keepInView(() => this.body.append(details));
  }

  // kind: "warning" (nothing relevant found), "error" or "muted"
  addNote(text, kind) {
    keepInView(() => this.body.insertBefore(el("p", `note note-${kind}`, text), this.sources));
  }

  finish() {
    this.statusLine.remove();
  }
}

// The NDJSON stream of /api/ask, one parsed event at a time
async function* readEvents(body) {
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const lines = buffer.split("\n");
    buffer = lines.pop();  // an incomplete last line waits for the next piece
    for (const line of lines) {
      if (line.trim()) yield JSON.parse(line);
    }
  }
  if (buffer.trim()) yield JSON.parse(buffer);
}

function canAsk() {
  return Boolean(state.backend && state.paper && !state.indexing && !state.controller);
}

async function ask({ question = "", task = null, label = question }) {
  if (!canAsk()) return;
  addUserMessage(label);
  const message = new AssistantMessage();
  message.setStatus("Thinking...");
  state.controller = new AbortController();
  updateControls();

  let finished = false;
  try {
    const res = await fetch(`${state.backend}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, task, history: task ? [] : state.history }),
      signal: state.controller.signal,
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `The backend answered with HTTP ${res.status}.`);
    }
    for await (const event of readEvents(res.body)) {
      if (event.type === "status") {
        message.setStatus(event.text);
      } else if (event.type === "sources") {
        message.setSources(event.sources);
      } else if (event.type === "text") {
        message.appendText(event.text);
      } else if (event.type === "notice" || event.type === "error") {
        message.addNote(event.text, event.type === "notice" ? "warning" : "error");
        finished = true;
      } else if (event.type === "done") {
        finished = true;
        // Whole-paper tasks are not part of the conversation, as in main.py
        if (!task) remember(question.replace(/^!\s*/, ""), message.text);
      }
    }
    if (!finished) message.addNote("The connection to the backend was lost before the answer was complete.", "error");
  } catch (err) {
    if (err.name === "AbortError") {
      message.addNote("Stopped.", "muted");
    } else if (err instanceof TypeError) {
      message.addNote("Could not reach the backend. Is server.py still running?", "error");
      connect();
    } else {
      message.addNote(err.message, "error");
    }
  } finally {
    message.finish();
    state.controller = null;
    updateControls();
  }
}

function remember(question, answer) {
  state.history.push([question, answer]);
  state.history = state.history.slice(-MAX_HISTORY_TURNS);
}

// An answer the page gives itself, e.g. for /help
function showLocalAnswer(userText, markdown) {
  addUserMessage(userText);
  const message = new AssistantMessage();
  message.appendText(markdown);
  message.finish();
  updateControls();
}

function newChat() {
  state.controller?.abort();
  state.history = [];
  els.messages.replaceChildren();
  els.empty.hidden = false;
  updateControls();
}

function submitQuestion() {
  const text = els.question.value.trim();
  if (!text || !canAsk()) return;
  els.question.value = "";
  autosize();

  // The same commands as in main.py
  const command = text.toLowerCase();
  if (command === "/help") return showLocalAnswer(text, HELP_TEXT);
  if (command === "/reset") return newChat();
  if (command === "/summary" || command === "/suggest") {
    const task = command.slice(1);
    return ask({ task, label: TASK_LABELS[task] });
  }
  if (command.startsWith("/")) {
    return showLocalAnswer(text, `Unknown command \`${text}\`. Type \`/help\` for the list of commands.`);
  }
  ask({ question: text });
}

function autosize() {
  els.question.style.height = "auto";
  els.question.style.height = `${Math.min(els.question.scrollHeight, 200)}px`;
}

function updateControls() {
  const ready = Boolean(state.backend && state.paper && !state.indexing);
  const busy = Boolean(state.controller);

  els.question.disabled = !ready;
  els.question.placeholder = !state.backend ? "Start the backend to begin"
    : state.indexing ? "Reading the paper…"
    : !state.paper ? "Upload a paper to start asking questions"
    : "Ask anything about this paper…";
  for (const button of document.querySelectorAll("[data-task], [data-question]")) {
    button.disabled = !ready || busy;
  }
  els.send.classList.toggle("stop", busy);
  els.send.setAttribute("aria-label", busy ? "Stop" : "Send");
  els.send.disabled = !busy && (!ready || !els.question.value.trim());
  els.newChat.disabled = !els.messages.children.length;
  els.replace.disabled = !state.backend || state.indexing;
}

/* ---------- Events ---------- */

els.question.addEventListener("input", () => {
  autosize();
  updateControls();
});

els.question.addEventListener("keydown", (event) => {
  // isComposing: this Enter confirms an IME candidate (e.g. Chinese input) and must not send
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!state.controller) submitQuestion();
  }
});

els.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.controller) state.controller.abort();
  else submitQuestion();
});

for (const button of document.querySelectorAll("[data-task]")) {
  button.addEventListener("click", () => ask({ task: button.dataset.task, label: TASK_LABELS[button.dataset.task] }));
}
for (const button of document.querySelectorAll("[data-question]")) {
  button.addEventListener("click", () => ask({ question: button.dataset.question }));
}
els.newChat.addEventListener("click", newChat);

els.messages.addEventListener("click", (event) => {
  const cite = event.target.closest(".cite");
  if (cite) return showSource(cite);
  const source = event.target.closest(".source");
  // Toggle the full text, unless the click ended a text selection
  if (source && !window.getSelection().toString()) source.classList.toggle("expanded");
});

els.fileInput.addEventListener("change", () => uploadPaper(els.fileInput.files[0]));
els.replace.addEventListener("click", () => els.fileInput.click());

// Drag and drop onto the upload section; the counter is needed because
// dragenter/dragleave also fire when the pointer moves between child elements
let dragDepth = 0;
els.upload.addEventListener("dragenter", (event) => {
  event.preventDefault();
  dragDepth++;
  if (state.backend && !state.indexing) els.upload.classList.add("dragging");
});
els.upload.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) els.upload.classList.remove("dragging");
});
els.upload.addEventListener("dragover", (event) => event.preventDefault());
els.upload.addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  els.upload.classList.remove("dragging");
  uploadPaper(event.dataTransfer.files[0]);
});
// A file dropped anywhere else must not make the browser leave the page to open it
window.addEventListener("dragover", (event) => event.preventDefault());
window.addEventListener("drop", (event) => event.preventDefault());

els.status.addEventListener("click", openSettings);
for (const button of document.querySelectorAll("[data-open-settings]")) {
  button.addEventListener("click", openSettings);
}
els.settingsReset.addEventListener("click", () => {
  storage.set("backend", null);
  els.settings.close();
  connect();
});
els.settings.addEventListener("close", () => {
  if (els.settings.returnValue !== "save") return;
  storage.set("backend", els.backendInput.value.trim().replace(/\/+$/, ""));
  connect();
});

// Coming back to the tab: check that the backend (and its paper) are still there
window.addEventListener("focus", connect);

updateControls();
connect();
