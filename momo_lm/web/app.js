"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const titles = {
  chat: "與 Momo 對話",
  agents: "本機代理工作",
  learn: "建立專屬知識",
  image: "本機圖像生成",
  speech: "離線語音工作室",
  weights: "觀察模型權重",
  mods: "擴充模組",
};
const agentStatuses = {
  pending: "等待中",
  running: "執行中",
  waiting_approval: "等待核准",
  completed: "已完成",
  failed: "失敗",
  cancelled: "已取消",
};
const agentProfiles = {
  training: "模型訓練",
  coding: "程式開發",
  workplace: "職場工作",
  copilot: "協作駕駛員",
};
const agentProfileHints = {
  training: "輸入「train: 訓練文字」並勾選模型訓練，可在單次核准後更新權重；其他目標只產生訓練檢查表。",
  coding: "輸入「read: 相對路徑」可讀檔；輸入「write: 相對路徑」後換行貼上內容，可在單次核准後寫入隔離工作區。",
  workplace: "職場工作會搜尋本機知識並產生草稿，不會傳送郵件、發布內容或連接外部服務。",
  copilot: "協作駕駛員會檢查本機模型、搜尋知識，再整理一份工作摘要。",
};
const terminalAgentStatuses = new Set(["completed", "failed", "cancelled"]);
const tokenStorageKey = "momo-lm-session-token";

const state = {
  token: "",
  status: null,
  activePanel: "chat",
  agents: [],
  selectedAgentId: null,
  selectedAgent: null,
  eventAgentId: null,
  lastEventSeq: 0,
  agentPollTimer: null,
  eventPollTimer: null,
  agentPollBusy: false,
  eventPollBusy: false,
  assetObjectUrls: { image: null, audio: null },
};

function makeElement(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function appendTextElement(parent, tag, className, text) {
  const node = makeElement(tag, className, text);
  parent.append(node);
  return node;
}

function toast(message) {
  const box = $("#toast");
  box.textContent = String(message);
  box.classList.add("show");
  clearTimeout(box.timer);
  box.timer = setTimeout(() => box.classList.remove("show"), 2800);
}

function isValidToken(token) {
  return typeof token === "string" && token.length >= 1 && token.length <= 1024 && /^[\x21-\x7e]+$/.test(token);
}

function sessionGetToken() {
  try {
    return sessionStorage.getItem(tokenStorageKey) || "";
  } catch {
    return "";
  }
}

function sessionSetToken(token) {
  try {
    if (token) sessionStorage.setItem(tokenStorageKey, token);
    else sessionStorage.removeItem(tokenStorageKey);
  } catch {
    // Browsers may disable session storage. The in-memory token still works.
  }
}

function takeStartupToken() {
  const current = new URL(window.location.href);
  const fragment = new URLSearchParams(current.hash.startsWith("#") ? current.hash.slice(1) : current.hash);
  const fragmentToken = fragment.get("token") || "";
  const queryToken = current.searchParams.get("token") || "";
  const candidate = fragmentToken || queryToken;
  let changed = false;

  if (fragment.has("token")) {
    fragment.delete("token");
    current.hash = fragment.toString() ? `#${fragment.toString()}` : "";
    changed = true;
  }
  if (current.searchParams.has("token")) {
    current.searchParams.delete("token");
    changed = true;
  }
  if (changed) history.replaceState(null, "", `${current.pathname}${current.search}${current.hash}`);

  if (candidate && isValidToken(candidate)) {
    sessionSetToken(candidate);
    return candidate;
  }
  return sessionGetToken();
}

function showAuth(message = "需要工作台存取權杖") {
  $("#authTitle").textContent = message;
  $("#authBanner").hidden = false;
}

function hideAuth() {
  $("#authBanner").hidden = true;
  $("#tokenInput").value = "";
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("Accept", "application/json");
  if (options.body !== undefined) headers.set("Content-Type", "application/json");
  if (state.token) headers.set("X-Momo-Token", state.token);

  const response = await fetch(path, {
    ...options,
    headers,
    credentials: "same-origin",
    redirect: "error",
  });
  const raw = await response.text();
  let data = {};
  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch {
      throw new Error(`伺服器回傳無效資料（HTTP ${response.status}）`);
    }
  }
  if (!response.ok) {
    if (response.status === 401) showAuth("存取權杖無效或已過期");
    const message = data && typeof data.error === "string" ? data.error : `HTTP ${response.status}`;
    throw new Error(message);
  }
  hideAuth();
  return data;
}

async function withButton(button, pendingText, operation) {
  const previous = button.textContent;
  button.disabled = true;
  button.textContent = pendingText;
  try {
    return await operation();
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error));
    return null;
  } finally {
    button.disabled = false;
    button.textContent = previous;
  }
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? new Intl.NumberFormat().format(number) : "—";
}

function formatTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, { dateStyle: "short", timeStyle: "medium" }).format(date);
}

function safeJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function safeAssetUrl(value, prefix) {
  const url = new URL(String(value), window.location.origin);
  if (url.origin !== window.location.origin || !url.pathname.startsWith(prefix)) {
    throw new Error("伺服器回傳了不允許的檔案位置");
  }
  return url.href;
}

async function authenticatedAssetUrl(value, prefix, kind, expectedMimePrefix) {
  const url = safeAssetUrl(value, prefix);
  const headers = new Headers();
  if (state.token) headers.set("X-Momo-Token", state.token);
  const response = await fetch(url, {
    headers,
    credentials: "same-origin",
    redirect: "error",
  });
  if (!response.ok) {
    if (response.status === 401) showAuth("存取權杖無效或已過期");
    let message = `檔案讀取失敗（HTTP ${response.status}）`;
    try {
      const data = await response.json();
      if (data && typeof data.error === "string") message = data.error;
    } catch {
      // Binary endpoints may return an empty or plain-text error body.
    }
    throw new Error(message);
  }
  const mime = response.headers.get("Content-Type") || "";
  if (!mime.toLowerCase().startsWith(expectedMimePrefix)) {
    throw new Error("伺服器回傳了不符合預期的檔案類型");
  }
  const blob = await response.blob();
  const previous = state.assetObjectUrls[kind];
  if (previous) URL.revokeObjectURL(previous);
  const objectUrl = URL.createObjectURL(blob);
  state.assetObjectUrls[kind] = objectUrl;
  return objectUrl;
}

function switchPanel(panelName) {
  if (!Object.hasOwn(titles, panelName)) return;
  state.activePanel = panelName;
  $$("#nav button").forEach((button) => button.classList.toggle("active", button.dataset.panel === panelName));
  $$(".panel").forEach((panel) => panel.classList.toggle("active", panel.id === panelName));
  $("#pageTitle").textContent = titles[panelName];
  if (panelName === "agents") {
    loadAgents({ silent: true });
    scheduleAgentPoll(2000);
    if (state.selectedAgentId) scheduleEventPoll(500);
  } else {
    clearTimeout(state.agentPollTimer);
    clearTimeout(state.eventPollTimer);
  }
}

function updateAgentProfileForm() {
  const profile = $("#agentProfile").value;
  $("#agentProfileHint").textContent = agentProfileHints[profile] || "";
  $$("[data-agent-profiles]").forEach((label) => {
    const profiles = String(label.dataset.agentProfiles || "").split(/\s+/);
    const available = profiles.includes(profile);
    label.hidden = !available;
    const checkbox = label.querySelector("input[type=checkbox]");
    if (checkbox) {
      checkbox.disabled = !available;
      if (!available) checkbox.checked = false;
    }
  });
}

$$("#nav button").forEach((button) => button.addEventListener("click", () => switchPanel(button.dataset.panel)));
$("#theme").addEventListener("click", () => document.body.classList.toggle("light"));
$("#agentProfile").addEventListener("change", updateAgentProfileForm);

$("#tokenForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const token = $("#tokenInput").value.trim();
  if (!isValidToken(token)) {
    toast("權杖必須是 1–1024 個可見 ASCII 字元");
    return;
  }
  state.token = token;
  sessionSetToken(token);
  const status = await loadStatus();
  if (status && state.activePanel === "agents") await loadAgents({ silent: true });
});

$("#chatInput").addEventListener("input", (event) => {
  event.target.style.height = "auto";
  event.target.style.height = `${Math.min(event.target.scrollHeight, 180)}px`;
});
$("#chatInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#chatForm").requestSubmit();
  }
});

function addMessage(role, text, meta = "") {
  const article = makeElement("article", `message ${role}`);
  const avatar = makeElement("div", "avatar", role === "momo" ? "M" : "Y");
  avatar.setAttribute("aria-hidden", "true");
  const box = makeElement("div");
  appendTextElement(box, "b", "", role === "momo" ? "Momo" : "You");
  appendTextElement(box, "p", "", text);
  if (meta) appendTextElement(box, "small", "", meta);
  article.append(avatar, box);
  $("#messages").append(article);
  article.scrollIntoView({ behavior: "smooth", block: "end" });
  return article;
}

$("#chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("#chatInput");
  const message = input.value.trim();
  if (!message) return;
  addMessage("user", message);
  input.value = "";
  input.style.height = "auto";
  const waiting = addMessage("momo", "思考與檢索本機資料中…");
  try {
    const result = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, learn: $("#learnToggle").checked }),
    });
    waiting.remove();
    const sources = Array.isArray(result.sources) ? result.sources.map(String) : [];
    const meta = sources.length ? `來源：${sources.join("、")}` : result.learned ? "已完成本機增量學習" : "";
    addMessage("momo", String(result.response || ""), meta);
    await loadStatus();
  } catch (error) {
    waiting.remove();
    addMessage("momo", `發生錯誤：${error instanceof Error ? error.message : String(error)}`);
  }
});

$("#ingestBtn").addEventListener("click", () => withButton($("#ingestBtn"), "學習中…", async () => {
  const text = $("#learnText").value.trim();
  if (!text) throw new Error("請先輸入學習資料");
  const train = $("#trainNow").checked;
  const path = train ? "/api/train" : "/api/ingest";
  const body = train
    ? { text, source: $("#source").value, epochs: Number($("#epochs").value) }
    : { text, source: $("#source").value, train: false };
  const result = await api(path, { method: "POST", body: JSON.stringify(body) });
  $("#learnResult").textContent = safeJson(result);
  toast("資料已存入本機");
  await loadStatus();
}));

$("#crawlBtn").addEventListener("click", () => withButton($("#crawlBtn"), "讀取中…", async () => {
  const result = await api("/api/crawl", {
    method: "POST",
    body: JSON.stringify({
      url: $("#crawlUrl").value,
      max_pages: Number($("#crawlPages").value),
      train: $("#crawlTrain").checked,
    }),
  });
  $("#learnResult").textContent = safeJson(result);
  toast(`已處理 ${formatNumber(result.visited)} 個頁面`);
  await loadStatus();
}));

$("#imageBtn").addEventListener("click", () => withButton($("#imageBtn"), "生成中…", async () => {
  const prompt = $("#imagePrompt").value.trim();
  if (!prompt) throw new Error("請輸入圖像描述");
  const size = Number($("#imageSize").value);
  const seedText = $("#imageSeed").value.trim();
  const stepsText = $("#imageSteps").value.trim();
  const payload = {
    prompt,
    negative_prompt: $("#negativePrompt").value.trim(),
    style: $("#imageStyle").value,
    quality: $("#imageQuality").value,
    width: size,
    height: size,
  };
  if (seedText) {
    const seed = Number(seedText);
    if (!Number.isSafeInteger(seed)) throw new Error("Seed 必須是安全整數");
    payload.seed = seed;
  }
  if (stepsText) {
    const steps = Number(stepsText);
    if (!Number.isInteger(steps) || steps < 1 || steps > 8) throw new Error("推理步數必須介於 1 到 8");
    payload.steps = steps;
  }
  const result = await api("/api/image", { method: "POST", body: JSON.stringify(payload) });
  const outputUrl = await authenticatedAssetUrl(result.url, "/generated/", "image", "image/");
  const image = $("#imageOutput");
  image.src = outputUrl;
  image.classList.add("show");
  $("#imageEmpty").hidden = true;
  $("#downloadImage").href = outputUrl;
  $("#downloadImage").classList.add("show");
  const meta = $("#imageMeta");
  meta.replaceChildren();
  const values = [
    result.style || payload.style,
    result.quality || payload.quality,
    `${formatNumber(result.width || size)} × ${formatNumber(result.height || size)}`,
    `${formatNumber(result.steps || payload.steps || ({ draft: 1, standard: 2, high: 4 }[payload.quality]))} 步`,
  ];
  values.forEach((value) => appendTextElement(meta, "span", "", value));
  meta.hidden = false;
  toast("圖像已在本機生成");
}));

$("#speechRate").addEventListener("input", (event) => {
  $("#rateOutput").textContent = event.target.value;
});
$("#speechBtn").addEventListener("click", () => withButton($("#speechBtn"), "產生中…", async () => {
  const text = $("#speechText").value.trim();
  if (!text) throw new Error("請輸入要朗讀的文字");
  const result = await api("/api/tts", {
    method: "POST",
    body: JSON.stringify({ text, rate: Number($("#speechRate").value) }),
  });
  const audio = $("#audioOutput");
  audio.src = await authenticatedAssetUrl(result.url, "/speech/", "audio", "audio/");
  audio.classList.add("show");
  $("#speechEngine").textContent = `引擎：${String(result.engine || "local")}`;
  audio.play().catch(() => {});
  toast("語音已產生");
}));

function addStat(parent, label, value) {
  const box = makeElement("div", "stat");
  appendTextElement(box, "small", "", label);
  appendTextElement(box, "strong", "", formatNumber(value));
  parent.append(box);
}

function renderStatus(data) {
  state.status = data;
  const weights = data && typeof data.weights === "object" && data.weights ? data.weights : {};
  const knowledge = data && typeof data.knowledge === "object" && data.knowledge ? data.knowledge : {};
  $("#statusDot").classList.remove("bad");
  $("#statusDot").classList.add("good");
  $("#statusText").textContent = "本機模型已就緒";
  $("#modelMeta").textContent = `${formatNumber(weights.parameters)} parameters`;
  if (typeof data.self_learning === "boolean") $("#learnToggle").checked = data.self_learning;

  const stats = $("#stats");
  stats.replaceChildren();
  addStat(stats, "文字模型參數", weights.parameters);
  addStat(stats, "訓練步數", weights.training_steps);
  addStat(stats, "已學習 tokens", weights.tokens_seen);
  addStat(stats, "知識片段", knowledge.documents);

  const layers = $("#layers");
  layers.replaceChildren();
  const layerEntries = weights.layers && typeof weights.layers === "object" ? Object.entries(weights.layers) : [];
  const parameterValues = layerEntries.map(([, layer]) => Number(layer && layer.parameters) || 0);
  const maximum = Math.max(1, ...parameterValues);
  for (const [name, layerValue] of layerEntries) {
    const layer = layerValue && typeof layerValue === "object" ? layerValue : {};
    const item = makeElement("div", "layer");
    appendTextElement(item, "b", "", name);
    const shape = Array.isArray(layer.shape) ? layer.shape.map(String).join(" × ") : "—";
    appendTextElement(item, "code", "", shape);
    const bar = makeElement("div", "bar");
    const fill = makeElement("i");
    const ratio = Math.max(2, Math.min(100, ((Number(layer.parameters) || 0) / maximum) * 100));
    fill.style.width = `${ratio}%`;
    bar.append(fill);
    item.append(bar);
    const std = Number(layer.std);
    appendTextElement(item, "small", "", Number.isFinite(std) ? `σ ${std.toFixed(4)}` : `${formatNumber(layer.parameters)} params`);
    layers.append(item);
  }
  if (!layerEntries.length) appendTextElement(layers, "p", "empty-state", "目前沒有可顯示的張量資料。");
  renderMods(data.mods || { loaded: [], errors: [] });
}

function renderMods(modsValue) {
  const mods = modsValue && typeof modsValue === "object" ? modsValue : {};
  const loaded = Array.isArray(mods.loaded) ? mods.loaded : [];
  const errors = Array.isArray(mods.errors) ? mods.errors : [];
  const list = $("#modList");
  list.replaceChildren();
  for (const value of loaded) {
    const mod = value && typeof value === "object" ? value : {};
    const item = makeElement("div", "mod");
    appendTextElement(item, "b", "", mod.name || "未命名模組");
    appendTextElement(item, "small", "", mod.version ? `v${mod.version}` : "");
    appendTextElement(item, "p", "", mod.description || "沒有說明");
    const commands = Array.isArray(mod.commands) ? mod.commands.map(String).join(" · ") : "";
    appendTextElement(item, "code", "", commands || "hooks only");
    list.append(item);
  }
  for (const value of errors) {
    const error = value && typeof value === "object" ? value : {};
    const item = makeElement("div", "mod mod-error");
    appendTextElement(item, "b", "", `載入失敗：${error.file || "unknown"}`);
    appendTextElement(item, "p", "", error.error || "未知錯誤");
    list.append(item);
  }
  if (!loaded.length && !errors.length) appendTextElement(list, "p", "empty-state", "尚未載入模組。可從儲存庫的 mods/ 範例開始。");
}

async function loadStatus() {
  try {
    const data = await api("/api/status");
    renderStatus(data);
    return data;
  } catch (error) {
    $("#statusDot").classList.remove("good");
    $("#statusDot").classList.add("bad");
    $("#statusText").textContent = "無法連接";
    $("#modelMeta").textContent = error instanceof Error ? error.message : "請檢查本機服務";
    return null;
  }
}

$("#refreshWeights").addEventListener("click", async () => {
  const result = await loadStatus();
  if (result) toast("權重狀態已更新");
});
$("#reloadMods").addEventListener("click", () => withButton($("#reloadMods"), "載入中…", async () => {
  const result = await api("/api/mods/reload", { method: "POST", body: "{}" });
  renderMods(result);
  toast("Mods 已重新載入");
}));

function statusClass(status) {
  return Object.hasOwn(agentStatuses, status) ? `status-${status}` : "status-unknown";
}

function renderAgentList(agents) {
  const list = $("#agentList");
  list.replaceChildren();
  if (!agents.length) {
    appendTextElement(list, "p", "empty-state", "目前沒有代理工作。左側可建立第一個工作。");
    return;
  }
  for (const agent of agents) {
    const button = makeElement("button", "agent-item");
    button.type = "button";
    if (agent.id === state.selectedAgentId) button.classList.add("selected");
    button.dataset.agentId = String(agent.id);
    const top = makeElement("span", "agent-item-top");
    appendTextElement(top, "span", "agent-profile", agentProfiles[agent.profile] || agent.profile || "代理");
    appendTextElement(top, "span", `status-chip ${statusClass(agent.status)}`, agentStatuses[agent.status] || agent.status || "未知");
    button.append(top);
    appendTextElement(button, "span", "agent-goal", agent.goal || "未命名工作");
    appendTextElement(button, "span", "agent-updated", `更新於 ${formatTime(agent.updated_at)}`);
    button.addEventListener("click", () => selectAgent(String(agent.id)));
    list.append(button);
  }
}

function usageValue(label, used, limit) {
  const box = makeElement("div", "usage-item");
  appendTextElement(box, "small", "", label);
  const hasLimit = Number.isFinite(Number(limit));
  appendTextElement(box, "strong", "", hasLimit ? `${formatNumber(used)} / ${formatNumber(limit)}` : formatNumber(used));
  return box;
}

function renderAgentDetail(agent) {
  if (!agent || String(agent.id) !== state.selectedAgentId) return;
  state.selectedAgent = agent;
  $("#agentDetailCard").hidden = false;
  $("#agentDetailProfile").textContent = `${agentProfiles[agent.profile] || agent.profile || "代理"} · ${String(agent.id)}`;
  $("#agentDetailGoal").textContent = agent.goal || "未命名工作";
  const status = $("#agentDetailStatus");
  status.className = `status-chip ${statusClass(agent.status)}`;
  status.textContent = agentStatuses[agent.status] || agent.status || "未知";
  $("#cancelAgent").hidden = terminalAgentStatuses.has(agent.status);

  const budgets = agent.budgets && typeof agent.budgets === "object" ? agent.budgets : {};
  const usage = $("#agentUsage");
  usage.replaceChildren(
    usageValue("步驟", agent.steps_used, budgets.max_steps),
    usageValue("工具呼叫", agent.tool_calls_used, budgets.max_tool_calls),
  );
  const capabilities = Array.isArray(agent.capabilities) ? agent.capabilities : [];
  const capabilityBox = makeElement("div", "usage-item capability-usage");
  appendTextElement(capabilityBox, "small", "", "允許能力");
  appendTextElement(capabilityBox, "strong", "", capabilities.length ? capabilities.map(String).join(" · ") : "受限預設");
  usage.append(capabilityBox);

  const pending = agent.pending_approval && typeof agent.pending_approval === "object" ? agent.pending_approval : null;
  const approvalBox = $("#approvalBox");
  approvalBox.hidden = !pending;
  if (pending) {
    $("#approvalTool").textContent = pending.tool || "未知工具";
    $("#approvalReason").textContent = pending.reason || "沒有提供理由";
    $("#approvalArguments").textContent = safeJson(pending.arguments || {});
    $("#approvalExpiry").textContent = `到期：${formatTime(pending.expires_at)}`;
    $("#approveAgent").disabled = false;
  }

  const outcome = $("#agentOutcome");
  const hasResult = agent.result !== null && agent.result !== undefined && agent.result !== "";
  const hasError = Boolean(agent.error);
  outcome.hidden = !hasResult && !hasError;
  if (hasError) {
    $("#outcomeTitle").textContent = "錯誤";
    $("#outcomeText").textContent = typeof agent.error === "string" ? agent.error : safeJson(agent.error);
  } else if (hasResult) {
    $("#outcomeTitle").textContent = "結果";
    $("#outcomeText").textContent = typeof agent.result === "string" ? agent.result : safeJson(agent.result);
  }
}

async function loadAgents({ silent = false } = {}) {
  try {
    const result = await api("/api/agents");
    state.agents = Array.isArray(result.agents) ? result.agents : [];
    renderAgentList(state.agents);
    if (state.selectedAgentId) {
      const selected = state.agents.find((agent) => String(agent.id) === state.selectedAgentId);
      if (selected) renderAgentDetail(selected);
      else await loadAgentDetail(state.selectedAgentId);
    }
    return result;
  } catch (error) {
    if (!silent) toast(error instanceof Error ? error.message : String(error));
    return null;
  }
}

async function loadAgentDetail(agentId) {
  try {
    const result = await api(`/api/agents/${encodeURIComponent(agentId)}`);
    if (result.agent) renderAgentDetail(result.agent);
    return result.agent || null;
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error));
    return null;
  }
}

function resetAgentEvents() {
  state.lastEventSeq = 0;
  $("#agentEvents").replaceChildren();
  $("#eventConnection").textContent = "讀取事件…";
}

function appendAgentEvents(events) {
  const list = $("#agentEvents");
  for (const event of events) {
    const sequence = Number(event.seq);
    if (Number.isFinite(sequence) && sequence <= state.lastEventSeq) continue;
    const item = makeElement("li", "event-item");
    const top = makeElement("div", "event-top");
    appendTextElement(top, "span", "event-type", event.type || "event");
    appendTextElement(top, "time", "", formatTime(event.created_at));
    item.append(top);
    appendTextElement(item, "p", "", event.message || "");
    const eventData = event.data && typeof event.data === "object" ? event.data : null;
    if (eventData && Object.keys(eventData).length) {
      const details = makeElement("details", "event-data");
      appendTextElement(details, "summary", "", "事件資料");
      appendTextElement(details, "pre", "", safeJson(eventData));
      item.append(details);
    }
    list.append(item);
    if (Number.isFinite(sequence)) state.lastEventSeq = Math.max(state.lastEventSeq, sequence);
  }
  while (list.children.length > 200) list.firstElementChild.remove();
  if (!list.children.length) appendTextElement(list, "li", "empty-state", "目前沒有事件。");
}

async function loadAgentEvents() {
  const agentId = state.selectedAgentId;
  if (!agentId || state.eventPollBusy) return;
  if (state.eventAgentId !== agentId) {
    state.eventAgentId = agentId;
    resetAgentEvents();
  }
  state.eventPollBusy = true;
  try {
    const result = await api(`/api/agents/${encodeURIComponent(agentId)}/events?after=${state.lastEventSeq}&limit=100`);
    if (state.selectedAgentId !== agentId) return;
    const events = Array.isArray(result.events) ? result.events : [];
    const empty = $("#agentEvents .empty-state");
    if (empty && events.length) empty.remove();
    appendAgentEvents(events);
    $("#eventConnection").textContent = events.length ? `最新事件 #${formatNumber(state.lastEventSeq)}` : `已同步 · ${formatTime(new Date())}`;
  } catch (error) {
    $("#eventConnection").textContent = error instanceof Error ? error.message : "事件讀取失敗";
  } finally {
    state.eventPollBusy = false;
  }
}

async function selectAgent(agentId) {
  state.selectedAgentId = String(agentId);
  state.eventAgentId = null;
  resetAgentEvents();
  renderAgentList(state.agents);
  await Promise.all([loadAgentDetail(state.selectedAgentId), loadAgentEvents()]);
  scheduleEventPoll(1500);
}

function optionalInteger(selector, minimum, maximum) {
  const value = $(selector).value.trim();
  if (!value) return null;
  const number = Number(value);
  if (!Number.isInteger(number) || number < minimum || number > maximum) throw new Error(`數值必須介於 ${minimum} 到 ${maximum}`);
  return number;
}

$("#agentCreateBtn").addEventListener("click", () => withButton($("#agentCreateBtn"), "建立中…", async () => {
  const goal = $("#agentGoal").value.trim();
  if (!goal) throw new Error("請先輸入代理工作目標");
  const maxSteps = optionalInteger("#agentMaxSteps", 1, 128);
  const maxToolCalls = optionalInteger("#agentMaxTools", 0, 128);
  const maxInputChars = optionalInteger("#agentMaxInput", 256, 1000000);
  const budgets = {};
  if (maxSteps !== null) budgets.max_steps = maxSteps;
  if (maxToolCalls !== null) budgets.max_tool_calls = maxToolCalls;
  if (maxInputChars !== null) budgets.max_input_chars = maxInputChars;
  const capabilities = $$('input[name="agentCapability"]:checked:not(:disabled)').map((input) => input.value);
  const payload = { goal, profile: $("#agentProfile").value };
  if (capabilities.length) payload.capabilities = capabilities;
  if (Object.keys(budgets).length) payload.budgets = budgets;
  const result = await api("/api/agents", { method: "POST", body: JSON.stringify(payload) });
  if (!result.agent || !result.agent.id) throw new Error("伺服器沒有回傳代理工作");
  $("#agentGoal").value = "";
  toast("代理工作已建立");
  await loadAgents({ silent: true });
  await selectAgent(String(result.agent.id));
}));

$("#refreshAgents").addEventListener("click", () => withButton($("#refreshAgents"), "更新中…", async () => {
  await loadAgents();
  if (state.selectedAgentId) await loadAgentEvents();
}));

$("#cancelAgent").addEventListener("click", () => withButton($("#cancelAgent"), "取消中…", async () => {
  if (!state.selectedAgentId) return;
  const result = await api(`/api/agents/${encodeURIComponent(state.selectedAgentId)}/cancel`, { method: "POST", body: "{}" });
  if (result.agent) renderAgentDetail(result.agent);
  toast("已送出取消要求");
  await Promise.all([loadAgents({ silent: true }), loadAgentEvents()]);
}));

$("#approveAgent").addEventListener("click", () => withButton($("#approveAgent"), "核准中…", async () => {
  const agent = state.selectedAgent;
  const pending = agent && agent.pending_approval;
  if (!state.selectedAgentId || !pending || !pending.id) throw new Error("這項核准要求已不存在");
  const result = await api(`/api/agents/${encodeURIComponent(state.selectedAgentId)}/approve`, {
    method: "POST",
    body: JSON.stringify({ approval_id: pending.id }),
  });
  if (result.agent) renderAgentDetail(result.agent);
  toast("已核准這一次工具呼叫");
  await Promise.all([loadAgents({ silent: true }), loadAgentEvents()]);
}));

function scheduleAgentPoll(delay = 2500) {
  clearTimeout(state.agentPollTimer);
  if (state.activePanel !== "agents" || document.hidden) return;
  state.agentPollTimer = setTimeout(async () => {
    if (state.agentPollBusy) return scheduleAgentPoll(1000);
    state.agentPollBusy = true;
    await loadAgents({ silent: true });
    state.agentPollBusy = false;
    scheduleAgentPoll(2500);
  }, delay);
}

function scheduleEventPoll(delay = 1500) {
  clearTimeout(state.eventPollTimer);
  if (state.activePanel !== "agents" || !state.selectedAgentId || document.hidden) return;
  state.eventPollTimer = setTimeout(async () => {
    await loadAgentEvents();
    scheduleEventPoll(1500);
  }, delay);
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && state.activePanel === "agents") {
    scheduleAgentPoll(0);
    scheduleEventPoll(0);
  }
});

window.addEventListener("beforeunload", () => {
  Object.values(state.assetObjectUrls).forEach((url) => {
    if (url) URL.revokeObjectURL(url);
  });
});

state.token = takeStartupToken();
if (!state.token) showAuth();
updateAgentProfileForm();
loadStatus();
