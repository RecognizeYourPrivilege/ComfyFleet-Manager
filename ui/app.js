/* ComfyFleet control UI. Calls the same-origin HTTP API only.
   Docker lifecycle stays in comfyfleet.control. */

const state = {
  busy: false,
  tab: "fleet",
  gpus: [],
  gpuError: "",
  selected: new Set(),
  drafts: new Map(),
  openEditors: new Set(),
  listSignature: "",
  timer: 0,
  galleryTimer: 0,
  gallery: {
    instance: "",
    items: [],
    instances: [],
    total: 0,
    signature: "",
    seq: 0,
    openKey: "",
  },
};

const banner = document.querySelector("#banner");
const toast = document.querySelector("#toast");
const list = document.querySelector("#list");
const empty = document.querySelector("#empty");
const updated = document.querySelector("#updated");
const sheet = document.querySelector("#sheet");
const sheetBanner = document.querySelector("#sheet-banner");
const gpuRow = document.querySelector("#gpus");
const gpuNote = document.querySelector("#gpu-note");
const fileInput = document.querySelector("#workflow-file");
const fileName = document.querySelector("#file-name");
const pathInput = document.querySelector("#workflow-path");
const instanceNameInput = document.querySelector("#instance-name");
const forceInput = document.querySelector("#force");
const instanceImageInput = document.querySelector("#instance-image");
const reserveInput = document.querySelector("#reserve-vram");
const headroomInput = document.querySelector("#vram-headroom");
const previewMethodInput = document.querySelector("#preview-method");
const previewSizeInput = document.querySelector("#preview-size");
const extraArgsInput = document.querySelector("#extra-args");
const gitUrlsInput = document.querySelector("#custom-node-git-urls");
const zipInput = document.querySelector("#custom-nodes-zip");
const zipName = document.querySelector("#zip-name");
const zipPacks = document.querySelector("#zip-packs");
const installMissingInput = document.querySelector("#install-missing-from-workflow");
const flagsDisclosure = document.querySelector("#comfy-flags");

const hostButton = document.querySelector("#host-menu-button");
const hostMenu = document.querySelector("#host-menu");

function setHostMenu(open) {
  hostMenu.hidden = !open;
  hostButton.setAttribute("aria-expanded", open ? "true" : "false");
}

hostButton.addEventListener("click", (event) => {
  event.stopPropagation();
  setHostMenu(hostMenu.hidden);
});
document.addEventListener("click", (event) => {
  if (!hostMenu.hidden && !event.target.closest(".host-menu")) setHostMenu(false);
});
document.querySelector("#fix-owner").addEventListener("click", () => {
  setHostMenu(false);
  fixOwnership();
});
document.querySelector("#prune-dangling").addEventListener("click", () => {
  setHostMenu(false);
  pruneDangling();
});

document.querySelector("#refresh").addEventListener("click", () => refresh());
document.querySelector("#logout").addEventListener("click", () => logout());
document.querySelector("#open-create").addEventListener("click", openSheet);
document.querySelector("#create-stopped").addEventListener("click", () => submitCreate(false));
document.querySelector("#create-start").addEventListener("click", () => submitCreate(true));
fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  fileName.textContent = file ? file.name : "No file chosen";
});
zipInput.addEventListener("change", () => {
  renderZipPacks();
});
sheet.addEventListener("click", (event) => {
  if (event.target.closest("[data-close]")) closeSheet();
});
document.addEventListener("keydown", (event) => {
  const confirmSheet = document.querySelector("#confirm");
  const lightbox = document.querySelector("#lightbox");
  if (event.key === "Escape") {
    if (!hostMenu.hidden) setHostMenu(false);
    if (typeof closeImportSurfaces === "function" && closeImportSurfaces()) return;
    if (confirmSheet && !confirmSheet.hidden) return;
    if (lightbox && !lightbox.hidden) {
      closeLightbox();
      return;
    }
    if (!sheet.hidden) closeSheet();
    return;
  }
  if (!lightbox || lightbox.hidden || (confirmSheet && !confirmSheet.hidden)) return;
  if (event.target && event.target.closest && event.target.closest("input, textarea, select")) return;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    moveLightbox(-1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    moveLightbox(1);
  }
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) return;
  refresh();
  if (state.tab === "gallery") refreshGallery();
});

refresh();
resumeImportOverlay();
state.timer = window.setInterval(() => {
  if (!state.busy && sheet.hidden && !document.hidden) refresh();
}, 10000);
state.galleryTimer = window.setInterval(() => {
  if (state.tab === "gallery" && !state.busy && !document.hidden) refreshGallery();
}, 5000);
bindLightbox();

async function refresh() {
  if (state.busy) return;
  const health = await call("/api/health");
  if (health.unreachable) {
    showBanner("Control backend unreachable. This page only talks to the ComfyFleet API on this host.");
    updated.textContent = "Not connected";
    return;
  }
  const gpus = await call("/api/gpus");
  if (gpus.sessionExpired) return;
  const instances = await call("/api/instances");
  if (instances.sessionExpired) return;
  if (isAuthFailure(gpus) || isAuthFailure(instances)) return;
  if (gpus.ok && Array.isArray(gpus.payload.gpus)) {
    state.gpus = gpus.payload.gpus;
    state.gpuError = "";
  } else {
    state.gpus = [];
    state.selected.clear();
    state.gpuError = gpus.error || "GPU probe failed.";
  }
  const messages = [];
  if (!gpus.ok) messages.push(`GPU probe failed. ${gpus.error}`);
  if (!instances.ok) messages.push(instances.error);
  if (messages.length) showBanner(messages.join(" "));
  else hide(banner);
  renderGpus();
  renderList(instances.ok ? instances.payload.instances : []);
  const clock = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  updated.textContent = `Updated ${clock}`;
}

function renderList(instances) {
  const rows = instances || [];
  const signature = JSON.stringify(rows);
  // Polling hits this every 10s. Rebuilding the cards remounts the flags
  // panel (so it collapses) and resets window scroll. Skip that when the
  // fleet snapshot is unchanged, and restore scroll if a real change lands.
  if (signature === state.listSignature && list.childElementCount === rows.length) return;
  const scrollX = window.scrollX;
  const scrollY = window.scrollY;
  const editorScroll = new Map();
  for (const editor of list.querySelectorAll(".flag-editor-body")) {
    const panel = editor.closest(".flag-editor");
    if (panel && panel.id) editorScroll.set(panel.id, editor.scrollTop);
  }
  const openDetails = new Set();
  for (const node of list.querySelectorAll("article")) {
    const heading = node.querySelector("h2");
    const details = node.querySelector("details");
    if (heading && details && details.open) openDetails.add(heading.textContent);
  }
  state.listSignature = signature;
  list.replaceChildren();
  empty.hidden = rows.length !== 0;
  for (const instance of rows) {
    list.append(instanceCard(instance));
  }
  for (const node of list.querySelectorAll("article")) {
    const heading = node.querySelector("h2");
    const details = node.querySelector("details");
    if (heading && details && openDetails.has(heading.textContent)) details.open = true;
  }
  window.scrollTo(scrollX, scrollY);
  for (const [id, top] of editorScroll) {
    const panel = document.getElementById(id);
    const body = panel && panel.querySelector(".flag-editor-body");
    if (body) body.scrollTop = top;
  }
}

function isRunning(instance) {
  return instance.status === "running";
}

function selectedCudaTag() {
  const picked = document.querySelector('input[name="cuda-tag"]:checked');
  return picked ? picked.value : "cu130";
}

function openTarget(instance) {
  if (!isRunning(instance)) return null;
  const port = Number(instance.port);
  if (!Number.isInteger(port) || port < 1 || port > 65535) return null;
  const hostname = window.location.hostname;
  if (!hostname) return null;
  const host = hostname.indexOf(":") === -1 ? hostname : "[" + hostname + "]";
  const protocol = window.location.protocol === "https:" ? "https:" : "http:";
  return protocol + "//" + host + ":" + String(port) + "/";
}

function instanceCard(instance) {
  const running = isRunning(instance);
  const url = openTarget(instance);
  const card = el("article", { className: "card glass" });
  const top = el("div", { className: "card-top" });
  top.append(el("h2", { text: instance.name }));
  const trailing = el("div", { className: "card-trailing" });
  const pill = el("span", {
    className: `pill ${running ? "running" : "stopped"}`,
    text: running ? "Running" : "Stopped",
  });
  trailing.append(pill);
  top.append(trailing);
  card.append(top);
  const gpuText = (instance.gpus || []).join(", ") || "none";
  const cudaText = instance.cuda_tag === "cu130" || instance.cuda_tag === "cu124"
    ? instance.cuda_tag
    : (instance.image || "unknown");
  card.append(el("p", {
    className: "meta",
    text: `Port ${instance.port} · GPU ${gpuText} · CUDA ${cudaText} · ${instance.status}`,
  }));
  const launchArgv = instance.launch && instance.launch.argv;
  if (Array.isArray(launchArgv) && launchArgv.length) {
    card.append(el("p", {
      className: "meta",
      text: `Comfy --listen 0.0.0.0 --port 8188 ${launchArgv.join(" ")}`,
    }));
  }
  if (url) {
    card.append(el("p", { className: "url-line", text: url }));
  }
  const actions = el("div", { className: "icon-actions" });
  actions.setAttribute("role", "group");
  actions.setAttribute("aria-label", `Actions for ${instance.name}`);
  const start = actionButton("Start", "Start", playIcon(), "green");
  start.disabled = Boolean(running);
  start.addEventListener("click", () => mutate(instance.name, "start", start, "Starting…"));
  const stop = actionButton("Stop", "Stop", stopIcon(), "gray");
  stop.disabled = !running;
  stop.addEventListener("click", () => mutate(instance.name, "stop", stop, "Stopping…"));
  const kill = actionButton("Force stop", "Force", forceStopIcon(), "orange");
  kill.disabled = !running;
  kill.addEventListener("click", () => mutate(instance.name, "force-stop", kill, "Killing…"));
  const open = actionButton("Open", "Shell", terminalIcon(), "blue");
  open.disabled = !running;
  open.addEventListener("click", () => openTerminal(instance.name));
  const comfy = actionButton("Open Comfy", "Comfy", comfyIcon(), "blue");
  comfy.disabled = !url;
  comfy.addEventListener("click", () => openInstance(url));
  const editor = instanceFlagEditor(instance);
  editor.id = `flags-${instance.name}`;
  const editing = state.openEditors.has(instance.name);
  editor.hidden = !editing;
  const edit = actionButton("Edit flags", "Flags", pencilIcon(), "yellow");
  edit.setAttribute("aria-expanded", editing ? "true" : "false");
  edit.setAttribute("aria-controls", editor.id);
  if (editing) edit.classList.add("on");
  edit.addEventListener("click", () => toggleFlagEditor(instance.name, edit, editor));
  const remove = actionButton(`Delete ${instance.name}`, "Delete", trashIcon(), "red");
  remove.addEventListener("click", () => confirmDelete(instance.name, remove));
  actions.append(start, stop, kill, open, comfy, edit, remove);
  card.append(actions);
  card.append(editor);
  const details = el("details");
  details.append(el("summary", { text: "Details" }));
  details.append(el("pre", { text: JSON.stringify(instance, null, 2) }));
  card.append(details);
  return card;
}

async function mutate(name, action, button, pending) {
  const icon = button.querySelector("svg");
  const caption = button.querySelector(".action-caption");
  const previous = icon ? button.getAttribute("aria-label") : button.textContent;
  const previousCaption = caption ? caption.textContent : "";
  state.busy = true;
  button.disabled = true;
  if (icon) button.setAttribute("aria-label", pending);
  else button.textContent = pending;
  if (caption) caption.textContent = pending;
  const result = await call(`/api/instances/${encodeURIComponent(name)}/${action}`, { method: "POST" });
  state.busy = false;
  if (icon) button.setAttribute("aria-label", previous);
  else button.textContent = previous;
  if (caption) caption.textContent = previousCaption;
  if (!result.ok) {
    showBanner(result.error);
    await refresh();
    return;
  }
  if (action === "delete") {
    showToast(`Deleted ${name}.`);
    hide(banner);
    await refresh();
    return;
  }
  const warning = result.payload.warning;
  const instance = result.payload.instance;
  showToast(warning ? `${actionLabel(action, instance)} ${warning}` : actionLabel(action, instance));
  hide(banner);
  await refresh();
}

function actionLabel(action, instance) {
  if (!instance) return action;
  if (action === "start") return `Started ${instance.name} on port ${instance.port}.`;
  if (action === "force-stop") return `Force-stopped ${instance.name}.`;
  return `Stopped ${instance.name}.`;
}

function openTerminal(name) {
  const page = `/terminal.html?name=${encodeURIComponent(name)}`;
  const opened = window.open(page, "_blank", "noopener");
  if (!opened) showToast(page);
}

async function confirmDelete(name, button) {
  const yes = await askConfirm(
    `Delete ${name}? This force-stops and removes only that instance container, and drops its fleet record. Other containers are not touched. Host files for this instance are kept.`
  );
  if (!yes) return;
  await mutate(name, "delete", button, "Deleting…");
}

function askConfirm(text, options) {
  const sheet = document.querySelector("#confirm");
  const message = document.querySelector("#confirm-text");
  const title = document.querySelector("#confirm-title");
  const yes = document.querySelector("#confirm-yes");
  const fields = document.querySelector("#confirm-fields");
  const userInput = document.querySelector("#confirm-user");
  const groupInput = document.querySelector("#confirm-group");
  const opts = options || {};
  title.textContent = opts.title || "Delete instance";
  yes.textContent = opts.yes || "Delete";
  message.textContent = text;
  const lightbox = document.querySelector("#lightbox");
  sheet.classList.toggle("over-lightbox", Boolean(lightbox && !lightbox.hidden));
  const showFields = Boolean(opts.fields);
  fields.hidden = !showFields;
  if (showFields) {
    userInput.value = "";
    groupInput.value = "";
  }
  sheet.hidden = false;
  return new Promise((resolve) => {
    function finish(value) {
      sheet.hidden = true;
      sheet.classList.remove("over-lightbox");
      fields.hidden = true;
      sheet.removeEventListener("click", onClick);
      document.removeEventListener("keydown", onKey);
      if (value && showFields) {
        resolve({ user: userInput.value.trim(), group: groupInput.value.trim() });
        return;
      }
      resolve(value);
    }
    function onClick(event) {
      if (event.target.closest("#confirm-yes")) finish(true);
      else if (event.target.closest("#confirm-no") || event.target.closest("[data-confirm-no]")) finish(false);
    }
    function onKey(event) {
      if (event.key === "Escape") finish(false);
    }
    sheet.addEventListener("click", onClick);
    document.addEventListener("keydown", onKey);
  });
}

function openInstance(url) {
  if (!url) return;
  const opened = window.open(url, "_blank", "noopener");
  if (!opened) copyUrl(url);
}

async function copyUrl(url) {
  if (!url) return;
  try {
    await navigator.clipboard.writeText(url);
    showToast(`Copied ${url}`);
  } catch {
    showToast(url);
  }
}

function openSheet() {
  hide(sheetBanner);
  renderGpus();
  sheet.hidden = false;
  document.body.style.overflow = "hidden";
}

function closeSheet() {
  sheet.hidden = true;
  document.body.style.overflow = "";
}

function renderGpus() {
  gpuRow.replaceChildren();
  if (state.gpuError) {
    gpuNote.textContent = state.gpuError;
    return;
  }
  if (!state.gpus.length) {
    gpuNote.textContent = "No GPUs reported yet.";
    return;
  }
  if (state.gpus.length === 1) state.selected.add(String(state.gpus[0].index));
  const known = new Set(state.gpus.map((gpu) => String(gpu.index)));
  for (const index of [...state.selected]) {
    if (!known.has(index)) state.selected.delete(index);
  }
  for (const gpu of state.gpus) {
    const index = String(gpu.index);
    const button = el("button", {
      className: "gpu",
      type: "button",
      role: "checkbox",
    });
    button.setAttribute("aria-checked", state.selected.has(index) ? "true" : "false");
    button.dataset.index = index;
    button.append(document.createTextNode(`GPU ${gpu.index}`));
    const memory = [gpu.name, gpu.memory].filter(Boolean).join(" · ");
    if (memory) button.append(el("small", { text: memory }));
    button.addEventListener("click", () => {
      if (state.selected.has(index)) state.selected.delete(index);
      else state.selected.add(index);
      button.setAttribute("aria-checked", state.selected.has(index) ? "true" : "false");
    });
    gpuRow.append(button);
  }
  gpuNote.textContent = state.gpus.length > 1
    ? "Choose one or more GPUs. Nothing is assumed on a multi-GPU host."
    : "Confirm the GPU for this instance.";
}

async function submitCreate(start) {
  if (state.busy) return;
  const file = fileInput.files && fileInput.files[0];
  const workflowPath = pathInput.value.trim();
  if (!file && !workflowPath) {
    showSheetError("A workflow JSON file is required. There is no built-in default.");
    return;
  }
  if (file && workflowPath) {
    showSheetError("Upload a workflow file or enter a host path, not both.");
    return;
  }
  if (file && !file.name.toLowerCase().endsWith(".json")) {
    showSheetError("Workflow must be a .json file.");
    return;
  }
  const chosen = [...state.selected];
  if (!chosen.length) {
    showSheetError(state.gpuError || "Select at least one GPU.");
    return;
  }
  const launch = readLaunch();
  const conflict = launchConflict(launch);
  if (conflict) {
    showSheetError(conflict);
    return;
  }
  const body = new FormData();
  if (file) body.append("workflow", file, file.name);
  if (workflowPath) body.append("workflow_path", workflowPath);
  body.append("name", instanceNameInput.value.trim());
  body.append("gpus", chosen.join(","));
  body.append("cuda_tag", selectedCudaTag());
  const imageOverride = instanceImageInput.value.trim();
  if (imageOverride) body.append("instance_image", imageOverride);
  body.append("start", start ? "true" : "false");
  body.append("force", forceInput.checked ? "true" : "false");
  body.append("vram", launch.vram);
  body.append("attention", launch.attention);
  body.append("flags", launch.flags.join(","));
  body.append("reserve_vram", reserveInput.value.trim());
  body.append("vram_headroom", headroomInput.value.trim());
  body.append("preview_method", previewMethodInput.value);
  body.append("preview_size", previewSizeInput.value.trim());
  body.append("extra_args", "");
  body.append("comfy_extra_args", extraArgsInput.value.trim());
  for (const url of gitUrlLines(gitUrlsInput.value)) {
    body.append("custom_node_git_urls", url);
  }
  const zips = [...(zipInput.files || [])];
  const zipNameInputs = [...document.querySelectorAll("#zip-packs input")];
  zips.forEach((zip, index) => {
    body.append("custom_nodes_zip", zip, zip.name);
    const typed = zipNameInputs[index] ? zipNameInputs[index].value.trim() : "";
    body.append("custom_nodes_zip_name", typed);
  });
  body.append("install_missing_from_workflow", installMissingInput.checked ? "true" : "false");
  state.busy = true;
  setCreatePending(true, start);
  const result = await call("/api/instances", { method: "POST", body });
  state.busy = false;
  setCreatePending(false, start);
  if (!result.ok) {
    showSheetError(result.error);
    return;
  }
  const instance = result.payload.instance;
  const mode = result.payload.started ? "Created and started" : "Created (not started)";
  const notices = responseNotices(result.payload);
  fileInput.value = "";
  fileName.textContent = "No file chosen";
  pathInput.value = "";
  instanceNameInput.value = "";
  forceInput.checked = false;
  instanceImageInput.value = "";
  gitUrlsInput.value = "";
  zipInput.value = "";
  renderZipPacks();
  installMissingInput.checked = true;
  resetLaunch();
  if (flagsDisclosure) flagsDisclosure.open = false;
  closeSheet();
  const noticeText = notices.length ? ` ${notices.join(" ")}` : "";
  showToast(`${mode}: ${instance.name} · port ${instance.port}.${noticeText}`);
  if (notices.length) showBanner(notices.join(" "));
  else hide(banner);
  await refresh();
  if (notices.length) showBanner(notices.join(" "));
}

function renderZipPacks() {
  const files = [...(zipInput.files || [])];
  zipPacks.replaceChildren();
  if (!files.length) {
    zipName.textContent = "No zip chosen";
    return;
  }
  zipName.textContent = files.length === 1 ? files[0].name : `${files.length} zips chosen`;
  files.forEach((file) => {
    const row = document.createElement("label");
    row.className = "zip-pack";
    const title = document.createElement("span");
    title.className = "zip-pack-file";
    title.textContent = file.name;
    const input = document.createElement("input");
    input.className = "text-input";
    input.type = "text";
    input.autocomplete = "off";
    input.spellcheck = false;
    input.placeholder = "Name from the zip";
    input.setAttribute("aria-label", `Folder name for ${file.name}`);
    row.append(title, input);
    zipPacks.append(row);
  });
}

function gitUrlLines(value) {
  return String(value || "")
    .split(/[\r\n,]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function responseNotices(payload) {
  const items = [];
  if (payload && payload.warning) items.push(String(payload.warning));
  if (payload && Array.isArray(payload.warnings)) {
    for (const item of payload.warnings) {
      if (item) items.push(String(item));
    }
  }
  return items;
}

function toggleFlagEditor(name, button, editor) {
  const open = editor.hidden;
  editor.hidden = !open;
  button.setAttribute("aria-expanded", open ? "true" : "false");
  button.classList.toggle("on", open);
  if (open) state.openEditors.add(name);
  else state.openEditors.delete(name);
}

function actionButton(label, caption, icon, tone) {
  const button = el("button", {
    className: `action-icon tone-${tone}`,
    type: "button",
  });
  const glyph = el("span", { className: "action-glyph" });
  glyph.append(icon);
  button.setAttribute("aria-label", label);
  button.append(glyph, el("span", { className: "action-caption", text: caption }));
  return button;
}

function strokeIcon(paths, filled) {
  const svgNs = "http:" + "//www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNs, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  for (const d of paths) {
    const path = document.createElementNS(svgNs, "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", filled ? "currentColor" : "none");
    path.setAttribute("stroke", filled ? "none" : "currentColor");
    path.setAttribute("stroke-width", filled ? "0" : "1.65");
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");
    svg.append(path);
  }
  return svg;
}

function playIcon() {
  return strokeIcon(["M8.2 5.6c-.7 0-1.2.5-1.2 1.2v10.4c0 .9 1 1.5 1.8 1l8.6-5.2c.7-.4.7-1.5 0-1.9L8.8 5.9c-.2-.1-.4-.3-.6-.3z"], true);
}

function stopIcon() {
  return strokeIcon(["M8.2 6.8h7.6a1.6 1.6 0 0 1 1.6 1.6v7.2a1.6 1.6 0 0 1-1.6 1.6H8.2a1.6 1.6 0 0 1-1.6-1.6V8.4a1.6 1.6 0 0 1 1.6-1.6z"], true);
}

function forceStopIcon() {
  return strokeIcon([
    "M12 4.2a7.8 7.8 0 1 0 0 15.6 7.8 7.8 0 0 0 0-15.6z",
    "M9 9l6 6",
    "M15 9l-6 6",
  ]);
}

function terminalIcon() {
  return strokeIcon([
    "M6.2 7.2h11.6a1.6 1.6 0 0 1 1.6 1.6v6.4a1.6 1.6 0 0 1-1.6 1.6H6.2a1.6 1.6 0 0 1-1.6-1.6V8.8a1.6 1.6 0 0 1 1.6-1.6z",
    "M8 11.1l2.2 1.6L8 14.3",
    "M11.6 14.3h3.4",
  ]);
}

function comfyIcon() {
  return strokeIcon([
    "M10 7H8.2A1.7 1.7 0 0 0 6.5 8.7v7.1A1.7 1.7 0 0 0 8.2 17.5h7.1a1.7 1.7 0 0 0 1.7-1.7V14",
    "M13.2 6.2H18v4.8",
    "M17.6 6.6l-7.2 7.2",
  ]);
}

function pencilIcon() {
  return strokeIcon([
    "M14.2 5.1a1.7 1.7 0 0 1 2.4 0l2.3 2.3a1.7 1.7 0 0 1 0 2.4L9.2 19.5 4.6 20.4l.9-4.6 8.7-10.7z",
    "M13 7.4l3.6 3.6",
  ]);
}

function trashIcon() {
  return strokeIcon([
    "M5 7.5h14",
    "M9.2 7.5V6a1.4 1.4 0 0 1 1.4-1.4h2.8A1.4 1.4 0 0 1 14.8 6v1.5",
    "M7.6 7.5l.7 11.1a1.4 1.4 0 0 0 1.4 1.3h4.6a1.4 1.4 0 0 0 1.4-1.3l.7-11.1",
  ]);
}

function readLaunch() {
  const vram = document.querySelector('input[name="vram"]:checked');
  const attention = document.querySelector('input[name="attention"]:checked');
  const flags = [...document.querySelectorAll('input[name="flag"]:checked')];
  return {
    vram: vram ? vram.value : "",
    attention: attention ? attention.value : "",
    flags: flags.map((node) => node.value),
    nodes: flags,
  };
}

function launchConflict(launch) {
  const buckets = new Map();
  function add(group, flag) {
    if (!group || !flag) return;
    const list = buckets.get(group) || [];
    list.push(flag);
    buckets.set(group, list);
  }
  add("vram", launch.vram);
  for (const node of launch.nodes) add(node.dataset.exclusive || "", node.value);
  for (const flags of buckets.values()) {
    if (flags.length > 1) {
      return `${flags.join(", ")} cannot be combined. ComfyUI accepts only one of that group.`;
    }
  }
  return "";
}

function resetLaunch() {
  const vram = document.querySelector('input[name="vram"][value=""]');
  const attention = document.querySelector('input[name="attention"][value=""]');
  if (vram) vram.checked = true;
  if (attention) attention.checked = true;
  for (const node of document.querySelectorAll('input[name="flag"]')) node.checked = false;
  reserveInput.value = "";
  headroomInput.value = "";
  previewMethodInput.value = "";
  previewSizeInput.value = "";
  extraArgsInput.value = "";
  renderFlagChips();
}

const FLAG_SECTIONS = [
  ["precision", "Precision"],
  ["caching", "Caching"],
  ["preview", "Preview"],
  ["vram", "VRAM"],
  ["misc", "Misc"],
];

function instanceFlagEditor(instance) {
  const draft = draftFor(instance);
  const box = el("div", { className: "flag-editor" });
  const body = el("div", { className: "flag-editor-body" });
  body.append(el("p", { className: "flag-sub", text: "ComfyUI flags" }));
  body.append(el("p", {
    className: "hint",
    text: "Click a flag to add it. × removes it. Apply stops this instance if it is running and recreates the same name, port, mounts, and workflow. Only the Comfy arguments change. The CUDA line stays. Changing cu130 versus cu124 requires a recreate from the create sheet.",
  }));
  const applied = el("div", { className: "chip-row" });
  const catalog = el("div");
  body.append(el("p", { className: "flag-sub", text: "VRAM" }));
  body.append(draftRadios(draft, "vram", `vram-${instance.name}`, radioValues("vram"), applied, catalog));
  body.append(draftNumber(draft, "reserve", "Reserve VRAM (GB)", "--reserve-vram"));
  body.append(draftNumber(draft, "headroom", "VRAM headroom (GB)", "--vram-headroom"));
  body.append(el("p", { className: "flag-sub", text: "Attention" }));
  body.append(draftRadios(draft, "attention", `attention-${instance.name}`, radioValues("attention"), applied, catalog));
  paintInstanceFlags(draft, applied, catalog);
  body.append(applied, catalog);
  const apply = el("button", { className: "btn secondary flag-apply", type: "button", text: "Apply" });
  apply.addEventListener("click", () => applyLaunch(instance.name, apply));
  box.append(body, apply);
  return box;
}

function draftNumber(draft, field, label, flag) {
  const block = el("div", { className: "flag-number" });
  block.append(el("p", { className: "field-label", text: label }));
  const input = document.createElement("input");
  input.className = "text-input";
  input.type = "number";
  input.min = "0";
  input.step = "any";
  input.inputMode = "decimal";
  input.placeholder = "empty = omit";
  input.setAttribute("aria-label", `${label} ${flag}`);
  input.value = draft[field] || "";
  input.addEventListener("input", () => {
    draft[field] = input.value.trim();
    draft.dirty = true;
  });
  block.append(input);
  return block;
}

function radioValues(name) {
  return [...document.querySelectorAll(`input[name="${name}"]`)].map((node) => node.value);
}

function draftRadios(draft, field, groupName, values, applied, catalog) {
  const group = el("div", { className: "choice-col" });
  for (const value of values) {
    const label = el("label", { className: "choice" });
    const input = document.createElement("input");
    input.type = "radio";
    input.name = groupName;
    input.value = value;
    input.checked = (draft[field] || "") === value;
    input.addEventListener("change", () => {
      if (!input.checked) return;
      draft[field] = value;
      if (field === "vram" && value) {
        draft.flags.delete("--cpu");
        draft.flags.delete("--gpu-only");
      }
      draft.dirty = true;
      paintInstanceFlags(draft, applied, catalog);
    });
    label.append(input, el("span", { text: value || "Default" }));
    group.append(label);
  }
  return group;
}

function draftFor(instance) {
  const existing = state.drafts.get(instance.name);
  if (existing && existing.dirty) return existing;
  const launch = instance.launch || {};
  const fresh = {
    vram: launch.vram || "",
    attention: launch.attention || "",
    flags: new Set(Array.isArray(launch.flags) ? launch.flags : []),
    preview: launch.preview_method || "",
    reserve: launch.reserve_vram == null ? "" : String(launch.reserve_vram),
    headroom: launch.vram_headroom == null ? "" : String(launch.vram_headroom),
    previewSize: launch.preview_size == null ? "" : String(launch.preview_size),
    extra: launch.extra_args || "",
    dirty: false,
  };
  state.drafts.set(instance.name, fresh);
  return fresh;
}

function paintInstanceFlags(draft, applied, catalog) {
  applied.replaceChildren();
  if (draft.vram) {
    applied.append(appliedChip(draft.vram, () => {
      draft.vram = "";
      draft.dirty = true;
      paintInstanceFlags(draft, applied, catalog);
    }));
  }
  if (draft.attention) {
    applied.append(appliedChip(draft.attention, () => {
      draft.attention = "";
      draft.dirty = true;
      paintInstanceFlags(draft, applied, catalog);
    }));
  }
  for (const flag of draft.flags) {
    applied.append(appliedChip(flag, () => {
      draft.flags.delete(flag);
      draft.dirty = true;
      paintInstanceFlags(draft, applied, catalog);
    }));
  }
  if (draft.preview) {
    applied.append(appliedChip(`--preview-method ${draft.preview}`, () => {
      draft.preview = "";
      draft.dirty = true;
      paintInstanceFlags(draft, applied, catalog);
    }));
  }
  catalog.replaceChildren();
  for (const [section, label] of FLAG_SECTIONS) {
    const row = el("div", { className: "chip-row" });
    if (section === "preview") {
      for (const value of ["auto", "latent2rgb", "taesd", "none"]) {
        row.append(catalogChip(`--preview-method ${value}`, draft.preview === value, () => {
          draft.preview = value;
          draft.dirty = true;
          paintInstanceFlags(draft, applied, catalog);
        }));
      }
    }
    for (const node of document.querySelectorAll(`input[name="flag"][data-section="${section}"]`)) {
      const flag = node.value;
      row.append(catalogChip(flag, draft.flags.has(flag) || draft.vram === flag || draft.attention === flag, () => {
        addDraftFlag(draft, node);
        paintInstanceFlags(draft, applied, catalog);
      }));
    }
    if (!row.childNodes.length) continue;
    catalog.append(el("p", { className: "flag-sub", text: label }));
    catalog.append(row);
  }
}

function addDraftFlag(draft, node) {
  const flag = node.value;
  const group = node.dataset.exclusive || "";
  if (group) {
    for (const other of document.querySelectorAll(`input[name="flag"][data-exclusive="${group}"]`)) {
      draft.flags.delete(other.value);
    }
    if (group === "vram") draft.vram = "";
  }
  if (flag === "--cpu" || flag === "--gpu-only") draft.vram = "";
  draft.flags.add(flag);
  draft.dirty = true;
}

async function applyLaunch(name, button) {
  const draft = state.drafts.get(name);
  if (!draft || state.busy) return;
  const previous = button.textContent;
  state.busy = true;
  button.disabled = true;
  button.textContent = "Applying…";
  const result = await call(`/api/instances/${encodeURIComponent(name)}/launch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      vram: draft.vram,
      attention: draft.attention,
      flags: [...draft.flags],
      reserve_vram: draft.reserve,
      vram_headroom: draft.headroom,
      preview_method: draft.preview,
      preview_size: draft.previewSize,
      extra_args: draft.extra,
    }),
  });
  state.busy = false;
  button.textContent = previous;
  if (!result.ok) {
    showBanner(result.error);
    await refresh();
    return;
  }
  draft.dirty = false;
  const instance = result.payload.instance;
  showToast(`Updated ${instance.name} on port ${instance.port}. Same name, mounts, and workflow.`);
  hide(banner);
  await refresh();
}

function renderFlagChips() {
  const applied = document.querySelector("#applied-flags");
  const catalog = document.querySelector("#flag-catalog");
  if (!applied || !catalog) return;
  applied.replaceChildren();
  for (const node of document.querySelectorAll('input[name="flag"]:checked')) {
    applied.append(appliedChip(node.value, () => {
      node.checked = false;
      renderFlagChips();
    }));
  }
  if (previewMethodInput.value) {
    applied.append(appliedChip(`--preview-method ${previewMethodInput.value}`, () => {
      previewMethodInput.value = "";
      renderFlagChips();
    }));
  }
  catalog.replaceChildren();
  for (const [section, label] of FLAG_SECTIONS) {
    const row = el("div", { className: "chip-row" });
    if (section === "preview") {
      for (const value of ["auto", "latent2rgb", "taesd", "none"]) {
        row.append(catalogChip(`--preview-method ${value}`, previewMethodInput.value === value, () => {
          previewMethodInput.value = value;
          renderFlagChips();
        }));
      }
    }
    for (const node of document.querySelectorAll(`input[name="flag"][data-section="${section}"]`)) {
      row.append(catalogChip(node.value, node.checked, () => addFlag(node)));
    }
    if (!row.childNodes.length) continue;
    catalog.append(el("p", { className: "flag-sub", text: label }));
    catalog.append(row);
  }
}

function addFlag(node) {
  const group = node.dataset.exclusive || "";
  if (group) {
    for (const other of document.querySelectorAll(`input[name="flag"][data-exclusive="${group}"]`)) {
      if (other !== node) other.checked = false;
    }
  }
  node.checked = true;
  renderFlagChips();
}

function appliedChip(label, onRemove) {
  const chip = el("span", { className: "chip on" });
  chip.append(document.createTextNode(label));
  const remove = el("button", { className: "chip-x", type: "button", text: "×" });
  remove.setAttribute("aria-label", `Remove ${label}`);
  remove.addEventListener("click", onRemove);
  chip.append(remove);
  return chip;
}

function catalogChip(label, selected, onAdd) {
  const button = el("button", { className: selected ? "chip on" : "chip", type: "button", text: label });
  button.addEventListener("click", onAdd);
  return button;
}

renderFlagChips();

function setCreatePending(pending, start) {
  const stopped = document.querySelector("#create-stopped");
  const started = document.querySelector("#create-start");
  stopped.disabled = pending;
  started.disabled = pending;
  if (!pending) {
    stopped.textContent = "Create";
    started.textContent = "Create & start";
    return;
  }
  if (start) started.textContent = "Creating…";
  else stopped.textContent = "Creating…";
}

function isAuthFailure(result) {
  if (!result || result.sessionExpired || result.status === 401) return true;
  const error = (result.error || "").toLowerCase();
  return error === "unauthorized" || error === "session expired";
}

async function fixOwnership() {
  const answer = await askConfirm(
    "Change ownership of /home/ComfyFleet/wildcards, /home/ComfyFleet/models, every /home/ComfyFleet/custom_nodes_* directory, and /home/ComfyFleet/files? Only those directories are walked. Symlinks into the image baked custom nodes are not followed.",
    { title: "Fix ownership", yes: "Fix ownership", fields: true }
  );
  if (!answer) return;
  state.busy = true;
  const result = await call("/api/host/fix-owner", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user: answer.user || "", group: answer.group || "" }),
  });
  state.busy = false;
  if (result.sessionExpired || isAuthFailure(result)) return;
  if (!result.ok) {
    showBanner(result.error || "Fix ownership failed.");
    return;
  }
  const paths = (result.payload && result.payload.paths) || [];
  const owner = (result.payload && result.payload.user) || "comfyuser";
  const group = (result.payload && result.payload.group) || "comfyuser";
  showToast(
    paths.length
      ? `Ownership updated on ${paths.length} paths for ${owner}:${group}.`
      : "No allowlisted directories were present."
  );
  hide(banner);
}

async function pruneDangling() {
  const yes = await askConfirm(
    "Remove stopped containers that are not ComfyFleet instances? Containers labeled comfyfleet.managed=true are kept, including stopped instances.",
    { title: "Prune dangling containers", yes: "Prune" }
  );
  if (!yes) return;
  state.busy = true;
  const result = await call("/api/host/prune-dangling", { method: "POST" });
  state.busy = false;
  if (result.sessionExpired || isAuthFailure(result)) return;
  if (!result.ok) {
    showBanner(result.error || "Prune failed.");
    return;
  }
  const removed = (result.payload && result.payload.removed) || [];
  showToast(
    removed.length
      ? `Removed ${removed.length} dangling container${removed.length === 1 ? "" : "s"}. Fleet instances were kept.`
      : "No dangling containers to remove. Fleet instances were kept."
  );
  hide(banner);
}

async function logout() {
  state.busy = true;
  try {
    await fetch("/api/logout", {
      method: "POST",
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    });
  } catch {
    /* still leave the fleet page */
  }
  window.location.assign("/login");
}

async function call(path, options) {
  try {
    const response = await fetch(path, {
      method: (options && options.method) || "GET",
      body: options && options.body,
      credentials: "same-origin",
      headers: Object.assign({ Accept: "application/json" }, (options && options.headers) || {}),
    });
    let payload = null;
    try { payload = await response.json(); } catch { payload = null; }
    if (response.status === 401) {
      const expired = payload && payload.error === "session expired";
      window.location.assign(expired ? "/login?expired=1" : "/login");
      return {
        ok: false,
        status: 401,
        sessionExpired: true,
        error: expired ? "session expired" : "unauthorized",
      };
    }
    if (!response.ok || !payload || payload.ok === false) {
      return {
        ok: false,
        status: response.status,
        error: (payload && payload.error) || `Request failed (${response.status}).`,
      };
    }
    return { ok: true, status: response.status, payload };
  } catch {
    return { ok: false, unreachable: true, error: "Control backend unreachable." };
  }
}

function showBanner(text) {
  banner.hidden = false;
  banner.textContent = text;
}

function showSheetError(text) {
  sheetBanner.hidden = false;
  sheetBanner.textContent = text;
}

function showToast(text) {
  toast.hidden = false;
  toast.textContent = text;
}

function hide(node) {
  node.hidden = true;
  node.textContent = "";
}

function el(tag, attrs) {
  const node = document.createElement(tag);
  if (!attrs) return node;
  if (attrs.className) node.className = attrs.className;
  if (attrs.type) node.type = attrs.type;
  if (attrs.role) node.setAttribute("role", attrs.role);
  if (attrs.text) node.textContent = attrs.text;
  return node;
}

const GALLERY_PAGE = 48;

function bindLightbox() {
  const stage = document.querySelector("#lb-stage");
  let startX = 0;
  stage.addEventListener("touchstart", (event) => {
    if (!event.changedTouches || !event.changedTouches.length) return;
    startX = event.changedTouches[0].clientX;
  }, { passive: true });
  stage.addEventListener("touchend", (event) => {
    if (document.querySelector("#lightbox").hidden) return;
    if (!event.changedTouches || !event.changedTouches.length) return;
    const dx = event.changedTouches[0].clientX - startX;
    if (Math.abs(dx) < 48) return;
    moveLightbox(dx < 0 ? 1 : -1);
  }, { passive: true });
  document.querySelector("#lb-close").addEventListener("click", closeLightbox);
  document.querySelector("#lb-scrim").addEventListener("click", closeLightbox);
  document.querySelector("#lb-prev").addEventListener("click", () => moveLightbox(-1));
  document.querySelector("#lb-next").addEventListener("click", () => moveLightbox(1));
  document.querySelector("#gallery-more").addEventListener("click", () => loadMoreGallery());
  document.querySelector("#tab-fleet").addEventListener("click", () => setTab("fleet"));
  document.querySelector("#tab-gallery").addEventListener("click", () => setTab("gallery"));
}

function setTab(tab) {
  state.tab = tab === "gallery" ? "gallery" : "fleet";
  document.body.dataset.tab = state.tab;
  const gallery = state.tab === "gallery";
  document.querySelector("#tab-fleet").setAttribute("aria-selected", gallery ? "false" : "true");
  document.querySelector("#tab-gallery").setAttribute("aria-selected", gallery ? "true" : "false");
  document.querySelector("#gallery").hidden = !gallery;
  document.querySelector("#list").hidden = gallery;
  document.querySelector(".dock").hidden = gallery;
  if (gallery) refreshGallery();
}

function galleryItemKey(item) {
  return item.instance + "\n" + item.path;
}

function galleryListUrl(offset, limit) {
  const params = new URLSearchParams();
  if (state.gallery.instance) params.set("instance", state.gallery.instance);
  params.set("offset", String(offset));
  params.set("limit", String(limit));
  return "/api/gallery?" + params.toString();
}

function mediaUrl(item, download) {
  const params = new URLSearchParams();
  params.set("instance", item.instance);
  params.set("path", item.path);
  if (item.mtime_ms != null) params.set("v", String(item.mtime_ms));
  if (download) params.set("download", "1");
  return "/api/gallery/media?" + params.toString();
}

function thumbUrl(item) {
  const params = new URLSearchParams();
  params.set("instance", item.instance);
  params.set("path", item.path);
  if (item.mtime_ms != null) params.set("v", String(item.mtime_ms));
  return "/api/gallery/thumb?" + params.toString();
}

async function refreshGallery() {
  const seq = ++state.gallery.seq;
  const limit = Math.min(240, Math.max(GALLERY_PAGE, state.gallery.items.length || GALLERY_PAGE));
  const result = await call(galleryListUrl(0, limit));
  if (seq !== state.gallery.seq) return;
  if (result.sessionExpired || isAuthFailure(result)) return;
  if (!result.ok) {
    showBanner(result.error || "Gallery failed.");
    return;
  }
  hide(banner);
  applyGallery(result.payload, false);
}

async function loadMoreGallery() {
  if (state.gallery.items.length >= state.gallery.total) return;
  const seq = state.gallery.seq;
  const result = await call(galleryListUrl(state.gallery.items.length, GALLERY_PAGE));
  if (seq !== state.gallery.seq) return;
  if (result.sessionExpired || isAuthFailure(result)) return;
  if (!result.ok) {
    showBanner(result.error || "Gallery failed.");
    return;
  }
  applyGallery(result.payload, true);
}

function applyGallery(payload, append) {
  const incoming = Array.isArray(payload.items) ? payload.items : [];
  state.gallery.instances = Array.isArray(payload.instances) ? payload.instances : [];
  state.gallery.total = Number(payload.total) || 0;
  if (append) {
    const seen = new Set(state.gallery.items.map(galleryItemKey));
    for (const item of incoming) {
      const key = galleryItemKey(item);
      if (!seen.has(key)) {
        seen.add(key);
        state.gallery.items.push(item);
      }
    }
    state.gallery.signature = JSON.stringify(state.gallery.items);
  } else {
    const signature = JSON.stringify(incoming);
    if (signature === state.gallery.signature) {
      renderGalleryChrome();
      syncLightbox();
      return;
    }
    state.gallery.signature = signature;
    state.gallery.items = incoming.slice();
  }
  renderGalleryChrome();
  renderGalleryGrid();
  syncLightbox();
}

function renderGalleryChrome() {
  const filters = document.querySelector("#gallery-filters");
  const current = state.gallery.instance;
  filters.replaceChildren();
  filters.append(galleryFilterChip("", "All", current === ""));
  for (const instance of state.gallery.instances) {
    filters.append(galleryFilterChip(instance.name, instance.name, current === instance.name));
  }
  const noun = state.gallery.total === 1 ? "file" : "files";
  const status = document.querySelector("#gallery-status");
  status.textContent = state.gallery.instances.length
    ? state.gallery.total + " " + noun
    : "No instance output folders yet.";
  document.querySelector("#gallery-more").hidden = state.gallery.items.length >= state.gallery.total;
  const empty = document.querySelector("#gallery-empty");
  empty.hidden = state.gallery.total !== 0;
  const copy = empty.querySelector("p");
  if (!state.gallery.instances.length) {
    copy.textContent = "No instance output folders under /home/ComfyFleet/files yet.";
  } else if (current) {
    copy.textContent = "No images or videos in this output folder yet.";
  } else {
    copy.textContent = "No images or videos in the instance output folders yet.";
  }
}

function galleryFilterChip(value, label, on) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = on ? "chip on" : "chip";
  button.textContent = label;
  button.setAttribute("aria-pressed", on ? "true" : "false");
  button.addEventListener("click", () => {
    if (state.gallery.instance === value) return;
    state.gallery.instance = value;
    state.gallery.items = [];
    state.gallery.signature = "";
    refreshGallery();
  });
  return button;
}

function renderGalleryGrid() {
  const grid = document.querySelector("#gallery-grid");
  const scrollX = window.scrollX;
  const scrollY = window.scrollY;
  grid.replaceChildren();
  for (const item of state.gallery.items) grid.append(galleryTile(item));
  window.scrollTo(scrollX, scrollY);
}

function galleryTile(item) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = item.kind === "image" ? "gallery-tile" : "gallery-tile is-playable";
  button.setAttribute("aria-label", item.name + ", " + item.instance);
  const img = document.createElement("img");
  img.alt = "";
  img.loading = "lazy";
  img.decoding = "async";
  img.src = thumbUrl(item);
  const fallback = document.createElement("span");
  fallback.className = "gallery-fallback";
  fallback.textContent = item.name;
  fallback.hidden = true;
  img.addEventListener("error", () => {
    img.hidden = true;
    fallback.hidden = false;
  });
  button.append(img, fallback);
  if (item.kind !== "image") {
    const badge = document.createElement("span");
    badge.className = "play-badge";
    badge.setAttribute("aria-hidden", "true");
    badge.append(playIcon());
    button.append(badge);
  }
  button.addEventListener("click", () => openLightbox(item));
  return button;
}

function openLightbox(item) {
  state.gallery.openKey = galleryItemKey(item);
  document.querySelector("#lightbox").hidden = false;
  document.body.style.overflow = "hidden";
  paintLightbox(item);
  document.querySelector("#lb-close").focus();
}

function paintLightbox(item) {
  document.querySelector("#lb-name").textContent = item.name;
  const place = item.path.indexOf("/") === -1 ? item.instance : item.instance + " / " + item.path;
  document.querySelector("#lb-meta").textContent = place + " · " + formatWhen(item.mtime_ms) + " · " + formatSize(item.size);
  const image = document.querySelector("#lb-image");
  const video = document.querySelector("#lb-video");
  video.pause();
  video.removeAttribute("src");
  video.load();
  image.removeAttribute("src");
  if (item.kind === "video") {
    image.hidden = true;
    video.hidden = false;
    video.src = mediaUrl(item, false);
  } else {
    video.hidden = true;
    image.hidden = false;
    image.alt = item.name;
    image.src = mediaUrl(item, false);
  }
  const open = document.querySelector("#lb-open");
  const url = openTarget(item);
  open.disabled = !url;
  open.title = url ? "Opens this instance in a new tab." : "Start the instance to open ComfyUI.";
  open.onclick = () => openInstance(url);
  document.querySelector("#lb-download").onclick = () => downloadGalleryItem(item);
  document.querySelector("#lb-delete").onclick = () => deleteGalleryItem(item);
  updateLightboxNav(item);
}

function updateLightboxNav(item) {
  const index = state.gallery.items.findIndex((row) => galleryItemKey(row) === galleryItemKey(item));
  document.querySelector("#lb-prev").disabled = index <= 0;
  const hasNext = index >= 0 && (index < state.gallery.items.length - 1 || state.gallery.items.length < state.gallery.total);
  document.querySelector("#lb-next").disabled = !hasNext;
}

function syncLightbox() {
  const box = document.querySelector("#lightbox");
  if (box.hidden || !state.gallery.openKey) return;
  const item = state.gallery.items.find((row) => galleryItemKey(row) === state.gallery.openKey);
  if (!item) {
    closeLightbox();
    showToast("That file is no longer in the gallery.");
    return;
  }
  updateLightboxNav(item);
}

function closeLightbox() {
  document.querySelector("#lightbox").hidden = true;
  const video = document.querySelector("#lb-video");
  video.pause();
  video.removeAttribute("src");
  video.load();
  document.querySelector("#lb-image").removeAttribute("src");
  state.gallery.openKey = "";
  if (document.querySelector("#confirm").hidden && sheet.hidden) document.body.style.overflow = "";
}

async function moveLightbox(delta) {
  const index = state.gallery.items.findIndex((row) => galleryItemKey(row) === state.gallery.openKey);
  if (index < 0) return;
  const next = index + delta;
  if (next < 0) return;
  if (next >= state.gallery.items.length) {
    if (delta > 0 && state.gallery.items.length < state.gallery.total) await loadMoreGallery();
    if (next >= state.gallery.items.length) return;
  }
  openLightbox(state.gallery.items[next]);
}

function downloadGalleryItem(item) {
  const link = document.createElement("a");
  link.href = mediaUrl(item, true);
  link.download = item.name;
  link.rel = "noopener";
  document.body.append(link);
  link.click();
  link.remove();
}

async function deleteGalleryItem(item) {
  const yes = await askConfirm(
    "Delete " + item.name + " from " + item.instance + "? This removes the file from that instance's output folder on disk.",
    { title: "Delete file", yes: "Delete file" }
  );
  if (!yes) return;
  state.busy = true;
  const result = await call("/api/gallery/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instance: item.instance, path: item.path }),
  });
  state.busy = false;
  if (result.sessionExpired || isAuthFailure(result)) return;
  if (!result.ok) {
    showBanner(result.error || "Delete failed.");
    return;
  }
  const key = galleryItemKey(item);
  const index = state.gallery.items.findIndex((row) => galleryItemKey(row) === key);
  state.gallery.items = state.gallery.items.filter((row) => galleryItemKey(row) !== key);
  state.gallery.total = Math.max(0, state.gallery.total - 1);
  state.gallery.signature = "";
  renderGalleryChrome();
  renderGalleryGrid();
  showToast("Deleted " + item.name + ".");
  hide(banner);
  if (document.querySelector("#lightbox").hidden) return;
  if (!state.gallery.items.length) {
    closeLightbox();
    return;
  }
  openLightbox(state.gallery.items[Math.min(Math.max(index, 0), state.gallery.items.length - 1)]);
}

function formatWhen(mtimeMs) {
  const date = new Date(Number(mtimeMs));
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString();
}

function formatSize(bytes) {
  const size = Number(bytes) || 0;
  if (size < 1024) return size + " B";
  if (size < 1048576) return (size / 1024).toFixed(1) + " KB";
  return (size / 1048576).toFixed(1) + " MB";
}

const importState = {
  draft: null,
  job: null,
  timer: 0,
  gpus: new Set(),
};

document.querySelector("#import-open").addEventListener("click", () => {
  setHostMenu(false);
  openImportPlan();
});
document.querySelector("#import-dupes-open").addEventListener("click", () => {
  setHostMenu(false);
  openDuplicates();
});
document.querySelector("#import-logs-open").addEventListener("click", () => {
  setHostMenu(false);
  openImportLogs();
});
document.querySelector("#import-plan").addEventListener("click", (event) => {
  if (event.target.closest("[data-plan-close]")) closeImportPlan();
});
document.querySelector("#import-overlay").addEventListener("click", (event) => {
  if (event.target.closest("[data-overlay-hide]")) hideImportOverlay();
});
document.querySelector("#import-dupes").addEventListener("click", (event) => {
  if (event.target.closest("[data-dupes-close]")) closeDialog("#import-dupes");
});
document.querySelector("#import-logs").addEventListener("click", (event) => {
  if (event.target.closest("[data-logs-close]")) closeDialog("#import-logs");
});
document.querySelector("#import-pill").addEventListener("click", () => reopenImport());
document.querySelector("#import-scan").addEventListener("click", () => scanImport());
document.querySelector("#import-start").addEventListener("click", () => beginImport());
document.querySelector("#import-overlay-start").addEventListener("click", () => beginImport());
document.querySelector("#import-pause").addEventListener("click", () => postImport("pause"));
document.querySelector("#import-resume").addEventListener("click", () => postImport("resume"));
document.querySelector("#import-cancel").addEventListener("click", () => postImport("cancel"));
document.querySelector("#import-dismiss").addEventListener("click", () => postImport("dismiss"));
document.querySelector("#import-remove-old").addEventListener("click", () => removeOldContainer());

function closeImportSurfaces() {
  const names = ["#import-logs", "#import-dupes", "#import-overlay", "#import-plan"];
  for (const name of names) {
    const node = document.querySelector(name);
    if (node && !node.hidden) {
      if (name === "#import-overlay") hideImportOverlay();
      else if (name === "#import-plan") closeImportPlan();
      else closeDialog(name);
      return true;
    }
  }
  return false;
}

function lockPage(locked) {
  const open = ["#sheet", "#confirm", "#lightbox", "#import-plan", "#import-overlay", "#import-dupes", "#import-logs"]
    .some((name) => {
      const node = document.querySelector(name);
      return node && !node.hidden;
    });
  document.body.style.overflow = locked || open ? "hidden" : "";
}

function closeDialog(selector) {
  const node = document.querySelector(selector);
  if (node) node.hidden = true;
  lockPage(false);
}

async function resumeImportOverlay() {
  const result = await call("/api/import/active");
  if (!result.ok || !result.payload || !result.payload.job) return;
  importState.job = result.payload.job;
  const status = importState.job.status;
  if (status === "running" || status === "paused" || status === "hashing" || status === "completed" || status === "cancelled" || status === "failed") {
    openImportOverlay();
  } else if (status === "awaiting_confirm") {
    showPlanForJob(importState.job);
  }
  startImportPoll();
}

function startImportPoll() {
  if (importState.timer) return;
  importState.timer = window.setInterval(pollImportJob, 700);
}

async function pollImportJob() {
  const job = importState.job;
  if (!job || job.dismissed) return;
  const result = await call("/api/import/jobs/" + encodeURIComponent(job.id));
  if (!result.ok || !result.payload) return;
  importState.job = result.payload.job;
  renderImportJob();
}

async function openImportPlan() {
  importState.draft = null;
  document.querySelector("#import-form").hidden = true;
  document.querySelector("#import-containers").hidden = false;
  document.querySelector("#import-plan-title").textContent = "Import container";
  hide(document.querySelector("#import-plan-banner"));
  document.querySelector("#import-plan").hidden = false;
  lockPage(true);
  const result = await call("/api/import/containers");
  const list = document.querySelector("#import-containers");
  list.replaceChildren();
  if (!result.ok) {
    showPlanError(result.error || "Could not list containers.");
    return;
  }
  const rows = (result.payload && result.payload.containers) || [];
  if (!rows.length) {
    list.append(el("p", { className: "hint", text: "No containers on this engine." }));
    return;
  }
  for (const row of rows) {
    const button = el("button", { className: "import-choice", type: "button" });
    button.innerHTML = "";
    const title = document.createElement("strong");
    title.textContent = row.name;
    const detail = document.createElement("small");
    detail.textContent = row.status + (row.managed ? " · fleet instance" : "");
    button.append(title, detail);
    button.addEventListener("click", () => inspectContainer(row.name));
    list.append(button);
  }
}

async function inspectContainer(name) {
  hide(document.querySelector("#import-plan-banner"));
  const result = await call("/api/import/inspect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ container: name }),
  });
  if (!result.ok) {
    showPlanError(result.error || "Could not read that container.");
    return;
  }
  importState.draft = result.payload;
  importState.gpus = new Set((result.payload.container.gpus || []).map((index) => String(index)));
  fillImportForm(result.payload);
  document.querySelector("#import-containers").hidden = true;
  document.querySelector("#import-form").hidden = false;
  document.querySelector("#import-counts").hidden = true;
  document.querySelector("#import-start").hidden = true;
  document.querySelector("#import-scan").hidden = false;
}

function fillImportForm(draft) {
  const container = draft.container;
  document.querySelector("#import-plan-title").textContent = "Import " + container.name;
  document.querySelector("#import-source").textContent =
    container.image + " · " + container.status + " · owner comfyuser";
  document.querySelector("#import-name").value = draft.suggested_name || "";
  document.querySelector("#import-port").value = String(draft.suggested_port || 8188);
  const move = document.querySelector("#import-mode-move");
  const running = container.status === "running";
  const unseen = (container.mounts || []).some((mount) => mount.visible === false);
  move.disabled = running || unseen;
  document.querySelector("#import-move-note").hidden = !running;
  document.querySelector("#import-unseen-note").hidden = !unseen;
  if (running || unseen) document.querySelector('input[name="import-mode"][value="copy"]').checked = true;
  document.querySelector('input[name="import-nodes"][value="as-is"]').checked = true;
  const workflow = document.querySelector("#import-workflow");
  workflow.replaceChildren();
  const flows = draft.workflows || [];
  if (!flows.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No workflow JSON found";
    workflow.append(option);
  }
  for (const item of flows) {
    const option = document.createElement("option");
    option.value = item.path;
    option.textContent = item.name;
    workflow.append(option);
  }
  const mounts = document.querySelector("#import-mounts");
  mounts.replaceChildren();
  for (const mount of container.mounts || []) {
    const row = document.createElement("div");
    row.className = "import-mount";
    const label = document.createElement("p");
    label.textContent = mount.source + " → " + mount.destination + (mount.visible === false ? " · docker cp" : "");
    const select = document.createElement("select");
    select.className = "text-input";
    select.dataset.source = mount.source;
    for (const role of ["models", "input", "output", "temp", "custom_nodes", "workflows", "wildcards", "user", "skip"]) {
      const option = document.createElement("option");
      option.value = role;
      option.textContent = role;
      if (role === mount.role) option.selected = true;
      select.append(option);
    }
    row.append(label, select);
    mounts.append(row);
  }
  const env = document.querySelector("#import-env");
  env.replaceChildren();
  const rows = container.env || [];
  if (!rows.length) env.append(el("div", { text: "No environment variables." }));
  for (const item of rows) {
    env.append(el("div", { text: item.key + "=" + item.value }));
  }
  const gpus = document.querySelector("#import-gpus");
  gpus.replaceChildren();
  const known = state.gpus.length ? state.gpus : (container.gpus || []).map((index) => ({ index, name: "GPU " + index }));
  if (!known.length) {
    gpus.append(el("p", { className: "hint", text: "No GPUs reported yet." }));
  }
  for (const gpu of known) {
    const id = "import-gpu-" + gpu.index;
    const label = document.createElement("label");
    label.className = "choice";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = id;
    input.value = String(gpu.index);
    input.checked = importState.gpus.has(String(gpu.index));
    input.addEventListener("change", () => {
      if (input.checked) importState.gpus.add(input.value);
      else importState.gpus.delete(input.value);
    });
    const text = document.createElement("span");
    text.textContent = gpu.name ? gpu.index + " " + gpu.name : "GPU " + gpu.index;
    label.append(input, text);
    gpus.append(label);
  }
}

function importRequestBody() {
  const draft = importState.draft;
  const mounts = [];
  for (const select of document.querySelectorAll("#import-mounts select")) {
    mounts.push({ source: select.dataset.source, role: select.value });
  }
  const mode = document.querySelector('input[name="import-mode"]:checked');
  const nodes = document.querySelector('input[name="import-nodes"]:checked');
  return {
    container: draft.container.name,
    name: document.querySelector("#import-name").value,
    port: Number(document.querySelector("#import-port").value),
    mode: mode ? mode.value : "copy",
    custom_nodes: nodes ? nodes.value : "as-is",
    cuda_tag: document.querySelector("#import-cuda").value,
    gpus: Array.from(importState.gpus).map((value) => Number(value)),
    workflow: document.querySelector("#import-workflow").value,
    mounts,
  };
}

async function scanImport() {
  const body = importRequestBody();
  if (!body.gpus.length) {
    showPlanError("Select at least one GPU.");
    return;
  }
  if (!body.workflow) {
    showPlanError("A workflow JSON is required.");
    return;
  }
  const result = await call("/api/import/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!result.ok) {
    showPlanError(result.error || "Could not start hashing.");
    return;
  }
  importState.job = result.payload.job;
  closeImportPlan(false);
  openImportOverlay();
  startImportPoll();
}

function showPlanForJob(job) {
  document.querySelector("#import-plan").hidden = false;
  document.querySelector("#import-containers").hidden = true;
  document.querySelector("#import-form").hidden = false;
  document.querySelector("#import-plan-title").textContent = "Review " + job.container;
  document.querySelector("#import-source").textContent = job.container + " → " + job.name + " · owner comfyuser";
  document.querySelector("#import-name").value = job.name || "";
  document.querySelector("#import-port").value = String(job.port || "");
  const mode = document.querySelector('input[name="import-mode"][value="' + job.mode + '"]');
  if (mode) mode.checked = true;
  const nodes = document.querySelector('input[name="import-nodes"][value="' + job.custom_nodes + '"]');
  if (nodes) nodes.checked = true;
  renderCounts(job.summary);
  document.querySelector("#import-scan").hidden = true;
  document.querySelector("#import-start").hidden = false;
  lockPage(true);
}

function renderCounts(summary) {
  const box = document.querySelector("#import-counts");
  if (!summary) {
    box.hidden = true;
    return;
  }
  box.hidden = false;
  box.textContent =
    "Skip " + summary.skip +
    " · duplicates " + summary.duplicates +
    " · rename " + summary.conflicts +
    " · transfer " + summary.transfer +
    " · space saved " + formatSize(summary.bytes_saved);
}

async function beginImport() {
  const job = importState.job;
  if (!job) return;
  const mode = document.querySelector('input[name="import-mode"]:checked');
  const result = await call("/api/import/jobs/" + encodeURIComponent(job.id) + "/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: mode ? mode.value : job.mode }),
  });
  if (!result.ok) {
    showPlanError(result.error || "Could not start the import.");
    showToast(result.error || "Could not start the import.");
    return;
  }
  importState.job = result.payload.job;
  closeImportPlan(false);
  openImportOverlay();
  startImportPoll();
}

function openImportOverlay() {
  document.querySelector("#import-overlay").hidden = false;
  document.querySelector("#import-pill").hidden = true;
  lockPage(true);
  renderImportJob();
}

function hideImportOverlay() {
  document.querySelector("#import-overlay").hidden = true;
  const job = importState.job;
  const pill = document.querySelector("#import-pill");
  if (job && !job.dismissed && job.status !== "dismissed") {
    pill.hidden = false;
    document.querySelector("#import-pill-label").textContent = pillText(job);
  } else {
    pill.hidden = true;
  }
  lockPage(false);
}

function closeImportPlan(showPill) {
  document.querySelector("#import-plan").hidden = true;
  lockPage(false);
  if (showPill !== false && importState.job && !importState.job.dismissed) {
    const pill = document.querySelector("#import-pill");
    pill.hidden = false;
    document.querySelector("#import-pill-label").textContent = pillText(importState.job);
  }
}

function reopenImport() {
  const job = importState.job;
  if (!job) return;
  document.querySelector("#import-pill").hidden = true;
  if (job.status === "awaiting_confirm") showPlanForJob(job);
  else openImportOverlay();
}

function renderImportJob() {
  const job = importState.job;
  if (!job) return;
  const overlay = document.querySelector("#import-overlay");
  if (!overlay.hidden) paintOverlay(job);
  else if (!document.querySelector("#import-pill").hidden) {
    document.querySelector("#import-pill-label").textContent = pillText(job);
  }
  if (job.status === "awaiting_confirm" && !document.querySelector("#import-plan").hidden) {
    renderCounts(job.summary);
    document.querySelector("#import-start").hidden = false;
    document.querySelector("#import-scan").hidden = true;
  }
  if (job.status === "awaiting_confirm" && overlay.hidden && document.querySelector("#import-plan").hidden && document.querySelector("#import-pill").hidden) {
    showPlanForJob(job);
  }
}

function paintOverlay(job) {
  document.querySelector("#import-overlay-title").textContent = "Import " + (job.name || "");
  document.querySelector("#import-phase").textContent = phaseLabel(job);
  const fileTotal = Number(job.current_size) || 0;
  const fileDone = Number(job.current_bytes) || 0;
  document.querySelector("#import-current").textContent = job.current_file
    ? job.current_file
    : (job.status === "completed" ? "Finished" : "Waiting");
  document.querySelector("#import-file-label").textContent = fileTotal
    ? formatSize(fileDone) + " / " + formatSize(fileTotal)
    : "";
  setBar("#import-file-bar", fileTotal ? fileDone / fileTotal : 0);
  const filesTotal = Number(job.files_total) || 0;
  const filesDone = Number(job.files_done) || 0;
  const bytesTotal = Number(job.bytes_total) || 0;
  const bytesDone = Number(job.bytes_done) || 0;
  document.querySelector("#import-overall-label").textContent =
    filesDone + " / " + filesTotal + " files · " + formatSize(bytesDone) + " / " + formatSize(bytesTotal);
  const byFiles = filesTotal ? filesDone / filesTotal : 0;
  const byBytes = bytesTotal ? bytesDone / bytesTotal : 0;
  setBar("#import-overall-bar", Math.max(byFiles, byBytes));
  const counts = job.counts || {};
  const stats = document.querySelector("#import-stats");
  stats.replaceChildren();
  addStat(stats, "Speed", formatRate(job.speed_current));
  addStat(stats, "Average", formatRate(job.speed_average));
  addStat(stats, "Elapsed", formatDuration(job.elapsed_s));
  addStat(stats, "ETA", job.eta_s == null ? "—" : formatDuration(job.eta_s));
  addStat(stats, "Moved", String(counts.moved || 0));
  addStat(stats, "Skipped", String(counts.skipped || 0));
  addStat(stats, "Duplicates", String(counts.renamed_dupes || 0));
  addStat(stats, "Conflicts", String(counts.conflicts || 0));
  addStat(stats, "Errors", String(counts.errors || 0));
  addStat(stats, "Free space", formatSize(job.free_bytes));
  const warn = document.querySelector("#import-space");
  warn.hidden = !job.low_space;
  const log = document.querySelector("#import-log");
  log.textContent = (job.log_tail || []).join("\n");
  log.scrollTop = log.scrollHeight;
  const terminal = job.status === "completed" || job.status === "cancelled" || job.status === "failed";
  const running = job.status === "running" || job.status === "hashing";
  document.querySelector("#import-pause").hidden = !running;
  document.querySelector("#import-resume").hidden = job.status !== "paused";
  document.querySelector("#import-cancel").hidden = terminal;
  document.querySelector("#import-dismiss").hidden = !terminal;
  document.querySelector("#import-overlay-start").hidden = job.status !== "awaiting_confirm";
  document.querySelector("#import-remove-old").hidden = !terminal || job.old_removed;
  if (job.summary && job.status === "awaiting_confirm") {
    document.querySelector("#import-current").textContent =
      "Skip " + job.summary.skip +
      ", duplicates " + job.summary.duplicates +
      ", rename " + job.summary.conflicts +
      ", transfer " + job.summary.transfer +
      ", space saved " + formatSize(job.summary.bytes_saved);
  }
  if (job.error) document.querySelector("#import-current").textContent = job.error;
}

function addStat(parent, label, value) {
  const wrap = document.createElement("div");
  const dt = document.createElement("dt");
  dt.textContent = label;
  const dd = document.createElement("dd");
  dd.textContent = value;
  wrap.append(dt, dd);
  parent.append(wrap);
}

function setBar(selector, ratio) {
  const width = Math.max(0, Math.min(1, ratio)) * 100;
  document.querySelector(selector).style.width = width.toFixed(1) + "%";
}

function phaseLabel(job) {
  if (job.status === "paused") return "Paused · " + (job.phase || "");
  if (job.status === "awaiting_confirm") return "Review";
  if (job.status === "completed") return "Completed";
  if (job.status === "cancelled") return "Cancelled";
  if (job.status === "failed") return "Failed";
  const phase = job.phase || job.status || "";
  if (phase === "hashing") return "Hashing";
  if (phase === "copying") return "Copying";
  if (phase === "verifying") return "Verifying";
  if (phase === "cleanup") return "Cleanup";
  return phase;
}

function pillText(job) {
  const total = Number(job.files_total) || 0;
  const done = Number(job.files_done) || 0;
  const pct = total ? Math.round((100 * done) / total) : 0;
  if (job.status === "awaiting_confirm") return "Review import";
  if (job.status === "completed") return "Import finished";
  if (job.status === "paused") return "Import paused " + pct + "%";
  return "Import " + pct + "%";
}

function formatRate(bytesPerSecond) {
  const rate = Number(bytesPerSecond) || 0;
  if (rate <= 0) return "—";
  return formatSize(rate) + "/s";
}

function formatDuration(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const secs = value % 60;
  if (hours) return hours + "h " + minutes + "m";
  if (minutes) return minutes + "m " + secs + "s";
  return secs + "s";
}

async function postImport(verb) {
  const job = importState.job;
  if (!job) return;
  const result = await call("/api/import/jobs/" + encodeURIComponent(job.id) + "/" + verb, { method: "POST" });
  if (!result.ok) {
    showToast(result.error || "Import request failed.");
    return;
  }
  importState.job = result.payload.job;
  if (verb === "dismiss") {
    document.querySelector("#import-overlay").hidden = true;
    document.querySelector("#import-pill").hidden = true;
    lockPage(false);
    return;
  }
  renderImportJob();
}

async function removeOldContainer() {
  const job = importState.job;
  if (!job) return;
  const yes = await askConfirm(
    "Remove the old container " + job.container + "? It must already be stopped. Host files are not deleted.",
    { title: "Remove old container", yes: "Remove old" }
  );
  if (!yes) return;
  await postImport("remove-old");
}

async function openDuplicates() {
  const dialog = document.querySelector("#import-dupes");
  dialog.hidden = false;
  lockPage(true);
  const list = document.querySelector("#import-dupe-list");
  list.replaceChildren();
  const result = await call("/api/import/duplicates");
  if (!result.ok) {
    list.append(el("p", { className: "hint", text: result.error || "Could not load duplicates." }));
    return;
  }
  const rows = (result.payload && result.payload.duplicates) || [];
  if (!rows.length) {
    list.append(el("p", { className: "hint", text: "No duplicates yet." }));
    return;
  }
  for (const row of rows) {
    const card = document.createElement("article");
    card.className = "import-dupe";
    card.append(el("p", { text: row.incoming_name + " → " + row.existing_name }));
    card.append(el("p", { text: row.existing_path }));
    card.append(el("p", { text: formatSize(row.size) + " · " + row.hash }));
    card.append(el("p", { text: row.import_id + " · " + row.date }));
    list.append(card);
  }
}

async function openImportLogs() {
  const dialog = document.querySelector("#import-logs");
  dialog.hidden = false;
  lockPage(true);
  const view = document.querySelector("#import-log-view");
  view.hidden = true;
  view.textContent = "";
  const list = document.querySelector("#import-log-list");
  list.replaceChildren();
  const result = await call("/api/import/jobs");
  if (!result.ok) {
    list.append(el("p", { className: "hint", text: result.error || "Could not load logs." }));
    return;
  }
  const rows = (result.payload && result.payload.jobs) || [];
  if (!rows.length) {
    list.append(el("p", { className: "hint", text: "No imports yet." }));
    return;
  }
  for (const row of rows) {
    const wrap = document.createElement("div");
    wrap.className = "import-choice";
    const title = document.createElement("strong");
    title.textContent = row.container + " → " + row.name;
    const detail = document.createElement("small");
    detail.textContent = row.status + " · " + row.created_at;
    const actions = document.createElement("div");
    actions.className = "sheet-actions";
    const viewButton = el("button", { className: "btn secondary", type: "button", text: "View" });
    viewButton.addEventListener("click", () => showImportLog(row.id));
    const download = document.createElement("a");
    download.className = "btn secondary";
    download.href = "/api/import/jobs/" + encodeURIComponent(row.id) + "/log";
    download.textContent = "Download";
    actions.append(viewButton, download);
    wrap.append(title, detail, actions);
    list.append(wrap);
  }
}

async function showImportLog(jobId) {
  const view = document.querySelector("#import-log-view");
  view.hidden = false;
  const response = await fetch("/api/import/jobs/" + encodeURIComponent(jobId) + "/log", {
    credentials: "same-origin",
    headers: { Accept: "text/plain" },
  });
  view.textContent = await response.text();
}

function showPlanError(text) {
  const banner = document.querySelector("#import-plan-banner");
  banner.hidden = false;
  banner.textContent = text;
}
