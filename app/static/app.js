// RodentTracker demo UI: one page, tabs in ANY-maze order. Plain JavaScript, no build step.
// Planned features (placeholders, Roadmap) come from planned.js.
const $ = (id) => document.getElementById(id);
const api = async (path, opts = {}) => {
  const r = await fetch(path, opts);
  if (r.status === 401) location.href = "/login";  // online demo: the session ended
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
};
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                         body: JSON.stringify(body) });

let changeTimer = null;
let video = null, info = null, setup = null, geom = null, img = new Image(), scale = 1;
let analysed = false, needsRerun = false, dragging = -1, downAt = null, pollTimer = null;
const HANDLE = 8;

// ---------- tabs ----------
function showTab(name) {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x.dataset.tab === name));
  document.querySelectorAll(".tab-page").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
  vid.pause(); rvid.pause();
  if (name === "validation") loadValidation();
  if (name === "tests") loadTests();
}
document.querySelectorAll(".tab").forEach((b) => b.onclick = () => showTab(b.dataset.tab));

// ---------- planned features: placeholders in their slots, the Roadmap tab, the show/hide switch ----------
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
function renderPlanned() {
  document.querySelectorAll("[data-planned-slot]").forEach((el) => {
    const items = PLANNED.filter((f) => f.slot === el.dataset.plannedSlot);
    if (el.tagName === "SELECT") {  // planned test types: disabled options
      el.insertAdjacentHTML("beforeend", items.map((f) =>
        `<option disabled class="planned-option" data-planned-id="${f.id}" title="${esc(f.what)}">`
        + `${esc(f.name)} (planned)</option>`).join(""));
      return;
    }
    el.innerHTML = items.map((f) =>
      `<div class="planned" data-planned-id="${f.id}" title="${esc(f.what)}">`
      + `<span class="name">${esc(f.name)}</span><span class="badge ${f.phase}">${PHASES[f.phase].label}</span>`
      + `<div class="what">${esc(f.what)}</div></div>`).join("");
  });
  // a click only expands the description; it never changes the setup
  document.querySelectorAll(".planned").forEach((p) => p.onclick = () => p.classList.toggle("open"));
  $("roadmap").innerHTML = Object.entries(PHASES).map(([key, ph]) => {
    const rows = PLANNED.filter((f) => f.phase === key);
    return `<h3><span class="badge ${key}">${ph.label}</span> ${esc(ph.title)}</h3><table>`
      + "<tr><th>Feature</th><th>Where</th><th>What it will do</th></tr>"
      + rows.map((f) => `<tr data-planned-id="${f.id}"><td>${esc(f.name)}</td><td>${slotName(f.slot)}</td>`
        + `<td>${esc(f.what)}</td></tr>`).join("") + "</table>";
  }).join("");
}
function slotName(slot) {
  const [tab, section] = slot.split(".");
  const names = { top: "", source: "Video source", test: "Test type", apparatus: "Apparatus", tracking: "Tracking",
                  zones: "Zones", behaviour: "Behaviour", testing: "Testing", reports: "Reports" };
  return [tab[0].toUpperCase() + tab.slice(1), names[section]].filter(Boolean).join(" › ");
}
function setShowPlanned(on) {
  document.body.classList.toggle("hide-planned", !on);
  $("show-planned").checked = on;
  if (!on && document.querySelector(".tab.active.planned-tab")) showTab("protocol");  // its tab just disappeared
  try { localStorage.setItem("showPlanned", on ? "1" : "0"); } catch (e) { /* storage blocked: keep the default */ }
}
$("show-planned").onchange = () => setShowPlanned($("show-planned").checked);
renderPlanned();
try { setShowPlanned(localStorage.getItem("showPlanned") !== "0"); } catch (e) { setShowPlanned(true); }

// ---------- tests: one per video until the experiment layer (Phase B) ----------
async function loadTests() {
  const vids = await api("/api/videos");
  table($("tests-table"), ["Test", "Video", "Animal", "Stage", "Trial", "Status"],
    vids.map((v, i) => [i + 1, v.video.replace(/^data\//, ""), "–", "–", "–",
                        v.analysed ? '<span class="ok">Analysed</span>' : "Not run"]));
  $("tests-table").querySelectorAll("tr").forEach((tr, i) => {
    if (!i) return;  // header
    const v = vids[i - 1];
    tr.dataset.video = v.video;
    tr.onclick = () => {  // switch at once; the video loads in the new tab
      showTab(v.analysed ? "results" : "protocol");
      if (v.video !== video) { $("video").value = v.video; openVideo(v.video); }
    };
  });
}

// ---------- videos ----------
async function loadVideos(select) {
  const vids = await api("/api/videos");
  $("video").innerHTML = vids.map((v) =>
    `<option value="${v.video}">${v.video.replace(/^data\//, "")}${v.analysed ? "  ✓" : ""}</option>`).join("");
  if (select) $("video").value = select;
  await openVideo($("video").value);
}
$("video").onchange = () => openVideo($("video").value);
$("upload").onchange = async () => {
  const f = $("upload").files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  const r = await api("/api/upload", { method: "POST", body: fd });
  await loadVideos(r.video);
};

let openSeq = 0;
async function openVideo(v) {
  const seq = ++openSeq;      // a newer openVideo call supersedes this one
  clearTimeout(changeTimer);  // a pending save belongs to the previous video
  analyseSeq++;               // ignore analysis replies for the previous video
  clearResults();
  vid.pause(); rvid.pause();
  video = v;
  seekToStart = true;
  const inf = await api("/api/info?video=" + encodeURIComponent(v));
  if (seq !== openSeq) return;  // the user picked another video meanwhile
  info = inf;
  setup = info.setup;
  analysed = info.analysed;
  needsRerun = info.needs_rerun;
  showNotice();
  fillForm();
  await openPlayback(seq);
  if (seq !== openSeq) return;
  await refreshGeometry();
  if (seq !== openSeq) return;
  showRun();
  if (analysed) analyse(); else clearResults();
}

function showNotice() {
  const n = $("notice");
  const src = setup.corners_source, maze = setup.test === "epm";
  const what = maze ? "Maze points" : "Corners";
  n.classList.toggle("warn", src === "none");
  n.innerHTML = src === "none"
    ? `<b>No saved setup for this camera.</b> ${what} are placed at a default position in the middle of the image — drag the red handles onto the ${maze ? "maze" : "floor corners"}.`
    : `<b>${what}: ${src}.</b> Check that the red outline sits on the ${maze ? "maze" : "arena floor"} and drag the handles if it does not.`;
  $("canvas-hint").innerHTML = maze
    ? (setup.maze_mode === "tips"
       ? "Drag <b>C1</b> and <b>C2</b> onto the ends of the <b>closed</b> arms and <b>O1</b>, <b>O2</b> onto the ends of the <b>open</b> arms (middle of each arm end). Set the arm width on the left."
       : "Drag each of the 12 corners of the '+' onto the maze.")
      + " Green = closed arms, red = open arms, blue = centre."
    : "Drag the 4 red handles onto the corners of the arena <b>floor</b> (where the walls meet the floor). Blue = centre zone, orange = corner zones, grey = grid.";
  $("maze-hint").textContent = setup.maze_mode === "tips"
    ? "Simple: the 12 corners are built from the tips and the width."
    : "Accurate for a tilted camera: every corner is placed by you.";
}

// ---------- form <-> setup ----------
function fillForm() {
  $("test").value = setup.test;
  document.querySelectorAll(".of-only").forEach((el) => el.classList.toggle("hidden", setup.test === "epm"));
  document.querySelectorAll(".epm-only").forEach((el) => el.classList.toggle("hidden", setup.test !== "epm"));
  document.querySelector(`input[name=maze_mode][value=${setup.maze_mode}]`).checked = true;
  $("arm_span_cm").value = setup.arm_span_cm;
  $("arm_width_cm").value = setup.arm_width_cm;
  $("width_cm").value = setup.width_cm;
  $("height_cm").value = setup.height_cm;
  $("grid_cm").value = setup.grid_cm;
  document.querySelector(`input[name=center_mode][value=${setup.center_mode}]`).checked = true;
  $("center_percent").value = setup.center_mode === "percent" ? setup.center_value : 50;
  $("center_cm").value = setup.center_mode === "cm" ? setup.center_value : 10;
  $("corner_zones").checked = setup.corner_zones;
  $("start_s").value = setup.start_s ?? "";
  $("end_s").value = setup.end_s ?? "";
}
function readForm() {
  setup.arm_span_cm = +$("arm_span_cm").value;
  setup.arm_width_cm = +$("arm_width_cm").value;
  setup.width_cm = +$("width_cm").value;
  setup.height_cm = +$("height_cm").value;
  setup.grid_cm = +$("grid_cm").value;
  setup.center_mode = document.querySelector("input[name=center_mode]:checked").value;
  setup.center_value = +(setup.center_mode === "percent" ? $("center_percent").value : $("center_cm").value);
  setup.corner_zones = $("corner_zones").checked;
  setup.start_s = $("start_s").value === "" ? null : +$("start_s").value;
  setup.end_s = $("end_s").value === "" ? null : +$("end_s").value;
}
["width_cm", "height_cm", "grid_cm", "center_percent", "center_cm", "corner_zones", "start_s", "end_s",
 "arm_span_cm", "arm_width_cm"]
  .forEach((id) => $(id).onchange = () => { readForm(); changed(); });
document.querySelectorAll("input[name=center_mode]").forEach((r) => r.onchange = () => { readForm(); changed(); });
$("test").onchange = () => { setup.test = $("test").value; fillForm(); showNotice(); changed(); };
document.querySelectorAll("input[name=maze_mode]").forEach((r) => r.onchange = () => {
  const mode = document.querySelector("input[name=maze_mode]:checked").value;
  if (mode === "corners" && geom?.points) setup.maze_points_px = JSON.parse(JSON.stringify(geom.points));
  if (mode === "tips" && geom?.tips) setup.maze_tips_px = JSON.parse(JSON.stringify(geom.tips));
  setup.maze_mode = mode;
  showNotice(); changed();
});
$("reset-zones").onclick = () => {
  setup.center_mode = "percent"; setup.center_value = 50; setup.toggled_cells = [];
  fillForm(); changed();
};
document.querySelectorAll("[data-auto]").forEach((b) => b.onclick = () => { $(b.dataset.auto).value = ""; readForm(); changed(); });
$("set-start").onclick = () => { $("start_s").value = (+$("scrub").value).toFixed(1); readForm(); changed(); };
$("set-end").onclick = () => { $("end_s").value = (+$("scrub").value).toFixed(1); readForm(); changed(); };

function changed() {  // any setup change: redraw now, save + re-analyse shortly after
  refreshGeometry();
  clearTimeout(changeTimer);
  const v = video, s = JSON.parse(JSON.stringify(setup));  // bind to this video's setup as it is now
  changeTimer = setTimeout(async () => {
    const r = await post("/api/setup?video=" + encodeURIComponent(v), s);
    if (v !== video) return;
    needsRerun = r.needs_rerun;
    showRun();
    if (analysed) analyse();
  }, 400);
}

// ---------- canvas ----------
const cv = $("canvas"), ctx = cv.getContext("2d"), vid = $("vid");
const rcv = $("res-canvas"), rctx = rcv.getContext("2d"), rvid = $("res-vid");
// Both tabs play the file itself in a hidden <video> and paint each frame onto their canvas with the overlays.
// If the browser cannot decode the file, the Setup slider falls back to still frames from the server.
let playable = false, seekToStart = false;
function frameSource() {
  if (playable) return vid.readyState >= 2 ? vid : null;
  return img.complete && img.naturalWidth ? img : null;
}
function loadVideo(v, src) {
  return new Promise((res) => {
    v.addEventListener("loadeddata", () => res(true), { once: true });
    v.addEventListener("error", () => res(false), { once: true });
    v.src = src;
  });
}
async function openPlayback(seq) {
  scale = Math.min(1.6, 960 / info.width);
  for (const c of [cv, rcv]) { c.width = info.width * scale; c.height = info.height * scale; }
  $("scrub").max = $("res-scrub").max = (info.frames / info.fps).toFixed(2);
  $("scrub").value = $("res-scrub").value = 0;
  $("scrub-time").textContent = $("res-time").textContent = "0.0 s";
  const src = "/api/video?video=" + encodeURIComponent(video);
  const ok = await loadVideo(vid, src);
  if (seq !== openSeq) return;
  playable = ok;
  $("play").disabled = $("rate").disabled = !ok;
  const why = ok ? "" : "Playback is not available for this video: the server did not send it (restart run_ui.bat "
    + "after an update) or this browser cannot play its format. The slider still steps through still frames.";
  $("play-note").textContent = why;
  $("res-canvas").classList.toggle("hidden", !ok);
  document.querySelector(".playback .timeline").classList.toggle("hidden", !ok);
  $("res-status").textContent = why;
  if (ok) { rvid.src = src; draw(); } else await loadFrame(0);
}
async function loadFrame(t) {  // still frame from the server, for files the browser cannot play
  await new Promise((res) => {
    img.onload = res;
    img.src = `/api/frame?video=${encodeURIComponent(video)}&t=${t}`;
  });
  draw();
}
function player(p) {  // play / pause / slider / speed for one video: p = {vid, play, scrub, time, rate, draw, still}
  let held = false;  // the user is dragging the slider; don't move it under their pointer
  const show = () => {
    p.time.textContent = p.vid.currentTime.toFixed(1) + " s";
    if (!held) p.scrub.value = p.vid.currentTime;
    p.draw();
  };
  const loop = () => { show(); if (!p.vid.paused) requestAnimationFrame(loop); };
  p.vid.addEventListener("play", () => { p.play.textContent = "Pause"; requestAnimationFrame(loop); });
  p.vid.addEventListener("pause", () => { p.play.textContent = "Play"; show(); });
  p.vid.addEventListener("seeked", show);
  p.play.onclick = () => (p.vid.paused ? p.vid.play() : p.vid.pause());
  p.rate.onchange = () => { p.vid.playbackRate = p.vid.defaultPlaybackRate = +p.rate.value; };
  p.scrub.onpointerdown = () => { held = true; };
  p.scrub.onpointerup = () => { held = false; };
  p.scrub.oninput = () => {
    p.time.textContent = (+p.scrub.value).toFixed(1) + " s";
    if (playable) p.vid.currentTime = +p.scrub.value; else if (p.still) p.still(+p.scrub.value);
  };
}
let scrubTimer = null;
player({ vid, play: $("play"), scrub: $("scrub"), time: $("scrub-time"), rate: $("rate"), draw: () => draw(),
         still: (t) => { clearTimeout(scrubTimer); scrubTimer = setTimeout(() => loadFrame(t), 120); } });
player({ vid: rvid, play: $("res-play"), scrub: $("res-scrub"), time: $("res-time"), rate: $("res-rate"),
         draw: () => drawResult() });

async function refreshGeometry() { geom = await post("/api/geometry", setup); draw(); drawResult(); }

function path(points, close = true, c = ctx) {
  c.beginPath();
  points.forEach(([x, y], i) => (i ? c.lineTo(x * scale, y * scale) : c.moveTo(x * scale, y * scale)));
  if (close) c.closePath();
}
const ZONE_COLOURS = { corners: ["rgba(224,123,0,.18)", "#e07b00"], centre: ["rgba(31,111,209,.22)", "#1f6fd1"],
                       open: ["rgba(214,32,32,.18)", "#d62020"], closed: ["rgba(46,157,79,.2)", "#2e9d4f"] };
function drawZones(c) {
  c.lineWidth = 1; c.strokeStyle = "rgba(160,160,160,.9)";
  geom.grid.forEach((l) => { path(l, false, c); c.stroke(); });
  c.lineWidth = 2;
  for (const [name, polys] of Object.entries(geom.zones)) {
    [c.fillStyle, c.strokeStyle] = ZONE_COLOURS[name];
    polys.forEach((p) => { path(p, true, c); c.fill(); c.stroke(); });
  }
}
function draw() {
  const src = frameSource();
  if (!src || !setup) return;
  ctx.drawImage(src, 0, 0, cv.width, cv.height);
  if (geom && dragging < 0) drawZones(ctx);
  const hs = handles();
  ctx.strokeStyle = "#d62020"; ctx.lineWidth = 2;
  if (setup.test !== "epm") { path(setup.corners_px); ctx.stroke(); }
  else if (dragging >= 0 && setup.maze_mode === "corners") { path(MAZE_ORDER.map((k) => setup.maze_points_px[k])); ctx.stroke(); }
  hs.forEach(({ pt: [x, y], label }) => {
    ctx.beginPath(); ctx.arc(x * scale, y * scale, HANDLE, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(214,32,32,.85)"; ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = "bold 9px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(label, x * scale, y * scale);
  });
}
const MAZE_ORDER = ["tl", "tr", "ctr", "rt", "rb", "cbr", "br", "bl", "cbl", "lb", "lt", "ctl"];
function handles() {  // draggable points for the current test: {pt (live array), label}
  if (setup.test !== "epm") return setup.corners_px.map((pt, i) => ({ pt, label: ["TL", "TR", "BR", "BL"][i] }));
  if (setup.maze_mode === "tips") return setup.maze_tips_px.map((pt, i) => ({ pt, label: ["C1", "O1", "C2", "O2"][i] }));
  return MAZE_ORDER.map((k) => ({ pt: setup.maze_points_px[k], label: k }));
}
function evPoint(e) {
  const r = cv.getBoundingClientRect();
  return [(e.clientX - r.left) * (cv.width / r.width) / scale, (e.clientY - r.top) * (cv.height / r.height) / scale];
}
function inside(pt, poly) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}
// Pointer capture: a drag released outside the image still ends (and is saved); points stay inside the image.
cv.onpointerdown = (e) => {
  const p = evPoint(e);
  downAt = p;
  dragging = handles().findIndex(({ pt: [x, y] }) => Math.hypot(x - p[0], y - p[1]) * scale <= HANDLE + 4);
  if (dragging >= 0) cv.setPointerCapture(e.pointerId);
};
cv.onpointermove = (e) => {
  if (dragging < 0) return;
  const [x, y] = evPoint(e), pt = handles()[dragging].pt;
  pt[0] = Math.max(0, Math.min(info.width, x));  // handles() returns the live arrays inside setup
  pt[1] = Math.max(0, Math.min(info.height, y));
  draw();
};
cv.onpointercancel = () => { if (dragging >= 0) { dragging = -1; changed(); } };
cv.onpointerup = (e) => {
  const p = evPoint(e);
  if (dragging >= 0) { dragging = -1; changed(); return; }
  if (downAt && Math.hypot(p[0] - downAt[0], p[1] - downAt[1]) * scale < 4 && geom) {  // click: toggle a grid square
    const cell = geom.cells.find((c) => inside(p, c.poly));
    if (cell) {
      const k = setup.toggled_cells.findIndex(([c, r]) => c === cell.col && r === cell.row);
      if (k >= 0) setup.toggled_cells.splice(k, 1); else setup.toggled_cells.push([cell.col, cell.row]);
      changed();
    }
  }
  downAt = null;
};

// ---------- run + results ----------
// The detector runs once per video. Everything else is recalculated from its cached detections, so the button is
// only offered before the first run, or when the arena / maze outline moved so far that the detections may miss.
function showRun() {
  $("run").classList.toggle("hidden", analysed && !needsRerun);
  $("run").textContent = analysed ? "Run analysis again" : "Run analysis";
  $("run-hint").textContent = !analysed
    ? "The detector has not run on this video yet (takes about as long as the video, longer online)."
    : needsRerun
      ? "The outline moved since the detector ran: run it again so the mouse is searched in the right place. "
        + "Until then the results use the earlier detections."
      : "Detections are cached: changes above update the results instantly.";
}
$("run").onclick = async () => {
  readForm();
  const r = await post("/api/run?video=" + encodeURIComponent(video), setup);
  if (!r.started && r.reason !== "already running") { alertBox(r.reason); return; }
  $("run").disabled = true;
  $("progress").classList.remove("hidden");
  poll();
};
async function poll() {
  clearTimeout(pollTimer);
  const s = await api("/api/status?video=" + encodeURIComponent(video));
  if (s.state === "running") {
    const pct = s.total ? Math.round(100 * s.done / s.total) : 0;
    $("progress").firstElementChild.style.width = pct + "%";
    $("progress").lastElementChild.textContent = `${pct}%` + (s.eta_s ? ` · about ${Math.ceil(s.eta_s / 60)} min left` : "");
    pollTimer = setTimeout(poll, 2000);
    return;
  }
  $("run").disabled = false;
  $("progress").classList.add("hidden");
  if (s.state === "error") { alertBox("The detector failed: " + s.error); return; }
  analysed = true;
  needsRerun = false;
  showRun();
  analyse();
  loadVideos(video);
}
function alertBox(msg) { $("run-hint").textContent = msg; }

function clearResults() {
  $("results").classList.add("hidden");
  $("results-empty").classList.remove("hidden");
  $("auto-times").textContent = "";
  track = null;
  $("start_s").placeholder = $("end_s").placeholder = "auto";  // the auto times belonged to the previous video
}
function table(el, header, rows) {
  el.innerHTML = (header ? "<tr>" + header.map((h) => `<th>${h}</th>`).join("") + "</tr>" : "")
    + rows.map((r) => "<tr>" + r.map((c) => `<td>${c ?? "–"}</td>`).join("") + "</tr>").join("");
}
let analyseSeq = 0;
async function analyse() {
  const seq = ++analyseSeq, v = video;
  $("auto-times").textContent = "Analysing…";
  let r;
  try { r = await post("/api/analyze?video=" + encodeURIComponent(v), setup); }
  catch (err) { if (seq === analyseSeq) alertBox(err.message); return; }
  if (seq !== analyseSeq || v !== video) return;  // superseded, or the user switched video
  $("auto-times").textContent = `Detected automatically: start ${r.auto_start_s} s (hand gone, mouse present), end ${r.auto_end_s} s.`;
  $("start_s").placeholder = `auto (${r.auto_start_s})`;
  $("end_s").placeholder = `auto (${r.auto_end_s})`;
  $("results-title").textContent = video.split("/").pop();
  table($("whole"), null, r.whole);
  table($("zones"), r.zone_header, r.zone_rows);
  $("img-track").src = r.images.track;
  $("img-heatmap").src = r.images.heatmap;
  $("img-per-minute").src = r.images.per_minute;
  const q = "?video=" + encodeURIComponent(video) + "&kind=";
  $("dl-csv").href = "/api/export" + q + "csv";
  $("dl-track").href = "/api/export" + q + "track";
  $("dl-pdf").href = "/api/export" + q + "pdf";
  $("results-empty").classList.add("hidden");
  $("results").classList.remove("hidden");
  track = r.track;
  if (seekToStart) { seekToStart = false; rvid.currentTime = r.summary.start_s; }  // open the player where the test begins
  drawResult();
}

// ---------- results playback: the video with the zones and the tracked position ----------
let track = null;  // per-frame track of the analysed window, from /api/analyze
const TRAIL_S = 2;
function trackIndex(t) {  // last tracked frame at or before t, or -1
  let lo = 0, hi = track.t.length - 1, k = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (track.t[m] <= t) { k = m; lo = m + 1; } else hi = m - 1;
  }
  return k;
}
function drawResult() {
  if (!playable || rvid.readyState < 2) return;
  rctx.drawImage(rvid, 0, 0, rcv.width, rcv.height);
  if (geom) drawZones(rctx);
  if (!track) { $("res-status").textContent = ""; return; }
  const t = rvid.currentTime, k = trackIndex(t), n = track.t.length - 1;
  if (k < 0 || (k === n && t > track.t[n] + 0.1)) {
    $("res-status").textContent =
      `${t.toFixed(1)} s · outside the analysed window (${track.t[0].toFixed(1)}–${track.t[n].toFixed(1)} s)`;
    return;
  }
  rctx.strokeStyle = "#ffd400"; rctx.lineWidth = 2; rctx.beginPath();
  let pen = false;
  for (let i = trackIndex(t - TRAIL_S) + 1; i <= k; i++) {  // the last two seconds of the path
    if (track.x[i] == null) { pen = false; continue; }
    const X = track.x[i] * scale, Y = track.y[i] * scale;
    if (pen) rctx.lineTo(X, Y); else rctx.moveTo(X, Y);
    pen = true;
  }
  rctx.stroke();
  if (track.x[k] == null) { $("res-status").textContent = `${t.toFixed(1)} s · mouse not found`; return; }
  rctx.beginPath(); rctx.arc(track.x[k] * scale, track.y[k] * scale, 6, 0, 2 * Math.PI);
  rctx.fillStyle = track.moving[k] ? "#ffd400" : "#ffffff"; rctx.fill();
  rctx.lineWidth = 2; rctx.strokeStyle = "#1d2433"; rctx.stroke();
  const zones = track.zone_names.filter((_, j) => track.zones[k][j]);
  $("res-status").textContent = `${t.toFixed(1)} s · in ${zones.join(" + ") || "no zone"} · `
    + (track.moving[k] ? "moving" : "not moving");
}

// ---------- validation ----------
let validationLoaded = false;
async function loadValidation() {
  if (validationLoaded) return;
  const v = await api("/api/validation");
  const card = (title, s, unit) =>
    `<div class="card">${title}<b>${s.mean_abs_pct}% avg difference</b>` +
    `<span class="muted">${s.mean_abs} ${unit} avg · worst ${s.max_abs_pct}% · r = ${s.r}</span></div>`;
  $("val-cards").innerHTML =
    card("Distance travelled", v.method.distance, "cm") +
    card("Centre time", v.method.centre_time, "s") +
    card("Centre transitions", v.method.transitions, "") +
    `<div class="card">Tracking<b>${v.frames.detected_pct}% frames found</b>` +
    `<span class="muted">median position error ${v.frames.median_error_cm} cm</span></div>` +
    `<div class="card">Full product engine<br><span class="muted">(tilted-camera calibration, our zones)</span>` +
    `<b>${v.engine.distance.mean_abs_pct}% / ${v.engine.centre_time.mean_abs} s</b>` +
    `<span class="muted">distance / centre time, avg difference</span></div>`;
  $("val-scatter").src = v.scatter;
  table($("val-table"),
    ["Video", "Found %", "Median error (cm)", "Distance ref (cm)", "Distance ours (cm)", "Centre ref (s)",
     "Centre ours (s)", "Transitions ref", "Transitions ours"],
    v.table.map((r) => [r.video, r.detected_pct, r.err_median_cm, r.ref_raw_distance_cm, r.ours_raw_distance_cm,
                        r.ref_center_time_s, r.ours_center_time_s, r.ref_center_transitions, r.ours_center_transitions]));
  validationLoaded = true;
}

loadVideos();
