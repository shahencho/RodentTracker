// RodentTracker demo UI: one page, three tabs. Plain JavaScript, no build step.
const $ = (id) => document.getElementById(id);
const api = async (path, opts = {}) => {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
};
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                         body: JSON.stringify(body) });

let changeTimer = null;
let video = null, info = null, setup = null, geom = null, img = new Image(), scale = 1;
let analysed = false, dragging = -1, downAt = null, pollTimer = null;
const HANDLE = 8;

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
  document.querySelectorAll(".tab-page").forEach((p) => p.classList.toggle("active", p.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "validation") loadValidation();
});

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

async function openVideo(v) {
  clearTimeout(changeTimer);  // a pending save belongs to the previous video
  analyseSeq++;               // ignore analysis replies for the previous video
  clearResults();
  video = v;
  info = await api("/api/info?video=" + encodeURIComponent(v));
  setup = info.setup;
  analysed = info.analysed;
  $("scrub").max = (info.frames / info.fps).toFixed(2);
  $("scrub").value = 0;
  showNotice();
  fillForm();
  await loadFrame(0);
  await refreshGeometry();
  $("run-hint").textContent = analysed ? "Detections are cached: changes below update the results instantly."
                                       : "The detector has not run on this video yet (takes about as long as the video).";
  if (analysed) analyse(); else clearResults();
}

function showNotice() {
  const n = $("notice");
  const src = setup.corners_source;
  n.classList.toggle("warn", src === "none");
  n.innerHTML = src === "none"
    ? "<b>No saved setup for this camera.</b> Corners are placed at a default position in the middle of the image — drag the 4 red handles onto the floor corners."
    : `<b>Corners: ${src}.</b> Check that the red outline sits on the arena floor and drag the handles if it does not.`;
}

// ---------- form <-> setup ----------
function fillForm() {
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
  setup.width_cm = +$("width_cm").value;
  setup.height_cm = +$("height_cm").value;
  setup.grid_cm = +$("grid_cm").value;
  setup.center_mode = document.querySelector("input[name=center_mode]:checked").value;
  setup.center_value = +(setup.center_mode === "percent" ? $("center_percent").value : $("center_cm").value);
  setup.corner_zones = $("corner_zones").checked;
  setup.start_s = $("start_s").value === "" ? null : +$("start_s").value;
  setup.end_s = $("end_s").value === "" ? null : +$("end_s").value;
}
["width_cm", "height_cm", "grid_cm", "center_percent", "center_cm", "corner_zones", "start_s", "end_s"]
  .forEach((id) => $(id).onchange = () => { readForm(); changed(); });
document.querySelectorAll("input[name=center_mode]").forEach((r) => r.onchange = () => { readForm(); changed(); });
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
    await post("/api/setup?video=" + encodeURIComponent(v), s);
    if (analysed && v === video) analyse();
  }, 400);
}

// ---------- canvas ----------
const cv = $("canvas"), ctx = cv.getContext("2d");
async function loadFrame(t) {
  await new Promise((res) => {
    img.onload = res;
    img.src = `/api/frame?video=${encodeURIComponent(video)}&t=${t}`;
  });
  scale = Math.min(1.6, 960 / img.width);
  cv.width = img.width * scale;
  cv.height = img.height * scale;
  draw();
}
let scrubTimer = null;
$("scrub").oninput = () => {
  $("scrub-time").textContent = (+$("scrub").value).toFixed(1) + " s";
  clearTimeout(scrubTimer);
  scrubTimer = setTimeout(() => loadFrame(+$("scrub").value), 120);
};

async function refreshGeometry() { geom = await post("/api/geometry", setup); draw(); }

function path(points, close = true) {
  ctx.beginPath();
  points.forEach(([x, y], i) => (i ? ctx.lineTo(x * scale, y * scale) : ctx.moveTo(x * scale, y * scale)));
  if (close) ctx.closePath();
}
function draw() {
  if (!img.complete || !setup) return;
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
  if (geom && dragging < 0) {
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(160,160,160,.9)";
    geom.grid.forEach((l) => { path(l, false); ctx.stroke(); });
    ctx.fillStyle = "rgba(224,123,0,.18)"; ctx.strokeStyle = "#e07b00"; ctx.lineWidth = 2;
    geom.corners.forEach((p) => { path(p); ctx.fill(); ctx.stroke(); });
    ctx.fillStyle = "rgba(31,111,209,.22)"; ctx.strokeStyle = "#1f6fd1";
    geom.centre.forEach((p) => { path(p); ctx.fill(); ctx.stroke(); });
  }
  ctx.strokeStyle = "#d62020"; ctx.lineWidth = 2;
  path(setup.corners_px); ctx.stroke();
  setup.corners_px.forEach(([x, y], i) => {
    ctx.beginPath(); ctx.arc(x * scale, y * scale, HANDLE, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(214,32,32,.85)"; ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = "bold 10px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(["TL", "TR", "BR", "BL"][i], x * scale, y * scale);
  });
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
cv.onmousedown = (e) => {
  const p = evPoint(e);
  downAt = p;
  dragging = setup.corners_px.findIndex(([x, y]) => Math.hypot(x - p[0], y - p[1]) * scale <= HANDLE + 4);
};
cv.onmousemove = (e) => {
  if (dragging < 0) return;
  setup.corners_px[dragging] = evPoint(e);
  draw();
};
cv.onmouseup = (e) => {
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
$("run").onclick = async () => {
  readForm();
  await post("/api/run?video=" + encodeURIComponent(video), setup);
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
  $("run-hint").textContent = "Detections are cached: changes below update the results instantly.";
  analyse();
  loadVideos(video);
}
function alertBox(msg) { $("run-hint").textContent = msg; }

function clearResults() {
  $("results").classList.add("hidden");
  $("results-empty").classList.remove("hidden");
  $("auto-times").textContent = "";
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
