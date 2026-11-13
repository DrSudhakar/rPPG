// SAMA live interface. Browser mode: each frame is reduced here to 30 skin-patch colour means, a
// motion index, the skin fraction and the face brightness; only those numbers are sent to the
// local server. No image, frame or video leaves this page. Lab-recorder mode: the server captures
// the webcam with ffmpeg and this page shows its preview and controls.
"use strict";

const $ = (id) => document.getElementById(id);
const LAYOUT = [];
for (let r = 0; r < 2; r++) for (let c = 0; c < 6; c++) LAYOUT.push([0.10 + 0.8 * c / 6, 0.06 + 0.12 * r, 0.8 / 6, 0.12]);
for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) LAYOUT.push([0.06 + 0.11 * c, 0.46 + 0.10 * r, 0.11, 0.10]);
for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) LAYOUT.push([0.61 + 0.11 * c, 0.46 + 0.10 * r, 0.11, 0.10]);

const S = {
  info: null, mode: "serve", sid: null, stream: null, running: false, recording: false,
  boxes: { face_box: [0.30, 0.15, 0.40, 0.60], white_box: null, chart_corners: null },
  drawMode: "face_box", drag: null, chartPts: [], state: null, pending: [], t0: null, prevThumb: null,
  lastBeat: -1, flashUntil: 0, sending: false, recStart: null, frameW: 320, frameH: 240,
};
const work = document.createElement("canvas");
const wctx = work.getContext("2d", { willReadFrequently: true });

async function api(path, body, method) {
  const opt = { method: method || (body !== undefined ? "POST" : "GET"), headers: { "Content-Type": "application/json" } };
  if (body !== undefined) opt.body = JSON.stringify(body);
  const r = await fetch(path, opt);
  return r.headers.get("content-type")?.includes("json") ? r.json() : null;
}

// ------------------------------------------------------------------ frame measurement (browser)
function skin(r, g, b) {
  const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
  const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
  const y = 0.299 * r + 0.587 * g + 0.114 * b;
  return cb >= 73 && cb <= 132 && cr >= 130 && cr <= 180 && y > 15;
}

function measure(img, W, H, box) {
  const x0 = Math.max(0, Math.floor(box[0] * W)), y0 = Math.max(0, Math.floor(box[1] * H));
  const x1 = Math.min(W, Math.floor((box[0] + box[2]) * W)), y1 = Math.min(H, Math.floor((box[1] + box[3]) * H));
  const bw = Math.max(2, x1 - x0), bh = Math.max(2, y1 - y0);
  const d = img.data;
  const at = (x, y) => 4 * ((y0 + y) * W + (x0 + x));
  let nSkin = 0, sumG = 0;
  const patches = [], pix = [];
  for (const [px, py, pw, ph] of LAYOUT) {
    const ya = Math.floor(py * bh), yb = Math.max(Math.floor((py + ph) * bh), ya + 1);
    const xa = Math.floor(px * bw), xb = Math.max(Math.floor((px + pw) * bw), xa + 1);
    let k = [0, 0, 0, 0], s = [0, 0, 0, 0], a = [0, 0, 0, 0], n = 0;
    for (let y = ya; y < yb && y < bh; y++) for (let x = xa; x < xb && x < bw; x++) {
      const i = at(x, y), r = d[i], g = d[i + 1], b = d[i + 2];
      n++; a[0] += r; a[1] += g; a[2] += b; a[3]++;
      const keep = Math.max(r, g, b) < 250 && Math.min(r, g, b) > 2;
      if (keep) { k[0] += r; k[1] += g; k[2] += b; k[3]++;
        if (skin(r, g, b)) { s[0] += r; s[1] += g; s[2] += b; s[3]++; } }
    }
    const use = s[3] >= Math.max(4, 0.3 * n) ? s : (k[3] ? k : a);
    patches.push(use[0] / use[3], use[1] / use[3], use[2] / use[3]);
    pix.push(use[3]);
  }
  // skin fraction, brightness and the 48x48 luma thumbnail for the motion index
  const T = 48, thumb = new Float32Array(T * T);
  for (let j = 0; j < T; j++) for (let i = 0; i < T; i++) {
    const x = Math.floor(i * bw / T), y = Math.floor(j * bh / T), q = at(x, y);
    const r = d[q], g = d[q + 1], b = d[q + 2];
    thumb[j * T + i] = 0.299 * r + 0.587 * g + 0.114 * b;
    if (skin(r, g, b)) nSkin++;
    sumG += g;
  }
  let motion = 0;
  if (S.prevThumb) { for (let i = 0; i < thumb.length; i++) motion += Math.abs(thumb[i] - S.prevThumb[i]); motion /= thumb.length; }
  S.prevThumb = thumb;
  return { patches: patches.map((v) => Math.round(v * 1000) / 1000), pix, motion, skin: nSkin / (T * T), bright: sumG / (T * T) };
}

function onFrame() {
  if (!S.running) return;
  const v = $("video");
  if (v.videoWidth) {
    S.frameW = 320; S.frameH = Math.round(320 * v.videoHeight / v.videoWidth);
    work.width = S.frameW; work.height = S.frameH;
    wctx.drawImage(v, 0, 0, S.frameW, S.frameH);
    const img = wctx.getImageData(0, 0, S.frameW, S.frameH);
    const m = measure(img, S.frameW, S.frameH, S.boxes.face_box);
    const t = (performance.now() - S.t0) / 1000;
    S.pending.push({ t: Math.round(t * 1000) / 1000, ...m });
  }
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) v.requestVideoFrameCallback(onFrame);
  else requestAnimationFrame(onFrame);
}

async function sendLoop() {
  if (!S.running) return;
  if (S.pending.length && !S.sending) {
    const batch = S.pending.splice(0, S.pending.length);
    S.sending = true;
    try {
      const r = await api(`/api/v1/session/${S.sid}/frames`, { frames: batch });
      if (r && r.state) render(r.state);
    } catch (e) { $("reason").textContent = "server not reachable"; }
    S.sending = false;
  }
  setTimeout(sendLoop, 250);
}

async function startBrowser() {
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: 30, facingMode: "user" }, audio: false });
  } catch (e) { alert("Camera not available: " + e.message); return; }
  const v = $("video");
  v.srcObject = S.stream; await v.play();
  const ita = $("ita").value;
  const r = await api("/api/v1/session", { participant_id: $("pid").value || null, ita: ita === "" ? null : Number(ita), knobs: readKnobs() });
  if (r.error) { alert(r.error); return; }
  S.sid = r.session; S.running = true; S.t0 = performance.now(); S.prevThumb = null; S.lastBeat = -1;
  $("idle").hidden = true;
  $("startBtn").disabled = true;
  onFrame(); sendLoop();
}

async function stopBrowser() {
  S.running = false;
  if (S.stream) S.stream.getTracks().forEach((t) => t.stop());
  if (S.sid) await api(`/api/v1/session/${S.sid}`, undefined, "DELETE");
  S.sid = null; $("startBtn").disabled = false; $("idle").hidden = false;
}

// ------------------------------------------------------------------ lab recorder
async function labPoll() {
  try {
    const st = await api("/api/v1/lab/state");
    if (st) {
      S.recording = !!st.recording;
      if (st.boxes) { S.boxes.white_box = st.boxes.white_box; S.boxes.chart_corners = st.boxes.chart_corners; if (!S.drag) S.boxes.face_box = st.boxes.face_box; }
      if (st.error) $("labMsg").textContent = "ffmpeg: " + st.error;
      else if (st.preview_error && !S.recording) $("labMsg").textContent = "camera: " + st.preview_error;
      if (st.white_check && st.white_check.clipped_frac !== undefined) {
        const w = st.white_check, el = $("whiteCheck");
        el.textContent = w.ok ? `card ok (mean ${w.mean.toFixed(0)})` : (w.clipped_frac >= 0.02 ? `card clipped ${(100 * w.clipped_frac).toFixed(0)} %` : "card too dark");
        el.className = "pill labOnly " + (w.ok ? "ok" : "bad");
      }
      if (st.t !== undefined && st.reason) render(st);
      $("recStart").disabled = S.recording; $("recStop").disabled = !S.recording;
      $("idle").hidden = S.recording;
    }
  } catch (e) { /* server restarting */ }
  setTimeout(labPoll, 400);
}

function labPreview() {
  // Polls for a camera frame whether or not recording has started: start_preview() (called once in
  // setupLab) opens the camera as soon as the page loads, so there is something to draw boxes on
  // before "Start recording" is pressed, and start()/stop() hand the same latest_frame back and forth.
  const img = $("preview");
  const nxt = new Image();
  nxt.onload = () => { img.src = nxt.src; img.hidden = false; S.frameW = nxt.naturalWidth; S.frameH = nxt.naturalHeight; setTimeout(labPreview, 90); };
  nxt.onerror = () => setTimeout(labPreview, 500);
  nxt.src = "/api/v1/lab/preview.jpg?" + Date.now();
}

async function labEvent(event, extra = {}) {
  const r = await api("/api/v1/lab/event", { event, ...extra });
  if (r.error) { $("labMsg").textContent = r.error; return; }
  const li = document.createElement("li");
  li.textContent = `${new Date(Number(r.wall_time) * 1000).toLocaleTimeString()}  ${event} ${r.block || ""} ${r.light || ""} ${r.note || ""}`;
  $("events").prepend(li);
}

function setupLab() {
  $("videoWrap").classList.add("lab");
  const sel = (id, list) => { $(id).innerHTML = list.map((x) => `<option>${x}</option>`).join(""); };
  sel("labBlock", S.info.blocks); sel("labLight", S.info.lights);
  $("labPid").value = S.info.participant || "";
  $("recStart").onclick = async () => {
    const r = await api("/api/v1/lab/start", { participant_id: $("labPid").value, operator: $("labOp").value });
    if (r.error) { $("labMsg").textContent = r.error; return; }
    await pushBoxes();
    $("labMsg").textContent = "Recording to " + r.session_dir; S.recStart = Date.now();
  };
  $("recStop").onclick = async () => { const r = await api("/api/v1/lab/stop", {}); $("labMsg").textContent = `Stopped: ${r.frames || 0} frames saved in ${r.session_dir || ""}`; S.recStart = null; };
  $("blockStart").onclick = () => labEvent("block_start", { block: $("labBlock").value, light: $("labLight").value });
  $("blockEnd").onclick = () => labEvent("block_end", { block: $("labBlock").value, light: $("labLight").value });
  $("syncMark").onclick = () => labEvent("sync_mark", { note: "LED flash / clap" });
  $("noteBtn").onclick = () => { labEvent("note", { note: $("noteText").value }); $("noteText").value = ""; };
  $("cardRefl").value = S.info.white_reflectance;
  $("cardRefl").onchange = pushBoxes;
  setInterval(() => { if (S.recStart) { const s = Math.floor((Date.now() - S.recStart) / 1000); $("recTimer").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; } }, 500);
  api("/api/v1/lab/preview_start", {});  // open the camera now so boxes can be drawn before recording starts
  labPoll(); labPreview();
}

async function pushBoxes() {
  if (S.mode !== "record") return;
  await api("/api/v1/lab/boxes", { face_box: S.boxes.face_box, white_box: S.boxes.white_box || [], chart_corners: S.boxes.chart_corners || [],
                                   white_reflectance: Number($("cardRefl").value) });
}

// ------------------------------------------------------------------ rendering
function lvl(el, level, text) { el.className = "pill " + (level || ""); el.textContent = text; }
const fmt = (x, d = 0) => (x === null || x === undefined || !isFinite(x)) ? "–" : Number(x).toFixed(d);

function render(st) {
  S.state = st;
  $("hr").textContent = st.shown ? fmt(st.hr) : "–";
  $("interval").textContent = st.shown && st.lo !== null && st.lo !== undefined ? `90 % interval ${fmt(st.lo)}–${fmt(st.hi)} bpm` : " ";
  $("reason").textContent = st.shown ? " " : (st.message || st.reason || "");
  $("methodTag").textContent = (st.method || "") + (st.method_note ? " — " + st.method_note : "");
  const h = st.hrv || {};
  $("rmssd").textContent = h.valid ? fmt(h.rmssd) : "–";
  $("sdnn").textContent = h.valid ? fmt(h.sdnn) : "–";
  $("pnn50").textContent = h.valid ? fmt(h.pnn50) : "–";
  $("hrvNote").textContent = h.valid ? `${h.n_beats} beats, ${fmt(100 * h.edited)} % edited` : (h.reason || " ");
  const q = st.quality || {};
  lvl($("qMotion"), q.motion_level, `motion ${fmt(q.motion, 1)}`);
  lvl($("qFace"), q.face_level, `face ${fmt(100 * (q.face || 0))} %`);
  lvl($("qLight"), q.light_level, `light ${fmt(q.light)}`);
  lvl($("qFps"), q.fps_level, `${fmt(q.fps)} fps`);
  const ul = $("warnings"); ul.innerHTML = "";
  for (const w of st.warnings || []) { const li = document.createElement("li"); li.className = w.level; li.textContent = w.text; ul.appendChild(li); }
  const mw = (st.warnings || []).find((w) => w.code === "MOTION");
  const mb = $("motionBanner");
  if (mw) { mb.hidden = false; mb.className = "motionBanner " + mw.level; mb.textContent = (mw.level === "bad" ? "⚠ " : "") + mw.text; } else mb.hidden = true;
  // beats: blink the dot with the rhythm of the newly detected beats
  const showBeats = (st.knobs || {}).show_beats !== false;
  const nb = (st.beats || []).filter((b) => b > S.lastBeat + 0.05);
  if (showBeats && nb.length) {
    const first = nb[0];
    nb.forEach((b) => setTimeout(() => { S.flashUntil = performance.now() + 140; }, Math.min(2000, (b - first) * 1000)));
  }
  if (st.beats && st.beats.length) S.lastBeat = Math.max(S.lastBeat, st.beats[st.beats.length - 1]);
  drawTrace(st, showBeats);
}

function drawTrace(st, showBeats) {
  const c = $("trace"), ctx = c.getContext("2d");
  const W = c.width = c.clientWidth * devicePixelRatio, H = c.height = 140 * devicePixelRatio;
  ctx.clearRect(0, 0, W, H);
  const tr = st.trace;
  if (!tr || !tr.y.length) return;
  const n = tr.y.length, ymax = Math.max(2.5, ...tr.y.map(Math.abs));
  const X = (i) => (i / (n - 1)) * (W - 10) + 5, Y = (v) => H / 2 - (v / ymax) * (H / 2 - 8);
  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue("--accent"); ctx.lineWidth = 2 * devicePixelRatio;
  ctx.beginPath(); tr.y.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)))); ctx.stroke();
  if (!showBeats) return;
  ctx.fillStyle = "#d62f2f";
  for (const b of st.beats || []) {
    const fi = (b - tr.t0) * tr.fs;
    if (fi < 0 || fi > n - 1) continue;
    const i0 = Math.floor(fi), f = fi - i0, v = tr.y[i0] * (1 - f) + (tr.y[Math.min(n - 1, i0 + 1)] || 0) * f;
    ctx.beginPath(); ctx.arc(X(fi), Y(v), 5 * devicePixelRatio, 0, 2 * Math.PI); ctx.fill();
  }
}

// ------------------------------------------------------------------ overlay: boxes, dot, drawing
function contentRect() {
  const wrap = $("overlay").getBoundingClientRect();
  const ar = S.frameW / S.frameH, war = wrap.width / wrap.height;
  let w = wrap.width, h = wrap.height, x = 0, y = 0;
  if (ar > war) { h = w / ar; y = (wrap.height - h) / 2; } else { w = h * ar; x = (wrap.width - w) / 2; }
  return { x, y, w, h };
}
const mirrored = () => S.mode !== "record";
function toDisp(bx) { const r = contentRect(); let x = bx[0]; if (mirrored()) x = 1 - bx[0] - bx[2]; return [r.x + x * r.w, r.y + bx[1] * r.h, bx[2] * r.w, bx[3] * r.h]; }
function toFrame(px, py) { const r = contentRect(); let x = (px - r.x) / r.w; const y = (py - r.y) / r.h; if (mirrored()) x = 1 - x; return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))]; }

function drawOverlay() {
  const c = $("overlay"), ctx = c.getContext("2d");
  const rect = c.getBoundingClientRect();
  c.width = rect.width * devicePixelRatio; c.height = rect.height * devicePixelRatio;
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const fb = S.boxes.face_box;
  if (fb) {
    const [x, y, w, h] = toDisp(fb);
    const ok = S.state && S.state.quality && S.state.quality.face_level === "ok";
    ctx.strokeStyle = ok ? "#3ddc84" : "#ffb300"; ctx.lineWidth = 2; ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "rgba(255,255,255,.85)"; ctx.font = "12px system-ui"; ctx.fillText("face", x + 4, y + 14);
    ctx.strokeStyle = "rgba(255,255,255,.25)"; ctx.lineWidth = 1;
    for (const [px, py, pw, ph] of LAYOUT) { const lx = mirrored() ? 1 - px - pw : px; ctx.strokeRect(x + lx * w, y + py * h, pw * w, ph * h); }
    if (performance.now() < S.flashUntil) { ctx.fillStyle = "#ff3b3b"; ctx.beginPath(); ctx.arc(x + w - 14, y + 14, 9, 0, 2 * Math.PI); ctx.fill(); }
    $("beatDot").classList.toggle("on", performance.now() < S.flashUntil);
  }
  const wb = S.boxes.white_box;
  if (wb && wb.length) { const [x, y, w, h] = toDisp(wb); ctx.setLineDash([5, 4]); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.strokeRect(x, y, w, h); ctx.setLineDash([]); ctx.fillStyle = "#fff"; ctx.fillText("card", x + 4, y + 14); }
  const cc = S.boxes.chart_corners;
  if (cc && cc.length) { ctx.fillStyle = "#00e5ff"; cc.forEach(([fx, fy]) => { const [x, y] = toDisp([fx, fy, 0, 0]); ctx.beginPath(); ctx.arc(x, y, 4, 0, 6.3); ctx.fill(); }); }
  if (S.drag) { ctx.strokeStyle = "#00e5ff"; ctx.lineWidth = 1.5; const d = S.drag; ctx.strokeRect(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1), Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0)); }
  requestAnimationFrame(drawOverlay);
}

function setupDrawing() {
  const c = $("overlay");
  document.querySelectorAll("input[name=drawMode]").forEach((r) => r.onchange = () => { S.drawMode = r.value; S.chartPts = []; });
  const pos = (e) => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  c.addEventListener("pointerdown", (e) => {
    const [x, y] = pos(e);
    if (S.drawMode === "chart_corners") {
      S.chartPts.push(toFrame(x, y));
      S.boxes.chart_corners = S.chartPts.slice();
      if (S.chartPts.length === 4) { pushBoxes(); S.chartPts = []; }
      return;
    }
    S.drag = { x0: x, y0: y, x1: x, y1: y }; c.setPointerCapture(e.pointerId);
  });
  c.addEventListener("pointermove", (e) => { if (S.drag) { const [x, y] = pos(e); S.drag.x1 = x; S.drag.y1 = y; } });
  c.addEventListener("pointerup", () => {
    const d = S.drag; S.drag = null;
    if (!d || Math.abs(d.x1 - d.x0) < 8 || Math.abs(d.y1 - d.y0) < 8) return;
    const a = toFrame(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1)), b = toFrame(Math.max(d.x0, d.x1), Math.max(d.y0, d.y1));
    const box = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.abs(b[0] - a[0]), Math.abs(b[1] - a[1])];
    S.boxes[S.drawMode] = box;
    S.prevThumb = null;
    pushBoxes();
  });
}

// ------------------------------------------------------------------ knobs
function readKnobs() {
  return { method: $("kMethod").value, hr_window_s: Number($("kHrWin").value), motion_warn: Number($("kWarn").value),
           motion_abstain: Number($("kAbst").value), update_s: Number($("kUpd").value), gate: $("kGate").checked,
           show_hrv: $("kHrv").checked, show_beats: $("kBeats").checked, require_still_for_hrv: $("kStill").checked };
}
function setupKnobs(k) {
  $("kMethod").value = k.method; $("kHrWin").value = k.hr_window_s; $("kWarn").value = k.motion_warn; $("kAbst").value = k.motion_abstain;
  $("kUpd").value = k.update_s; $("kGate").checked = k.gate; $("kHrv").checked = k.show_hrv; $("kBeats").checked = k.show_beats;
  $("kStill").checked = k.require_still_for_hrv;
  document.querySelectorAll(".knobs input, .knobs select").forEach((el) => el.onchange = async () => {
    const body = readKnobs();
    if (S.mode === "record") await api("/api/v1/lab/knobs", body);
    else if (S.sid) await api(`/api/v1/session/${S.sid}/knobs`, body);
  });
}

async function init() {
  S.info = await api("/api/v1/info");
  S.mode = S.info.mode;
  S.boxes.face_box = S.info.face_box;
  $("modeTag").textContent = S.mode === "record" ? "lab recorder" : "live";
  $("dispS").textContent = S.info.display_s;
  document.querySelectorAll(".labOnly").forEach((el) => el.hidden = S.mode !== "record");
  document.querySelectorAll(".browserOnly").forEach((el) => el.hidden = S.mode === "record");
  $("video").hidden = S.mode === "record";
  setupKnobs(S.info.knobs);
  const m = S.info.model;
  $("modelInfo").textContent = m.loaded ? `SAMA model ${m.version}, trained on ${(m.trained_on || []).join(", ")} (${m.provenance}); gate ${m.gate_type}.`
                                        : "No SAMA model loaded: POS/CHROM only (train one with `python -m sama train`).";
  setupDrawing();
  requestAnimationFrame(drawOverlay);
  if (S.mode === "record") setupLab();
  else { $("startBtn").onclick = startBrowser; $("stopBtn").onclick = stopBrowser; }
}
init();
