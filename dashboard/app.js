"use strict";
// Monsoon Branches dashboard. "What-if shock": calls the forecast server's /simulate twice (observed start, shocked
// start) and compares the two 10-day rollouts. "Forecast check": compares the unshocked rollout with what happened and
// with the grader's two baselines (repeat the last day, date average). No build step, no dependencies.
// Grid rows run south -> north.

const NC = 6, NLAT = 32, NLON = 36, CELLS = NLAT * NLON, STEPS = 10;
const LAT0 = 6.5, LON0 = 65.5;
const T2M = 0, TP = 1, U10 = 2, V10 = 3, MSL = 4, TCWV = 5;
const STD = [13.4636, 7.5533, 2.8990, 2.4270, 619.02, 20.494];   // stats.json: the scorer's shock-energy units
const DRY = 2.5;                                                  // mm/day: a dry day
const HEAVY = 10;                                                 // mm/day: the grader's heavy-rain threshold (rain > 10)
const SRC = {                                                     // forecast check: the model and the grader's baselines
  model: { n: "Our model", c: "--series-1" },
  persist: { n: "Repeat last day", c: "--series-2" },
  clim: { n: "Date average", c: "--series-3" },
};
const ERRWORDS = { tp: ["too dry", "too wet"], t2m: ["too cold", "too warm"], wind: ["too weak", "too strong"], msl: ["too low", "too high"], tcwv: ["too dry", "too moist"] };
const WLAT = (() => { const c = [...Array(NLAT)].map((_, i) => Math.cos((LAT0 + i) * Math.PI / 180)), m = c.reduce((a, b) => a + b) / NLAT; return c.map(v => v / m); })();

const KINDS = {
  wind: { min: -100, max: 100, step: 5, def: -40, fmt: v => `${sign(v)}${v}% of the current wind` },
  warm: { min: 0, max: 6, step: 0.5, def: 3, fmt: v => `+${v} K and +${(5 * v).toFixed(0)}% vapour` },
  moist: { min: -50, max: 50, step: 5, def: 15, fmt: v => `${sign(v)}${v}% of the current vapour` },
  pressure: { min: -10, max: 10, step: 1, def: -4, fmt: v => `${sign(v)}${v} hPa` },
};
const PRESETS = {
  monsoon_stall: { kind: "wind", strength: -40, box: [8, 20, 65, 75] },
  warm_ocean: { kind: "warm", strength: 3, box: [8, 20, 82, 94] },
};
const FIELDS = {
  tp: { label: "Rain", unit: "mm/day", seq: "blue", div: "brownblue", fixed: [0, 100], sqrt: true, ticks: [0, 2.5, 10, 25, 50, 100], floor: 0.5, dec: 1 },
  dry: { label: "Longest dry spell", unit: "days", seq: "orange", div: "bluebrown", fixed: [0, 10], ticks: [0, 2, 4, 6, 8, 10], floor: 1, dec: 0, whole: true },
  t2m: { label: "Temperature", unit: "°C", seq: "orange", div: "bluered", floor: 0.05, dec: 1 },
  wind: { label: "Wind speed", unit: "m/s", seq: "blue", div: "bluered", floor: 0.05, dec: 1, arrows: true },
  msl: { label: "Sea-level pressure", unit: "hPa", seq: "blue", div: "bluered", floor: 0.05, dec: 1 },
  tcwv: { label: "Water vapour", unit: "kg/m²", seq: "blue", div: "brownblue", floor: 0.1, dec: 1 },
};
const RAMPS = {
  light: {
    blue: ["#f5f8fc", "#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"],
    orange: ["#fdf6f0", "#fde0cc", "#fbc4a1", "#f8a273", "#f47230", "#d85a1c", "#b04512", "#83320c"],
    bluered: ["#1757ac", "#519ced", "#b6d5f8", "#efefef", "#f9c3c3", "#ec6b6b", "#b72a2b"],
    brownblue: ["#7a4a12", "#b77c2f", "#e6c99a", "#efefef", "#b6d5f8", "#519ced", "#1757ac"],
    bluebrown: ["#1757ac", "#519ced", "#b6d5f8", "#efefef", "#e6c99a", "#b77c2f", "#7a4a12"],
    mask: "#e9e9e6",
  },
  dark: {
    blue: ["#1b1f26", "#0e3d7e", "#1757ac", "#2370cd", "#338ced", "#6aabf1", "#9dc8f7", "#cce3fd"],
    orange: ["#221c18", "#6b2c0c", "#94400f", "#c4561a", "#e76624", "#f08a5d", "#f7b48e", "#fcd9c2"],
    bluered: ["#6aabf1", "#2f78cf", "#1b3f6e", "#383838", "#6e2526", "#c84243", "#f08a8a"],
    brownblue: ["#e0b169", "#a86d22", "#5c3d17", "#383838", "#1b3f6e", "#2f78cf", "#6aabf1"],
    bluebrown: ["#6aabf1", "#2f78cf", "#1b3f6e", "#383838", "#5c3d17", "#a86d22", "#e0b169"],
    mask: "#262626",
  },
};

const $ = s => document.querySelector(s);
const sign = v => (v > 0 ? "+" : "");
const state = {
  meta: null, coast: [], lsm: null, ex: null, kind: "wind", strength: -40, box: [8, 20, 65, 75], preset: "monsoon_stall",
  field: "tp", day: 1, baseMode: "forecast", base: null, shock: null, cell: null, busy: false, scales: null, drag: null,
  mode: "shock", src: "model", errMode: "error", scope: "event", area: null, check: null,
  layer: { shock: "change", check: "forecast" }, runId: 0, wind: true,
};
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)");
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const dparts = s => { const [y, m, d] = s.split("-").map(Number); return { y, m, d }; };
const shortDate = s => { const p = dparts(s); return `${p.d} ${MON[p.m - 1]}`; };
function fmtRange(a, b) {
  const p = dparts(a), q = dparts(b);
  return p.m === q.m ? `${p.d}–${q.d} ${MON[q.m - 1]} ${q.y}` : `${p.d} ${MON[p.m - 1]} – ${q.d} ${MON[q.m - 1]} ${q.y}`;
}

// ---------- data helpers ----------
function b64f32(s) {
  const bin = atob(s), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
}
function nest(a, shape, off = 0) {
  if (shape.length === 1) {
    const out = new Array(shape[0]);
    for (let k = 0; k < shape[0]; k++) out[k] = Math.round(a[off + k] * 1000) / 1000;
    return out;
  }
  const inner = shape.slice(1).reduce((x, y) => x * y, 1), out = new Array(shape[0]);
  for (let k = 0; k < shape[0]; k++) out[k] = nest(a, shape.slice(1), off + k * inner);
  return out;
}
function flatten(nested, n) {
  const out = new Float32Array(n); let k = 0;
  (function rec(x) { if (Array.isArray(x)) for (const y of x) rec(y); else out[k++] = x; })(nested);
  if (k !== n) throw new Error(`rollout has ${k} values, expected ${n}`);
  return out;
}
const inBox = (i, j, b) => { const la = LAT0 + i, lo = LON0 + j; return la >= b[0] && la <= b[1] && lo >= b[2] && lo <= b[3]; };
const chan = (arr, t, c) => arr.subarray((t * NC + c) * CELLS, (t * NC + c + 1) * CELLS);

function makePert(last, kind, s, box) {
  const d = new Float32Array(NC * CELLS);
  for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
    if (!inBox(i, j, box)) continue;
    const p = i * NLON + j;
    if (kind === "wind") { d[U10 * CELLS + p] = s / 100 * last[U10 * CELLS + p]; d[V10 * CELLS + p] = s / 100 * last[V10 * CELLS + p]; }
    else if (kind === "warm") { d[T2M * CELLS + p] = s; d[TCWV * CELLS + p] = 0.05 * s * last[TCWV * CELLS + p]; }
    else if (kind === "moist") d[TCWV * CELLS + p] = s / 100 * last[TCWV * CELLS + p];
    else if (kind === "pressure") d[MSL * CELLS + p] = s * 100;
  }
  return d;
}

// ---------- server ----------
function serverUrl() { return $("#server").value.replace(/\/+$/, ""); }
async function checkHealth() {
  const pill = $("#status");
  pill.className = "pill pill-wait"; pill.textContent = "connecting…";
  try {
    const r = await fetch(serverUrl() + "/health", { method: "POST" });
    const js = await r.json();
    if (js.status !== "ready") throw new Error(js.status);
    pill.className = "pill pill-ok"; pill.textContent = `${js.model_name} · ready`;
    return true;
  } catch (e) {
    pill.className = "pill pill-bad"; pill.textContent = "server offline";
    $("#runinfo").textContent = $("#checkinfo").textContent = "Start the server from the repo root: python serve.py --port 8000";
    return false;
  }
}
async function simulate(init, doy, pert) {
  const a = 2 * Math.PI * doy / 365;
  const body = { init_states: nest(init, [3, NC, NLAT, NLON]), doy_sincos: [Math.sin(a), Math.cos(a)], steps: STEPS,
    perturbation: pert ? nest(pert, [NC, NLAT, NLON]) : null };
  const r = await fetch(serverUrl() + "/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const js = await r.json();
  return { out: flatten(js.rollout, STEPS * NC * CELLS), ms: js.inference_ms };
}

async function run() {
  if (state.busy || !state.ex) return;
  state.busy = true; $("#run").disabled = true; $("#runinfo").textContent = "Running both forecasts…";
  const ex = state.ex, init = ex.win.subarray(0, 3 * NC * CELLS), last = ex.win.subarray(2 * NC * CELLS, 3 * NC * CELLS);
  const pert = makePert(last, state.kind, state.strength, state.box);
  const t0 = performance.now();
  try {
    const [b, s] = await Promise.all([simulate(init, ex.doy, null), simulate(init, ex.doy, pert)]);
    state.base = b.out; state.shock = s.out; state.runId++;
    const wall = (performance.now() - t0) / 1000;
    $("#runinfo").textContent = `2 model runs · ${wall.toFixed(2)} s round trip · ${(b.ms + s.ms).toFixed(0)} ms on the server`;
    $("#checkinfo").textContent = `Forecast from the live model · ${(b.ms / 1000).toFixed(2)} s on the server`;
    state.scales = null; render();
  } catch (e) {
    $("#runinfo").textContent = $("#checkinfo").textContent = `Forecast failed (${e.message}). Is the server running?`;
    checkHealth();
  } finally {
    state.busy = false; $("#run").disabled = false;
  }
}

// ---------- derived fields ----------
function observed() { return state.ex.win.subarray(3 * NC * CELLS); }
function fieldDay(src, field, t) {
  const out = new Float32Array(CELLS);
  if (field === "tp") out.set(chan(src, t, TP));
  else if (field === "t2m") { const a = chan(src, t, T2M); for (let p = 0; p < CELLS; p++) out[p] = a[p] - 273.15; }
  else if (field === "wind") { const u = chan(src, t, U10), v = chan(src, t, V10); for (let p = 0; p < CELLS; p++) out[p] = Math.hypot(u[p], v[p]); }
  else if (field === "msl") { const a = chan(src, t, MSL); for (let p = 0; p < CELLS; p++) out[p] = a[p] / 100; }
  else if (field === "tcwv") out.set(chan(src, t, TCWV));
  else if (field === "dry") out.set(drySpell(src));
  return out;
}
function longestRun(series) { let best = 0, cur = 0; for (const x of series) { cur = x < DRY ? cur + 1 : 0; best = Math.max(best, cur); } return best; }
function drySpell(src) {
  const out = new Float32Array(CELLS);
  for (let p = 0; p < CELLS; p++) {
    if (state.lsm[p] < 0.5) { out[p] = NaN; continue; }
    const s = []; for (let t = 0; t < STEPS; t++) s.push(src[(t * NC + TP) * CELLS + p]);
    out[p] = longestRun(s);
  }
  return out;
}
function quantile(arr, q) {
  const v = Array.from(arr).filter(Number.isFinite).sort((a, b) => a - b);
  return v.length ? v[Math.min(v.length - 1, Math.floor(q * v.length))] : 0;
}
function derive() {
  // colour scales are fixed across the 10 days so the slider compares like with like
  const f = FIELDS[state.field], days = f.whole ? [0] : [...Array(STEPS).keys()];
  let seq;
  if (f.fixed) seq = f.fixed;
  else {
    const all = [];
    for (const t of days) for (const src of [state.base, state.shock, observed()]) all.push(...fieldDay(src, state.field, t));
    seq = [quantile(all, 0.01), quantile(all, 0.99)];
    if (state.field === "wind") seq[0] = 0;
  }
  const diffs = [];
  for (const t of days) { const a = fieldDay(state.base, state.field, t), b = fieldDay(state.shock, state.field, t); for (let p = 0; p < CELLS; p++) diffs.push(Math.abs(b[p] - a[p])); }
  let L = Math.max(quantile(diffs, 0.985), f.floor);
  if (f.whole) L = Math.max(1, Math.ceil(L));
  state.scales = { seq, L };
}

// ---------- colour ----------
function theme() {
  const t = document.documentElement.dataset.theme;
  return t ? t : (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
const hex = h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
function ramp(stops, x) {
  x = Math.min(1, Math.max(0, x));
  const s = x * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(s)), f = s - i;
  const a = hex(stops[i]), b = hex(stops[i + 1]);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function seqPos(v) {
  const f = FIELDS[state.field], [lo, hi] = state.scales.seq;
  if (f.sqrt) return Math.sqrt(Math.max(0, v) / hi);
  return (v - lo) / Math.max(1e-9, hi - lo);
}
function seqColor(v) { return ramp(RAMPS[theme()][FIELDS[state.field].seq], seqPos(v)); }
function divColor(v) { return ramp(RAMPS[theme()][FIELDS[state.field].div], 0.5 + 0.5 * v / state.scales.L); }

// ---------- maps ----------
const panels = {};
function setupPanels() {
  for (const fig of document.querySelectorAll(".map")) {
    const name = fig.dataset.panel, canvas = fig.querySelector("canvas");
    panels[name] = { canvas, fig };
    canvas.addEventListener("pointerdown", e => onDown(e, canvas));
    canvas.addEventListener("pointermove", e => onMove(e, canvas, name));
    canvas.addEventListener("pointerup", e => onUp(e, canvas));
    canvas.addEventListener("pointerleave", () => { $("#tip").hidden = true; });
  }
  new ResizeObserver(() => render()).observe($(".maps"));
}
function cellAt(e, canvas) {
  const r = canvas.getBoundingClientRect();
  const j = Math.floor((e.clientX - r.left) / r.width * NLON), i = NLAT - 1 - Math.floor((e.clientY - r.top) / r.height * NLAT);
  return (i >= 0 && i < NLAT && j >= 0 && j < NLON) ? [i, j] : null;
}
function onDown(e, canvas) { const c = cellAt(e, canvas); if (!c) return; state.drag = { a: c, b: c }; canvas.setPointerCapture(e.pointerId); }
function onMove(e, canvas, name) {
  const c = cellAt(e, canvas);
  if (state.drag && c) { state.drag.b = c; render(); }
  showTip(e, c, name);
}
function onUp(e, canvas) {
  const d = state.drag; state.drag = null;
  if (!d) return;
  const [i0, i1] = [Math.min(d.a[0], d.b[0]), Math.max(d.a[0], d.b[0])], [j0, j1] = [Math.min(d.a[1], d.b[1]), Math.max(d.a[1], d.b[1])];
  const big = i1 - i0 + j1 - j0 >= 2, deg = [LAT0 + i0 - 0.5, LAT0 + i1 + 0.5, LON0 + j0 - 0.5, LON0 + j1 + 0.5];
  if (state.mode === "check") { if (big) state.area = deg; else { state.cell = d.a; state.area = null; } render(); return; }
  if (big) { state.box = deg; state.preset = null; syncControls(); run(); }
  else { state.cell = d.a; render(); }
}
function showTip(e, c, name) {
  const tip = $("#tip");
  if (state.mode === "check") { checkTip(e, c, tip); return; }
  if (!c || !state.base) { tip.hidden = true; return; }
  const f = FIELDS[state.field], t = state.day - 1, p = c[0] * NLON + c[1];
  const a = fieldDay(baseSource(), state.field, t)[p], b = fieldDay(state.shock, state.field, t)[p], a0 = fieldDay(state.base, state.field, t)[p];
  const fmt = v => Number.isFinite(v) ? v.toFixed(f.dec) : "–";
  const lab = state.baseMode === "observed" ? "Observed" : "Baseline";
  tip.innerHTML = `<b>${(LAT0 + c[0]).toFixed(1)}°N ${(LON0 + c[1]).toFixed(1)}°E</b><br>${lab}: ${fmt(a)} ${f.unit}<br>Shocked: ${fmt(b)} ${f.unit}<br>Change vs baseline: ${Number.isFinite(b - a0) ? sign(b - a0) : ""}${fmt(b - a0)} ${f.unit}`;
  tip.hidden = false;
  tip.style.left = `${Math.min(innerWidth - 250, e.clientX + 14)}px`; tip.style.top = `${e.clientY + 14}px`;
}
function baseSource() { return state.baseMode === "observed" ? observed() : state.base; }

function drawPanel(name, values, colorFn, uv) {
  const { canvas } = panels[name], wrap = canvas.parentElement;
  const dpr = devicePixelRatio || 1, W = Math.round(wrap.clientWidth * dpr), H = Math.round(wrap.clientHeight * dpr);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext("2d"), cs = getComputedStyle(document.documentElement);
  const img = new ImageData(NLON, NLAT), mask = hex(RAMPS[theme()].mask);
  for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) {
    const v = values[i * NLON + j], col = Number.isFinite(v) ? colorFn(v) : mask, k = ((NLAT - 1 - i) * NLON + j) * 4;
    img.data[k] = col[0]; img.data[k + 1] = col[1]; img.data[k + 2] = col[2]; img.data[k + 3] = 255;
  }
  const off = new OffscreenCanvas(NLON, NLAT); off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false; ctx.clearRect(0, 0, W, H); ctx.drawImage(off, 0, 0, W, H);
  const X = lon => (lon - 65) / 36 * W, Y = lat => (38 - lat) / 32 * H, cw = W / NLON, ch = H / NLAT;
  // coastline
  ctx.strokeStyle = cs.getPropertyValue("--coast"); ctx.lineWidth = 1 * dpr; ctx.lineJoin = "round";
  for (const line of state.coast) { ctx.beginPath(); line.forEach(([lo, la], k) => (k ? ctx.lineTo(X(lo), Y(la)) : ctx.moveTo(X(lo), Y(la)))); ctx.stroke(); }
  // wind arrows
  if (uv) {
    ctx.strokeStyle = cs.getPropertyValue("--text"); ctx.globalAlpha = 0.55; ctx.lineWidth = 1 * dpr;
    for (let i = 1; i < NLAT; i += 3) for (let j = 1; j < NLON; j += 3) {
      const p = i * NLON + j, u = uv[0][p], v = uv[1][p], sp = Math.hypot(u, v);
      if (sp < 0.3) continue;
      const len = Math.min(1.4, sp / 8) * 3 * cw * 0.5, x = (j + 0.5) * cw, y = (NLAT - 1 - i + 0.5) * ch, dx = u / sp * len, dy = -v / sp * len;
      ctx.beginPath(); ctx.moveTo(x - dx / 2, y - dy / 2); ctx.lineTo(x + dx / 2, y + dy / 2);
      const ang = Math.atan2(dy, dx), hl = 3 * dpr;
      ctx.lineTo(x + dx / 2 - hl * Math.cos(ang - 0.5), y + dy / 2 - hl * Math.sin(ang - 0.5));
      ctx.moveTo(x + dx / 2, y + dy / 2); ctx.lineTo(x + dx / 2 - hl * Math.cos(ang + 0.5), y + dy / 2 - hl * Math.sin(ang + 0.5));
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  // shock box (or the box being drawn)
  let bi;
  if (state.drag) { const d = state.drag; bi = [Math.min(d.a[0], d.b[0]), Math.max(d.a[0], d.b[0]), Math.min(d.a[1], d.b[1]), Math.max(d.a[1], d.b[1])]; }
  else bi = state.mode === "check" ? (state.area ? boxCells(state.area) : null) : boxCells(state.box);
  if (bi && bi[1] >= bi[0] && bi[3] >= bi[2]) {
    ctx.strokeStyle = cs.getPropertyValue("--box"); ctx.lineWidth = 2 * dpr; ctx.setLineDash(state.drag ? [4 * dpr, 3 * dpr] : []);
    ctx.strokeRect(bi[2] * cw, (NLAT - 1 - bi[1]) * ch, (bi[3] - bi[2] + 1) * cw, (bi[1] - bi[0] + 1) * ch); ctx.setLineDash([]);
  }
  if (state.cell && !(state.mode === "check" && state.area)) {
    const [i, j] = state.cell;
    ctx.strokeStyle = cs.getPropertyValue("--surface"); ctx.lineWidth = 3 * dpr; ctx.strokeRect(j * cw, (NLAT - 1 - i) * ch, cw, ch);
    ctx.strokeStyle = cs.getPropertyValue("--sel"); ctx.lineWidth = 1.5 * dpr; ctx.strokeRect(j * cw, (NLAT - 1 - i) * ch, cw, ch);
  }
}
function boxCells(b) {
  const i0 = Math.max(0, Math.ceil(b[0] - LAT0)), i1 = Math.min(NLAT - 1, Math.floor(b[1] - LAT0));
  const j0 = Math.max(0, Math.ceil(b[2] - LON0)), j1 = Math.min(NLON - 1, Math.floor(b[3] - LON0));
  return [i0, i1, j0, j1];
}

function legend(el, title, stops, ticks, pos, fmt) {
  const grad = stops.map((s, k) => `${s} ${(100 * k / (stops.length - 1)).toFixed(0)}%`).join(",");
  el.innerHTML = `<div class="title">${title}</div><div class="bar" style="background:linear-gradient(90deg,${grad})"></div>` +
    `<div class="ticks" style="position:relative;height:14px">${ticks.map(t => `<span style="position:absolute;left:${(100 * pos(t)).toFixed(1)}%;transform:translateX(-${pos(t) > 0.98 ? 100 : pos(t) < 0.02 ? 0 : 50}%)">${fmt(t)}</span>`).join("")}</div>`;
}

function render() {
  if (!state.ex) return;
  renderMain(); renderDays(); update3D();
}
function renderMain() {
  const f = FIELDS[state.field], t = state.day - 1;
  if (state.mode === "check") { renderCheck(); return; }
  if (!state.base) { renderCell(); return; }
  if (!state.scales) derive();
  const a = fieldDay(baseSource(), state.field, t), b = fieldDay(state.shock, state.field, t), a0 = fieldDay(state.base, state.field, t);
  const d = new Float32Array(CELLS); for (let p = 0; p < CELLS; p++) d[p] = b[p] - a0[p];
  const uv = f.arrows ? [src => [chan(src, t, U10), chan(src, t, V10)]].map(g => [g(baseSource()), g(state.shock)])[0] : null;
  drawPanel("base", a, seqColor, uv && uv[0]);
  drawPanel("shock", b, seqColor, uv && uv[1]);
  drawPanel("diff", d, divColor, null);
  const R = RAMPS[theme()], [lo, hi] = state.scales.seq;
  const ticks = f.ticks || niceTicks(lo, hi);
  legend($("#legend-seq"), `${f.label} (${f.unit}), shared by both forecasts`, R[f.seq], ticks, v => Math.min(1, Math.max(0, seqPos(v))), v => fmtNum(v));
  const L = state.scales.L, dl = f.div === "bluebrown" ? ["shorter", "longer"] : f.div === "brownblue" ? ["drier", "wetter"] : ["lower", "higher"];
  legend($("#legend-div"), `Change (${f.unit}): ${dl[0]} ← 0 → ${dl[1]}`, R[f.div], [-L, 0, L], v => 0.5 + 0.5 * v / L, v => `${sign(v)}${fmtNum(v)}`);
  renderCell(); renderSummary();
}
function fmtNum(v) { const a = Math.abs(v); return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(0) : a >= 1 ? (+v.toFixed(1)).toString() : (+v.toFixed(2)).toString(); }
function niceTicks(lo, hi, n = 4) {
  const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw)), step = [1, 2, 2.5, 5, 10].map(x => x * mag).find(x => x >= raw);
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

// ---------- cell chart ----------
function series(src, p) { const s = []; for (let t = 0; t < STEPS; t++) s.push(src[(t * NC + TP) * CELLS + p]); return s; }
function renderCell() {
  const el = $("#chart");
  if (!state.cell || !state.base) { el.innerHTML = ""; $("#cell-sub").textContent = state.base ? "Click a cell to compare its 10-day rainfall." : ""; return; }
  const [i, j] = state.cell, p = i * NLON + j, land = state.lsm[p] >= 0.5;
  const sb = series(state.base, p), ss = series(state.shock, p), so = series(observed(), p);
  const db = longestRun(sb), ds = longestRun(ss), tb = sb.reduce((x, y) => x + y, 0), ts = ss.reduce((x, y) => x + y, 0);
  $("#cell-title").textContent = `${(LAT0 + i).toFixed(1)}°N ${(LON0 + j).toFixed(1)}°E · ${land ? "land" : "sea"}`;
  $("#cell-sub").innerHTML = land
    ? `Longest dry spell (&lt; ${DRY} mm/day): ${db} → ${ds} days <b>(${sign(ds - db)}${ds - db} dry days)</b> · 10-day rain ${tb.toFixed(0)} → ${ts.toFixed(0)} mm`
    : `10-day rain ${tb.toFixed(0)} → ${ts.toFixed(0)} mm · dry-spell indicator is computed for land cells`;
  const W = el.clientWidth || 600, H = el.clientHeight || 220, m = { l: 36, r: 86, t: 10, b: 24 };
  const ymax = Math.max(10, ...sb, ...ss, ...so) * 1.1, x = k => m.l + (k / (STEPS - 1)) * (W - m.l - m.r), y = v => H - m.b - (v / ymax) * (H - m.t - m.b);
  const path = s => s.map((v, k) => `${k ? "L" : "M"}${x(k).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const yt = niceTicks(0, ymax);
  const day = state.day - 1, cs = getComputedStyle(document.documentElement);
  const lines = [
    { s: ss, c: cs.getPropertyValue("--series-2"), n: "Shocked", dash: "" },
    { s: sb, c: cs.getPropertyValue("--series-1"), n: "Baseline", dash: "" },
    { s: so, c: cs.getPropertyValue("--observed"), n: "Observed", dash: "4 3" },
  ];
  // direct labels at the line ends, nudged apart
  const ends = lines.map(l => ({ n: l.n, y: y(l.s[STEPS - 1]) })).sort((a, b) => a.y - b.y);
  for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 13) ends[k].y = ends[k - 1].y + 13;
  const anim = drawIn(el, [state.ex.id, state.runId, i, j].join("|"));
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="10-day rainfall: baseline, shocked and observed">
    ${clipDef(el, W, H, anim)}
    <g class="grid">${yt.map(v => `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>`).join("")}</g>
    <rect x="${x(day) - 6}" y="${m.t}" width="12" height="${H - m.t - m.b}" fill="var(--surface-2)"/>
    <line class="ref" x1="${m.l}" x2="${W - m.r}" y1="${y(DRY)}" y2="${y(DRY)}"/>
    <text class="axis" x="${m.l + 4}" y="${y(DRY) - 4}" font-size="10" fill="var(--text-3)">dry day &lt; ${DRY}</text>
    <g class="axis">${yt.map(v => `<text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`).join("")}
      ${[...Array(STEPS).keys()].map(k => `<text x="${x(k)}" y="${H - 6}" text-anchor="middle">${k + 1}</text>`).join("")}</g>
    <g clip-path="url(#clip-${el.id})">${lines.map(l => `<path d="${path(l.s)}" fill="none" stroke="${l.c}" stroke-width="2" stroke-dasharray="${l.dash}" stroke-linejoin="round"/>`).join("")}</g>
    ${ends.map(e => `<text class="lbl" x="${W - m.r + 8}" y="${e.y + 4}">${e.n}</text>`).join("")}
    <g id="hover" visibility="hidden"><line y1="${m.t}" y2="${H - m.b}" stroke="var(--text-3)"/></g>
    <rect id="hit" x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>
  </svg>`;
  startDrawIn(el, anim);
  const hit = el.querySelector("#hit"), hov = el.querySelector("#hover"), tip = $("#tip");
  hit.addEventListener("pointermove", e => {
    const r = hit.getBoundingClientRect(), k = Math.round((e.clientX - r.left) / r.width * (STEPS - 1));
    hov.setAttribute("visibility", "visible"); hov.firstElementChild.setAttribute("x1", x(k)); hov.firstElementChild.setAttribute("x2", x(k));
    tip.innerHTML = `<b>Day ${k + 1} · ${state.ex.dates[3 + k]}</b><br>Baseline ${sb[k].toFixed(1)} · Shocked ${ss[k].toFixed(1)} · Observed ${so[k].toFixed(1)} mm/day`;
    tip.hidden = false; tip.style.left = `${Math.min(innerWidth - 250, e.clientX + 14)}px`; tip.style.top = `${e.clientY + 14}px`;
  });
  hit.addEventListener("pointerleave", () => { hov.setAttribute("visibility", "hidden"); tip.hidden = true; });
  $("#cell-table").innerHTML = `<caption>Rainfall (mm/day) at the selected cell</caption><tr><th>Day</th><th>Baseline</th><th>Shocked</th><th>Observed</th></tr>` +
    sb.map((v, k) => `<tr><td>${k + 1}</td><td>${v.toFixed(1)}</td><td>${ss[k].toFixed(1)}</td><td>${so[k].toFixed(1)}</td></tr>`).join("");
}

// ---------- summary: the scorer's view of the response ----------
function nearMask(box) {
  const [i0, i1, j0, j1] = boxCells(box); let m = new Uint8Array(CELLS);
  for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) m[i * NLON + j] = 1;
  for (let it = 0; it < 3; it++) {
    const o = m.slice();
    for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) if (m[i * NLON + j]) {
      if (i > 0) o[(i - 1) * NLON + j] = 1; if (i < NLAT - 1) o[(i + 1) * NLON + j] = 1;
      if (j > 0) o[i * NLON + j - 1] = 1; if (j < NLON - 1) o[i * NLON + j + 1] = 1;
    }
    m = o;
  }
  return m;
}
function renderSummary() {
  const near = nearMask(state.box), share = [];
  for (let t = 0; t < STEPS; t++) {
    let tot = 0, inside = 0;
    for (let c = 0; c < NC; c++) { const a = chan(state.base, t, c), b = chan(state.shock, t, c); for (let p = 0; p < CELLS; p++) { const e = ((b[p] - a[p]) / STD[c]) ** 2; tot += e; if (near[p]) inside += e; } }
    share.push(tot > 0 ? inside / tot : NaN);
  }
  let landN = 0, dRain = 0, wetter = 0, drier = 0, longer = 0, shorter = 0, maxLonger = 0;
  const db = drySpell(state.base), ds = drySpell(state.shock);
  for (let p = 0; p < CELLS; p++) {
    if (state.lsm[p] < 0.5) continue;
    landN++;
    let tb = 0, ts = 0; for (let t = 0; t < STEPS; t++) { tb += state.base[(t * NC + TP) * CELLS + p]; ts += state.shock[(t * NC + TP) * CELLS + p]; }
    dRain += ts - tb; if (ts - tb > 5) wetter++; if (ts - tb < -5) drier++;
    const k = ds[p] - db[p]; if (k > 0) { longer++; maxLonger = Math.max(maxLonger, k); } if (k < 0) shorter++;
  }
  const pct = v => Number.isFinite(v) ? `${(100 * v).toFixed(0)}%` : "–";
  const rows = [
    ["Response within 3 cells of the box, day 1", pct(share[0])],
    ["… on day 4 / day 8", `${pct(share[3])} / ${pct(share[7])}`],
    ["10-day rain over land, mean change", `${sign(dRain / landN)}${(dRain / landN).toFixed(2)} mm`],
    ["Land cells > 5 mm wetter / drier", `${wetter} / ${drier}`],
    ["Land cells with a longer dry spell", `${longer}${longer ? ` (up to +${maxLonger} d)` : ""}`],
    ["Land cells with a shorter dry spell", `${shorter}`],
  ];
  setCounting($("#summary"), "dd:not(.note)", rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("") +
    `<dd class="note">Response share uses the scorer's energy: the sum over all six fields of (change ÷ field std)². A falling share means the effect is spreading out from the box.</dd>`);
}

// ---------- forecast check: the model against what happened and the grader's two baselines ----------
function persistence(ex) {
  const last = ex.win.subarray(2 * NC * CELLS, 3 * NC * CELLS), out = new Float32Array(STEPS * NC * CELLS);
  for (let t = 0; t < STEPS; t++) out.set(last, t * NC * CELLS);
  return out;
}
function srcArr(name) { return name === "model" ? state.base : name === "persist" ? state.ex.persist : state.ex.clim; }
function sources() { return Object.keys(SRC).filter(k => srcArr(k)); }
const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function setMode(m) {
  state.mode = m; document.body.dataset.mode = m; $("#tip").hidden = true;
  if (m === "check" && state.field === "dry") state.field = "tp";
  state.scales = null; syncControls(); render();
  try { history.replaceState(null, "", m === "check" ? "#check" : location.pathname + location.search); } catch (e) { /* file:// or sandboxed */ }
}
function rmseDay(src, field, t) {      // latitude-weighted RMSE over the whole map, as the grader's rmse_lat
  const a = fieldDay(src, field, t), o = fieldDay(observed(), field, t); let s = 0;
  for (let i = 0; i < NLAT; i++) for (let j = 0; j < NLON; j++) { const p = i * NLON + j; s += WLAT[i] * (a[p] - o[p]) ** 2; }
  return Math.sqrt(s / CELLS);
}
function heavyDay(src, t) {           // 1 hit, 2 missed, 3 false alarm, 0 neither (rain > 10 mm/day)
  const a = chan(src, t, TP), o = chan(observed(), t, TP), cls = new Uint8Array(CELLS); let h = 0, m = 0, f = 0;
  for (let p = 0; p < CELLS; p++) {
    const fc = a[p] > HEAVY, ob = o[p] > HEAVY;
    if (fc && ob) { cls[p] = 1; h++; } else if (ob) { cls[p] = 2; m++; } else if (fc) { cls[p] = 3; f++; }
  }
  return { cls, h, m, f, csi: h + m + f ? h / (h + m + f) : NaN };
}
function csiAll(src) {
  let h = 0, m = 0, f = 0;
  for (let t = 0; t < STEPS; t++) { const d = heavyDay(src, t); h += d.h; m += d.m; f += d.f; }
  return h + m + f ? h / (h + m + f) : NaN;
}
function deriveCheck() {
  const f = FIELDS[state.field], srcs = sources().map(srcArr), obs = observed();
  let seq;
  if (f.fixed) seq = f.fixed;
  else {
    const all = [];
    for (let t = 0; t < STEPS; t++) for (const src of [obs, ...srcs]) all.push(...fieldDay(src, state.field, t));
    seq = [quantile(all, 0.01), quantile(all, 0.99)];
    if (state.field === "wind") seq[0] = 0;
  }
  const errs = [];
  for (let t = 0; t < STEPS; t++) {
    const o = fieldDay(obs, state.field, t);
    for (const src of srcs) { const a = fieldDay(src, state.field, t); for (let p = 0; p < CELLS; p++) errs.push(Math.abs(a[p] - o[p])); }
  }
  state.scales = { seq, L: Math.max(quantile(errs, 0.985), f.floor) };
}
function ranges(days) {               // [1,2,3,4,6,8] -> "1–4, 6, 8"
  const out = [];
  for (let k = 0; k < days.length; k++) {
    let e = k; while (e + 1 < days.length && days[e + 1] === days[e] + 1) e++;
    out.push(e > k ? `${days[k]}–${days[e]}` : `${days[k]}`); k = e;
  }
  return out.length ? out.join(", ") : "none";
}
const lowestDays = (m, p, c) => m.map((v, k) => (v < Math.min(p[k], c ? c[k] : Infinity) ? k + 1 : 0)).filter(Boolean);
const fmtF = (v, dec) => (Number.isFinite(v) ? v.toFixed(dec) : "–");

function renderCheck() {
  const f = FIELDS[state.field], t = state.day - 1, obs = observed(), src = srcArr(state.src);
  if (!state.scales) deriveCheck();
  const heavy = state.field === "tp" && state.errMode === "heavy";
  const blank = new Float32Array(CELLS).fill(NaN), uv = s => (f.arrows && s ? [chan(s, t, U10), chan(s, t, V10)] : null);
  drawPanel("base", fieldDay(obs, state.field, t), seqColor, uv(obs));
  drawPanel("shock", src ? fieldDay(src, state.field, t) : blank, seqColor, uv(src));
  $("#errcap").textContent = heavy ? `Rain > ${HEAVY} mm/day` : state.field === "tp" ? "Forecast − observed" : "Error (forecast − observed)";
  const R = RAMPS[theme()], [lo, hi] = state.scales.seq;
  legend($("#legend-seq"), `${f.label} (${f.unit}), shared by observed and forecast`, R[f.seq], f.ticks || niceTicks(lo, hi),
    v => Math.min(1, Math.max(0, seqPos(v))), v => fmtNum(v));
  if (heavy && src) {
    const d = heavyDay(src, t), col = ["--none", "--series-1", "--series-2", "--series-3"].map(v => hex(cssVar(v)));
    drawPanel("diff", d.cls, v => col[v], null);
    const cats = [["--series-1", "Hit", d.h], ["--series-2", "Missed", d.m], ["--series-3", "False alarm", d.f]];
    $("#legend-div").innerHTML = `<div class="title">Heavy rain (&gt; ${HEAVY} mm/day), ${SRC[state.src].n.toLowerCase()} vs observed, day ${state.day}</div>
      <div class="cats">${cats.map(([c, n, k]) => `<span><i class="sq" style="background:var(${c})"></i>${n} ${k}</span>`).join("")}
      <span>CSI ${fmtF(d.csi, 2)}</span></div>`;
  } else {
    const d = new Float32Array(CELLS);
    if (src) { const a = fieldDay(src, state.field, t), o = fieldDay(obs, state.field, t); for (let p = 0; p < CELLS; p++) d[p] = a[p] - o[p]; } else d.fill(NaN);
    drawPanel("diff", d, divColor, null);
    const L = state.scales.L, w = ERRWORDS[state.field];
    legend($("#legend-div"), `Error (${f.unit}): forecast ${w[0]} ← 0 → ${w[1]}`, R[f.div], [-L, 0, L], v => 0.5 + 0.5 * v / L, v => `${sign(v)}${fmtNum(v)}`);
  }
  renderTiles(); renderErrChart(); renderPointChart();
}

function renderTiles() {
  const el = $("#tiles"), f = FIELDS[state.field];
  if (!state.base) {
    el.innerHTML = `<div class="tile" style="grid-column:1/-1"><div class="k">Waiting for the forecast server</div>
      <div class="s">The maps show the baselines meanwhile. Start the server from the repo root: python serve.py --port 8000</div></div>`;
    return;
  }
  const err = k => (srcArr(k) ? [...Array(STEPS).keys()].map(t => rmseDay(srcArr(k), state.field, t)) : null);
  const em = err("model"), ep = err("persist"), ec = err("clim");
  const best0 = Math.min(ep[0], ec ? ec[0] : Infinity), gain = 1 - em[0] / best0;
  const days = lowestDays(em, ep, ec);
  const cm = csiAll(state.base), cp = csiAll(state.ex.persist), cc = state.ex.clim ? csiAll(state.ex.clim) : NaN;
  const tiles = [
    [`${f.label} error on day 1, whole map`, `${fmtF(em[0], f.dec + 1)} ${f.unit}`,
      `<span class="${gain >= 0 ? "good" : "bad"}">${Math.abs(100 * gain).toFixed(0)}% ${gain >= 0 ? "lower" : "higher"}</span> than the better baseline (${fmtF(best0, f.dec + 1)})`],
    [`Days our model has the lowest ${f.label.toLowerCase()} error`, ranges(days), `of 10, this event, against both baselines`],
    [`Heavy-rain hit score (CSI), days 1–10`, Number.isFinite(cm) ? cm.toFixed(2) : "no heavy rain",
      `rain > ${HEAVY} mm/day, higher is better · repeat last day ${fmtF(cp, 2)} · date average ${fmtF(cc, 2)}`],
  ];
  if (state.check) {
    const T = state.check.test;
    tiles.push([`All ${T.windows} test windows, ${T.years.replace("-", "–")}`, `${(100 * T.mean_skill).toFixed(0)}% lower error`,
      `than the better baseline, averaged over 6 fields and days 1, 4, 8`]);
  }
  setCounting(el, ".v", tiles.map(([k, v, s]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join(""));
}

function key(el, series) {
  el.innerHTML = series.map(s => `<span><i class="sw" style="border-top:2px ${s.dash ? "dashed" : "solid"} var(${s.c})"></i>${s.n}</span>`).join("");
}
function lineChart(el, tableEl, o) {
  // o: { series: [{ n, c (css var), v: [10], dash }], unit, dec, zero, label }
  const W = el.clientWidth || 600, H = el.clientHeight || 230, m = { l: 44, r: 108, t: 10, b: 24 };
  const vals = o.series.flatMap(s => s.v).filter(Number.isFinite);
  if (!vals.length) { el.innerHTML = ""; return; }
  let lo = o.zero ? 0 : Math.min(...vals), hi = Math.max(...vals);
  if (o.zero) hi = Math.max(hi * 1.12, 1e-3);
  else { const pad = Math.max((hi - lo) * 0.12, 10 ** -o.dec); lo -= pad; hi += pad; }
  const x = k => m.l + (k / (STEPS - 1)) * (W - m.l - m.r), y = v => H - m.b - ((v - lo) / (hi - lo)) * (H - m.t - m.b);
  const yt = niceTicks(lo, hi).filter(v => v >= lo && v <= hi), day = state.day - 1;
  const path = s => s.v.map((v, k) => `${k ? "L" : "M"}${x(k).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const ends = o.series.map(s => ({ n: s.n, y: y(s.v[STEPS - 1]) })).sort((a, b) => a.y - b.y);
  for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 13) ends[k].y = ends[k - 1].y + 13;
  const over = ends.length ? ends[ends.length - 1].y - (H - m.b) : 0;
  if (over > 0) for (const e of ends) e.y -= over;
  const anim = drawIn(el, o.anim);
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.label}">
    ${clipDef(el, W, H, anim)}
    <g class="grid">${yt.map(v => `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>`).join("")}</g>
    <rect x="${x(day) - 6}" y="${m.t}" width="12" height="${H - m.t - m.b}" fill="var(--surface-2)"/>
    <g class="axis">${yt.map(v => `<text x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${fmtNum(v)}</text>`).join("")}
      ${[...Array(STEPS).keys()].map(k => `<text x="${x(k)}" y="${H - 6}" text-anchor="middle">${k + 1}</text>`).join("")}</g>
    <g clip-path="url(#clip-${el.id})">${o.series.map(s => `<path d="${path(s)}" fill="none" stroke="var(${s.c})" stroke-width="2" stroke-dasharray="${s.dash ? "4 3" : ""}" stroke-linejoin="round"/>`).join("")}</g>
    ${ends.map(e => `<text class="lbl" x="${W - m.r + 8}" y="${e.y + 4}">${e.n}</text>`).join("")}
    <g class="hover" visibility="hidden"><line y1="${m.t}" y2="${H - m.b}" stroke="var(--text-3)"/>
      ${o.series.map(s => `<circle r="4" fill="var(${s.c})" stroke="var(--surface)" stroke-width="2"/>`).join("")}</g>
    <rect class="hit" x="${m.l - 8}" y="${m.t}" width="${W - m.l - m.r + 16}" height="${H - m.t - m.b}" fill="transparent"/>
  </svg>`;
  startDrawIn(el, anim);
  const hit = el.querySelector(".hit"), hov = el.querySelector(".hover"), tip = $("#tip");
  hit.addEventListener("pointermove", e => {
    const r = hit.getBoundingClientRect(), k = Math.max(0, Math.min(STEPS - 1, Math.round((e.clientX - r.left - 8) / (r.width - 16) * (STEPS - 1))));
    hov.setAttribute("visibility", "visible");
    const ln = hov.querySelector("line"); ln.setAttribute("x1", x(k)); ln.setAttribute("x2", x(k));
    hov.querySelectorAll("circle").forEach((c, i) => { c.setAttribute("cx", x(k)); c.setAttribute("cy", y(o.series[i].v[k])); });
    tip.innerHTML = `<b>Day ${k + 1} · ${state.ex.dates[3 + k]}</b>` + o.series.map(s => `<br>${s.n}: ${fmtF(s.v[k], o.dec)} ${o.unit}`).join("");
    tip.hidden = false; tip.style.left = `${Math.min(innerWidth - 250, e.clientX + 14)}px`; tip.style.top = `${e.clientY + 14}px`;
  });
  hit.addEventListener("pointerleave", () => { hov.setAttribute("visibility", "hidden"); tip.hidden = true; });
  tableEl.innerHTML = `<caption>${o.label} (${o.unit})</caption><tr><th>Day</th>${o.series.map(s => `<th>${s.n}</th>`).join("")}</tr>` +
    [...Array(STEPS).keys()].map(k => `<tr><td>${k + 1}</td>${o.series.map(s => `<td>${fmtF(s.v[k], o.dec)}</td>`).join("")}</tr>`).join("");
}

function renderErrChart() {
  const f = FIELDS[state.field], el = $("#err-chart");
  let series, sub;
  if (state.scope === "test" && state.check) {
    const T = state.check.test, r = T.rmse[state.field];
    series = Object.keys(SRC).map(k => ({ n: SRC[k].n, c: SRC[k].c, v: r[k] }));
    sub = `Pooled over ${T.windows} forecasts started every 3rd day in ${T.years.replace("-", "–")}. Our model is lowest on days ${ranges(lowestDays(r.model, r.persist, r.clim))}.`;
    if (state.field === "tp") {
      const at = (a, k) => a[k].toFixed(2), b = k => Math.max(T.csi.persist[k], T.csi.clim[k]).toFixed(2);
      sub += ` Heavy-rain CSI on days 1 / 4 / 8: ${at(T.csi.model, 0)} / ${at(T.csi.model, 3)} / ${at(T.csi.model, 7)}, better baseline ${b(0)} / ${b(3)} / ${b(7)}.`;
    }
  } else {
    if (!state.base) { el.innerHTML = ""; $("#err-legend").innerHTML = ""; $("#err-sub").textContent = "Needs the forecast server."; return; }
    series = sources().map(k => ({ n: SRC[k].n, c: SRC[k].c, v: [...Array(STEPS).keys()].map(t => rmseDay(srcArr(k), state.field, t)) }));
    const v = Object.fromEntries(sources().map((k, i) => [k, series[i].v]));
    sub = `This event, whole map, latitude-weighted as the grader does it. Our model is lowest on days ${ranges(lowestDays(v.model, v.persist, v.clim))}.`;
  }
  $("#err-sub").textContent = sub;
  key($("#err-legend"), series);
  lineChart(el, $("#err-table"), { series, unit: f.unit, dec: f.dec + 1, zero: true, label: `${f.label} error (RMSE) by forecast day`,
    anim: [state.field, state.ex.id, state.runId, state.scope].join("|") });
}

function renderPointChart() {
  const f = FIELDS[state.field], el = $("#pt-chart");
  let cells, title;
  if (state.area) {
    const [i0, i1, j0, j1] = boxCells(state.area); cells = [];
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) cells.push(i * NLON + j);
    title = `Area ${LAT0 + i0}–${LAT0 + i1}°N, ${LON0 + j0}–${LON0 + j1}°E · ${cells.length} cells`;
  } else if (state.cell) {
    const [i, j] = state.cell, p = i * NLON + j; cells = [p];
    title = `${(LAT0 + i).toFixed(1)}°N ${(LON0 + j).toFixed(1)}°E · ${state.lsm[p] >= 0.5 ? "land" : "sea"}`;
  } else { $("#pt-title").textContent = "Pick a cell on any map"; el.innerHTML = ""; return; }
  const ser = src => [...Array(STEPS).keys()].map(t => { const a = fieldDay(src, state.field, t); return cells.reduce((s, p) => s + a[p], 0) / cells.length; });
  const obs = ser(observed());
  const series = [{ n: "Observed", c: "--observed", v: obs, dash: true }, ...sources().map(k => ({ n: SRC[k].n, c: SRC[k].c, v: ser(srcArr(k)) }))];
  $("#pt-title").textContent = `${title} · ${f.label}`;
  const name = s => s.n === "Observed" ? "observed" : s.n.toLowerCase();
  if (state.field === "tp") {
    const tot = s => s.v.reduce((a, b) => a + b, 0);
    $("#pt-sub").textContent = `10-day rain: ${series.map(s => `${name(s)} ${tot(s).toFixed(0)}`).join(" · ")} mm` + (state.area ? "" : ". Drag on a map to average over an area.");
  } else {
    const mae = s => s.v.reduce((a, v, k) => a + Math.abs(v - obs[k]), 0) / STEPS;
    $("#pt-sub").textContent = `Average miss over the 10 days: ${series.slice(1).map(s => `${name(s)} ${fmtF(mae(s), f.dec + 1)}`).join(" · ")} ${f.unit}`;
  }
  key($("#pt-legend"), series);
  lineChart(el, $("#pt-table"), { series, unit: f.unit, dec: f.dec + 1, zero: state.field === "tp", label: `${f.label}, ${title}`,
    anim: [state.field, state.ex.id, state.runId, title].join("|") });
}

function checkTip(e, c, tip) {
  if (!c || !state.ex) { tip.hidden = true; return; }
  const f = FIELDS[state.field], t = state.day - 1, p = c[0] * NLON + c[1], src = srcArr(state.src);
  const o = fieldDay(observed(), state.field, t)[p], a = src ? fieldDay(src, state.field, t)[p] : NaN, fmt = v => fmtF(v, f.dec);
  let last = `Error: ${Number.isFinite(a - o) ? sign(a - o) : ""}${fmt(a - o)} ${f.unit}`;
  if (state.field === "tp" && state.errMode === "heavy" && src) last = ["Neither above 10 mm/day", "Hit", "Missed", "False alarm"][heavyDay(src, t).cls[p]];
  tip.innerHTML = `<b>${(LAT0 + c[0]).toFixed(1)}°N ${(LON0 + c[1]).toFixed(1)}°E</b><br>Observed: ${fmt(o)} ${f.unit}<br>${SRC[state.src].n}: ${fmt(a)} ${f.unit}<br>${last}`;
  tip.hidden = false;
  tip.style.left = `${Math.min(innerWidth - 250, e.clientX + 14)}px`; tip.style.top = `${e.clientY + 14}px`;
}


// ---------- timeline ----------
let playTimer = null;
const STEP_MS = 1100;
function buildDays() {
  const el = $("#days");
  el.innerHTML = [...Array(STEPS).keys()].map(k => `<button data-day="${k + 1}" aria-label="Day ${k + 1}, ${state.ex.dates[3 + k]}"><b>${k + 1}</b><span>${shortDate(state.ex.dates[3 + k])}</span><i></i></button>`).join("");
  for (const b of el.querySelectorAll("button")) b.addEventListener("click", () => { stopPlay(); state.day = +b.dataset.day; render(); });
}
function renderDays() {
  const whole = !!FIELDS[state.field].whole;
  for (const b of $("#days").querySelectorAll("button")) { b.setAttribute("aria-pressed", String(!whole && +b.dataset.day === state.day)); b.disabled = whole; }
  $("#play").disabled = whole;
  if (whole) stopPlay();
}
function startPlay() {
  if (state.day >= STEPS) state.day = 1;
  const tl = $(".timeline"); tl.style.setProperty("--step", `${STEP_MS}ms`); tl.classList.add("playing");
  $("#play").setAttribute("aria-label", "Pause");
  render();
  playTimer = setInterval(() => { if (state.day >= STEPS) { stopPlay(); return; } state.day++; render(); }, STEP_MS);
}
function stopPlay() {
  if (playTimer) clearInterval(playTimer);
  playTimer = null; $(".timeline").classList.remove("playing"); $("#play").setAttribute("aria-label", "Play the 10 days");
}

// ---------- 3D relief view (globe.js) ----------
let view3d = null;
const k3 = { data: "", box: "", cell: "", theme: "", run: 0 };
function init3D(terrain) {
  const go = () => {
    if (view3d || !window.MB3D || !terrain) return;
    try {
      view3d = window.MB3D.createTerrainView($("#hero3d"), {
        terrain, coast: state.coast, nlat: NLAT, nlon: NLON,
        onHover: (c, e) => { if (!c) { $("#tip").hidden = true; return; } showTip(e, c, "3d"); },
        onPick: c => { state.cell = c; if (state.mode === "check") state.area = null; render(); },
      });
      render();
    } catch (err) { console.error(err); $("#hero-fallback").hidden = false; }
  };
  if (window.MB3D) go(); else window.addEventListener("mb3d-ready", go, { once: true });
  setTimeout(() => { if (!view3d) $("#hero-fallback").hidden = false; }, 5000);
}
const heroLayer = () => state.layer[state.mode];
function heroSpec() {
  const L = heroLayer();
  if (state.mode === "shock") {
    if (!state.base) return null;
    if (!state.scales) derive();
    const src = { baseline: state.base, shocked: state.shock, change: state.shock, observed: observed() }[L];
    const vals = L === "change"
      ? t => { const a = fieldDay(state.base, state.field, t), b = fieldDay(state.shock, state.field, t); for (let p = 0; p < CELLS; p++) b[p] -= a[p]; return b; }
      : t => fieldDay(src, state.field, t);
    return { vals, color: L === "change" ? divColor : seqColor, wind: src, div: L === "change" };
  }
  if (!state.scales) deriveCheck();
  const src = srcArr(state.src), obs = observed();
  if (L === "observed") return { vals: t => fieldDay(obs, state.field, t), color: seqColor, wind: obs };
  if (!src) return null;
  if (L === "forecast") return { vals: t => fieldDay(src, state.field, t), color: seqColor, wind: src };
  if (state.field === "tp" && state.errMode === "heavy") {
    const col = ["--none", "--series-1", "--series-2", "--series-3"].map(v => hex(cssVar(v)));
    return { vals: t => heavyDay(src, t).cls, color: v => col[v], wind: src, nearest: true, heavy: true };
  }
  return { vals: t => { const a = fieldDay(src, state.field, t), o = fieldDay(obs, state.field, t); for (let p = 0; p < CELLS; p++) a[p] -= o[p]; return a; }, color: divColor, wind: src, div: true };
}
function rgba(values, colorFn) {
  const out = new Uint8Array(CELLS * 4), mask = hex(RAMPS[theme()].mask);
  for (let p = 0; p < CELLS; p++) {
    const v = values[p], c = Number.isFinite(v) ? colorFn(v) : mask;
    out[4 * p] = c[0]; out[4 * p + 1] = c[1]; out[4 * p + 2] = c[2]; out[4 * p + 3] = 255;
  }
  return out;
}
function legend3D(spec) {
  const el = $("#legend-3d"), f = FIELDS[state.field], R = RAMPS[theme()], L = heroLayer();
  if (!spec) { el.innerHTML = `<div class="title">Waiting for the forecast server</div>`; return; }
  const names = { change: "change after the shock", shocked: "shocked forecast", baseline: "baseline forecast", observed: "observed",
    forecast: SRC[state.src].n.toLowerCase(), error: "forecast error" };
  if (spec.heavy) {
    el.innerHTML = `<div class="title">Heavy rain (&gt; ${HEAVY} mm/day), ${SRC[state.src].n.toLowerCase()} vs observed</div><div class="cats">` +
      [["--series-1", "Hit"], ["--series-2", "Missed"], ["--series-3", "False alarm"]].map(([c, n]) => `<span><i class="sq" style="background:var(${c})"></i>${n}</span>`).join("") + `</div>`;
    return;
  }
  if (spec.div) {
    const Lr = state.scales.L, w = state.mode === "check" ? ERRWORDS[state.field].map(x => `forecast ${x}`)
      : f.div === "bluebrown" ? ["shorter", "longer"] : f.div === "brownblue" ? ["drier", "wetter"] : ["lower", "higher"];
    legend(el, `${f.label}, ${names[L]} (${f.unit}): ${w[0]} ← 0 → ${w[1]}`, R[f.div], [-Lr, 0, Lr], v => 0.5 + 0.5 * v / Lr, v => `${sign(v)}${fmtNum(v)}`);
    return;
  }
  const [lo, hi] = state.scales.seq;
  legend(el, `${f.label}, ${names[L]} (${f.unit})`, R[f.seq], f.ticks || niceTicks(lo, hi), v => Math.min(1, Math.max(0, seqPos(v))), v => fmtNum(v));
}
function update3D() {
  if (!view3d || !state.ex) return;
  const th = theme();
  if (th !== k3.theme) { view3d.setTheme(th); k3.theme = th; k3.data = ""; }
  const spec = heroSpec(), f = FIELDS[state.field];
  const key = [state.mode, heroLayer(), state.field, state.ex.id, state.runId, state.src, state.errMode, th, JSON.stringify(state.scales)].join("|");
  if (!spec) { if (k3.data !== "none") { view3d.clear(); k3.data = "none"; legend3D(null); } }
  else if (key !== k3.data) {
    const frames = (f.whole ? [0] : [...Array(STEPS).keys()]).map(t => rgba(spec.vals(t), spec.color));
    const winds = [...Array(STEPS).keys()].map(t => ({ u: chan(spec.wind, t, U10), v: chan(spec.wind, t, V10) }));
    view3d.setData({ key, frames, winds, nearest: !!spec.nearest });
    k3.data = key; legend3D(spec);
    $("#note-3d").textContent = `Relief exaggerated ${view3d.exaggeration}×. ` + (spec.nearest ? "Each 1° cell is shown as it is scored."
      : "Colours blend between 1° cells; the flat maps below show the exact grid.");
  }
  view3d.setDay(state.day);
  const box = state.mode === "shock" ? state.box : state.area;
  const bkey = `${state.mode}|${box ? box.join() : ""}|${state.mode === "shock" ? state.runId : ""}`;
  if (bkey !== k3.box) {
    const pulse = state.mode === "shock" && state.runId !== k3.run && state.runId > 0;
    view3d.setBox(box, { kind: state.mode === "shock" ? "shock" : "area", pulse });
    k3.box = bkey; k3.run = state.runId;
  }
  const cell = state.mode === "check" && state.area ? null : state.cell, ckey = cell ? cell.join() : "";
  if (ckey !== k3.cell) { view3d.setCell(cell); k3.cell = ckey; }
}

// ---------- motion that answers an action ----------
const NUM = /[−+-]?\d+(?:\.\d+)?/g;
const nums = t => (t.match(NUM) || []);
function setCounting(root, sel, html) {
  if (root._html === html) return;                 // nothing changed: leave the DOM (and any running count) alone
  const old = [...root.querySelectorAll(sel)].map(e => e.textContent);
  root.innerHTML = html; root._html = html;
  if (REDUCED.matches) return;
  root.querySelectorAll(sel).forEach((el, k) => {
    const a = nums(old[k] || ""), b = nums(el.textContent);
    if (a.length !== 1 || b.length !== 1 || a[0] === b[0]) return;
    const from = parseFloat(a[0].replace("−", "-")), to = parseFloat(b[0].replace("−", "-")), text = el.textContent;
    const dec = (b[0].split(".")[1] || "").length, minus = b[0].startsWith("−") ? "−" : "-", plus = b[0].startsWith("+");
    const fmt = v => (v < 0 ? minus : plus ? "+" : "") + Math.abs(v).toFixed(dec);
    const t0 = performance.now(), dur = 700;
    const step = now => {
      const k2 = Math.min(1, (now - t0) / dur), e = 1 - (1 - k2) ** 3;
      el.textContent = text.replace(b[0], fmt(from + (to - from) * e));
      if (k2 < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}
function drawIn(el, key) {                          // true when this chart shows new data
  if (el.dataset.k === key) return false;
  el.dataset.k = key; return !REDUCED.matches;
}
const clipDef = (el, W, H, anim) => `<defs><clipPath id="clip-${el.id}"><rect x="0" y="-20" width="${W}" height="${H + 40}">${anim
  ? `<animate attributeName="width" from="0" to="${W}" dur="0.9s" begin="indefinite" fill="freeze" calcMode="spline" keyTimes="0;1" keySplines="0.25 0.6 0.3 1"/>` : ""}</rect></clipPath></defs>`;
function startDrawIn(el, anim) { if (anim) { const a = el.querySelector("clipPath animate"); if (a && a.beginElement) a.beginElement(); } }

// ---------- controls ----------
function syncControls() {
  const k = KINDS[state.kind], s = $("#strength");
  $("#kind").value = state.kind; s.min = k.min; s.max = k.max; s.step = k.step; s.value = state.strength;
  $("#strength-out").textContent = k.fmt(state.strength);
  [$("#la0").value, $("#la1").value, $("#lo0").value, $("#lo1").value] = state.box.map(v => +v.toFixed(1));
  for (const b of document.querySelectorAll("#presets button")) b.setAttribute("aria-checked", String(b.dataset.preset === state.preset));
  for (const b of document.querySelectorAll("#fields button")) b.setAttribute("aria-selected", String(b.dataset.field === state.field));
  for (const b of document.querySelectorAll("#basemode button")) b.setAttribute("aria-pressed", String(b.dataset.mode === state.baseMode));
  for (const b of document.querySelectorAll(".ex")) b.setAttribute("aria-checked", String(state.ex && b.dataset.id === state.ex.id));
  for (const b of document.querySelectorAll("#modes button")) b.setAttribute("aria-selected", String(b.dataset.mode === state.mode));
  for (const b of document.querySelectorAll("#srcmode button")) {
    b.setAttribute("aria-pressed", String(b.dataset.src === state.src));
    b.disabled = b.dataset.src === "clim" && !(state.ex && state.ex.clim);
  }
  for (const b of document.querySelectorAll("#errmode button")) b.setAttribute("aria-pressed", String(b.dataset.err === state.errMode));
  for (const b of document.querySelectorAll("#scope button")) {
    b.setAttribute("aria-pressed", String(b.dataset.scope === state.scope));
    if (b.dataset.scope === "test") { b.disabled = !state.check; if (state.check) b.textContent = `All ${state.check.test.windows} test windows`; }
  }
  $("#errmode").style.display = state.field === "tp" ? "" : "none";
  for (const b of document.querySelectorAll("#layers-shock button")) b.setAttribute("aria-pressed", String(b.dataset.layer === state.layer.shock));
  for (const b of document.querySelectorAll("#layers-check button")) b.setAttribute("aria-pressed", String(b.dataset.layer === state.layer.check));
  $("#windbtn").setAttribute("aria-pressed", String(state.wind));
}
let debounce;
const rerun = () => { clearTimeout(debounce); debounce = setTimeout(run, 250); };
function bind() {
  for (const b of document.querySelectorAll("#presets button")) b.addEventListener("click", () => {
    const p = PRESETS[b.dataset.preset]; Object.assign(state, { kind: p.kind, strength: p.strength, box: [...p.box], preset: b.dataset.preset });
    syncControls(); run();
  });
  $("#kind").addEventListener("change", e => { state.kind = e.target.value; state.strength = KINDS[state.kind].def; state.preset = null; syncControls(); rerun(); });
  $("#strength").addEventListener("input", e => { state.strength = +e.target.value; state.preset = null; syncControls(); rerun(); });
  for (const id of ["la0", "la1", "lo0", "lo1"]) $("#" + id).addEventListener("change", () => {
    const v = ["la0", "la1", "lo0", "lo1"].map(k => +$("#" + k).value);
    state.box = [Math.min(v[0], v[1]), Math.max(v[0], v[1]), Math.min(v[2], v[3]), Math.max(v[2], v[3])]; state.preset = null; syncControls(); rerun();
  });
  $("#run").addEventListener("click", run);
  for (const b of document.querySelectorAll("#fields button")) b.addEventListener("click", () => { state.field = b.dataset.field; state.scales = null; syncControls(); render(); });
  for (const b of document.querySelectorAll("#basemode button")) b.addEventListener("click", () => { state.baseMode = b.dataset.mode; syncControls(); render(); });
  $("#days").addEventListener("keydown", e => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault(); stopPlay();
    state.day = Math.max(1, Math.min(STEPS, state.day + (e.key === "ArrowRight" ? 1 : -1))); render();
    $(`#days button[data-day="${state.day}"]`).focus();
  });
  for (const b of document.querySelectorAll("#layers-shock button, #layers-check button")) b.addEventListener("click", () => {
    state.layer[state.mode] = b.dataset.layer; syncControls(); render();
  });
  for (const b of document.querySelectorAll("#viewbtns button")) b.addEventListener("click", () => view3d && view3d.view(b.dataset.view));
  $("#windbtn").addEventListener("click", () => { state.wind = !state.wind; syncControls(); if (view3d) view3d.setWind(state.wind); });
  for (const b of document.querySelectorAll("#modes button")) b.addEventListener("click", () => setMode(b.dataset.mode));
  for (const b of document.querySelectorAll("#srcmode button")) b.addEventListener("click", () => { state.src = b.dataset.src; syncControls(); render(); });
  for (const b of document.querySelectorAll("#errmode button")) b.addEventListener("click", () => { state.errMode = b.dataset.err; syncControls(); render(); });
  for (const b of document.querySelectorAll("#scope button")) b.addEventListener("click", () => { state.scope = b.dataset.scope; syncControls(); render(); });
  $("#play").addEventListener("click", () => (playTimer ? stopPlay() : startPlay()));
  $("#server").addEventListener("change", () => { try { localStorage.setItem("mb-server", $("#server").value); } catch (e) { /* storage may be blocked */ } checkHealth().then(ok => ok && run()); });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
}
function selectExample(id) {
  const m = state.meta.examples.find(e => e.id === id);
  if (!m.win) m.win = b64f32(m.b64);
  if (!m.persist) m.persist = persistence(m);
  if (m.clim === undefined) m.clim = state.check && state.check.clim[m.id] ? b64f32(state.check.clim[m.id]) : null;
  state.ex = m; state.base = state.shock = null; state.scales = null;
  if (!state.cell) state.cell = [12, 8];   // 18.5N 73.5E: Konkan/Western Ghats land cell, inside the Arabian Sea box
  buildDays(); syncControls(); render(); run();
}

async function init() {
  try { const s = localStorage.getItem("mb-server"); if (s) $("#server").value = s; } catch (e) { /* storage may be blocked */ }
  const opt = url => fetch(url).then(r => (r.ok ? r.json() : null)).catch(() => null);
  const [meta, coast, check, terrain] = await Promise.all([fetch("data/examples.json").then(r => r.json()), fetch("data/coast.json").then(r => r.json()),
    opt("data/check.json"), opt("data/terrain.json")]);
  state.meta = meta; state.coast = coast; state.lsm = b64f32(meta.land_sea_mask); state.check = check;
  if (location.hash === "#check") { state.mode = "check"; document.body.dataset.mode = "check"; }
  const box = $("#examples");
  box.innerHTML = meta.examples.map(e => `<button class="ex" role="radio" data-id="${e.id}"><b>${e.title}</b><span>Forecast for ${fmtRange(e.dates[3], e.dates[12])}</span><span class="note">${e.note}</span></button>`).join("");
  for (const b of box.querySelectorAll(".ex")) b.addEventListener("click", () => selectExample(b.dataset.id));
  setupPanels(); bind(); syncControls(); init3D(terrain);
  await checkHealth();
  selectExample(meta.examples[0].id);
}
init();
