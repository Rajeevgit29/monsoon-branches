// Monsoon Branches 3D view: ERA5 relief of the forecast window (0.25 deg) with the forecast draped on it, live wind
// trails from the displayed run, the shock box as a column and orientation labels. Display only: every number the
// dashboard reports comes from the 1-degree grid in app.js; this module just draws what app.js hands it.
import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const LAT_S = 6, LAT_N = 38, LON_W = 65, LON_E = 101, CLAT = 22, CLON = 83;
const KM = 0.5;                                   // scene units per km of height (1 unit = 1 degree, about 111 km)
const EXAG = Math.round(111 * KM);                 // vertical exaggeration shown to the viewer
const BASE = -0.9;                                // bottom of the slab
const TRAIL = 14, NPART = 2400, SPEED = 0.0055;   // wind trails: points per trail, count, degrees per frame per m/s
const reduced = matchMedia("(prefers-reduced-motion: reduce)");

const X = lon => lon - CLON, Z = lat => -(lat - CLAT);
const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);

const PLACES = [
  { n: "Arabian Sea", lat: 14.5, lon: 66.5, k: "sea" }, { n: "Bay of Bengal", lat: 15, lon: 89, k: "sea" },
  { n: "Himalaya", lat: 30.2, lon: 81.5, k: "land" }, { n: "Tibetan Plateau", lat: 33.5, lon: 89, k: "land" },
  { n: "Western Ghats", lat: 16.2, lon: 74.6, k: "land" }, { n: "Thar Desert", lat: 26.8, lon: 71.2, k: "land" },
  { n: "Mumbai", lat: 19.08, lon: 72.88, k: "city" }, { n: "Delhi", lat: 28.61, lon: 77.21, k: "city" },
  { n: "Kolkata", lat: 22.57, lon: 88.36, k: "city" }, { n: "Chennai", lat: 13.08, lon: 80.27, k: "city" },
  { n: "Bengaluru", lat: 12.97, lon: 77.59, k: "city" },
];

function b64(s, Type) {
  const bin = atob(s), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Type(u8.buffer);
}

export function createTerrainView(host, opts) {
  const { terrain, coast, nlat, nlon } = opts;
  const T = { nlat: terrain.nlat, nlon: terrain.nlon, h: b64(terrain.height_m, Int16Array), land: b64(terrain.land, Uint8Array) };
  const heightAt = (lat, lon) => {           // bilinear, metres
    const fi = Math.min(T.nlat - 1.001, Math.max(0, (lat - terrain.lat0) / terrain.step));
    const fj = Math.min(T.nlon - 1.001, Math.max(0, (lon - terrain.lon0) / terrain.step));
    const i = Math.floor(fi), j = Math.floor(fj), a = fi - i, b = fj - j, H = (r, c) => T.h[r * T.nlon + c];
    return (H(i, j) * (1 - a) + H(i + 1, j) * a) * (1 - b) + (H(i, j + 1) * (1 - a) + H(i + 1, j + 1) * a) * b;
  };
  const Y = (lat, lon) => (heightAt(lat, lon) / 1000) * KM;

  // ---------- renderer, scene, camera ----------
  const canvas = document.createElement("canvas");
  host.appendChild(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(34, 1, 0.5, 400);
  const VIEWS = { tilt: { p: new THREE.Vector3(0, 30.5, 41.5), t: new THREE.Vector3(0, -1.2, 4.2) }, top: { p: new THREE.Vector3(0, 60, 0.01), t: new THREE.Vector3(0, 0, 0) } };
  camera.position.copy(VIEWS.tilt.p);
  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(VIEWS.tilt.t);
  Object.assign(controls, { enableDamping: true, dampingFactor: 0.08, minDistance: 16, maxDistance: 90, maxPolarAngle: 1.32, screenSpacePanning: true });
  scene.add(new THREE.AmbientLight(0xffffff, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(-20, 30, -20); scene.add(sun);

  // ---------- terrain ----------
  const pos = new Float32Array(T.nlat * T.nlon * 3), uv = new Float32Array(T.nlat * T.nlon * 2), idx = [];
  for (let i = 0; i < T.nlat; i++) for (let j = 0; j < T.nlon; j++) {
    const k = i * T.nlon + j, lat = terrain.lat0 + i * terrain.step, lon = terrain.lon0 + j * terrain.step;
    pos[3 * k] = X(lon); pos[3 * k + 1] = (T.h[k] / 1000) * KM; pos[3 * k + 2] = Z(lat);
    uv[2 * k] = (lon - LON_W) / (LON_E - LON_W); uv[2 * k + 1] = (lat - LAT_S) / (LAT_N - LAT_S);
    if (i < T.nlat - 1 && j < T.nlon - 1) { const a = k, b = k + 1, c = k + T.nlon, d = c + 1; idx.push(a, b, c, b, d, c); }   // counter-clockwise from above
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  geo.setIndex(idx); geo.computeVertexNormals();
  const landTex = new THREE.DataTexture(T.land, T.nlon, T.nlat, THREE.RedFormat, THREE.UnsignedByteType);
  Object.assign(landTex, { magFilter: THREE.LinearFilter, minFilter: THREE.LinearFilter, needsUpdate: true });

  const blankData = new Uint8Array(nlat * nlon * 4);
  const mkTex = data => { const t = new THREE.DataTexture(data, nlon, nlat, THREE.RGBAFormat); t.magFilter = t.minFilter = THREE.LinearFilter; t.needsUpdate = true; return t; };
  let frames = [mkTex(blankData)];
  const uniforms = {
    uA: { value: frames[0] }, uB: { value: frames[0] }, uMix: { value: 0 }, uLand: { value: landTex },
    uLight: { value: new THREE.Vector3(-0.55, 0.75, -0.45).normalize() }, uKm: { value: KM }, uReady: { value: 0 },
    uBlank: { value: new THREE.Color("#cfd9da") },
  };
  const terrainMat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: `varying vec2 vUv; varying vec3 vN; varying float vH;
      void main() { vUv = uv; vN = normal; vH = position.y; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `uniform sampler2D uA, uB, uLand; uniform float uMix, uKm, uReady; uniform vec3 uLight, uBlank;
      varying vec2 vUv; varying vec3 vN; varying float vH;
      void main() {
        vec4 a = texture2D(uA, vUv), b = texture2D(uB, vUv);
        vec3 c = mix(uBlank, mix(a.rgb, b.rgb, uMix), uReady);
        float land = texture2D(uLand, vUv).r;
        float lit = 0.58 + 0.62 * max(dot(normalize(vN), uLight), 0.0);
        float relief = smoothstep(0.0, 0.04, vH);
        c *= mix(1.0, lit, relief);
        float h = vH / uKm, w = fwidth(h), dd = abs(fract(h + 0.5) - 0.5);        // a contour line every 1000 m
        float line = (1.0 - smoothstep(0.0, 1.3 * w, dd)) * step(0.6, h);
        c = mix(c, c * 0.8, line * 0.3);
        c *= mix(0.93, 1.0, land);                                              // sea a touch deeper than land
        gl_FragColor = vec4(c, 1.0);
      }`,
  });
  const mesh = new THREE.Mesh(geo, terrainMat);
  scene.add(mesh);

  // slab walls under the relief, so it reads as a model of the region
  const wallMat = new THREE.MeshLambertMaterial({ color: 0x9fb2b6, side: THREE.DoubleSide });
  const walls = new THREE.Group();
  const edge = (pts) => {
    const p = [], ix = [];
    pts.forEach(([lat, lon], k) => { p.push(X(lon), Y(lat, lon), Z(lat), X(lon), BASE, Z(lat)); if (k) { const a = 2 * (k - 1); ix.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); } });
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3)); g.setIndex(ix); g.computeVertexNormals();
    walls.add(new THREE.Mesh(g, wallMat));
  };
  const along = (a, b, n) => [...Array(n + 1).keys()].map(k => [a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n]);
  edge(along([LAT_S, LON_W], [LAT_S, LON_E], 144)); edge(along([LAT_N, LON_W], [LAT_N, LON_E], 144));
  edge(along([LAT_S, LON_W], [LAT_N, LON_W], 128)); edge(along([LAT_S, LON_E], [LAT_N, LON_E], 128));
  const bottom = new THREE.Mesh(new THREE.PlaneGeometry(LON_E - LON_W, LAT_N - LAT_S), wallMat);
  bottom.rotation.x = Math.PI / 2; bottom.position.set(0, BASE, Z(CLAT)); walls.add(bottom);
  scene.add(walls);

  // soft shadow under the slab
  const sh = document.createElement("canvas"); sh.width = sh.height = 128;
  const sg = sh.getContext("2d"), grad = sg.createRadialGradient(64, 64, 10, 64, 64, 64);
  grad.addColorStop(0, "rgba(0,0,0,0.55)"); grad.addColorStop(1, "rgba(0,0,0,0)"); sg.fillStyle = grad; sg.fillRect(0, 0, 128, 128);
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(52, 46), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(sh), transparent: true, depthWrite: false, opacity: 0.5 }));
  shadow.rotation.x = -Math.PI / 2; shadow.position.set(0, BASE - 0.05, 0.6); scene.add(shadow);

  // ---------- coastline and graticule, draped ----------
  const lineMat = new THREE.LineBasicMaterial({ color: 0x0f2730, transparent: true, opacity: 0.6 });
  const gridMat = new THREE.LineBasicMaterial({ color: 0x0f2730, transparent: true, opacity: 0.16 });
  const drape = (lines, mat, lift) => {
    const p = [];
    for (const line of lines) for (let k = 1; k < line.length; k++) {
      const [lo0, la0] = line[k - 1], [lo1, la1] = line[k];
      p.push(X(lo0), Y(la0, lo0) + lift, Z(la0), X(lo1), Y(la1, lo1) + lift, Z(la1));
    }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    const ls = new THREE.LineSegments(g, mat); scene.add(ls); return ls;
  };
  drape(coast, lineMat, 0.03);
  const grat = [];
  for (const la of [10, 20, 30]) grat.push(along([la, LON_W], [la, LON_E], 144).map(([a, b]) => [b, a]));
  for (const lo of [70, 80, 90, 100]) grat.push(along([LAT_S, lo], [LAT_N, lo], 128).map(([a, b]) => [b, a]));
  drape(grat, gridMat, 0.02);

  // ---------- wind trails ----------
  const P = new Float32Array(NPART * TRAIL * 3), head = new Int16Array(NPART), age = new Float32Array(NPART), life = new Float32Array(NPART);
  const plat = new Float32Array(NPART), plon = new Float32Array(NPART);
  const segPos = new Float32Array(NPART * (TRAIL - 1) * 6), segA = new Float32Array(NPART * (TRAIL - 1) * 2);
  const pgeo = new THREE.BufferGeometry();
  pgeo.setAttribute("position", new THREE.BufferAttribute(segPos, 3).setUsage(THREE.DynamicDrawUsage));
  pgeo.setAttribute("aAlpha", new THREE.BufferAttribute(segA, 1).setUsage(THREE.DynamicDrawUsage));
  const pmat = new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(0x0f2730) }, uOpacity: { value: 0.75 } }, transparent: true, depthWrite: false,
    vertexShader: `attribute float aAlpha; varying float vA; void main() { vA = aAlpha; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `uniform vec3 uColor; uniform float uOpacity; varying float vA; void main() { gl_FragColor = vec4(uColor, vA * uOpacity); }`,
  });
  const trails = new THREE.LineSegments(pgeo, pmat); trails.frustumCulled = false; scene.add(trails);
  let winds = null, windOn = true;
  const spawn = p => {
    plat[p] = LAT_S + 0.3 + Math.random() * (LAT_N - LAT_S - 0.6); plon[p] = LON_W + 0.3 + Math.random() * (LON_E - LON_W - 0.6);
    age[p] = 0; life[p] = 70 + Math.random() * 90; head[p] = 0;
    const x = X(plon[p]), y = Y(plat[p], plon[p]) + 0.12, z = Z(plat[p]);
    for (let k = 0; k < TRAIL; k++) { const o = (p * TRAIL + k) * 3; P[o] = x; P[o + 1] = y; P[o + 2] = z; }
  };
  for (let p = 0; p < NPART; p++) { spawn(p); age[p] = Math.random() * life[p]; }
  const UV = new Float32Array(2);
  const windAt = (w, lat, lon) => {          // bilinear on the 1-degree cell centres, into UV
    const fi = Math.min(nlat - 1.001, Math.max(0, lat - (LAT_S + 0.5))), fj = Math.min(nlon - 1.001, Math.max(0, lon - (LON_W + 0.5)));
    const i = Math.floor(fi), j = Math.floor(fj), a = fi - i, b = fj - j, k00 = i * nlon + j, k10 = k00 + nlon;
    const w00 = (1 - a) * (1 - b), w10 = a * (1 - b), w01 = (1 - a) * b, w11 = a * b;
    UV[0] = w.u[k00] * w00 + w.u[k10] * w10 + w.u[k00 + 1] * w01 + w.u[k10 + 1] * w11;
    UV[1] = w.v[k00] * w00 + w.v[k10] * w10 + w.v[k00 + 1] * w01 + w.v[k10 + 1] * w11;
  };
  function stepWind(frac, dayA, dayB, steps) {
    const wa = winds[dayA], wb = winds[dayB];
    for (let s = 0; s < steps; s++) for (let p = 0; p < NPART; p++) {
      age[p]++;
      if (age[p] > life[p] || plat[p] < LAT_S || plat[p] > LAT_N || plon[p] < LON_W || plon[p] > LON_E) { spawn(p); continue; }
      windAt(wa, plat[p], plon[p]); const ua = UV[0], va = UV[1];
      windAt(wb, plat[p], plon[p]); const u = ua + (UV[0] - ua) * frac, v = va + (UV[1] - va) * frac;
      plon[p] += (u * SPEED) / Math.cos((plat[p] * Math.PI) / 180); plat[p] += v * SPEED;
      head[p] = (head[p] + 1) % TRAIL;
      const o = (p * TRAIL + head[p]) * 3;
      P[o] = X(plon[p]); P[o + 1] = Y(plat[p], plon[p]) + 0.12; P[o + 2] = Z(plat[p]);
    }
    let q = 0;
    for (let p = 0; p < NPART; p++) {
      const fade = Math.min(1, age[p] / 12, (life[p] - age[p]) / 12);
      for (let k = 0; k < TRAIL - 1; k++) {
        const o0 = (p * TRAIL + ((head[p] - k + TRAIL) % TRAIL)) * 3, o1 = (p * TRAIL + ((head[p] - k - 1 + TRAIL) % TRAIL)) * 3;
        const s6 = q * 6;
        segPos[s6] = P[o0]; segPos[s6 + 1] = P[o0 + 1]; segPos[s6 + 2] = P[o0 + 2];
        segPos[s6 + 3] = P[o1]; segPos[s6 + 4] = P[o1 + 1]; segPos[s6 + 5] = P[o1 + 2];
        const al = fade * (1 - k / (TRAIL - 1));
        segA[q * 2] = al; segA[q * 2 + 1] = al * 0.8; q++;
      }
    }
    pgeo.attributes.position.needsUpdate = true; pgeo.attributes.aAlpha.needsUpdate = true;
  }

  // ---------- shock box / averaging area, selected cell, ripples ----------
  const boxGroup = new THREE.Group(); scene.add(boxGroup);
  const boxFill = new THREE.MeshBasicMaterial({ color: 0xf47230, transparent: true, opacity: 0.14, depthWrite: false });
  const boxEdge = new THREE.LineBasicMaterial({ color: 0xf47230, transparent: true, opacity: 0.95 });
  let boxKind = "shock", themeNow = "light";
  function boxColors() {
    const dark = themeNow === "dark", c = boxKind === "area" ? (dark ? 0xe8f0f2 : 0x0f2730) : (dark ? 0xe76624 : 0xf47230);
    boxFill.color.setHex(c); boxEdge.color.setHex(c); boxFill.opacity = boxKind === "area" ? 0.08 : 0.14;
  }
  function setBox(b, o = {}) {
    boxGroup.clear(); boxKind = o.kind || "shock"; boxColors();
    if (!b) return;
    const [la0, la1, lo0, lo1] = b, w = lo1 - lo0, d = la1 - la0;
    let top = 0; for (let la = la0; la <= la1; la += 0.5) for (let lo = lo0; lo <= lo1; lo += 0.5) top = Math.max(top, Y(la, lo));
    const hgt = top + 2.6, g = new THREE.BoxGeometry(w, hgt - BASE, d);
    const m = new THREE.Mesh(g, boxFill); m.position.set(X((lo0 + lo1) / 2), (hgt + BASE) / 2, Z((la0 + la1) / 2));
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(g), boxEdge); e.position.copy(m.position);
    boxGroup.add(m, e);
    if (o.pulse && !reduced.matches) ripple((la0 + la1) / 2, (lo0 + lo1) / 2, Math.max(w, d));
  }
  const ripples = [];
  function ripple(lat, lon, size) {
    for (let k = 0; k < 3; k++) {
      const r = new THREE.Mesh(new THREE.RingGeometry(0.92, 1, 96), new THREE.MeshBasicMaterial({ color: 0xf47230, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }));
      r.rotation.x = -Math.PI / 2; r.position.set(X(lon), Y(lat, lon) + 0.25, Z(lat));
      scene.add(r); ripples.push({ r, t0: performance.now() + k * 260, size });
    }
  }
  const cellGroup = new THREE.Group(); scene.add(cellGroup);
  const pinMat = new THREE.LineBasicMaterial({ color: 0x0f2730 }), dotMat = new THREE.MeshBasicMaterial({ color: 0x0f2730 });
  function setCell(c) {
    cellGroup.clear();
    if (!c) return;
    const lat = LAT_S + c[0] + 0.5, lon = LON_W + c[1] + 0.5, y = Y(lat, lon);
    const sq = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5]].map(([a, b]) => new THREE.Vector3(X(lon + b), Y(lat + a, lon + b) + 0.06, Z(lat + a)));
    cellGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(sq), pinMat));
    cellGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(X(lon), y, Z(lat)), new THREE.Vector3(X(lon), y + 2.2, Z(lat))]), pinMat));
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.16, 16, 12), dotMat); dot.position.set(X(lon), y + 2.2, Z(lat)); cellGroup.add(dot);
  }

  // ---------- labels (HTML, projected every frame) ----------
  const labelLayer = document.createElement("div"); labelLayer.className = "g-labels"; host.appendChild(labelLayer);
  const labels = PLACES.map(p => {
    const el = document.createElement("span"); el.className = `g-label g-${p.k}`; el.textContent = p.n; labelLayer.appendChild(el);
    return { el, v: new THREE.Vector3(X(p.lon), Y(p.lat, p.lon) + (p.k === "city" ? 0.35 : 0.6), Z(p.lat)) };
  });
  for (const la of [10, 20, 30]) { const el = document.createElement("span"); el.className = "g-label g-grid"; el.textContent = `${la}°N`; labelLayer.appendChild(el); labels.push({ el, v: new THREE.Vector3(X(LON_W) - 0.9, 0, Z(la)) }); }
  for (const lo of [70, 80, 90, 100]) { const el = document.createElement("span"); el.className = "g-label g-grid"; el.textContent = `${lo}°E`; labelLayer.appendChild(el); labels.push({ el, v: new THREE.Vector3(X(lo), BASE, Z(LAT_S) + 1.0) }); }
  const tmp = new THREE.Vector3();
  function placeLabels(W, H) {
    for (const l of labels) {
      tmp.copy(l.v).project(camera);
      const vis = tmp.z < 1 && Math.abs(tmp.x) < 1.05 && Math.abs(tmp.y) < 1.05;
      l.el.style.opacity = vis ? "" : "0";
      if (vis) l.el.style.transform = `translate(${((tmp.x + 1) / 2) * W}px, ${((1 - tmp.y) / 2) * H}px) translate(-50%, -50%)`;
    }
  }

  // ---------- data and days ----------
  let dayNow = 0, dayTarget = 0, whole = false, dataKey = null;
  function setData(d) {
    if (d.key === dataKey) return;
    dataKey = d.key;
    for (const f of frames) f.dispose();
    frames = d.frames.map(fr => { const t = mkTex(fr); if (d.nearest) t.magFilter = t.minFilter = THREE.NearestFilter; return t; });
    winds = d.winds; whole = frames.length === 1;
    if (whole) dayNow = dayTarget = 0;
    uniforms.uReady.value = 1;
    applyDay();
    if (reduced.matches && winds && windOn) staticTrails();
  }
  function applyDay() {
    const a = Math.floor(dayNow), b = Math.min(frames.length - 1, a + 1);
    uniforms.uA.value = frames[Math.min(a, frames.length - 1)]; uniforms.uB.value = frames[b]; uniforms.uMix.value = whole ? 0 : dayNow - a;
  }
  function setDay(d) {
    dayTarget = whole ? 0 : d - 1;
    if (reduced.matches) { dayNow = dayTarget; applyDay(); }
  }
  function staticTrails() { for (let p = 0; p < NPART; p++) spawn(p); const a = Math.floor(dayNow); stepWind(0, a, a, TRAIL); }

  // ---------- camera moves ----------
  let move = null;
  function view(name, ms = 900) {
    const v = VIEWS[name];
    if (reduced.matches || !ms) { camera.position.copy(v.p); controls.target.copy(v.t); controls.update(); return; }
    move = { p0: camera.position.clone(), t0: controls.target.clone(), p1: v.p, t1: v.t, s: performance.now(), ms };
  }

  // ---------- pointer: hover and pick ----------
  const ray = new THREE.Raycaster(), ndc = new THREE.Vector2();
  let down = null;
  const cellFromEvent = e => {
    const r = canvas.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hit = ray.intersectObject(mesh)[0];
    if (!hit || !hit.uv) return null;
    const lat = LAT_S + hit.uv.y * (LAT_N - LAT_S), lon = LON_W + hit.uv.x * (LON_E - LON_W);
    return [Math.min(nlat - 1, Math.max(0, Math.floor(lat - LAT_S))), Math.min(nlon - 1, Math.max(0, Math.floor(lon - LON_W)))];
  };
  canvas.addEventListener("pointerdown", e => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
  canvas.addEventListener("pointermove", e => { if (!down && opts.onHover) opts.onHover(cellFromEvent(e), e); });
  canvas.addEventListener("pointerleave", () => opts.onHover && opts.onHover(null));
  canvas.addEventListener("pointerup", e => {
    if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5 && performance.now() - down.t < 500) { const c = cellFromEvent(e); if (c && opts.onPick) opts.onPick(c); }
    down = null;
  });

  // ---------- theme ----------
  function setTheme(t) {
    const dark = t === "dark", ink = dark ? 0xe8f0f2 : 0x0f2730;
    lineMat.color.setHex(ink); gridMat.color.setHex(ink); lineMat.opacity = dark ? 0.55 : 0.6; gridMat.opacity = dark ? 0.14 : 0.16;
    pmat.uniforms.uColor.value.setHex(dark ? 0xffffff : 0x0f2730); pmat.uniforms.uOpacity.value = dark ? 0.7 : 0.62;
    pinMat.color.setHex(ink); dotMat.color.setHex(ink);
    wallMat.color.setHex(dark ? 0x22343b : 0xa9bbbf); shadow.material.opacity = dark ? 0.7 : 0.45;
    uniforms.uBlank.value.set(dark ? "#22343b" : "#cfd9da");
    themeNow = t; boxColors();
  }

  // ---------- loop ----------
  let visible = true, last = performance.now(), W = 0, H = 0, started = performance.now();
  new IntersectionObserver(es => { visible = es[0].isIntersecting; }).observe(host);
  const resize = () => {
    W = host.clientWidth; H = host.clientHeight; renderer.setSize(W, H, false);
    camera.aspect = W / Math.max(1, H);
    camera.fov = Math.min(68, 34 * Math.max(1, 1.5 / camera.aspect));      // narrow screens: widen the view so the whole region fits
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(host); resize();
  function frame(now) {
    requestAnimationFrame(frame);
    if (!visible || document.hidden) { last = now; return; }
    const dt = Math.min(0.05, (now - last) / 1000); last = now;
    if (move) {
      const k = Math.min(1, (now - move.s) / move.ms), e = ease(k);
      camera.position.lerpVectors(move.p0, move.p1, e); controls.target.lerpVectors(move.t0, move.t1, e);
      if (k >= 1) move = null;
    }
    controls.update();
    if (!reduced.matches && dayNow !== dayTarget) {
      const speed = Math.max(1.8, Math.abs(dayTarget - dayNow) / 0.55);
      dayNow = dayNow < dayTarget ? Math.min(dayTarget, dayNow + speed * dt) : Math.max(dayTarget, dayNow - speed * dt);
      applyDay();
    }
    trails.visible = !!(winds && windOn);
    if (winds && windOn && !reduced.matches) { const a = Math.floor(dayNow), b = Math.min(winds.length - 1, a + 1); stepWind(dayNow - a, a, b, 1); }
    if (boxKind === "shock" && boxGroup.children.length) boxFill.opacity = reduced.matches ? 0.14 : 0.1 + 0.06 * Math.sin(now / 420);
    for (let k = ripples.length - 1; k >= 0; k--) {
      const r = ripples[k], t = (now - r.t0) / 1500;
      if (t < 0) continue;
      if (t >= 1) { scene.remove(r.r); r.r.geometry.dispose(); r.r.material.dispose(); ripples.splice(k, 1); continue; }
      const s = r.size * (0.6 + 2.2 * t); r.r.scale.set(s, s, s); r.r.material.opacity = 0.8 * (1 - t);
    }
    renderer.render(scene, camera);
    placeLabels(W, H);
  }
  // opening move: from straight overhead down to the tilted view
  if (!reduced.matches) { camera.position.set(0, 75, 0.01); controls.target.set(0, 0, 0); controls.update(); view("tilt", 2000); }
  requestAnimationFrame(frame);

  return {
    setData, setDay, setBox, setCell, setTheme, view, exaggeration: EXAG,
    clear() { uniforms.uReady.value = 0; winds = null; dataKey = null; },
    setWind(on) { windOn = on; if (on && reduced.matches && winds) staticTrails(); },
  };
}

window.MB3D = { createTerrainView };
window.dispatchEvent(new Event("mb3d-ready"));
