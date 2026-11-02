// AMOC 3D block — interactive version of tools/block3d.py.
// z is up. Horizontal units are km on the LAEA plane; depths are stored in km
// and the whole scene sits in a group whose z-scale is the vertical exaggeration,
// so the slider is one assignment and the lighting follows the real geometry.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const css = getComputedStyle(document.documentElement);
const tok = (n) => css.getPropertyValue(n).trim();
const col = (n) => new THREE.Color(tok(n));

// --- look — adjust here --------------------------------------------------------------
const FLOOR_STOPS = [[0, col("--floor0")], [-500, col("--floor1")], [-1500, col("--floor2")],
                     [-3000, col("--floor3")], [-4500, col("--floor4")], [-5600, col("--floor5")]];
const RIBBON_W = [0.9, 3.4];                 // px, slow → fast (× devicePixelRatio below)
const ARROW_W = { 1: 5, 2: 9, 3: 14 };       // px per Thickness class
const HEAD = { len: 5.0, half: 3.2, gap: 1.4 };   // arrowhead size in km per px of line width; gap: space between the inflow tip head and the ribbon
const LIGHT_DIR = new THREE.Vector3(-1, 1, 1.4);   // light from the north-west, above
const DEFAULT = { az: 180, el: 41, exag: 85 };

// data files live next to this script (web/data/), wherever the page itself is served from
const DATA_V = "19";   // bump after re-running export_web.py so browsers drop their cached data files
const D = name => new URL("data/" + name + "?v=" + DATA_V, import.meta.url);
const stage = document.getElementById("stage");
const status = document.getElementById("status");
const dpr = Math.min(window.devicePixelRatio || 1, 2);

const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: false });
renderer.setPixelRatio(dpr);
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(col("--paper"), 1);
stage.appendChild(renderer.domElement);

const labelRenderer = new CSS2DRenderer({ element: document.getElementById("labels") });
labelRenderer.setSize(innerWidth, innerHeight);

const scene = new THREE.Scene();
const world = new THREE.Group();           // everything; world.scale.z = exaggeration
scene.add(world);

let persp = new THREE.PerspectiveCamera(28, innerWidth / innerHeight, 10, 60000);
let ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -60000, 60000);
persp.up.set(0, 0, 1); ortho.up.set(0, 0, 1);
let camera = persp;
let controls = null;

scene.add(new THREE.AmbientLight(0xffffff, 1.95));   // adjust floor brightness here (with the sun below)
const sun = new THREE.DirectionalLight(0xffffff, 1.75);
sun.position.copy(LIGHT_DIR).multiplyScalar(5000);
scene.add(sun, sun.target);

window.amoc3d = { scene, world, renderer, lights: { sun } };   // handy for tuning from the console (setOrtho added below)
const lineMats = [];                        // LineMaterials need the viewport size
const layers = {};                          // name → Group
function layer(name) { if (!layers[name]) { layers[name] = new THREE.Group(); world.add(layers[name]); } return layers[name]; }

// --------------------------------------------------------------------------------------
async function load() {
  const [meta, floorBuf, lines] = await Promise.all([
    fetch(D("meta.json")).then(r => r.json()),
    fetch(D("floor.f32")).then(r => r.arrayBuffer()),
    fetch(D("lines.json")).then(r => r.json()),
  ]);
  const floor = new Float32Array(floorBuf);
  buildFloor(meta, floor);
  buildFaces(meta, floor);
  buildRibbons(lines);
  buildArrows(lines.arrows);
  buildPlumes(lines.plumes);
  buildOsnap(lines.osnap);
  buildLabels(lines.labels);
  buildFront(lines.front || []);
  buildDepthScale(meta);
  const cx = (meta.x0 + meta.x1) / 2, cy = (meta.y0 + meta.y1) / 2;
  setupCamera(new THREE.Vector3(cx, cy - 150, 0));
  setExag(DEFAULT.exag);
  status.textContent = "";   // the status line only shows "loading…" and errors
  animate();
}

// --- seafloor -------------------------------------------------------------------------
function rampColor(z) {
  const s = FLOOR_STOPS;
  if (z >= s[0][0]) return s[0][1];
  for (let i = 1; i < s.length; i++) {
    if (z >= s[i][0]) { const t = (z - s[i - 1][0]) / (s[i][0] - s[i - 1][0]); return s[i - 1][1].clone().lerp(s[i][1], t); }
  }
  return s[s.length - 1][1];
}
function buildFloor(m, floor) {
  const { nx, ny, x0, y0, cell } = m;
  const pos = new Float32Array(nx * ny * 3), colr = new Float32Array(nx * ny * 3);
  const c = new THREE.Color();
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const k = j * nx + i, z = floor[k];
    pos[3 * k] = x0 + (i + 0.5) * cell; pos[3 * k + 1] = y0 + (j + 0.5) * cell; pos[3 * k + 2] = z / 1000;
    c.copy(rampColor(z)); colr[3 * k] = c.r; colr[3 * k + 1] = c.g; colr[3 * k + 2] = c.b;
  }
  const idx = new Uint32Array((nx - 1) * (ny - 1) * 6); let n = 0;
  for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const a = j * nx + i, b = a + 1, cc = a + nx, d = cc + 1;
    idx[n++] = a; idx[n++] = b; idx[n++] = d; idx[n++] = a; idx[n++] = d; idx[n++] = cc;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.BufferAttribute(colr, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true }));
  layer("floor").add(mesh);
}

// --- cut faces and block bottom ------------------------------------------------------------
function strip(top, bottom) {
  // two polylines of equal length → triangle strip geometry
  const n = top.length, pos = new Float32Array(n * 6), idx = [];
  for (let i = 0; i < n; i++) {
    pos.set(top[i], 6 * i); pos.set(bottom[i], 6 * i + 3);
    if (i < n - 1) { const a = 2 * i; idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
  return g;
}
function buildFaces(m, floor) {
  const { nx, ny, x0, y0, x1, y1, cell, zbase } = m;
  const at = (i, j) => floor[j * nx + i] / 1000;
  const edges = [];
  // south, north (along x), west, east (along y); each: list of [x, y, floorZ]
  const south = [[x0, y0, at(0, 0)]], north = [[x0, y1, at(0, ny - 1)]];
  for (let i = 0; i < nx; i++) { const x = x0 + (i + 0.5) * cell; south.push([x, y0, at(i, 0)]); north.push([x, y1, at(i, ny - 1)]); }
  south.push([x1, y0, at(nx - 1, 0)]); north.push([x1, y1, at(nx - 1, ny - 1)]);
  const west = [[x0, y0, at(0, 0)]], east = [[x1, y0, at(nx - 1, 0)]];
  for (let j = 0; j < ny; j++) { const y = y0 + (j + 0.5) * cell; west.push([x0, y, at(0, j)]); east.push([x1, y, at(nx - 1, j)]); }
  west.push([x0, y1, at(0, ny - 1)]); east.push([x1, y1, at(nx - 1, ny - 1)]);
  edges.push(south, north, west, east);

  const crustMat = new THREE.MeshLambertMaterial({ color: col("--face-crust"), side: THREE.DoubleSide });
  for (const e of edges) {
    const fl = e.map(p => [p[0], p[1], p[2]]), base = e.map(p => [p[0], p[1], zbase / 1000]);
    layer("faces").add(new THREE.Mesh(strip(fl, base), crustMat));   // crust only: the water column above the floor is left open
    // the floor profile as a hairline on the face
    const g = new THREE.BufferGeometry().setFromPoints(fl.map(p => new THREE.Vector3(...p)));
    layer("faces").add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: col("--ink-soft") })));
  }
  const bottom = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0),
    new THREE.MeshBasicMaterial({ color: col("--block-bottom"), side: THREE.DoubleSide }));
  bottom.position.set((x0 + x1) / 2, (y0 + y1) / 2, zbase / 1000);
  layer("faces").add(bottom);
  // block outline
  const o = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]];
  for (const z of [0, zbase / 1000]) {
    const g = new THREE.BufferGeometry().setFromPoints(o.map(p => new THREE.Vector3(p[0], p[1], z)));
    layer("faces").add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: col("--ink-soft") })));
  }
  for (const p of o.slice(0, 4)) {
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(p[0], p[1], 0), new THREE.Vector3(p[0], p[1], zbase / 1000)]);
    layer("faces").add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: col("--ink-soft") })));
  }
}

// --- VIKING ribbons and schematic arrows (screen-space line widths) --------------------------
function fatLine(points, colors, widthPx) {
  const g = new LineGeometry();
  g.setPositions(points);
  if (colors) g.setColors(colors);
  const mat = new LineMaterial({ color: 0xffffff, vertexColors: !!colors, linewidth: widthPx, worldUnits: false,
    resolution: new THREE.Vector2(innerWidth, innerHeight) });
  if (!colors) mat.color = new THREE.Color(0xffffff);
  lineMats.push(mat);
  const l = new Line2(g, mat); l.computeLineDistances();
  return l;
}
function buildRibbons(lines) {
  const pal = { surface: ["--surf-slow", "--surf"], intermediate: ["--mid-slow", "--mid"], deep: ["--deep-slow", "--deep"] };
  for (const [name, band] of Object.entries(lines.bands)) {
    const grp = layer(name);
    const c0 = col(pal[name][0]), c1 = col(pal[name][1]), c = new THREE.Color();
    for (const rec of band.lines) {
      const pts = [], cols = []; let sMean = 0;
      for (let i = 0; i < rec.p.length; i++) {
        const [x, y, z] = rec.p[i], t = Math.pow(Math.min(1, rec.s[i]), 0.7); sMean += t;
        pts.push(x, y, z / 1000 + (name === "surface" ? 0.004 : 0));
        c.copy(c0).lerp(c1, t); cols.push(c.r, c.g, c.b);
      }
      sMean /= rec.p.length;
      grp.add(fatLine(pts, cols, (RIBBON_W[0] + (RIBBON_W[1] - RIBBON_W[0]) * sMean) * 1.0));
    }
  }
}
// surface red → purple → deep navy along the sinking ribbons (Fig. 1.1 depth panel)
function convColor(s, out) {
  const a = col("--arrowSurf"), b = col("--plume"), c = col("--arrowDeep");
  return s < 0.5 ? out.copy(a).lerp(b, s / 0.5) : out.copy(b).lerp(c, (s - 0.5) / 0.5);
}
function buildArrows(arrows) {
  const grp = layer("arrows");
  const tmp = new THREE.Color();
  for (const a of arrows) {
    const color = col(a.cls === "surface" ? "--arrowSurf" : "--arrowDeep");
    const wpx = a.type === "Eddy" ? 3 : ARROW_W[a.w];
    const pts = [], cols = [];
    a.p.forEach(([x, y, z], i) => {
      pts.push(x, y, z / 1000 + (a.cls === "surface" ? 0.005 : 0));
      if (a.cls === "conv") { convColor(a.s[i], tmp); cols.push(tmp.r, tmp.g, tmp.b); }
    });
    const l = fatLine(pts, a.cls === "conv" ? cols : null, wpx);
    if (a.cls !== "conv") { l.material.color = color; l.material.vertexColors = false; l.material.needsUpdate = true; }
    grp.add(l);
    // arrowhead at point `at`, pointing from `from` towards `at`
    const addHead = (from, at) => {
      const [x1, y1, z1] = at, [x0, y0] = from;
      const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L;
      const len = HEAD.len * wpx, half = HEAD.half * wpx, z = z1 / 1000 + (a.cls === "surface" ? 0.006 : 0);
      const tip = [x1 + ux * len * 0.7, y1 + uy * len * 0.7], base = [x1 - ux * len * 0.3, y1 - uy * len * 0.3];
      const tri = new Float32Array([tip[0], tip[1], z, base[0] - uy * half, base[1] + ux * half, z, base[0] + uy * half, base[1] - ux * half, z]);
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(tri, 3));
      grp.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide })));
    };
    if (a.type === "Current" && a.head !== false) addHead(a.p[a.p.length - 2], a.p[a.p.length - 1]);
    if (a.type === "Current" && a.head_start) {        // inflow stub: a head outside the tip, its point just touching the ribbon
      const [x0, y0, z0] = a.p[0], [x1, y1] = a.p[1];
      const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L, len = HEAD.len * wpx;
      const back = len * 0.7 + HEAD.gap * wpx;                    // addHead puts the point 0.7·len ahead of `at`; plus the gap
      const at = [x0 - ux * back, y0 - uy * back, z0];
      addHead([at[0] - ux, at[1] - uy, z0], at);
    }
  }
}

function buildPlumes(plumes) {
  const grp = layer("plumes");
  const mat = new THREE.MeshBasicMaterial({ color: col("--plume"), transparent: true, opacity: 0.28, depthWrite: false, side: THREE.DoubleSide });
  for (const p of plumes) {                       // each site: an outline (block km) extruded from the surface down to zb
    const h = -p.zb / 1000;
    const shape = new THREE.Shape(p.outline.map(([x, y]) => new THREE.Vector2(x, y)));
    const g = new THREE.ExtrudeGeometry(shape, { depth: h, bevelEnabled: false });
    const m = new THREE.Mesh(g, mat);
    m.position.z = -h;
    grp.add(m);
  }
}

// the OSNAP observing array: dashed line on the sea surface, hairline mooring posts down to the floor
function buildOsnap(osnap) {
  const grp = layer("osnap");
  const ink = col("--osnap");
  for (const line of osnap.lines) {
    const pts = []; for (const [x, y] of line) pts.push(x, y, 0.007);
    const l = fatLine(pts, null, 1.6);
    Object.assign(l.material, { color: ink, vertexColors: false, dashed: true, dashSize: 36, gapSize: 22 });  // dash lengths in km — adjust here
    l.material.defines.USE_DASH = ""; l.material.needsUpdate = true;
    grp.add(l);
  }
  const seg = [];
  for (const [x, y, zf] of osnap.moorings) seg.push(x, y, zf / 1000, x, y, 0);
  const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(seg), 3));
  grp.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: ink, transparent: true, opacity: 0.75 })));
}

// place and flow labels (CSS, so they stay crisp and selectable) — styles in style.css by kind
function buildLabels(labels) {
  const grp = layer("labels");
  for (const L of labels) {
    const div = document.createElement("div"); div.className = `map-label kind-${L.kind}`;
    div.innerHTML = L.text.replace("\n", "<br>");
    const o = new CSS2DObject(div); o.position.set(L.x, L.y, L.kind === "land" ? 0.015 : 0.002);
    grp.add(o);
  }
}

// in/outflow labels at the tips of the arrow stubs protruding from the front face, with a short leader
function buildFront(front) {
  const grp = layer("labels");
  const mat = new THREE.LineBasicMaterial({ color: col("--ink") });
  for (const F of front) {
    const z = F.z / 1000 + (F.cls === "surface" ? 0.005 : 0), lead = 110, side = F.side || 1;   // leader length km — adjust here; side: away from the ribbon
    const drop = 70;   // km toward the viewer (down the page) from the tip to the label — adjust here
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(F.x + 20 * side, F.y - 10, z), new THREE.Vector3(F.x + lead * side, F.y - drop, z)]);
    grp.add(new THREE.Line(g, mat));
    const div = document.createElement("div"); div.className = "map-label kind-flow kind-front"; div.textContent = F.text;
    const o = new CSS2DObject(div); o.position.set(F.x + (lead + 15) * side, F.y - drop, z); o.center.set(side > 0 ? 0 : 1, 0.5);
    grp.add(o);
  }
}

function buildDepthScale(m) {
  const grp = layer("depthScale");
  const { x0, y0 } = m;
  const mat = new THREE.LineBasicMaterial({ color: col("--ink") });
  for (let k = 0; k <= 5; k++) {
    const z = -k;
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x0, y0, z), new THREE.Vector3(x0 - 40, y0 - 40, z)]);
    grp.add(new THREE.Line(g, mat));
    const div = document.createElement("div"); div.className = "depth-label";
    div.textContent = k === 0 ? "0 m" : `${k},000 m`;
    const lab = new CSS2DObject(div); lab.position.set(x0 - 60, y0 - 60, z); lab.center.set(1, 0.5);
    grp.add(lab);
  }
}

// --- camera ---------------------------------------------------------------------------------
let target = new THREE.Vector3();
function setupCamera(t) {
  target.copy(t);
  makeControls();
  setView(DEFAULT.az, DEFAULT.el, 7600);
}
function makeControls() {
  if (controls) controls.dispose();
  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(target);
  controls.maxPolarAngle = Math.PI / 2 - 0.02;   // never below the sea surface
  controls.enableDamping = true; controls.dampingFactor = 0.12;
  controls.addEventListener("change", updateCamOut);
}
function setView(azDeg, elDeg, dist) {
  dist = dist ?? camera.position.distanceTo(controls.target);
  const az = THREE.MathUtils.degToRad(azDeg), el = THREE.MathUtils.degToRad(Math.min(elDeg, 89));
  const r = dist * Math.cos(el);
  camera.position.set(controls.target.x + r * Math.sin(az), controls.target.y + r * Math.cos(az), controls.target.z + dist * Math.sin(el));
  // azimuth 180 = camera due south of the target, looking north
  camera.lookAt(controls.target); controls.update(); updateCamOut();
}
function camAngles() {
  const d = camera.position.clone().sub(controls.target);
  const az = (THREE.MathUtils.radToDeg(Math.atan2(d.x, d.y)) + 360) % 360;
  const el = THREE.MathUtils.radToDeg(Math.atan2(d.z, Math.hypot(d.x, d.y)));
  return { az, el, dist: d.length() };
}
function updateCamOut() {
  const { az, el, dist } = camAngles();
  document.getElementById("camOut").textContent = `az ${az.toFixed(0)}° · elev ${el.toFixed(0)}° · ${(dist).toFixed(0)} km`;
}
function setOrtho(on) {
  const { az, el, dist } = camAngles();
  const from = camera;
  camera = on ? ortho : persp;
  if (on) {
    const h = dist * Math.tan(THREE.MathUtils.degToRad(persp.fov / 2)), a = innerWidth / innerHeight;
    ortho.left = -h * a; ortho.right = h * a; ortho.top = h; ortho.bottom = -h; ortho.zoom = 1; ortho.updateProjectionMatrix();
  }
  camera.position.copy(from.position);
  makeControls();
  setView(az, el, dist);
}
window.amoc3d.setOrtho = setOrtho;   // parallel projection, as in the print figure: amoc3d.setOrtho(true)

function setExag(v) {
  world.scale.z = v;
  document.getElementById("exagOut").textContent = `${v}×`;
}

// --- UI ------------------------------------------------------------------------------------
document.getElementById("exag").addEventListener("input", e => setExag(parseFloat(e.target.value)));
document.querySelectorAll("[data-cam]").forEach(b => b.addEventListener("click", () => {
  const [az, el] = b.dataset.cam.split(",").map(Number); setView(az, el);
}));
document.getElementById("reset").addEventListener("click", () => {
  document.getElementById("exag").value = DEFAULT.exag; setExag(DEFAULT.exag); setView(DEFAULT.az, DEFAULT.el, 7600);
});
const panel = document.getElementById("panel"), collapseBtn = document.getElementById("collapse");
collapseBtn.addEventListener("click", () => {
  const open = panel.classList.toggle("collapsed") === false;
  collapseBtn.setAttribute("aria-expanded", String(open)); collapseBtn.title = open ? "Collapse panel" : "Expand panel";
});
document.querySelectorAll("[data-layer]").forEach(cb => cb.addEventListener("change", () => {
  const g = layers[cb.dataset.layer]; if (g) g.visible = cb.checked;
}));
document.getElementById("save").addEventListener("click", () => {
  renderer.render(scene, camera);
  const { az, el } = camAngles();
  const a = document.createElement("a");
  a.download = `amoc3d_az${az.toFixed(0)}_el${el.toFixed(0)}_x${world.scale.z}.png`;
  a.href = renderer.domElement.toDataURL("image/png"); a.click();
});
addEventListener("resize", () => {
  persp.aspect = innerWidth / innerHeight; persp.updateProjectionMatrix();
  if (camera === ortho) { const h = ortho.top, a = innerWidth / innerHeight; ortho.left = -h * a; ortho.right = h * a; ortho.updateProjectionMatrix(); }
  renderer.setSize(innerWidth, innerHeight); labelRenderer.setSize(innerWidth, innerHeight);
  for (const m of lineMats) m.resolution.set(innerWidth, innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}

// apply the initial layer visibility from the checkboxes once things exist
load().then(() => {
  document.querySelectorAll("[data-layer]").forEach(cb => { const g = layers[cb.dataset.layer]; if (g) g.visible = cb.checked; });
}).catch(err => { status.textContent = "failed: " + err.message; console.error(err); });
