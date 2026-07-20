// scene.js: two independent Three.js URDF canvases. Self-rendered geometry-once +
// per-frame 4x4 transform stream; no JS kinematics. Loaded directly by index.html
// (not ES-imported); each exposes itself on window for the rest of the app.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// Self-rendered URDF canvas — mirrors viser's "geometry once + transform stream":
// load each geometry payload once into a mesh per arm, then each frame apply the
// backend's 4x4 world matrices (computed by yourdfpy) straight onto the meshes.
// No JS kinematics.
const Scene3D = (() => {
  const canvas = document.getElementById("gl");
  const COL = { paper: 0xFFFFFF, surf: 0xFFFFFF, ink: 0x111111, signal: 0xFF4D00,
                rule: 0xCFCFCF };
  const MESH_FALLBACK = 0xE6EBED;
  let renderer, scene, camera, controls;
  const meshes = {};
  const ghostMeshes = {};
  let armNames = [];
  let ready = false;
  let ghostVisible = false;

  function init() {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xFFFFFF);

    camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    camera.up.set(0, 0, 1);              // URDF is Z-up
    camera.position.set(0.9, -0.9, 0.7);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.2);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    // Hemisphere fill (sky/ground) for soft ambient gradient + a strong key and a
    // gentle rim, so the grey arm reads as shaped metal instead of a flat silhouette.
    scene.add(new THREE.HemisphereLight(0xffffff, 0xbfc4c9, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.25); key.position.set(1, -1, 2); scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.45); fill.position.set(-1, 1, 1); scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffffff, 0.35); rim.position.set(-0.5, -1.2, 0.6); scene.add(rim);

    // drafting ground grid (Z-up: rotate the XY grid into place)
    const grid = new THREE.GridHelper(2, 20, COL.rule, COL.rule);
    grid.rotation.x = Math.PI / 2;
    grid.material.opacity = 0.5; grid.material.transparent = true;
    scene.add(grid);
    // world axes accent
    const axes = new THREE.AxesHelper(0.12); scene.add(axes);

    resize();
    window.addEventListener("resize", resize);
    animate();
  }

  function resize() {
    const r = canvas.parentElement.getBoundingClientRect();
    renderer.setSize(r.width, r.height, false);
    camera.aspect = r.width / Math.max(r.height, 1);
    camera.updateProjectionMatrix();
  }

  // Smoothing time constant (seconds). Each render frame the meshes ease toward
  // the latest polled pose by alpha = 1 - exp(-dt/TAU), so sparse 120ms scene
  // updates become continuous 60fps motion. Smaller = snappier, larger = floatier.
  const SMOOTH_TAU = 0.06;
  let _lastFrame = performance.now();

  function easeLayer(layer, alpha) {
    for (const arm of armNames) {
      const armMeshes = layer[arm]; if (!armMeshes) continue;
      for (const name in armMeshes) {
        const mesh = armMeshes[name];
        const u = mesh.userData;
        if (!u.tPos) continue;
        mesh.position.lerp(u.tPos, alpha);
        mesh.quaternion.slerp(u.tQuat, alpha);
        mesh.scale.copy(u.tScale);
        mesh.matrix.compose(mesh.position, mesh.quaternion, mesh.scale);
        mesh.matrixWorldNeedsUpdate = true;
      }
    }
  }

  function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    const dt = Math.min((now - _lastFrame) / 1000, 0.1);
    _lastFrame = now;
    if (ready) {
      const alpha = 1 - Math.exp(-dt / SMOOTH_TAU);
      easeLayer(meshes, alpha);
      if (ghostVisible) easeLayer(ghostMeshes, alpha);
    }
    controls.update();
    renderer.render(scene, camera);
  }

  async function load() {
    const meta = await (await fetch("/api/meshes")).json();
    if (!meta.available) {
      document.getElementById("canvas-empty").querySelector(".big").textContent = "3D UNAVAILABLE";
      document.getElementById("canvas-empty").querySelector("div:last-child").textContent = "URDF not found on server";
      return;
    }
    armNames = (meta.arms && meta.arms.length) ? meta.arms : ["arm"];
    armNames.forEach((arm) => {
      meshes[arm] = {};
      ghostMeshes[arm] = {};
    });
    function decodeMesh(buffer) {
      const view = new DataView(buffer);
      const magic = Array.from({ length: 8 }, (_, i) => String.fromCharCode(view.getUint8(i))).join("");
      if (magic !== "EVAMESH1") throw new Error("bad mesh payload");
      const nVertices = view.getUint32(8, true);
      const nFaces = view.getUint32(12, true);
      let offset = 16;
      const positions = new Float32Array(buffer, offset, nVertices * 3); offset += nVertices * 12;
      const normals = new Float32Array(buffer, offset, nVertices * 3); offset += nVertices * 12;
      const indices = new Uint32Array(buffer, offset, nFaces * 3);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
      geo.setIndex(new THREE.BufferAttribute(indices, 1));
      geo.computeBoundingSphere();
      return geo;
    }
    const loadOne = async (file) => {
      const resp = await fetch("/meshes/" + encodeURIComponent(file));
      if (!resp.ok) throw new Error(`mesh ${file}: HTTP ${resp.status}`);
      return decodeMesh(await resp.arrayBuffer());
    };
    const solidMats = {};
    function meshColor(m) {
      const c = m.color;
      // URDF colors are authored sRGB; interpret them as such so the renderer's
      // linear pipeline doesn't darken the greys into mud.
      if (Array.isArray(c) && c.length >= 3) return new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace);
      return new THREE.Color().setHex(MESH_FALLBACK, THREE.SRGBColorSpace);
    }
    function solidMaterial(m) {
      const c = Array.isArray(m.color) ? m.color.slice(0, 3).map((x) => Number(x).toFixed(4)).join(",") : "fallback";
      if (!solidMats[c]) {
        solidMats[c] = new THREE.MeshStandardMaterial(
          { color: meshColor(m), metalness: 0.25, roughness: 0.55, side: THREE.DoubleSide });
      }
      return solidMats[c];
    }

    // Download each distinct geometry payload once in parallel.
    const files = [...new Set(meta.meshes.map((m) => m.file))];
    const geos = {};
    await Promise.all(files.map(async (file) => {
      const geo = await loadOne(file);
      geos[file] = geo;
    }));

    const ghostMat = new THREE.MeshStandardMaterial(
      { color: 0x6FA8DC, metalness: 0.0, roughness: 0.9,
        transparent: true, opacity: 0.28, depthWrite: false });

    for (const m of meta.meshes) {
      const geo = geos[m.file];
      for (const arm of armNames) {
        const mesh = new THREE.Mesh(geo, solidMaterial(m));
        mesh.matrixAutoUpdate = false;   // we drive the matrix directly each frame
        meshes[arm][m.name] = mesh;
        scene.add(mesh);
        // Translucent ghost twin: only shown when a payload carries a ghost layer
        // (MANUAL tab). Hidden by default so DEBUG sees a single solid arm.
        const ghost = new THREE.Mesh(geo, ghostMat);
        ghost.matrixAutoUpdate = false;
        ghost.visible = false;
        ghostMeshes[arm][m.name] = ghost;
        scene.add(ghost);
      }
    }
    ready = true;
    document.getElementById("canvas-empty").style.display = "none";
  }

  function show3DError(error) {
    console.error("3D scene unavailable", error);
    const empty = document.getElementById("canvas-empty");
    empty.style.display = "flex";
    empty.querySelector(".big").textContent = "3D UNAVAILABLE";
    const message = error && error.message ? error.message : String(error || "unknown error");
    empty.querySelector("div:last-child").textContent = message;
  }

  const _m4 = new THREE.Matrix4();
  const _m4Next = new THREE.Matrix4();
  const _p0 = new THREE.Vector3();
  const _p1 = new THREE.Vector3();
  const _q0 = new THREE.Quaternion();
  const _q1 = new THREE.Quaternion();
  const _s0 = new THREE.Vector3();
  const _s1 = new THREE.Vector3();

  function matrixFromFloats(matrix, floats, o) {
    matrix.set(floats[o], floats[o + 1], floats[o + 2], floats[o + 3],
               floats[o + 4], floats[o + 5], floats[o + 6], floats[o + 7],
               floats[o + 8], floats[o + 9], floats[o + 10], floats[o + 11],
               floats[o + 12], floats[o + 13], floats[o + 14], floats[o + 15]);
  }

  function ensureMeshTarget(mesh) {
    const u = mesh.userData;
    if (!u.tPos) {
      u.tPos = new THREE.Vector3();
      u.tQuat = new THREE.Quaternion();
      u.tScale = new THREE.Vector3(1, 1, 1);
    }
    return u;
  }

  function seedMeshTarget(mesh, u) {
    if (!u.seeded) {
      // First sample: jump straight there so we don't ease in from the origin.
      mesh.position.copy(u.tPos);
      mesh.quaternion.copy(u.tQuat);
      mesh.scale.copy(u.tScale);
      mesh.matrix.compose(mesh.position, mesh.quaternion, mesh.scale);
      mesh.matrixWorldNeedsUpdate = true;
      u.seeded = true;
    }
  }

  function setMeshTarget(mesh, immediate = false) {
    const u = ensureMeshTarget(mesh);
    _m4.decompose(u.tPos, u.tQuat, u.tScale);
    if (immediate) {
      mesh.position.copy(u.tPos);
      mesh.quaternion.copy(u.tQuat);
      mesh.scale.copy(u.tScale);
      mesh.matrix.compose(mesh.position, mesh.quaternion, mesh.scale);
      mesh.matrixWorldNeedsUpdate = true;
    } else {
      seedMeshTarget(mesh, u);
    }
  }

  function setMeshInterpolatedTarget(mesh, floats, o0, o1, a, immediate = false) {
    const u = ensureMeshTarget(mesh);
    matrixFromFloats(_m4, floats, o0);
    matrixFromFloats(_m4Next, floats, o1);
    _m4.decompose(_p0, _q0, _s0);
    _m4Next.decompose(_p1, _q1, _s1);
    u.tPos.copy(_p0).lerp(_p1, a);
    u.tQuat.copy(_q0).slerp(_q1, a);
    u.tScale.copy(_s0).lerp(_s1, a);
    if (immediate) {
      mesh.position.copy(u.tPos);
      mesh.quaternion.copy(u.tQuat);
      mesh.scale.copy(u.tScale);
      mesh.matrix.compose(mesh.position, mesh.quaternion, mesh.scale);
      mesh.matrixWorldNeedsUpdate = true;
    } else {
      seedMeshTarget(mesh, u);
    }
  }

  function applyLayer(layer, arms) {
    for (const arm of armNames) {
      const T = arms[arm]; if (!T) continue;
      const armMeshes = layer[arm]; if (!armMeshes) continue;
      for (const name in T) {
        const mesh = armMeshes[name]; if (!mesh) continue;
        const r = T[name];   // 4x4 row-major
        _m4.set(r[0][0],r[0][1],r[0][2],r[0][3],
                r[1][0],r[1][1],r[1][2],r[1][3],
                r[2][0],r[2][1],r[2][2],r[2][3],
                r[3][0],r[3][1],r[3][2],r[3][3]);
        setMeshTarget(mesh);
      }
    }
  }

  function applyTransformFrame(parts, geoms, floats, nGeoms, frameIndex, immediate = false) {
    if (!ready) return;
    if (!nGeoms) return;
    const nFrames = Math.floor(floats.length / (nGeoms * 16));
    if (!nFrames) return;
    const frame = Math.max(0, Math.min(Number(frameIndex) || 0, nFrames - 1));
    const frame0 = Math.floor(frame);
    const frame1 = Math.min(frame0 + 1, nFrames - 1);
    const a = frame - frame0;
    const base0 = frame0 * nGeoms * 16;
    const base1 = frame1 * nGeoms * 16;
    for (let g = 0; g < nGeoms; g++) {
      const armMeshes = meshes[parts[g]]; if (!armMeshes) continue;
      const mesh = armMeshes[geoms[g]]; if (!mesh) continue;
      const o0 = base0 + g * 16;
      if (a > 1e-6 && frame1 !== frame0) {
        setMeshInterpolatedTarget(mesh, floats, o0, base1 + g * 16, a, immediate);
      } else {
        matrixFromFloats(_m4, floats, o0);
        setMeshTarget(mesh, immediate);
      }
    }
    if (!_framed) { _framed = true; requestAnimationFrame(fitCameraToArm); }
    setGhostVisible(false);
  }

  // One-shot auto-framing: after the arm receives its first real pose, fit the
  // camera so the whole robot fills the viewport. Keeps the current view direction
  // (so the operator's chosen angle is preserved) and only adjusts distance/target.
  let _framed = false;
  const _fitBox = new THREE.Box3();
  const _fitTmp = new THREE.Box3();
  const _fitCtr = new THREE.Vector3();
  const _fitSz = new THREE.Vector3();
  const _fitDir = new THREE.Vector3();
  function fitCameraToArm() {
    scene.updateMatrixWorld(true);
    _fitBox.makeEmpty();
    for (const arm of armNames) {
      const am = meshes[arm]; if (!am) continue;
      for (const name in am) {
        _fitTmp.setFromObject(am[name]);
        if (!_fitTmp.isEmpty()) _fitBox.union(_fitTmp);
      }
    }
    if (_fitBox.isEmpty()) return;
    _fitBox.getCenter(_fitCtr);
    _fitBox.getSize(_fitSz);
    const maxDim = Math.max(_fitSz.x, _fitSz.y, _fitSz.z) || 1;
    const fov = camera.fov * Math.PI / 180;
    let dist = (maxDim / 2) / Math.tan(fov / 2) * 1.6;   // 1.6 = breathing room
    _fitDir.copy(camera.position).sub(controls.target);
    if (_fitDir.lengthSq() < 1e-6) _fitDir.set(0.9, -0.9, 0.7);
    _fitDir.normalize();
    controls.target.copy(_fitCtr);
    camera.position.copy(_fitCtr).addScaledVector(_fitDir, dist);
    camera.near = Math.max(dist / 200, 0.001);
    camera.far = dist * 200;
    camera.updateProjectionMatrix();
    controls.update();
  }

  // Poll handler: applies the latest per-mesh world transforms onto the solid arms.
  // A "ghost" layer (MANUAL tab) overlays the staged target qpos translucently;
  // when absent the ghost twins stay hidden so DEBUG shows a single solid arm.
  function applyTransforms(payload) {
    if (!ready || !payload) return;
    const command = payload.command || payload.arms;
    if (command) applyLayer(meshes, command);
    if (command && !_framed) { _framed = true; requestAnimationFrame(fitCameraToArm); }
    if (payload.ghost) {
      setGhostVisible(true);
      applyLayer(ghostMeshes, payload.ghost);
    } else {
      setGhostVisible(false);
    }
  }

  function setGhostVisible(on) {
    ghostVisible = on;
    for (const arm of armNames) {
      const g = ghostMeshes[arm]; if (!g) continue;
      for (const name in g) g[name].visible = on;
    }
  }

  async function boot3D() {
    try {
      init();
      await load();
    } catch (error) {
      show3DError(error);
    }
  }

  boot3D();
  return { applyTransforms, applyTransformFrame, resize };
})();
window.Scene3D = Scene3D;


// RESULT-tab replay canvas (#rt-gl): geometry-once + per-frame 4x4, driven by the
// replay clock (setFrame jumps straight to a frame's pose). Independent of Scene3D.
const ReplayScene = (() => {
  let renderer, scene, camera, controls, canvas;
  const meshes = {};
  let armNames = [], ready = false, loading = false;
  const INITIAL_CHUNK_FRAMES = 30;
  const TRANSFORM_CHUNK_FRAMES = 60;
  let xfKey = "";
  let xfGeneration = 0;
  let xfTotal = 0;
  let xfChunks = new Map();
  let xfLoads = new Map();
  const _m4Next = new THREE.Matrix4();
  const _rp0 = new THREE.Vector3(), _rp1 = new THREE.Vector3();
  const _rq0 = new THREE.Quaternion(), _rq1 = new THREE.Quaternion();
  const _rs0 = new THREE.Vector3(), _rs1 = new THREE.Vector3();
  function init() {
    canvas = document.getElementById("rt-gl");
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    scene = new THREE.Scene(); scene.background = new THREE.Color(0xffffff);
    camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    camera.up.set(0, 0, 1); camera.position.set(0.9, -0.9, 0.7);
    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.2); controls.enableDamping = true; controls.dampingFactor = 0.08;
    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const k = new THREE.DirectionalLight(0xffffff, 1.1); k.position.set(1, -1, 2); scene.add(k);
    const f = new THREE.DirectionalLight(0xffffff, 0.4); f.position.set(-1, 1, 1); scene.add(f);
    const grid = new THREE.GridHelper(2, 20, 0xCFCFCF, 0xCFCFCF);
    grid.rotation.x = Math.PI / 2; grid.material.opacity = 0.5; grid.material.transparent = true;
    scene.add(grid); scene.add(new THREE.AxesHelper(0.12));
    resize(); window.addEventListener("resize", resize); loop();
  }
  function resize() {
    if (!canvas) return;
    const r = canvas.parentElement.getBoundingClientRect();
    renderer.setSize(r.width, r.height, false);
    camera.aspect = r.width / Math.max(r.height, 1); camera.updateProjectionMatrix();
  }
  function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); }
  function decodeMesh(buffer) {
    const v = new DataView(buffer);
    const magic = Array.from({ length: 8 }, (_, i) => String.fromCharCode(v.getUint8(i))).join("");
    if (magic !== "EVAMESH1") throw new Error("bad mesh");
    const nV = v.getUint32(8, true), nF = v.getUint32(12, true);
    let off = 16;
    const pos = new Float32Array(buffer, off, nV * 3); off += nV * 12;
    const nrm = new Float32Array(buffer, off, nV * 3); off += nV * 12;
    const idx = new Uint32Array(buffer, off, nF * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(nrm, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1)); geo.computeBoundingSphere();
    return geo;
  }
  async function ensureLoaded() {
    if (ready || loading) return ready;
    loading = true;
    if (!renderer) init();
    const meta = await (await fetch("/api/meshes")).json();
    if (!meta.available) { loading = false; return false; }
    armNames = (meta.arms && meta.arms.length) ? meta.arms : ["arm"];
    armNames.forEach((a) => { meshes[a] = {}; });
    const files = [...new Set(meta.meshes.map((m) => m.file))];
    const geos = {};
    await Promise.all(files.map(async (file) => {
      const resp = await fetch("/meshes/" + encodeURIComponent(file));
      if (resp.ok) geos[file] = decodeMesh(await resp.arrayBuffer());
    }));
    const mats = {};
    const matFor = (m) => {
      const c = Array.isArray(m.color) ? m.color.slice(0, 3) : null;
      const key = c ? c.map((x) => Number(x).toFixed(3)).join(",") : "fb";
      if (!mats[key]) mats[key] = new THREE.MeshStandardMaterial({
        color: c ? new THREE.Color(c[0], c[1], c[2]) : new THREE.Color(0xE6EBED),
        metalness: 0.05, roughness: 0.72, side: THREE.DoubleSide });
      return mats[key];
    };
    for (const m of meta.meshes) {
      if (!geos[m.file]) continue;
      for (const arm of armNames) {
        const mesh = new THREE.Mesh(geos[m.file], matFor(m));
        mesh.matrixAutoUpdate = false; meshes[arm][m.name] = mesh; scene.add(mesh);
      }
    }
    ready = true; loading = false; requestAnimationFrame(resize);
    return true;
  }
  const _m4 = new THREE.Matrix4();
  function matrixFromFloats(mat, floats, offset) {
    mat.set(
      floats[offset], floats[offset + 1], floats[offset + 2], floats[offset + 3],
      floats[offset + 4], floats[offset + 5], floats[offset + 6], floats[offset + 7],
      floats[offset + 8], floats[offset + 9], floats[offset + 10], floats[offset + 11],
      floats[offset + 12], floats[offset + 13], floats[offset + 14], floats[offset + 15],
    );
  }

  function apply(arms) {
    if (!ready || !arms) return;
    for (const arm of armNames) {
      const T = arms[arm]; if (!T) continue;
      const am = meshes[arm]; if (!am) continue;
      for (const name in T) {
        const mesh = am[name]; if (!mesh) continue;
        const r = T[name];
        _m4.set(r[0][0],r[0][1],r[0][2],r[0][3], r[1][0],r[1][1],r[1][2],r[1][3],
                r[2][0],r[2][1],r[2][2],r[2][3], r[3][0],r[3][1],r[3][2],r[3][3]);
        mesh.matrix.copy(_m4);
        mesh.matrixWorldNeedsUpdate = true;
      }
    }
  }

  function transformKey(episodeIndex, model) {
    return String(model || "") + "|" + String(episodeIndex);
  }

  function transformRange(frame) {
    const index = Math.max(0, Math.floor(Number(frame) || 0));
    if (index < INITIAL_CHUNK_FRAMES) return [0, INITIAL_CHUNK_FRAMES];
    const start = INITIAL_CHUNK_FRAMES
      + Math.floor((index - INITIAL_CHUNK_FRAMES) / TRANSFORM_CHUNK_FRAMES)
        * TRANSFORM_CHUNK_FRAMES;
    return [start, TRANSFORM_CHUNK_FRAMES];
  }

  function parseTransformBlob(buf, start) {
    const view = new DataView(buf);
    const magic = Array.from({ length: 8 }, (_, i) => String.fromCharCode(view.getUint8(i))).join("");
    if (magic !== "EVAXFRM1") throw new Error("bad transform magic");
    const nFrames = view.getUint32(8, true);
    const nGeoms = view.getUint32(12, true);
    const hdrLen = view.getUint32(16, true);
    const keys = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 20, hdrLen)));
    const floats = new Float32Array(buf.slice(20 + hdrLen));
    if (floats.length !== nFrames * nGeoms * 16) throw new Error("bad transform length");
    const parts = [];
    const geoms = [];
    keys.forEach((key) => {
      const slash = key.indexOf("/");
      parts.push(key.slice(0, slash));
      geoms.push(key.slice(slash + 1));
    });
    return { start, nFrames, nGeoms, parts, geoms, floats };
  }

  function resetTransformChunks(key) {
    if (xfKey === key) return;
    xfKey = key;
    xfGeneration += 1;
    xfTotal = 0;
    xfChunks = new Map();
    xfLoads = new Map();
    window.__evaResultUrdfAppliedFrame = null;
  }

  async function loadChunk(key, start, count) {
    const rangeKey = `${key}:${start}:${count}`;
    if (xfChunks.has(rangeKey)) return true;
    const pending = xfLoads.get(rangeKey);
    if (pending) return pending;
    const generation = xfGeneration;
    const mq = key.split("|", 1)[0] ? "&model=" + encodeURIComponent(key.split("|", 1)[0]) : "";
    const promise = fetch(
      "/api/episode_transforms?episode_index="
      + encodeURIComponent(key.split("|").pop()) + mq
      + `&start=${start}&count=${count}`,
    ).then(async (resp) => {
      if (!resp.ok) throw new Error(`transform HTTP ${resp.status}`);
      const actualStart = Number(resp.headers.get("X-EVA-Transform-Start")) || start;
      const total = Number(resp.headers.get("X-EVA-Transform-Total")) || 0;
      const chunk = parseTransformBlob(await resp.arrayBuffer(), actualStart);
      if (generation !== xfGeneration || key !== xfKey) return false;
      xfTotal = total;
      xfChunks.set(rangeKey, chunk);
      return true;
    }).catch(() => false).finally(() => {
      xfLoads.delete(rangeKey);
    });
    xfLoads.set(rangeKey, promise);
    return promise;
  }

  async function loadEpisode(episodeIndex, model) {
    const ok = await ensureLoaded();
    const empty = document.getElementById("tp-urdf-empty");
    if (empty) empty.style.display = ok ? "none" : "flex";
    if (!ok) return false;
    const key = transformKey(episodeIndex, model);
    resetTransformChunks(key);
    return loadChunk(key, 0, INITIAL_CHUNK_FRAMES);
  }

  function applyTransformFrame(frameIndex) {
    if (!ready) return false;
    const frame = Math.max(0, Number(frameIndex) || 0);
    const chunk = [...xfChunks.values()].find((candidate) => (
      frame >= candidate.start && frame < candidate.start + candidate.nFrames
    ));
    if (!chunk) return false;
    const frameLocal = frame - chunk.start;
    const frame0 = Math.floor(frameLocal);
    const frame1 = Math.min(frame0 + 1, chunk.nFrames - 1);
    const a = frameLocal - frame0;
    const base0 = frame0 * chunk.nGeoms * 16;
    const base1 = frame1 * chunk.nGeoms * 16;
    for (let g = 0; g < chunk.nGeoms; g++) {
      const armMeshes = meshes[chunk.parts[g]]; if (!armMeshes) continue;
      const mesh = armMeshes[chunk.geoms[g]]; if (!mesh) continue;
      const o0 = base0 + g * 16;
      if (a > 1e-6 && frame1 !== frame0) {
        matrixFromFloats(_m4, chunk.floats, o0);
        matrixFromFloats(_m4Next, chunk.floats, base1 + g * 16);
        _m4.decompose(_rp0, _rq0, _rs0);
        _m4Next.decompose(_rp1, _rq1, _rs1);
        _rp0.lerp(_rp1, a);
        _rq0.slerp(_rq1, a);
        _rs0.lerp(_rs1, a);
        mesh.matrix.compose(_rp0, _rq0, _rs0);
      } else {
        matrixFromFloats(_m4, chunk.floats, o0);
        mesh.matrix.copy(_m4);
      }
      mesh.matrixWorldNeedsUpdate = true;
    }
    return true;
  }

  let _seq = 0;
  async function setFrame(episodeIndex, frame, model) {
    const key = transformKey(episodeIndex, model);
    if (xfKey !== key) await loadEpisode(episodeIndex, model);
    const generation = xfGeneration;
    if (applyTransformFrame(frame)) {
      window.__evaResultUrdfAppliedFrame = Number(frame);
      const [start, count] = transformRange(frame);
      const nextStart = start === 0 ? INITIAL_CHUNK_FRAMES : start + count;
      if (nextStart < xfTotal) loadChunk(key, nextStart, TRANSFORM_CHUNK_FRAMES);
      return;
    }
    const [start, count] = transformRange(frame);
    await loadChunk(key, start, count);
    if (generation === xfGeneration && xfKey === key && applyTransformFrame(frame)) {
      window.__evaResultUrdfAppliedFrame = Number(frame);
    }
  }
  function resizeSoon() { if (ready) requestAnimationFrame(resize); }
  return { ensureLoaded, loadEpisode, setFrame, applyTransformFrame, resizeSoon };
})();
window.ReplayScene = ReplayScene;
