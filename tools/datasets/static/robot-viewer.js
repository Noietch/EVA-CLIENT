let THREE;
let OrbitControls;
let threeModules;
const geometryCache = new Map();

function encodePath(value) {
  return String(value).split("/").map(encodeURIComponent).join("/");
}

function loadThree() {
  if (!threeModules) {
    threeModules = Promise.all([
      import("three"),
      import("three/addons/controls/OrbitControls.js"),
    ]).then(([three, controls]) => {
      THREE = three;
      OrbitControls = controls.OrbitControls;
    });
  }
  return threeModules;
}

class RobotViewer {
  constructor(canvas, empty) {
    this.canvas = canvas;
    this.empty = empty;
    this.meshes = {};
    this.arms = [];
    this.chunks = new Map();
    this.loads = new Map();
    this.disposed = false;
    this.episodeIndex = null;
    this.batch = "";
    this.bounds = null;
    this.init();
  }

  init() {
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xffffff);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0.9, -0.9, 0.72);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.target.set(0, 0, 0.2);
    this.controls.enableDamping = true;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xbfc4c9, 0.7));
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(1, -1, 2);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.45);
    fill.position.set(-1, 1, 1);
    this.scene.add(fill);
    const grid = new THREE.GridHelper(2, 20, 0xcfcfcf, 0xcfcfcf);
    grid.rotation.x = Math.PI / 2;
    grid.material.opacity = 0.45;
    grid.material.transparent = true;
    this.scene.add(grid, new THREE.AxesHelper(0.12));
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvas.parentElement);
    this.animate();
  }

  animate() {
    if (this.disposed) return;
    this.animationFrame = requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    if (this.disposed || !this.canvas.parentElement) return;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false);
    this.camera.aspect = Math.max(rect.width, 1) / Math.max(rect.height, 1);
    this.camera.updateProjectionMatrix();
    if (this.bounds) {
      const sphere = this.bounds.getBoundingSphere(new THREE.Sphere());
      const halfFov = Math.atan(Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))
        * Math.min(1, this.camera.aspect));
      const direction = this.camera.position.clone().sub(this.controls.target).normalize();
      this.controls.target.copy(sphere.center);
      this.camera.position.copy(sphere.center).addScaledVector(direction, sphere.radius / Math.sin(halfFov) * 1.1);
      this.controls.update();
    }
  }

  async load(robotType, review, episodeIndex) {
    if (!robotType) {
      this.setEmpty("3D UNAVAILABLE", "plan meta 缺少 robot_type");
      return;
    }
    this.batch = review.batch_id;
    this.episodeIndex = Number.isInteger(episodeIndex) ? episodeIndex : null;
    try {
      const response = await fetch(
        "/api/robots/" + encodeURIComponent(robotType) + "/meshes",
      );
      if (!response.ok) throw new Error("robot meta HTTP " + response.status);
      const meta = await response.json();
      if (this.disposed) return;
      this.arms = meta.arms.length ? meta.arms : ["arm"];
      for (const arm of this.arms) this.meshes[arm] = {};
      const geometry = await this.loadGeometry(robotType, meta.meshes);
      if (this.disposed) return;
      this.installMeshes(meta.meshes, geometry);
      this.applyInitial(meta.initial);
      this.scene.updateMatrixWorld(true);
      this.bounds = new THREE.Box3();
      for (const arm of Object.values(this.meshes)) {
        for (const mesh of Object.values(arm)) this.bounds.expandByObject(mesh);
      }
      this.empty.hidden = true;
      this.resize();
      if (this.episodeIndex != null) await this.applyEpisodeFrame(0);
    } catch (error) {
      this.setEmpty("3D UNAVAILABLE", error.message || String(error));
    }
  }

  async loadGeometry(robotType, meshes) {
    const geometry = {};
    const files = [...new Set(meshes.map((item) => item.file))];
    await Promise.all(files.map(async (file) => {
      const key = robotType + "/" + file;
      let pending = geometryCache.get(key);
      if (!pending) {
        pending = fetch(
          "/api/robots/" + encodeURIComponent(robotType) + "/meshes/" + encodePath(file),
        ).then(async (response) => {
          if (!response.ok) throw new Error("mesh HTTP " + response.status);
          return this.decodeMesh(await response.arrayBuffer());
        }).catch((error) => {
          geometryCache.delete(key);
          throw error;
        });
        geometryCache.set(key, pending);
      }
      geometry[file] = await pending;
    }));
    return geometry;
  }

  decodeMesh(buffer) {
    const view = new DataView(buffer);
    let magic = "";
    for (let index = 0; index < 8; index += 1) magic += String.fromCharCode(view.getUint8(index));
    if (magic !== "EVAMESH1") throw new Error("bad mesh payload");
    const vertexCount = view.getUint32(8, true);
    const faceCount = view.getUint32(12, true);
    let offset = 16;
    const positions = new Float32Array(buffer, offset, vertexCount * 3);
    offset += vertexCount * 12;
    const normals = new Float32Array(buffer, offset, vertexCount * 3);
    offset += vertexCount * 12;
    const indices = new Uint32Array(buffer, offset, faceCount * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    return geometry;
  }

  installMeshes(meshSpecs, geometry) {
    const materials = new Map();
    for (const spec of meshSpecs) {
      const colorKey = Array.isArray(spec.color) ? spec.color.slice(0, 3).join(",") : "fallback";
      if (!materials.has(colorKey)) {
        const color = Array.isArray(spec.color)
          ? new THREE.Color().setRGB(spec.color[0], spec.color[1], spec.color[2], THREE.SRGBColorSpace)
          : new THREE.Color(0xe6ebed);
        materials.set(colorKey, new THREE.MeshStandardMaterial({
          color,
          metalness: 0.18,
          roughness: 0.62,
          side: THREE.DoubleSide,
        }));
      }
      for (const arm of this.arms) {
        const mesh = new THREE.Mesh(geometry[spec.file], materials.get(colorKey));
        mesh.matrixAutoUpdate = false;
        this.meshes[arm][spec.name] = mesh;
        this.scene.add(mesh);
      }
    }
  }

  applyInitial(transforms) {
    const matrix = new THREE.Matrix4();
    for (const arm of this.arms) {
      for (const [name, rows] of Object.entries(transforms[arm] || {})) {
        const mesh = this.meshes[arm] && this.meshes[arm][name];
        if (!mesh) continue;
        matrix.set(...rows.flat());
        mesh.matrix.copy(matrix);
        mesh.matrixWorldNeedsUpdate = true;
      }
    }
  }

  async applyEpisodeFrame(frame) {
    if (this.episodeIndex == null || this.disposed) return;
    const start = Math.floor(frame / 120) * 120;
    let chunk = this.chunks.get(start);
    if (!chunk) {
      chunk = await this.loadTransformChunk(start);
      if (!chunk || this.disposed) return;
    }
    const localFrame = frame - chunk.start;
    if (localFrame < 0 || localFrame >= chunk.frameCount) return;
    const matrix = new THREE.Matrix4();
    for (let geom = 0; geom < chunk.geomCount; geom += 1) {
      const mesh = this.meshes[chunk.parts[geom]] && this.meshes[chunk.parts[geom]][chunk.geoms[geom]];
      if (!mesh) continue;
      const offset = (localFrame * chunk.geomCount + geom) * 16;
      matrix.set(...chunk.floats.slice(offset, offset + 16));
      mesh.matrix.copy(matrix);
      mesh.matrixWorldNeedsUpdate = true;
    }
  }

  async loadTransformChunk(start) {
    if (this.loads.has(start)) return this.loads.get(start);
    const promise = fetch(
      "/api/batches/" + encodeURIComponent(this.batch)
        + "/episodes/" + this.episodeIndex
        + "/transforms?start=" + start + "&count=120",
    ).then(async (response) => {
      if (!response.ok) throw new Error("transform HTTP " + response.status);
      const actual = Number(response.headers.get("X-EVA-Transform-Start")) || start;
      const chunk = this.decodeTransforms(await response.arrayBuffer(), actual);
      this.chunks.set(actual, chunk);
      return chunk;
    }).catch(() => null).finally(() => this.loads.delete(start));
    this.loads.set(start, promise);
    return promise;
  }

  decodeTransforms(buffer, start) {
    const view = new DataView(buffer);
    let magic = "";
    for (let index = 0; index < 8; index += 1) magic += String.fromCharCode(view.getUint8(index));
    if (magic !== "EVAXFRM1") throw new Error("bad transform payload");
    const frameCount = view.getUint32(8, true);
    const geomCount = view.getUint32(12, true);
    const headerLength = view.getUint32(16, true);
    const keys = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, headerLength)));
    const floats = new Float32Array(buffer.slice(20 + headerLength));
    const parts = [];
    const geoms = [];
    for (const key of keys) {
      const slash = key.indexOf("/");
      parts.push(key.slice(0, slash));
      geoms.push(key.slice(slash + 1));
    }
    return { start, frameCount, geomCount, parts, geoms, floats };
  }

  setEmpty(title, message) {
    this.empty.hidden = false;
    const heading = document.createElement("b");
    heading.textContent = title;
    const detail = document.createElement("small");
    detail.textContent = message;
    this.empty.replaceChildren(heading, detail);
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.animationFrame);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
  }
}


export { loadThree, RobotViewer };
