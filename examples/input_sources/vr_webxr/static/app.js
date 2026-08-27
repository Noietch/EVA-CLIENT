import * as THREE from "/three.module.min.js";

const statusLabel = document.getElementById("status");
const statusDot = document.getElementById("status-dot");
const enterButton = document.getElementById("enter-vr");
const errorLabel = document.getElementById("error");
const hapticStatus = document.getElementById("haptic-status");
if (hapticStatus) hapticStatus.textContent = "HAPTIC L:- R:-";
const query = new URLSearchParams(window.location.search);
const token = query.get("token") || "";
const sessionMode = query.get("mode") === "ar" ? "immersive-ar" : "immersive-vr";
const passthrough = sessionMode === "immersive-ar";

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: passthrough });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(passthrough ? 0x000000 : 0x111513, passthrough ? 0 : 1);
renderer.xr.enabled = true;
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 30);
camera.position.set(1.8, 1.3, 2.2);
camera.lookAt(0, 0.8, 0);
scene.add(camera);

const hapticHudCanvas = document.createElement("canvas");
hapticHudCanvas.width = 1024;
hapticHudCanvas.height = 128;
const hapticHudContext = hapticHudCanvas.getContext("2d");
const hapticHudTexture = new THREE.CanvasTexture(hapticHudCanvas);
const hapticHud = new THREE.Sprite(new THREE.SpriteMaterial({
  map: hapticHudTexture,
  transparent: true,
  depthTest: false,
  depthWrite: false,
}));
hapticHud.scale.set(0.9, 0.1125, 1);
hapticHud.position.set(0, 0.25, -1.4);
camera.add(hapticHud);

function drawHapticHud(text) {
  if (!hapticHudContext) return;
  hapticHudContext.clearRect(0, 0, hapticHudCanvas.width, hapticHudCanvas.height);
  hapticHudContext.fillStyle = "rgba(17, 21, 19, 0.86)";
  hapticHudContext.fillRect(0, 0, hapticHudCanvas.width, hapticHudCanvas.height);
  hapticHudContext.fillStyle = "#f0f3ef";
  hapticHudContext.font = "600 42px sans-serif";
  hapticHudContext.textBaseline = "middle";
  hapticHudContext.fillText(text, 32, hapticHudCanvas.height / 2);
  hapticHudTexture.needsUpdate = true;
}

drawHapticHud("HAPTIC L:? R:?");

scene.add(new THREE.HemisphereLight(0xf3f4e9, 0x39433d, 2.2));
const light = new THREE.DirectionalLight(0xffc987, 2.0);
light.position.set(1.5, 3, 2);
scene.add(light);
const floor = new THREE.GridHelper(8, 32, 0x718078, 0x303a35);
floor.visible = !passthrough;
scene.add(floor);

const controllerGeometry = new THREE.BoxGeometry(0.04, 0.04, 0.12);
const controllerColors = {
  left: 0x5ebd87,
  right: 0xd9a441,
  none: 0xb8c0bc,
};

function addControllerGrip(index) {
  const grip = renderer.xr.getControllerGrip(index);
  const material = new THREE.MeshStandardMaterial({
    color: controllerColors.none,
    roughness: 0.55,
    metalness: 0.1,
  });
  const body = new THREE.Mesh(controllerGeometry, material);
  body.position.z = -0.03;
  grip.add(body);
  grip.visible = false;
  grip.addEventListener("connected", (event) => {
    const handedness = event.data?.handedness || "none";
    material.color.setHex(controllerColors[handedness] || controllerColors.none);
    grip.visible = true;
  });
  grip.addEventListener("disconnected", () => {
    grip.visible = false;
  });
  scene.add(grip);
}

addControllerGrip(0);
addControllerGrip(1);

let socket = null;
let xrSession = null;
let referenceSpace = null;
let referenceSpaceType = "local-floor";
let seq = 0;
let reconnectTimer = null;
let lastHapticCapabilityKey = null;

function setConnection(online, label) {
  statusLabel.textContent = label;
  statusDot.classList.toggle("online", online);
  enterButton.disabled = !online || !navigator.xr;
}

function describeGamepad(gamepad) {
  return {
    hasGamepad: Boolean(gamepad),
    mapping: gamepad?.mapping || "",
    hapticActuators: Number(gamepad?.hapticActuators?.length || 0),
    vibrationActuator: Boolean(gamepad?.vibrationActuator),
  };
}

function updateHapticDiagnostics(reason) {
  const sources = [];
  const hands = { left: null, right: null };
  if (xrSession) {
    for (const source of xrSession.inputSources) {
      const info = {
        hand: source.handedness || "none",
        profiles: source.profiles || [],
        ...describeGamepad(source.gamepad),
      };
      sources.push(info);
      if (info.hand === "left" || info.hand === "right") hands[info.hand] = info;
    }
  }
  const formatHand = (info) => {
    if (!info) return "-";
    if (!info.hasGamepad) return "no-gp";
    if (info.hapticActuators > 0 || info.vibrationActuator) return "yes";
    return "no";
  };
  if (hapticStatus) {
    hapticStatus.textContent = `HAPTIC L:${formatHand(hands.left)} R:${formatHand(hands.right)}`;
  }
  drawHapticHud(`HAPTIC L:${formatHand(hands.left)} R:${formatHand(hands.right)}`);
  const capabilityKey = sources
    .filter((source) => source.hand === "left" || source.hand === "right")
    .map((source) => `${source.hand}:${source.hasGamepad}:${source.hapticActuators}:${source.vibrationActuator}`)
    .sort()
    .join(",");
  if (capabilityKey !== lastHapticCapabilityKey) {
    lastHapticCapabilityKey = capabilityKey;
    console.info("[WebXR haptics] capabilities", { reason, sources });
  }
  return { hands, sources };
}

function pulseGamepad(gamepad, intensity, durationMs) {
  const info = describeGamepad(gamepad);
  const attempts = [];
  const actuator = gamepad?.hapticActuators?.[0];
  if (actuator?.pulse) {
    attempts.push("hapticActuator.pulse");
    try {
      Promise.resolve(actuator.pulse(intensity, durationMs))
        .then(() => console.info("[WebXR haptics] hapticActuator.pulse resolved"))
        .catch((error) => console.warn("[WebXR haptics] hapticActuator.pulse rejected", error));
    } catch (error) {
      console.warn("[WebXR haptics] hapticActuator.pulse threw", error);
    }
  }
  const vibration = gamepad?.vibrationActuator;
  if (vibration?.playEffect) {
    attempts.push("vibrationActuator.playEffect");
    try {
      Promise.resolve(vibration.playEffect("dual-rumble", {
        duration: durationMs,
        strongMagnitude: intensity,
        weakMagnitude: intensity,
      }))
        .then(() => console.info("[WebXR haptics] vibrationActuator.playEffect resolved"))
        .catch((error) => console.warn("[WebXR haptics] vibrationActuator.playEffect rejected", error));
    } catch (error) {
      console.warn("[WebXR haptics] vibrationActuator.playEffect threw", error);
    }
  }
  return { ...info, attempts };
}

function pulseControllers(intensity, durationMs) {
  if (!xrSession) return;
  for (const source of xrSession.inputSources) {
    pulseGamepad(source.gamepad, intensity, durationMs);
  }
}

function pulseController(hand, intensity, durationMs) {
  if (!xrSession || (hand !== "left" && hand !== "right")) return [];
  const matches = [];
  for (const source of xrSession.inputSources) {
    if (source.handedness === hand) {
      matches.push({ hand, ...pulseGamepad(source.gamepad, intensity, durationMs) });
    }
  }
  return matches;
}

function connect() {
  clearTimeout(reconnectTimer);
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${window.location.host}/ws?token=${encodeURIComponent(token)}`);
  socket.addEventListener("open", () => setConnection(true, xrSession ? "STREAMING" : "READY"));
  socket.addEventListener("close", () => {
    setConnection(false, "DISCONNECTED");
    reconnectTimer = setTimeout(connect, 1000);
  });
  socket.addEventListener("error", () => setConnection(false, "CONNECTION ERROR"));
  socket.addEventListener("message", (message) => {
    let payload;
    try { payload = JSON.parse(message.data); } catch (_) { return; }
    if (payload.type === "haptic") {
      const intensity = Math.max(0, Math.min(1, Number(payload.intensity) || 0));
      const durationMs = Math.max(1, Math.min(1000, Number(payload.duration_ms) || 80));
      const matches = pulseController(payload.hand, intensity, durationMs);
      console.info("[WebXR haptics] request", {
        hand: payload.hand,
        intensity,
        durationMs,
        matches,
      });
      return;
    }
    if (payload.type !== "event_ack") return;
    statusLabel.textContent = payload.accepted ? (payload.message || "ACCEPTED") : "REJECTED";
    errorLabel.textContent = payload.accepted ? "" : (payload.message || "COMMAND REJECTED");
    pulseControllers(payload.accepted ? 0.45 : 1.0, payload.accepted ? 90 : 240);
  });
}

function poseValue(pose) {
  if (!pose) return null;
  const { position, orientation } = pose.transform;
  return {
    position: [position.x, position.y, position.z],
    orientation_xyzw: [orientation.x, orientation.y, orientation.z, orientation.w],
  };
}

function controllerValue(source, frame) {
  const pose = source.gripSpace && referenceSpace
    ? frame.getPose(source.gripSpace, referenceSpace)
    : null;
  return {
    valid: Boolean(pose),
    ...(poseValue(pose) || {}),
    profiles: source.profiles || [],
    mapping: source.gamepad?.mapping || "",
    buttons: Array.from(source.gamepad?.buttons || []).map((button) => ({
      pressed: Boolean(button.pressed),
      touched: Boolean(button.touched),
      value: Number(button.value || 0),
    })),
    axes: Array.from(source.gamepad?.axes || []).map(Number),
  };
}

function sendFrame(timestamp, frame) {
  if (!frame || !xrSession || !referenceSpace || socket?.readyState !== WebSocket.OPEN) return;
  updateHapticDiagnostics("frame");
  if (socket.bufferedAmount > 65536) return;
  const controllers = {
    left: { valid: false, profiles: [], buttons: [], axes: [] },
    right: { valid: false, profiles: [], buttons: [], axes: [] },
  };
  for (const source of xrSession.inputSources) {
    if (source.handedness === "left" || source.handedness === "right") {
      controllers[source.handedness] = controllerValue(source, frame);
    }
  }
  socket.send(JSON.stringify({
    type: "frame",
    version: 1,
    seq: seq++,
    client_time_ms: timestamp,
    reference_space: referenceSpaceType,
    controllers,
  }));
}

renderer.setAnimationLoop((timestamp, frame) => {
  sendFrame(timestamp, frame);
  renderer.render(scene, camera);
});

enterButton.addEventListener("click", async () => {
  errorLabel.textContent = "";
  try {
    xrSession = await navigator.xr.requestSession(sessionMode, { requiredFeatures: ["local-floor"] });
    referenceSpace = await xrSession.requestReferenceSpace("local-floor");
    referenceSpaceType = "local-floor";
    await renderer.xr.setSession(xrSession);
    updateHapticDiagnostics("session-start");
    xrSession.addEventListener("inputsourceschange", () => updateHapticDiagnostics("inputsourceschange"));
    setConnection(socket?.readyState === WebSocket.OPEN, "STREAMING");
    enterButton.textContent = "VR ACTIVE";
    enterButton.disabled = true;
    xrSession.addEventListener("end", () => {
      xrSession = null;
      referenceSpace = null;
      updateHapticDiagnostics("session-end");
      enterButton.textContent = passthrough ? "ENTER MR" : "ENTER VR";
      setConnection(socket?.readyState === WebSocket.OPEN, "READY");
    }, { once: true });
  } catch (error) {
    errorLabel.textContent = error instanceof Error ? error.message : String(error);
  }
});

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

if (passthrough) enterButton.textContent = "ENTER MR";
if (!navigator.xr) errorLabel.textContent = "WEBXR UNAVAILABLE";
connect();
