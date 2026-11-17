import { $, S, apiGet, apiPost } from "./core.js";
import { t } from "./i18n.js";

const DEVICE_KIND_LABELS = {
  robot: "device.kind.robot",
  teleop: "device.kind.teleop",
  camera: "device.kind.camera",
};

const DEVICE_STATE_LABELS = {
  Stopping: "device.state.stopping",
  Failed: "device.state.failed",
  Running: "device.state.running",
  "Waiting for headset": "device.state.waiting",
  Starting: "device.state.starting",
  Stopped: "device.state.stopped",
};

const DEVICE_CATALOG_LABELS = {
  None: "device.none",
  "Robot camera stream": "device.robotCameraStream",
};

function catalogLabel(value) {
  const key = DEVICE_CATALOG_LABELS[String(value)] || "";
  return key ? t(key) : value;
}

class DevicePanel {
  constructor() {
    this.data = null;
    this.pending = false;
    this.status = {processes: {}};
    this.busy = new Set();
    this.killBusy = false;
    this.polling = false;
    this.lastPoll = 0;
    const kill = $("device-kill-all");
    if (kill) kill.onclick = () => this.killAll();
  }

  async load() {
    this.pending = true;
    try {
      this.data = await apiGet("/api/devices");
      if (this.data.ok === false) throw new Error(this.data.error);
      this.render();
    } catch (error) { window.alert(error.message); }
    finally { this.pending = false; }
  }

  render() {
    const {catalog, selected} = this.data;
    $("device-selections").replaceChildren();
    for (const kind of ["robot", "teleop", "camera"]) {
      const row = document.createElement("div");
      row.className = "device-row";
      const label = document.createElement("label");
      label.htmlFor = "device-select-" + kind;
      label.textContent = t(DEVICE_KIND_LABELS[kind]);
      const indicator = document.createElement("span");
      indicator.id = "device-state-" + kind;
      indicator.className = "device-state";
      indicator.setAttribute("role", "status");
      label.prepend(indicator);
      const select = document.createElement("select");
      select.id = label.htmlFor;
      for (const [id, spec] of Object.entries(catalog[kind])) {
        if (kind === "robot") continue;
        if (kind === "camera" && spec.hidden) continue;
        const transport = S.CFG?.transport_type;
        const label = kind === "camera" && id === "external"
          ? ({ros1: "ROS 1", ros2: "ROS 2", zmq: "ZMQ"}[transport] || catalogLabel(spec.label))
          : catalogLabel(spec.label);
        select.add(new Option(label, id));
      }
      select.value = selected[kind];
      if (kind === "robot") {
        select.add(new Option(t("device.real"), "real"));
        select.add(new Option(t("device.fake"), "fake"));
        select.value = this.data.values.robot.mode || "real";
        select.title = catalogLabel(catalog.robot[selected.robot].label);
      }
      select.onchange = () => this.select(kind, select.value);
      const actions = document.createElement("div");
      actions.className = "device-actions";
      const toggle = document.createElement("button");
      toggle.id = "device-toggle-" + kind;
      toggle.type = "button";
      toggle.className = "btn";
      toggle.onclick = () => this.command(kind);
      actions.append(toggle);
      if (kind === "teleop" && selected.teleop === "vr_webxr") {
        const pico = document.createElement("button");
        pico.id = "device-open-pico";
        pico.type = "button";
        pico.className = "btn";
        pico.textContent = t("device.webxr");
        pico.title = t("device.openWebxr");
        pico.onclick = () => this.openPico();
        actions.append(pico);
      }
      row.append(label, select, actions);
      $("device-selections").append(row);
    }
    this.updateControls();
  }

  async select(kind, value) {
    const selected = {...this.data.selected};
    const values = structuredClone(this.data.values);
    if (kind === "robot") {
      values.robot.mode = value;
    } else selected[kind] = value;
    this.pending = true;
    this.updateControls();
    try {
      const before = await apiGet("/api/device_settings");
      const response = await apiPost("/api/device_selection", kind === "robot" ? {selected, values} : {selected});
      if (response.ok === false) throw new Error(response.error);
      for (let attempt = 0; attempt < 120; attempt++) {
        await new Promise(resolve => setTimeout(resolve, 250));
        let status;
        try { status = await apiGet("/api/device_settings", {timeoutMs: 1000}); }
        catch { continue; }
        if (status.state === "failed") throw new Error(status.error);
        if (status.boot_id !== before.boot_id) { location.reload(); return; }
      }
      throw new Error(t("device.selectionTimedOut"));
    } catch (error) {
      window.alert(error.message);
      this.pending = false;
      this.render();
    }
  }

  async waitForOperation(kind, requestId, action) {
    for (let attempt = 0; attempt < 240; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 500));
      this.status = await apiGet("/api/device_settings");
      const operation = this.status.operations?.[kind];
      if (!operation || operation.id !== requestId) continue;
      if (operation.state === "failed") throw new Error(operation.error);
      if (operation.state === "ready" && action === "start" && this.status.ready?.[kind]) return;
      if (action === "stop" && operation.state === "stopped" && !(kind in this.status.processes)) return;
      if (kind in this.status.processes && this.status.processes[kind] !== null) {
        throw new Error(this.status.error || "Device process exited");
      }
    }
    throw new Error("Device command timed out");
  }

  async command(kind, {throwOnError = false} = {}) {
    this.busy.add(kind);
    this.updateControls();
    try {
      const action = this.status.processes[kind] === null ? "stop" : "start";
      const result = await apiPost("/api/device_" + action, {component: kind});
      if (result.ok === false) throw new Error(result.error);
      if (typeof result.request_id !== "string" || !result.request_id) {
        throw new Error(t("device.serverOutOfDate"));
      }
      // Queue acknowledgement is not device readiness.
      await this.waitForOperation(kind, result.request_id, action);
    } catch (error) {
      if (throwOnError) throw error;
      window.alert(error.message);
    }
    finally { this.busy.delete(kind); this.updateControls(); }
  }

  async ensureStarted(kind) {
    this.status = await apiGet("/api/device_settings");
    const running = this.status.processes?.[kind] === null;
    if (running && this.status.ready?.[kind]) return;

    const operation = this.status.operations?.[kind];
    if (operation && ["queued", "starting"].includes(operation.state)) {
      await this.waitForOperation(kind, operation.id, "start");
      return;
    }
    if (running) throw new Error(`${kind} is running but not ready`);
    await this.command(kind, {throwOnError: true});
  }

  async connectPico() {
    if (this.data?.selected?.teleop !== "eva_pico") return;
    this.busy.add("pico");
    this.updateControls();
    try {
      // The native launcher owns ADB reverse and APK startup. Keep that separate
      // from the idempotent device-service start so a second click cannot create
      // another VR ZMQ bridge.
      await this.ensureStarted("robot");
      await this.ensureStarted("teleop");
      await this.openPico();
    } catch (error) { window.alert(error.message); }
    finally { this.busy.delete("pico"); this.updateControls(); }
  }

  async openPico() {
    this.busy.add("pico");
    this.updateControls();
    try {
      const result = await apiPost("/api/device_open_pico", {}, {timeoutMs: 20000});
      if (result.ok === false) throw new Error(result.error);
    } catch (error) { window.alert(error.message); }
    finally { this.busy.delete("pico"); this.updateControls(); }
  }

  async killAll() {
    if (!window.confirm(t("device.killConfirm"))) return;
    this.killBusy = true;
    this.updateControls();
    try {
      const result = await apiPost("/api/device_kill_all", {}, {timeoutMs: 30000});
      if (result.ok === false) throw new Error(result.error);
      this.status = await apiGet("/api/device_settings");
    } catch (error) { window.alert(error.message); }
    finally {
      this.busy.clear();
      this.killBusy = false;
      this.updateControls();
    }
  }

  updateControls() {
    if (!this.data) return;
    const moving = !!S.STATUS.collection_teleop_armed || !!S.STATUS.manual_publish_active ||
      S.STATUS.session_status === "running";
    for (const kind of ["robot", "teleop", "camera"]) {
      const running = this.status.processes[kind] === null;
      const ready = running && this.status.ready?.[kind];
      const isVr = kind === "teleop" && ["vr_webxr", "eva_pico"].includes(this.data.selected.teleop);
      const connected = !isVr || !!S.STATUS.teleop?.connected;
      const button = $("device-toggle-" + kind);
      const operation = this.status.operations?.[kind];
      const stopping = operation?.state === "stopping";
      const transitioning = ["queued", "starting", "stopping"].includes(operation?.state);
      const failed = operation?.state === "failed" || (kind in this.status.processes && !running);
      const state = stopping ? "Stopping" : failed ? "Failed" : ready ? (connected ? "Running" : "Waiting for headset") : running || this.busy.has(kind) || transitioning ? "Starting" : "Stopped";
      const stateLabel = t(DEVICE_STATE_LABELS[state]);
      const indicator = $("device-state-" + kind);
      indicator.dataset.state = state;
      indicator.title = state === "Failed" ? operation?.error || this.status.error || stateLabel : stateLabel;
      indicator.setAttribute("aria-label", stateLabel);
      button.textContent = stopping ? t("device.button.stopping") : (kind === "robot" && running && failed) ? t("device.button.stop") : ready ? (connected ? t("device.button.stop") : t("device.button.cancel")) : running || this.busy.has(kind) || transitioning ? t("device.button.starting") : t("device.button.start");
      button.classList.toggle("danger", !!ready && connected);
      button.disabled = this.killBusy || this.pending || this.busy.has(kind) || transitioning || (running && !ready && !failed) || (!running && moving);
      const spec = this.data.catalog[kind][this.data.selected[kind]];
      const launchable = spec.launch || spec.separate || (kind === "robot" && this.data.values.robot.mode === "fake");
      if (!launchable) button.disabled = true;
      button.title = !launchable ? t("device.noLocalProcess") : "";
      $("device-select-" + kind).disabled = this.killBusy || this.pending || moving || this.busy.size > 0 ||
        Object.values(this.status.processes).some(code => code === null);
    }
    const kill = $("device-kill-all");
    if (kill) kill.disabled = this.killBusy || this.pending;
    const pico = $("device-open-pico");
    if (pico) {
      const vrConnected = !!S.STATUS.teleop?.connected;
      pico.disabled = this.pending || this.busy.has("pico") ||
        this.status.processes.teleop !== null || !this.status.ready?.teleop || vrConnected;
      pico.title = vrConnected ? t("device.webxrStreaming") : t("device.openWebxr");
    }
    const nativePico = $("device-connect-pico");
    if (nativePico) {
      const teleopRunning = this.status.processes.teleop === null;
      const teleopReady = teleopRunning && !!this.status.ready?.teleop;
      const teleopOperation = this.status.operations?.teleop;
      const teleopTransitioning = ["queued", "starting", "stopping"].includes(teleopOperation?.state);
      const vrConnected = !!S.STATUS.teleop?.connected;
      nativePico.disabled = this.pending || this.killBusy || this.busy.has("pico") ||
        this.busy.has("teleop") || teleopTransitioning || (teleopReady && vrConnected);
      nativePico.textContent = vrConnected ? "PICO Connected" : "Connect PICO";
      nativePico.title = vrConnected
        ? "Native PICO input is connected"
        : teleopReady
          ? "Connect the native PICO APK over ADB"
          : "Start the native PICO input and connect it over ADB";
    }
  }

  update() {
    if (!this.data && !this.pending) this.load();
    this.updateControls();
    if (this.polling || Date.now() - this.lastPoll < 1000) return;
    this.polling = true;
    this.lastPoll = Date.now();
    apiGet("/api/device_settings").then(status => {
      this.status = status;
      this.updateControls();
    }).catch(() => {}).finally(() => { this.polling = false; });
  }
}

const panel = new DevicePanel();
export function renderDeviceSettings() { panel.update(); }
