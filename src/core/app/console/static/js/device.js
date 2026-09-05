import { $, S, apiGet, apiPost } from "./core.js";

class DevicePanel {
  constructor() {
    this.data = null;
    this.pending = false;
    this.dirty = false;
    this.error = "";
    this.polling = false;
    this.lastPoll = 0;
    $("device-workspace").onsubmit = event => { event.preventDefault(); this.apply(); };
    for (const action of ["start", "stop"]) {
      $("device-" + action).onclick = async () => {
        try {
          const result = await apiPost("/api/device_" + action, {component: $("device-component").value});
          if (result.ok === false) throw new Error(result.error);
          this.error = "";
        } catch (error) { this.error = error.message; }
      };
    }
    $("device-profile-load").onclick = () => this.loadProfile({path: $("device-profile-select").value});
    $("device-profile-upload").onclick = () => $("device-profile-file").click();
    $("device-profile-file").onchange = async event => {
      const file = event.target.files[0];
      if (file) await this.loadProfile({content: await file.text()});
      event.target.value = "";
    };
  }

  async load(selected) {
    this.pending = true;
    try {
      const query = selected ? "?" + new URLSearchParams(selected) : "";
      const data = await apiGet("/api/devices" + query);
      if (data.ok === false) throw new Error(data.error);
      this.data = data;
      this.render();
    } catch (error) {
      this.error = error.message;
    } finally {
      this.pending = false;
    }
  }

  render() {
    const {catalog, selected, values} = this.data;
    $("device-selections").replaceChildren();
    $("device-fields").replaceChildren();
    for (const kind of ["robot", "teleop", "camera"]) {
      const label = document.createElement("label");
      label.className = "device-setting";
      label.textContent = kind.toUpperCase();
      const select = document.createElement("select");
      select.id = "device-select-" + kind;
      for (const [id, spec] of Object.entries(catalog[kind])) {
        if (spec.robots && !spec.robots.includes(selected.robot)) continue;
        select.add(new Option(spec.label, id));
      }
      select.value = selected[kind];
      select.onchange = async () => {
        const next = {...selected, [kind]: select.value};
        if (kind === "robot") {
          for (const other of ["teleop", "camera"]) {
            const spec = catalog[other][next[other]];
            if (spec.robots && !spec.robots.includes(next.robot)) {
              next[other] = Object.keys(catalog[other]).find(id => catalog[other][id].default);
            }
          }
        }
        this.dirty = true;
        await this.load(next);
      };
      label.appendChild(select);
      $("device-selections").appendChild(label);
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = catalog[kind][selected[kind]].label + " settings";
      details.appendChild(summary);
      this.fields(details, values[kind], [], catalog[kind][selected[kind]].choices || {});
      $("device-fields").appendChild(details);
    }
    const camera = catalog.camera[selected.camera];
    $("device-profile").hidden = !camera.profile_field;
    const profiles = $("device-profile-select");
    profiles.replaceChildren();
    const active = values.camera.settings[camera.profile_field];
    const paths = new Set([...(camera.profiles || []), ...(active ? [active] : [])]);
    for (const path of paths) profiles.add(new Option(path.split("/").pop(), path));
    if (active) profiles.value = active;
  }

  fields(host, values, path, choices) {
    for (const [key, value] of Object.entries(values)) {
      if (key === "type") continue;
      const next = [...path, key];
      if (value !== null && typeof value === "object") {
        const group = document.createElement("fieldset");
        const legend = document.createElement("legend");
        legend.textContent = key.replaceAll("_", " ");
        group.appendChild(legend);
        this.fields(group, value, next, choices);
        if (Array.isArray(value)) {
          const add = document.createElement("button");
          add.type = "button"; add.className = "btn"; add.textContent = "+";
          add.title = "Add value"; add.setAttribute("aria-label", "Add " + key);
          add.onclick = () => {
            value.push(key === "gripper_limits_override" ? 0 : "");
            this.dirty = true; this.render();
          };
          group.appendChild(add);
        }
        host.appendChild(group);
        continue;
      }
      const row = document.createElement("label");
      row.className = "device-setting";
      const label = document.createElement("span");
      label.textContent = key.replaceAll("_", " ");
      const options = choices[key] || (key === "controller" ? ["left", "right"] :
        key === "mode" ? ["binary", "analog", "linear", "toggle"] : null);
      const input = document.createElement(options ? "select" : "input");
      if (options) for (const option of options) input.add(new Option(option, option));
      else {
        input.type = typeof value === "number" ? "number" : typeof value === "boolean" ? "checkbox" : "text";
        if (input.type === "number") input.step = "any";
      }
      input.value = value ?? "";
      if (input.type === "checkbox") input.checked = value;
      input.setAttribute("aria-label", next.join(" / "));
      input.oninput = () => {
        values[key] = input.type === "checkbox" ? input.checked :
          input.type === "number" ? Number(input.value) : input.value;
        this.dirty = true;
      };
      row.append(label, input);
      if (Array.isArray(values)) {
        const remove = document.createElement("button");
        remove.type = "button"; remove.className = "btn"; remove.textContent = "−";
        remove.title = "Remove value"; remove.setAttribute("aria-label", "Remove " + key);
        remove.onclick = event => {
          event.preventDefault(); values.splice(Number(key), 1);
          this.dirty = true; this.render();
        };
        row.appendChild(remove);
      }
      host.appendChild(row);
    }
  }

  async loadProfile(body) {
    try {
      const result = await apiPost("/api/camera_profile", {camera: this.data.selected.camera, ...body});
      if (result.ok === false) throw new Error(result.error);
      Object.assign(this.data.values.camera.settings, result.settings);
      $("device-profile-values").textContent = JSON.stringify(result.data, null, 2);
      this.dirty = true;
      this.render();
    } catch (error) {
      this.error = error.message;
    }
  }

  async apply() {
    this.error = "";
    this.pending = true;
    $("device-result").textContent = "APPLYING";
    try {
      const before = await apiGet("/api/device_settings");
      const response = await apiPost("/api/device_selection", {
        selected: this.data.selected, values: this.data.values,
      });
      if (response.ok === false) throw new Error(response.error);
      for (let attempt = 0; attempt < 120; attempt++) {
        await new Promise(resolve => setTimeout(resolve, 250));
        let status;
        try { status = await apiGet("/api/device_settings", {timeoutMs: 1000}); }
        catch { continue; }
        if (status.state === "failed") throw new Error(status.error);
        if (status.boot_id !== before.boot_id) {
          location.reload();
          return;
        }
      }
      throw new Error("Client restart timed out");
    } catch (error) {
      this.error = error.message;
      this.pending = false;
    }
  }

  update() {
    if (!this.data && !this.pending) this.load();
    const moving = !!S.STATUS.collection_teleop_armed || !!S.STATUS.manual_publish_active ||
      S.STATUS.session_status === "running";
    $("device-editors").disabled = this.pending || moving;
    $("device-start").disabled = this.pending || this.dirty || moving;
    if (!this.pending) $("device-result").textContent = this.error || (this.dirty ? "UNSAVED" : "");
    if (this.polling || Date.now() - this.lastPoll < 1000) return;
    this.polling = true;
    this.lastPoll = Date.now();
    apiGet("/api/device_settings").then(status => {
      $("device-log").textContent = status.log;
      $("device-open-input").hidden = !status.browser_url;
      $("device-open-input").href = status.browser_url || "#";
      if (!this.pending && !this.dirty) $("device-result").textContent = this.error || status.error || status.state.toUpperCase();
      const component = $("device-component").value;
      if (component && status.processes[component] === null) $("device-start").disabled = true;
    }).catch(error => {
      if (!this.pending) $("device-result").textContent = error.message;
    }).finally(() => { this.polling = false; });
  }
}

const panel = new DevicePanel();
export function renderDeviceSettings() { panel.update(); }
