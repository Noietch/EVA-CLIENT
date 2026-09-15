"""Device command polling with isolated responses, without hardware or a browser."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def test_device_command_polling():
    node = shutil.which("node")
    assert node is not None, "Install Node.js to run device panel tests"
    result = subprocess.run(
        [node, "--experimental-vm-modules", "--input-type=module"],
        cwd=ROOT,
        input=r"""
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

async function check(ack, statuses, expectedError = null) {
  const alerts = [];
  let polls = 0;
  const context = vm.createContext({
    window: {alert: message => alerts.push(message)},
    setTimeout: callback => callback(),
  });
  const core = new vm.SyntheticModule(["$", "S", "apiGet", "apiPost"], function () {
    this.setExport("$", () => null);
    this.setExport("S", {STATUS: {}});
    this.setExport("apiPost", async () => ack);
    this.setExport("apiGet", async () => {
      assert.ok(polls < statuses.length, "Unexpected extra status poll");
      return statuses[polls++];
    });
  }, {context});
  const source = fs.readFileSync("src/core/app/console/static/js/device.js", "utf8");
  const module = new vm.SourceTextModule(source + "\nexport { panel };", {context});
  await module.link(() => core);
  await module.evaluate();
  const panel = module.namespace.panel;
  await panel.command("robot");
  assert.equal(panel.busy.size, 0);
  assert.equal(polls, statuses.length);
  if (expectedError) {
    assert.equal(alerts.length, 1);
    assert.match(alerts[0], expectedError);
    assert.ok(!alerts[0].includes("Cannot read properties"));
  } else {
    assert.deepEqual(alerts, []);
  }
}

const ready = {
  processes: {robot: null}, ready: {robot: true},
  operations: {robot: {id: "new", state: "ready"}},
};
await check({ok: true}, [], /server is out of date/);
await check({ok: true, request_id: ""}, [], /server is out of date/);
await check({ok: true, request_id: "new"}, [
  {processes: {}},
  {processes: {}, operations: {}},
  {processes: {}, operations: {robot: null}},
  {...ready, operations: {robot: {id: "old", state: "failed", error: "old error"}}},
  ready,
]);
await check({ok: true, request_id: "new"}, [
  {processes: {}, operations: {robot: {id: "new", state: "failed", error: "Homing failed"}}},
], /Homing failed/);
""",
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_connect_pico_ensures_robot_and_teleop_once():
    node = shutil.which("node")
    assert node is not None, "Install Node.js to run device panel tests"
    result = subprocess.run(
        [node, "--experimental-vm-modules", "--input-type=module"],
        cwd=ROOT,
        input=r"""
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

const calls = [];
const statuses = [
  {processes: {}, ready: {}, operations: {}},
  {processes: {}, ready: {robot: true}, operations: {robot: {id: "robot-1", state: "ready"}}},
  {processes: {robot: null}, ready: {robot: true}, operations: {}},
  {processes: {robot: null}, ready: {robot: true, teleop: true}, operations: {teleop: {id: "teleop-1", state: "ready"}}},
];
let statusIndex = 0;
const context = vm.createContext({
  window: {alert: message => { throw new Error(`unexpected alert: ${message}`); }},
  setTimeout: callback => callback(),
});
const core = new vm.SyntheticModule(["$", "S", "apiGet", "apiPost"], function () {
  this.setExport("$", () => null);
  this.setExport("S", {STATUS: {teleop: {connected: false}}});
  this.setExport("apiGet", async path => {
    assert.equal(path, "/api/device_settings");
    assert.ok(statusIndex < statuses.length, "unexpected status poll");
    return statuses[statusIndex++];
  });
  this.setExport("apiPost", async (path, body) => {
    calls.push({path, body});
    if (path === "/api/device_start") return {ok: true, request_id: body.component + "-1"};
    if (path === "/api/device_open_pico") return {ok: true};
    throw new Error(`unexpected POST ${path}`);
  });
}, {context});
const source = fs.readFileSync("src/core/app/console/static/js/device.js", "utf8");
const module = new vm.SourceTextModule(source + "\nexport { panel };", {context});
await module.link(() => core);
await module.evaluate();
const panel = module.namespace.panel;
panel.updateControls = () => {};
panel.data = {selected: {teleop: "eva_pico"}};
panel.status = {processes: {}, ready: {}, operations: {}};
await panel.connectPico();
assert.equal(JSON.stringify(calls), JSON.stringify([
  {path: "/api/device_start", body: {component: "robot"}},
  {path: "/api/device_start", body: {component: "teleop"}},
  {path: "/api/device_open_pico", body: {}},
]));
assert.equal(statusIndex, statuses.length);

statuses.push(
  {processes: {robot: null, teleop: null}, ready: {robot: true, teleop: true}, operations: {}},
  {processes: {robot: null, teleop: null}, ready: {robot: true, teleop: true}, operations: {}},
);
calls.length = 0;
await panel.connectPico();
assert.equal(JSON.stringify(calls), JSON.stringify([
  {path: "/api/device_open_pico", body: {}},
]));
""",
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
