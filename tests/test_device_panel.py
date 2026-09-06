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
        input=r'''
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
''',
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
