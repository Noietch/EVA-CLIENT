"""Exercise the live camera DOM transitions without camera hardware."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_camera_warning_retains_panes_and_reconnects():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the live UI state test")
    source = Path("src/core/app/console/static/js/replay.js").read_text()
    poll = source.split("async function pollFrame()", 1)[1].split("let liveSeriesPolling", 1)[0]
    script = r'''
const assert = require("node:assert/strict");
let framePolling = false;
const S = {ACTIVE_TAB: "collect"}, LIVE = {replayMode: false};
const liveStageActive = () => true, syncGripperState = () => {};
let payload, images = [], builds = 0;
const apiGet = async () => payload;
const strip = {querySelectorAll: () => images};
const $ = () => strip;
function replaceCamStripContent(html) {
  builds++;
  images = [...html.matchAll(/data-key="([^"]+)"/g)].map((match) => ({
    dataset: {key: match[1]},
    attributes: {}, stale: false,
    closest() {return {classList: {toggle: (_, stale) => {this.stale = stale;}}};},
    removeAttribute(key) {delete this.attributes[key];},
    hasAttribute(key) {return key in this.attributes;},
    set src(value) {this.attributes.src = value;},
  }));
}
'''
    script += "\nasync function pollFrame()" + poll
    script += r'''
(async () => {
  payload = {cameras: ["left", "right"], camera_health: {
    left: {stale: false}, right: {stale: false}}};
  await pollFrame();
  assert.equal(images.length, 2);
  const original = images[0];
  assert(original.hasAttribute("src"));
  payload.cameras = ["right"];
  payload.camera_health.left.stale = true;
  await pollFrame();
  assert.equal(images[0], original);
  assert.equal(builds, 1);
  assert.equal(original.stale, true);
  assert(!original.hasAttribute("src"));
  assert.equal(images[1].stale, false);
  assert(images[1].hasAttribute("src"));
  payload.camera_health.left.stale = false;
  await pollFrame();
  assert.equal(original.stale, false);
  assert(original.hasAttribute("src"));
  LIVE.replayMode = true;
  payload.camera_health.left.stale = true;
  await pollFrame();
  assert.equal(original.stale, false);
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)
