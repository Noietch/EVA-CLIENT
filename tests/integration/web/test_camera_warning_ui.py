"""Exercise the live camera DOM transitions without camera hardware."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_camera_streams_stay_open_across_polls():
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
    attributes: {},
    removeAttribute(key) {delete this.attributes[key];},
    hasAttribute(key) {return key in this.attributes;},
    set src(value) {this.attributes.src = value;},
  }));
}
'''
    script += "\nasync function pollFrame()" + poll
    script += r'''
(async () => {
  payload = {cameras: ["left", "right"]};
  await pollFrame();
  assert.equal(images.length, 2);
  const original = images[0];
  assert.equal(original.attributes.src, "/api/camera/left");
  await pollFrame();
  assert.equal(images[0], original);
  assert.equal(builds, 1);
  assert(original.hasAttribute("src"));
  payload.cameras = ["right"];
  await pollFrame();
  assert.equal(images.length, 1);
  assert.equal(images[0].attributes.src, "/api/camera/right");
  LIVE.replayMode = true;
  payload.cameras = [];
  await pollFrame();
  assert.equal(images.length, 1);
})().catch(error => {console.error(error); process.exitCode = 1;});
'''
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)
