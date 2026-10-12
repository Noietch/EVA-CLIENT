const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Video extends EventTarget {
  constructor(duration = 20, ranges = [[0, 1]]) {
    super();
    this.duration = duration;
    this.ranges = ranges;
    this.readyState = 3;
    this.currentTime = 0;
    this.error = null;
    this.dataset = {src: "/clip.mp4"};
    this.attributes = {};
    this.plays = 0;
  }

  get buffered() {
    return {
      length: this.ranges.length,
      start: (index) => this.ranges[index][0],
      end: (index) => this.ranges[index][1],
    };
  }

  getAttribute(name) { return this.attributes[name]; }
  setAttribute(name, value) { this.attributes[name] = value; }
  closest() { return null; }
  play() { this.plays += 1; return Promise.resolve(); }
  pause() {}
  progress(ranges) {
    this.ranges = ranges;
    this.dispatchEvent(new Event("progress"));
  }
}

function loadPlayer(kind, videos, elapsed = 0) {
  const file = kind === "console"
    ? "src/core/app/console/static/js/replay.js"
    : "tools/datasets/static/review.js";
  const source = fs.readFileSync(path.join(__dirname, "../..", file), "utf8")
    .replace(/import[\s\S]*?from "[^"]+";/g, "")
    .replace(/export\s*\{[^}]*\};/g, "");
  const controls = {
    "review-play": {textContent: "▶"},
    "review-range": {value: "0", disabled: false},
  };
  const state = {
    playing: false,
    review: {series: {state: [[0], [1]], timestamp: [elapsed, elapsed + 10]}},
  };
  const context = vm.createContext({
    AbortController,
    window: {},
    LIVE: {cursor: elapsed, n: 21, playTime: Array.from({length: 21}, (_, i) => i)},
    $: (id) => id === "cam-strip" ? {querySelectorAll: () => videos} : controls[id],
    document: {
      getElementById: (id) => controls[id],
      querySelectorAll: () => videos,
    },
    clientTrace() {},
    performance: {now: () => 0},
    requestAnimationFrame: () => 1,
    cancelAnimationFrame() {},
  });
  const expose = kind === "console"
    ? `globalThis.wait = (video, signal) => {
         replayVideoAbortController = {signal}; return waitForVideoReady(video);
       }; globalThis.waitAll = waitForStageVideosReady;
       globalThis.setSignal = (signal) => {replayVideoAbortController = {signal};};`
    : `configureReview({app: state, showToast() {}, translate: (key) => key});
       globalThis.wait = (video, signal, elapsed) => waitForReviewVideoBuffer(video, signal, elapsed);
       globalThis.toggle = togglePlayback; globalThis.stop = stopPlayback;`;
  context.state = state;
  vm.runInContext(source + "\n" + expose, context);
  return {context, controls, state};
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

for (const kind of ["console", "tools"]) {
  test(kind + ": waits for five buffered seconds, even after canplay", async () => {
    const video = new Video();
    const {context} = loadPlayer(kind, [video]);
    let ready = false;
    const pending = context.wait(video, new AbortController().signal, 0).then(() => { ready = true; });
    video.dispatchEvent(new Event("canplay"));
    await flush();
    assert.equal(ready, false);
    video.progress([[0, 4.9]]);
    await flush();
    assert.equal(ready, false);
    video.progress([[0, 5]]);
    await pending;
    assert.equal(ready, true);
  });

  test(kind + ": buffers the entire short clip and the remaining tail", async () => {
    for (const [duration, elapsed, ranges] of [[3, 0, [[0, 3]]], [20, 18, [[18, 20]]]]) {
      const video = new Video(duration, ranges);
      video.currentTime = elapsed;
      const {context} = loadPlayer(kind, [video], elapsed);
      await context.wait(video, new AbortController().signal, elapsed);
      assert.equal(video.currentTime, elapsed);
    }
  });

  test(kind + ": requires a continuous buffer at the seek position", async () => {
    const video = new Video(20, [[0, 10], [12, 18]]);
    const {context} = loadPlayer(kind, [video], 9);
    let ready = false;
    const pending = context.wait(video, new AbortController().signal, 9).then(() => { ready = true; });
    await flush();
    assert.equal(ready, false);
    assert.equal(video.currentTime, 9);
    video.progress([[9, 14]]);
    await pending;
  });

  test(kind + ": abort and media errors release pending buffer waits", async () => {
    const video = new Video();
    const {context} = loadPlayer(kind, [video]);
    const controller = new AbortController();
    const pending = context.wait(video, controller.signal, 0);
    controller.abort();
    await pending;
    const failed = context.wait(video, new AbortController().signal, 0);
    video.error = {code: 4};
    video.dispatchEvent(new Event("error"));
    assert.equal(await failed, false);
  });

  test(kind + ": seek past a short camera clip fails instead of waiting forever", async () => {
    const video = new Video(3, [[0, 3]]);
    const {context} = loadPlayer(kind, [video], 4);
    assert.equal(await context.wait(video, new AbortController().signal, 4), false);
  });

  test(kind + ": one camera error releases the other camera's buffer wait", async () => {
    const videos = [new Video(), new Video()];
    const {context, state, controls} = loadPlayer(kind, videos);
    const pending = kind === "console" ? context.waitAll() : context.toggle();
    await flush();
    videos[0].error = {code: 4};
    videos[0].dispatchEvent(new Event("error"));
    assert.equal(await pending, kind === "console" ? false : undefined);
    if (kind === "tools") {
      assert.equal(state.playing, false);
      assert.equal(controls["review-range"].disabled, false);
      assert.equal(controls["review-play"].textContent, "▶");
    }
    assert.deepEqual(videos.map((video) => video.plays), [0, 0]);
  });
}

test("console: every camera must finish buffering", async () => {
  const videos = [new Video(20, [[0, 5]]), new Video()];
  const {context} = loadPlayer("console", videos);
  let ready = false;
  const pending = context.waitAll().then((value) => { ready = value; });
  await flush();
  assert.equal(ready, false);
  videos[1].progress([[0, 5]]);
  await pending;
  assert.equal(ready, true);
});

test("console: leaving the episode cancels every camera buffer wait", async () => {
  const videos = [new Video(), new Video()];
  const {context} = loadPlayer("console", videos);
  const controller = new AbortController();
  context.setSignal(controller.signal);
  const pending = context.waitAll();
  controller.abort();
  assert.equal(await pending, undefined);
  videos.forEach((video) => video.progress([[0, 5]]));
  assert.deepEqual(videos.map((video) => video.plays), [0, 0]);
});

test("tools: all cameras start together and cancellation prevents stale autoplay", async () => {
  const videos = [new Video(20, [[0, 5]]), new Video()];
  const {context, state, controls} = loadPlayer("tools", videos);
  const pending = context.toggle();
  await flush();
  assert.equal(state.playing, false);
  assert.equal(videos[0].plays, 0);
  assert.equal(controls["review-range"].disabled, true);
  videos[1].progress([[0, 5]]);
  await pending;
  assert.equal(state.playing, true);
  assert.deepEqual(videos.map((video) => video.plays), [1, 1]);
  context.stop();
  videos[1].ranges = [[0, 1]];
  const cancelled = context.toggle();
  context.stop();
  videos[1].progress([[0, 5]]);
  await cancelled;
  assert.equal(state.playing, false);
  assert.equal(controls["review-range"].disabled, false);
  assert.deepEqual(videos.map((video) => video.plays), [1, 1]);
});
