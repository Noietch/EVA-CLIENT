const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {getEventListeners} = require("node:events");

async function loadCore(fetch) {
  const context = vm.createContext({
    AbortController,
    setTimeout,
    clearTimeout,
    fetch,
    window: {addEventListener() {}},
    sessionStorage: {getItem: () => "test-client"},
    console: {info() {}},
  });
  const source = fs.readFileSync(
    path.join(__dirname, "../../src/core/app/console/static/js/core.js"), "utf8",
  );
  const module = new vm.SourceTextModule(source, {context});
  await module.link(() => { throw new Error("core.js should not import dependencies"); });
  await module.evaluate();
  return module.namespace.apiGet;
}

test("主动取消 GET 会中止 fetch，且不提交异常 trace", async () => {
  const requests = [];
  const apiGet = await loadCore((url, {signal}) => {
    requests.push(url);
    return new Promise((resolve, reject) => {
      signal.addEventListener("abort", () => {
        const error = new Error("Cancelled");
        error.name = "AbortError";
        reject(error);
      }, {once: true});
    });
  });
  const controller = new AbortController();
  const request = apiGet("/api/collection_slots", {signal: controller.signal});
  controller.abort();
  await assert.rejects(request, {name: "AbortError"});
  assert.deepEqual(requests, ["/api/collection_slots"]);
  assert.equal(getEventListeners(controller.signal, "abort").length, 0);
});

test("无超时 GET 仍传递外部 signal，成功后清理转发监听", async () => {
  const signals = [];
  const apiGet = await loadCore(async (url, {signal}) => {
    signals.push(signal);
    return {ok: true, status: 200, text: async () => '{"ok":true}'};
  });
  const controller = new AbortController();
  await apiGet("/api/collection_slots", {timeoutMs: 0, signal: controller.signal});
  assert.equal(signals[0], controller.signal);
  await apiGet("/api/collection_slots", {signal: controller.signal});
  assert.equal(getEventListeners(controller.signal, "abort").length, 0);
});
