const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../../src/core/app/console/static/js/replay.js'), 'utf8');
const logic = source.slice(source.indexOf('async function readCameraJpeg('), source.indexOf('async function refreshLiveCameraImage('));
const context = vm.createContext({ Uint8Array, Error });
vm.runInContext(logic + '\nthis.readCameraJpeg = readCameraJpeg;', context);
function feed(chunks) {
  let canceled = false;
  let index = 0;
  return { get canceled() { return canceled; }, ok: true,
    body: { getReader() { return {
      async read() { return index < chunks.length ? { value: new Uint8Array(chunks[index++]), done: false } : { done: true }; },
      async cancel() { canceled = true; },
    }; } },
  };
}

test('reads one JPEG across split markers and releases the multipart connection', async () => {
  const response = feed([[45,45,65,255], [216,1,2,3,255], [217,13,10,45,45,255,216,7,255,217]]);
  const jpeg = await context.readCameraJpeg(response);
  assert.deepEqual(Array.from(jpeg), [255,216,1,2,3,255,217]);
  assert.equal(response.canceled, true);
});

test('releases a truncated stream rather than occupying a connection indefinitely', async () => {
  const response = feed([[45,255,216,1,2]]);
  await assert.rejects(context.readCameraJpeg(response), /no complete JPEG/);
  assert.equal(response.canceled, true);
});

test('rejects unavailable feeds', async () => {
  await assert.rejects(context.readCameraJpeg({ ok: false }), /unavailable/);
});
