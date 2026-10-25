const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

const source = fs.readFileSync(path.resolve(__dirname, '../../src/core/app/console/static/js/local_history.js'), 'utf8');
const flush = () => new Promise(setImmediate);

class Element {
  constructor(tag = 'div') {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.attributes = {};
    const classes = new Set();
    this.classList = { toggle: (name, enabled) => enabled ? classes.add(name) : classes.delete(name) };
  }
  appendChild(child) { this.children.push(child); return child; }
  replaceChildren(...children) { this.children = children; }
  setAttribute(name, value) { this.attributes[name] = value; }
  querySelectorAll(tag) {
    return this.children.flatMap(child => [...(child.tag === tag ? [child] : []), ...child.querySelectorAll(tag)]);
  }
  focus() {}
}

async function harness({ recorded = true, failReplay = false, runs = 1 } = {}) {
  const elements = new Map();
  const $ = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const S = { ACTIVE_TAB: 'debug', CFG: {}, STATUS: { local_history: { enabled: true, record: false } } };
  const run = { id: 'rollout', prompt: 'Pick up a cup', model: 'test', mode: 'real', status: 'stopped', record: recorded, playable: recorded };
  const posts = [];
  const deleted = new Set();
  let played = 0;
  let historyReads = 0;
  const core = {
    $, S, LIVE: {}, setCommandMetadata() {}, releaseCameraImage() {},
    async apiGet(url) {
      if (url === '/api/local_sessions') historyReads++;
      if (url === '/api/local_sessions') return { sessions: [{ id: 'session', title: 'Pick up a cup', rollouts: runs, runs: Array.from({ length: runs }, (_, index) => ({ ...run, id: `rollout-${index}` })).filter(row => !deleted.has(row.id)) }] };
      return { session: { title: 'Pick up a cup', rollouts: Array.from({ length: runs }, (_, index) => ({ ...run, id: `rollout-${index}` })) } };
    },
    async apiPost(url, payload) {
      posts.push({ url, payload });
      if (url === '/api/delete_local_run') deleted.add(payload.rollout_id);
      if (url === '/api/restore_local_run') deleted.delete(payload.rollout_id);
      if (failReplay) return { ok: false, error: 'Recording cannot be read' };
      return { ok: true, dataset_dir: '/tmp/recorded-rollout' };
    },
  };
  const context = vm.createContext({ document: { hidden: false, createElement: tag => new Element(tag), querySelectorAll: () => [] }, Date, encodeURIComponent });
  async function moduleFor(exports) {
    const module = new vm.SyntheticModule(Object.keys(exports), function() {
      for (const [name, value] of Object.entries(exports)) this.setExport(name, value);
    }, { context });
    await module.link(() => {});
    await module.evaluate();
    return module;
  }
  const main = await moduleFor({ setActiveTab: tab => { S.ACTIVE_TAB = tab; } });
  const dependencies = {
    './core.js': await moduleFor(core),
    './replay.js': await moduleFor({ exitReplayMode() {}, loadMountedReplaySeries: async () => true, replayPlay: () => { played++; } }),
  };
  const module = new vm.SourceTextModule(source, { context, importModuleDynamically: async () => main });
  await module.link(id => dependencies[id]);
  await module.evaluate();
  module.namespace.initLocalHistory();
  await flush();
  return { $, S, posts, played: () => played, button: $('debug-session-list').children[0].children[0], row: $('debug-session-list').children[0], historyReads: () => historyReads, applyStatus: module.namespace.applyLocalHistoryStatus };
}

test('clicking a session with one run directly opens Replay', async () => {
  const h = await harness();
  h.button.onclick();
  await flush();
  assert.equal(h.S.ACTIVE_TAB, 'replay');
  assert.deepEqual(h.posts.map(row => row.url), ['/api/local_replay', '/api/tab_switch', '/api/select_mode', '/api/load_replay_dataset']);
  assert.equal(h.$('replay-dataset-input').value, '/tmp/recorded-rollout');
  assert.equal(h.played(), 1);
});

test('multiple runs render as one flat list and open the selected recording', async () => {
  const h = await harness({ runs: 2 });
  const runs = h.$('debug-session-list').children;
  assert.equal(runs.length, 2);
  assert.ok(runs.every(row => row.tag === 'div'));
  runs[1].children[0].onclick();
  await flush();
  assert.equal(h.S.ACTIVE_TAB, 'replay');
  assert.equal(h.posts[0].payload.rollout_id, 'rollout-1');
});

test('an unrecorded run is marked unavailable for playback', async () => {
  const h = await harness({ recorded: false });
  assert.equal(h.button.disabled, true);
  assert.equal(h.S.ACTIVE_TAB, 'debug');
  assert.equal(h.posts.length, 0);
  assert.equal(h.button.children[1].children[1].textContent, 'test · Not recorded');
});

test('click reports a playback failure without an unhandled rejection', async () => {
  const h = await harness({ failReplay: true });
  h.button.onclick();
  await flush();
  assert.equal(h.$('debug-history-status').textContent, 'Recording cannot be read');
  assert.equal(h.posts.length, 1);
});

test('trash deletes only the chosen run without opening Replay, and Undo restores it', async () => {
  const h = await harness();
  await h.row.children[1].onclick({ stopPropagation() {} });
  assert.equal(h.posts[0].url, '/api/delete_local_run');
  assert.equal(h.posts[0].payload.rollout_id, 'rollout-0');
  assert.equal(h.S.ACTIVE_TAB, 'debug');
  assert.equal(h.$('debug-session-list').children.length, 0);
  const undo = h.$('debug-history-status').children[1];
  assert.equal(undo.textContent, 'Undo');
  await undo.onclick();
  assert.equal(h.posts[1].url, '/api/restore_local_run');
  assert.equal(h.$('debug-session-list').children.length, 1);
});

test('starting and stopping refresh history immediately instead of waiting for the timer', async () => {
  const h = await harness();
  assert.equal(h.historyReads(), 1);
  h.S.STATUS.local_history.rollout_id = 'active-run';
  h.applyStatus(h.S.STATUS);
  await flush();
  assert.equal(h.historyReads(), 2);
  h.applyStatus(h.S.STATUS);
  await flush();
  assert.equal(h.historyReads(), 2);
  h.S.STATUS.local_history.rollout_id = null;
  h.applyStatus(h.S.STATUS);
  await flush();
  assert.equal(h.historyReads(), 3);
});

test('deleting and restoring a finished row stays available while another run is active', async () => {
  const h = await harness();
  h.S.STATUS.local_history.rollout_id = 'another-active-run';
  h.applyStatus(h.S.STATUS);
  await flush();
  const row = h.$('debug-session-list').children[0];
  assert.equal(row.children[0].disabled, true);
  assert.equal(row.children[1].disabled, false);
  await row.children[1].onclick({ stopPropagation() {} });
  assert.equal(h.posts[0].url, '/api/delete_local_run');
  await h.$('debug-history-status').children[1].onclick();
  assert.equal(h.posts[1].url, '/api/restore_local_run');
  assert.equal(h.S.STATUS.local_history.rollout_id, 'another-active-run');
});
