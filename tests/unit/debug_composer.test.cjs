const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const source = fs.readFileSync(path.join(__dirname, '../../src/core/app/console/static/js/run.js'), 'utf8');
const logic = source.slice(source.indexOf('// A send is an explicit intent;'), source.indexOf('function formatImageHz'));
function harness(status = {}, respond = async () => ({ ok: true })) {
  const nodes = new Map();
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, {
      value: '', dataset: {}, style: {}, parentElement: {}, children: [],
      setAttribute(k, v) { this[k] = v; },
      replaceChildren(...children) { this.children = children; },
      appendChild(child) { this.children.push(child); },
      get selectedOptions() { return this.children.filter(x => x.value === this.value); },
      focus() {},
    });
    return nodes.get(id);
  };
  const calls = [];
  const S = { CFG: { models: [{ index: 0, name: 'Alpha' }, { index: 1, name: 'Beta' }] }, ACTIVE_TAB: 'debug', STATUS: { selected_task: 'pour soybean', cli_mode: 'sim', selected_strategy: 'sync', web_phase: 'unset', ...status } };
  const context = {
    S, $: node, Date, console,
    document: { createElement() { return {}; }, createElementNS() { return { setAttribute() {}, appendChild() {} }; } },
    setCommandMetadata() {}, evalEnabled: () => false,
    uiMode: mode => mode === 'debug_sim_real' ? 'step' : mode,
    updateGuide() {}, renderSetupCtl() {}, mark() {},
    apiPost: async (url, body) => { calls.push({ url, body }); return respond(url, body); },
  };
  const setupActions = source.slice(source.indexOf('function pauseSetup()'), source.indexOf('function renderSetupCtl()'));
  vm.runInNewContext(logic + setupActions + '\nthis.actions = { changeDebugMode, newDebugRun, submitDebugPrompt, syncDebugComposer, cancelDebugIntent, renderDebugModels, syncDebugModelSelection, syncDebugDefaultMode };', context);
  node('debug-prompt-input').value = 'pour soybean';
  return { S, node, calls, ...context.actions, send: () => context.actions.submitDebugPrompt({ preventDefault() {} }) };
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test('a prepared Debug session in idle phase dispatches the send', async () => {
  const h = harness({ cli_mode: 'real', is_setup_done: true, web_phase: 'idle' });
  await h.send();
  await tick();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/run']);
});

test('send validates configuration before requesting robot actions', async () => {
  const h = harness({ selected_strategy: null });
  await h.send();
  assert.equal(h.calls.length, 0);
  assert.match(h.node('debug-prompt-status').textContent, /策略/);
});

test('changed prompt waits for matching task and setup before running once', async () => {
  const h = harness({ is_setup_done: true, web_phase: 'ready' });
  h.node('debug-prompt-input').value = 'put cup';
  await h.send();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/select_task']);
  h.syncDebugComposer(h.S.STATUS); // stale ready status from the previous task
  assert.equal(h.calls.length, 1);
  h.S.STATUS = { ...h.S.STATUS, selected_task: 'put cup', is_setup_done: false, web_phase: 'starting' };
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(h.calls.length, 1);
  h.S.STATUS = { ...h.S.STATUS, is_setup_done: true, web_phase: 'ready' };
  h.syncDebugComposer(h.S.STATUS);
  await tick();
  h.syncDebugComposer(h.S.STATUS);
  await h.send(); // duplicate click while waiting for run acknowledgement
  assert.deepEqual(h.calls.map(x => x.url), ['/api/select_task', '/api/run']);
});

test('Approve requires a separate send after preview is available', async () => {
  const h = harness({ cli_mode: 'debug_sim_real', is_setup_done: true, web_phase: 'ready', pending_chunk: null });
  await h.send();
  await tick();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/step_infer']);
  h.S.STATUS = { ...h.S.STATUS, pending_chunk: 8 };
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(h.node('debug-prompt-apply')['aria-label'], 'Approve');
  assert.equal(h.node('debug-prompt-input').readOnly, true);
  await h.send();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/step_infer', '/api/step_commit']);
});

test('same send button stops a running session', async () => {
  const h = harness({ session_status: 'running', web_phase: 'running' });
  await h.send();
  assert.equal(h.calls[0].url, '/api/halt');
});

test('reset, tab switch, and setup failure cancel queued run intent', async () => {
  for (const reason of ['reset', 'tab', 'error']) {
    const h = harness();
    await h.send();
    if (reason === 'reset') h.cancelDebugIntent();
    if (reason === 'tab') { h.S.ACTIVE_TAB = 'collect'; h.syncDebugComposer(h.S.STATUS); h.S.ACTIVE_TAB = 'debug'; }
    if (reason === 'error') h.syncDebugComposer({ ...h.S.STATUS, last_error: 'Policy disconnected' });
    h.syncDebugComposer({ ...h.S.STATUS, is_setup_done: true, web_phase: 'ready' });
    await tick();
    assert.equal(h.calls.length, 0, reason);
  }
});

test('rejected task selection never starts a run and allows retry', async () => {
  const h = harness({}, async () => ({ ok: false, error: 'Rejected' }));
  h.node('debug-prompt-input').value = 'new task';
  await h.send();
  h.syncDebugComposer({ ...h.S.STATUS, selected_task: 'new task', is_setup_done: true, web_phase: 'ready' });
  assert.deepEqual(h.calls.map(x => x.url), ['/api/select_task']);
  assert.equal(h.node('debug-prompt-apply').disabled, false);
  assert.equal(h.node('debug-prompt-status').textContent, 'Rejected');
});


test('model picker waits for backend acknowledgement and blocks sends during switch', async () => {
  const h = harness({ model_index: 0 });
  h.renderDebugModels();
  const select = h.node('debug-model-list');
  assert.deepEqual(select.children.map(x => x.textContent), ['Alpha', 'Beta']);
  select.value = '1';
  await select.onchange();
  assert.equal(h.calls[0].url, '/api/select_model');
  assert.equal(h.calls[0].body.index, 1);
  await h.send();
  assert.equal(h.calls.length, 1);
  assert.equal(h.node('debug-prompt-apply').disabled, true);
  h.S.STATUS = { ...h.S.STATUS, model_index: 1 };
  h.syncDebugModelSelection(h.S.STATUS);
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(select.value, '1');
  assert.equal(select.title, 'Beta');
  assert.equal(h.node('debug-prompt-apply').disabled, false);
});

test('running sessions cannot select another model', async () => {
  const h = harness({ session_status: 'running', model_index: 0 });
  h.renderDebugModels();
  const select = h.node('debug-model-list');
  assert.equal(select.disabled, true);
  select.value = '1';
  await select.onchange();
  assert.equal(h.calls.length, 0);
  assert.equal(select.value, '0');
});

test('a rejected model change restores selection and the send button', async () => {
  const h = harness({ model_index: 0 }, async () => ({ ok: false, error: 'Busy' }));
  h.renderDebugModels();
  const select = h.node('debug-model-list');
  select.value = '1';
  await select.onchange();
  assert.equal(select.value, '0');
  assert.equal(select.disabled, false);
  assert.equal(h.node('debug-prompt-apply').disabled, false);
  assert.equal(h.node('debug-prompt-status').textContent, 'Busy');
});

test('a queued send cannot run on a model changed by another client', async () => {
  const h = harness({ model_index: 0 });
  await h.send();
  h.syncDebugComposer({ ...h.S.STATUS, model_index: 1, is_setup_done: true, web_phase: 'ready' });
  await tick();
  assert.equal(h.calls.length, 0);
  assert.equal(h.node('debug-prompt-apply').disabled, false);
  assert.match(h.node('debug-prompt-status').textContent, /模型已切换/);
});


test('Approve becomes Stop while requesting the next preview, without auto-approving it', async () => {
  const h = harness({ cli_mode: 'step', pending_chunk: 8, step_preview_version: 1,
    step_operation: '', local_history: { rollout_id: 'one-run' }, is_setup_done: true });
  await h.send();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/step_commit']);
  h.S.STATUS = { ...h.S.STATUS, pending_chunk: null, step_operation: 'previewing' };
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(h.node('debug-prompt-apply').disabled, false);
  assert.equal(h.node('debug-prompt-apply').title, 'Stop');
  h.S.STATUS = { ...h.S.STATUS, pending_chunk: 8, step_operation: '', step_preview_version: 2 };
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(h.node('debug-prompt-apply').disabled, false);
  assert.equal(h.node('debug-prompt-apply')['aria-label'], 'Approve');
  assert.equal(h.calls.length, 1);
  await h.send();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/step_commit', '/api/step_commit']);
});


test('a live older server requests one new preview only after the full approved chunk executes', async () => {
  const h = harness({ cli_mode: 'step', pending_chunk: 8, step_index: 0, is_setup_done: true });
  await h.send();
  h.S.STATUS = { ...h.S.STATUS, pending_chunk: null, step_index: 8 };
  h.syncDebugComposer(h.S.STATUS);
  await tick();
  h.syncDebugComposer(h.S.STATUS);
  assert.deepEqual(h.calls.map(x => x.url), ['/api/step_commit', '/api/step_infer']);
  h.S.STATUS = { ...h.S.STATUS, pending_chunk: 8 };
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(h.node('debug-prompt-apply')['aria-label'], 'Approve');
  assert.equal(h.calls.length, 2);
});

test('cancel prevents a follow-up preview after an older server finishes its chunk', async () => {
  const h = harness({ cli_mode: 'step', pending_chunk: 8, step_index: 0, is_setup_done: true });
  await h.send();
  h.cancelDebugIntent();
  h.S.STATUS = { ...h.S.STATUS, pending_chunk: null, step_index: 8 };
  h.syncDebugComposer(h.S.STATUS);
  await tick();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/step_commit']);
});

test('loading the composer waits for backend mode and preserves a prepared approval session', async () => {
  const h = harness({ cli_mode: undefined });
  h.S.CFG.modes = ['real', 'sim', 'step'];
  h.syncDebugDefaultMode(h.S.STATUS);
  h.syncDebugDefaultMode({ cli_mode: 'step', is_setup_done: true, step_index: 48 });
  assert.equal(h.calls.length, 0);
  h.syncDebugDefaultMode({ cli_mode: 'select' });
  h.syncDebugDefaultMode({ cli_mode: 'select' });
  assert.equal(h.calls.length, 1);
  assert.equal(h.calls[0].body.mode, 'step');
});

test('Reset pause and resume use the main composer button without starting a run', async () => {
  const h = harness({ cli_mode: 'real', setup_stage: 'resetting to home position', web_phase: 'resetting' });
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(h.node('debug-prompt-apply').title, 'Stop');
  assert.equal(h.node('debug-prompt-apply').disabled, false);
  await h.send();
  assert.equal(h.calls[0].url, '/api/halt');
  assert.equal(h.node('debug-prompt-apply').disabled, true);
  h.S.STATUS = { ...h.S.STATUS, setup_stage: '', web_phase: 'idle' };
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(h.node('debug-prompt-apply').title, 'Resume');
  await h.send();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/halt', '/api/reset']);
});

test('pausing preparation cannot dispatch a run until explicitly resumed', async () => {
  const h = harness({ cli_mode: 'real', is_setup_done: false });
  await h.send();
  h.S.STATUS = { ...h.S.STATUS, setup_stage: 'warming up model', web_phase: 'starting' };
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(h.node('debug-prompt-apply').title, 'Stop');
  await h.send();
  h.S.STATUS = { ...h.S.STATUS, setup_stage: '', web_phase: 'ready', is_setup_done: true };
  h.syncDebugComposer(h.S.STATUS);
  assert.deepEqual(h.calls.map(x => x.url), ['/api/halt']);
  await h.send();
  await tick();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/halt', '/api/run']);
});

test('the main action can stop an approval operation without sending another approval', async () => {
  const h = harness({ cli_mode: 'step', step_operation: 'executing', local_history: { rollout_id: 'active' } });
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(h.node('debug-prompt-apply').title, 'Stop');
  assert.equal(h.node('debug-prompt-apply').disabled, false);
  assert.equal(h.node('debug-approval-stop').hidden, true);
  await h.send();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/halt']);
});

test('New run clears the composer and waits for acknowledgement before allowing send', async () => {
  const h = harness({ cli_mode: 'real', local_history: { generation: 4, current_rollout_id: 'current' } });
  await h.newDebugRun();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/new_debug_run']);
  assert.equal(h.node('debug-prompt-input').value, '');
  assert.equal(h.node('debug-prompt-apply').disabled, true);
  h.node('debug-prompt-input').value = 'next task';
  await h.send();
  assert.equal(h.calls.length, 1);
  h.S.STATUS = { ...h.S.STATUS, selected_task: null, local_history: { generation: 5, current_rollout_id: null } };
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(h.node('debug-prompt-apply').disabled, false);
});

test('a stopped current run offers Continue and cannot resume while its recording is saving', async () => {
  const h = harness({ cli_mode: 'real', local_history: { current_rollout_id: 'current', saving: true } });
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(h.node('debug-prompt-apply').title, 'Saving recording…');
  assert.equal(h.node('debug-prompt-apply').disabled, true);
  await h.send();
  assert.equal(h.calls.length, 0);
  h.S.STATUS.local_history.saving = false;
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(h.node('debug-prompt-apply').title, 'Continue run');
  assert.equal(h.node('debug-prompt-apply').disabled, false);
});

test('a mid-run permission switch waits for the backend and retains a usable Stop action', async () => {
  const h = harness({ cli_mode: 'real', session_status: 'running', is_setup_done: true, step_index: 12,
    local_history: { current_rollout_id: 'same-run', rollout_id: 'same-run' } });
  await h.changeDebugMode('step');
  assert.equal(h.S.STATUS.cli_mode, 'real');
  assert.equal(h.node('debug-prompt-apply').title, 'Stop');
  assert.equal(h.node('debug-prompt-apply').disabled, false);
  await h.changeDebugMode('step');
  assert.deepEqual(h.calls.map(x => x.url), ['/api/select_mode']);
  h.S.STATUS = { ...h.S.STATUS, cli_mode: 'step', session_status: 'ready',
    local_history: { current_rollout_id: 'same-run', rollout_id: null } };
  h.syncDebugComposer(h.S.STATUS);
  assert.equal(h.node('debug-prompt-apply').title, 'Continue run');
  assert.equal(h.S.STATUS.step_index, 12);
  assert.equal(h.calls.length, 1);
});

test('granting Full access discards a queued approval intent and requires an explicit send', async () => {
  const h = harness({ cli_mode: 'step', pending_chunk: 8, is_setup_done: true, web_phase: 'idle',
    local_history: { current_rollout_id: 'same-run' } });
  await h.send();
  await h.changeDebugMode('real');
  h.S.STATUS = { ...h.S.STATUS, cli_mode: 'real', pending_chunk: null, step_index: 2, session_status: 'ready' };
  h.syncDebugComposer(h.S.STATUS);
  await tick();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/step_commit', '/api/select_mode']);
  await h.send();
  await tick();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/step_commit', '/api/select_mode', '/api/run']);
});

test('a rejected permission change keeps the actual mode and existing approval usable', async () => {
  const h = harness({ cli_mode: 'step', pending_chunk: 8, is_setup_done: true }, async () => ({ ok: false, error: 'Cannot switch' }));
  await h.changeDebugMode('real');
  assert.equal(h.S.STATUS.cli_mode, 'step');
  assert.equal(h.node('debug-prompt-apply').title, 'Approve');
  assert.equal(h.node('debug-prompt-apply').disabled, false);
  assert.match(h.node('debug-prompt-status').textContent, /Cannot switch/);
});

test('stopping during a permission switch sends Halt instead of toggling execution', async () => {
  const h = harness({ cli_mode: 'real', session_status: 'running' });
  await h.changeDebugMode('step');
  await h.send();
  assert.deepEqual(h.calls.map(x => x.url), ['/api/select_mode', '/api/halt']);
});
