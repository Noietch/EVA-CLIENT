// Dataset manager card: per-task-set transfers against the Hugging Face repo.
// Mirrors the EVA-CLIENT console card; rows come from /api/transfers.

const $ = (id) => document.getElementById(id);

let context = null;
let datasets = [];
let remote = {};
let job = null;
let jobs = [];
let pollTimer = null;
let requesting = false;
let loaded = false;
let sortKey = "name";
let sortAscending = true;
let robotFilter = "";
let robots = [];
const selected = new Set();

const SORT_KEYS = ["status", "name", "collected", "total", "progress", "pending", "accept",
  "fail", "unreviewed", "cloud_episodes", "cloud_tasks", "cloud_accept", "cloud_fail",
  "cloud_unreviewed", "verify_data", "verify_task", "verify_qc"];
const RESOURCES = ["local_data", "local_qc", "local_task", "remote_data", "remote_qc",
  "remote_task"];
const VERIFY_ORDER = { same: 0, absent: 1, unknown: 2, different: 3 };
const STATUS_ORDER = { error: 0, warn: 1, unknown: 2, ok: 3 };

const t = (key) => context.translate(key);
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g,
  (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
const busy = () => requesting;
const running = () => jobs.some((item) => item.state === "running");
// Automatic QC sweeps every dataset when nothing is selected; the other
// actions only ever run on the selection.
const sweepAll = (action) => action === "auto_qc" && !selected.size;
const syncActionButtons = () => {
  document.querySelectorAll("[data-dm-action]").forEach((button) => {
    button.disabled = busy() || (!selected.size && button.dataset.dmAction !== "auto_qc");
  });
  $("dm-refresh").disabled = busy() || !datasets.length;
};
const sortOption = (key, ascending) => key === "name" ? (ascending ? "name" : "name-desc")
  : `${key}${ascending ? "-asc" : ""}`;

export function configureDatasetTransfer(deps) {
  context = deps;
}

function sortValue(row) {
  const cloud = remote[row.name];
  if (sortKey === "status") return STATUS_ORDER[statusOf(row).state];
  if (sortKey === "name") return row.name;
  if (sortKey === "progress") {
    return row.total > 0 && row.collected != null ? row.collected / row.total : null;
  }
  if (sortKey === "pending") {
    return row.total != null && row.collected != null ? Math.max(0, row.total - row.collected) : null;
  }
  if (sortKey === "cloud_episodes") return cloud?.checked_at ? cloud.episodes : null;
  if (sortKey === "cloud_tasks") return cloud?.checked_at ? cloud.task_count : null;
  if (sortKey.startsWith("cloud_")) return cloud?.qc?.[sortKey.slice(6)] ?? null;
  if (sortKey.startsWith("verify_")) {
    const state = cloud?.verification?.[sortKey.slice(7)]?.state;
    return VERIFY_ORDER[state] ?? null;
  }
  return row[sortKey] ?? null;
}

function acceptSnapshot(data) {
  jobs = data.jobs || (data.job ? [data.job] : []);
  job = jobs.at(-1) || null;
  remote = data.remote || {};
}

function visibleRows() {
  const query = $("dm-search").value.trim().toLowerCase();
  return datasets.filter((row) => (!robotFilter || row.robot === robotFilter)
      && `${row.name} ${row.robot || ""}`.toLowerCase().includes(query))
    .sort((a, b) => {
      const av = sortValue(a);
      const bv = sortValue(b);
      if (av == null || bv == null) {
        return av == null && bv == null ? a.name.localeCompare(b.name) : av == null ? 1 : -1;
      }
      const delta = sortKey === "name" ? av.localeCompare(bv) : av - bv;
      return delta * (sortAscending ? 1 : -1) || a.name.localeCompare(b.name);
    });
}

function sortTitle(key, active) {
  const label = t(`dm.sort.${key}`);
  const order = active && sortAscending ? t("dm.desc") : t("dm.asc");
  return `${t("dm.sortPrefix")}${label}${order}${t("dm.sortSuffix")}`;
}

// How many files/rows the cloud and the local copy disagree on.
function verifyDiff(cloud) {
  const verification = cloud?.verification;
  if (!verification) return null;
  return ["data", "task", "qc"].reduce((total, kind) => {
    const item = verification[kind];
    if (!item) return total;
    return total + ["missing_local", "local_only", "changed", "unknown"]
      .reduce((count, field) => count + (item[field]?.length || 0), 0);
  }, 0);
}

function statusOf(row) {
  const cloud = remote[row.name];
  const pending = row.total != null && row.collected != null
    ? Math.max(0, row.total - row.collected) : null;
  const verification = cloud?.verification;
  const mismatch = Boolean(verification && ["data", "task", "qc"].some((kind) =>
    verification[kind] && !["same", "absent"].includes(verification[kind].state)));
  const issues = [];
  if (cloud?.error) issues.push(t("dm.issue.cloud"));
  if (mismatch) issues.push(t("dm.issue.verify"));
  if (cloud?.stale) issues.push(t("dm.issue.stale"));
  if (row.fail) issues.push(`${t("dm.col.fail")} ${row.fail}`);
  if (row.unreviewed) issues.push(`${t("dm.col.unreviewed")} ${row.unreviewed}`);
  if (pending) issues.push(`${t("dm.sort.pending")} ${pending}`);
  const state = cloud?.error || mismatch || row.fail ? "error"
    : issues.length ? "warn"
    : cloud?.checked_at ? "ok" : "unknown";
  return { state, issues };
}

function verification(item) {
  if (!item) return `<span class="dm-muted">${t("dm.verifyUnchecked")}</span>`;
  const label = {
    same: t("dm.verify.same"), different: t("dm.verify.different"),
    absent: t("dm.verify.absent"), unknown: t("dm.verify.unknown"),
  }[item.state];
  const details = [
    [t("dm.missingLocal"), item.missing_local], [t("dm.localOnly"), item.local_only],
    [t("dm.changed"), item.changed], [t("dm.unknown"), item.unknown],
  ].filter(([, values]) => values?.length)
    .map(([title, values]) => `${title}: ${values.join(", ")}`).join("\n");
  return `<span class="dm-check ${item.state === "same" ? "ok" : "err"}" title="${escapeHtml(details)}">${label}</span>`;
}

function jobMarkup(task, expandedJobs) {
  const failures = task.results.filter((item) => !item.ok).length;
  const active = task.state === "running";
  const stopping = task.stopping || task.stop_requested;
  const eta = active ? task.waiting ? t("dm.waiting") : task.eta == null ? t("dm.estimating")
    : t("dm.aboutMinutes").replace("{n}", Math.ceil(task.eta / 60)) : "0";
  const waiting = task.waiting
    ? `${t("dm.waitingResources")}${(task.waiting_resources || [])
      .map((name) => t(`dm.resource.${name}`)).join("、")} · ` : "";
  const detail = active
    ? `${waiting}${task.current}${task.detail ? ` · ${task.detail}` : ""}`
    : task.state === "stopped" ? t("dm.halted") : t("dm.done");
  return `<div class="dm-job" data-dm-job="${escapeHtml(task.id || "legacy")}">
      <strong>${t(`dm.action.${task.action}`)} · ${task.completed}/${task.total}</strong>
      <span class="dm-job-detail">${escapeHtml(detail)}${failures ? ` · ${failures} ${t("dm.failedCount")}` : ""}</span>
      <span>ETA ${eta}</span>
      ${active ? `<button class="btn" data-dm-stop="${escapeHtml(task.id)}" ${stopping ? "disabled" : ""}>${stopping ? t("dm.stopping") : t("dm.stopNext")}</button>` : ""}
      <progress max="${task.total || 1}" value="${task.completed}" aria-label="${t(`dm.action.${task.action}`)}"></progress>
      ${failures ? `<details${expandedJobs.has(task.id || "legacy") ? " open" : ""}><summary>${t("dm.failureDetail")} (${failures})</summary>${task.results.filter((item) => !item.ok).map((item) => `<div>${escapeHtml(item.dataset)}：${escapeHtml(item.error)}</div>`).join("")}</details>` : ""}
    </div>`;
}

export function renderTransfers() {
  if (!loaded) return;
  const rows = visibleRows();
  $("dm-sort").value = sortOption(sortKey, sortAscending);
  document.querySelectorAll("[data-dm-sort]").forEach((button) => {
    const active = button.dataset.dmSort === sortKey;
    button.closest("th").setAttribute("aria-sort",
      active ? (sortAscending ? "ascending" : "descending") : "none");
    button.dataset.direction = active ? (sortAscending ? "asc" : "desc") : "none";
    button.title = sortTitle(button.dataset.dmSort, active);
  });
  const results = new Map(jobs.flatMap((task) => task.results.map((item) =>
    [item.dataset, { ...item, action: task.action }])));
  const focused = document.activeElement;
  const focusKey = ["dmSelect", "dmOpen", "dmRow"].find((key) => focused?.dataset?.[key]);
  const focusName = focusKey ? focused.dataset[focusKey] : null;
  const markup = rows.map((row) => {
    const cloud = remote[row.name];
    const result = results.get(row.name);
    const status = statusOf(row);
    const diff = verifyDiff(cloud);
    const ratio = row.total ? Math.min(100, Math.round(row.collected / row.total * 100)) : 0;
    const name = escapeHtml(row.name);
    const classes = [selected.has(row.name) ? "dm-selected" : "", `dm-${status.state}`].filter(Boolean).join(" ");
    const outcome = result
      ? (result.ok
        ? `${t(`dm.action.${result.action}`)}${t("dm.doneSuffix")}${result.detail ? ` · ${result.detail}` : ""}`
        : result.error)
      : cloud?.error || "";
    return `<tr data-dm-row="${name}" tabindex="0" aria-label="${t("dm.selectPrefix")}${name}" aria-selected="${selected.has(row.name)}" class="${classes}">
      <td><input type="checkbox" data-dm-select="${name}" aria-label="${t("dm.selectPrefix")}${name}" ${selected.has(row.name) ? "checked" : ""}></td>
      <td><button class="dm-name" data-dm-open="${name}" title="${t("dm.openHint")}${name}">${name}</button><small class="dm-dataset-id">${escapeHtml(row.robot || "")}</small>${row.error ? `<small class="dm-fail">${escapeHtml(row.error)}</small>` : ""}</td>
      <td>${row.collected ?? "--"}</td><td>${row.total ?? "--"}</td><td>${row.total > 0 ? `${ratio}%` : "--"}<progress max="100" value="${ratio}" aria-label="${name}"></progress></td>
      <td class="dm-accept">${row.accept ?? "--"}</td><td class="dm-fail">${row.fail ?? "--"}</td><td class="dm-unreviewed">${row.unreviewed ?? "--"}</td>
      <td data-dm-field="cloud_episodes">${cloud?.checked_at ? cloud.episodes : "--"}</td><td data-dm-field="cloud_tasks">${cloud?.checked_at ? cloud.task_count ?? "--" : "--"}</td>
      <td class="dm-accept" data-dm-field="cloud_accept">${cloud?.qc?.accept ?? "--"}</td><td class="dm-fail" data-dm-field="cloud_fail">${cloud?.qc?.fail ?? "--"}</td><td class="dm-unreviewed" data-dm-field="cloud_unreviewed">${cloud?.qc?.unreviewed ?? "--"}</td>
      <td>${verification(cloud?.verification?.data)}</td>
      <td>${verification(cloud?.verification?.task)}</td>
      <td>${verification(cloud?.verification?.qc)}</td>
      <td data-dm-field="status"><span class="dm-state dm-state-${status.state}">${t(`dm.status.${status.state}`)}</span>
        <small class="dm-issues">${escapeHtml(status.issues.length ? status.issues.join(" · ") : t("dm.issue.none"))}</small>
        <small class="dm-refresh">${cloud?.checked_at ? escapeHtml(new Date(cloud.checked_at * 1000).toLocaleString()) : t("dm.notRefreshed")}${diff === null ? "" : ` · ${t("dm.diff")} ${diff}`}</small>
        <span class="${result?.ok === false || cloud?.error ? "dm-fail" : ""}">${escapeHtml(outcome)}</span></td>
    </tr>`;
  }).join("");
  if ($("dm-rows").innerHTML !== markup) {
    $("dm-rows").innerHTML = markup;
    if (focusKey) {
      const replacement = [...$("dm-rows").querySelectorAll("tr, input, button")]
        .find((element) => element.dataset[focusKey] === focusName);
      replacement?.focus({ preventScroll: true });
    }
  }
  $("dm-empty").hidden = rows.length > 0;
  $("dm-selected").textContent =
    `${t("dm.selected")} ${selected.size} · ${t("dm.shown")} ${rows.length} / ${datasets.length}`;
  $("dm-clear").disabled = !selected.size;
  const all = $("dm-select-all");
  const allSelected = rows.length > 0 && rows.every((row) => selected.has(row.name));
  all.setAttribute("aria-pressed", String(allSelected));
  all.textContent = `${allSelected ? t("dm.deselectAll") : t("dm.selectAll")} (${rows.length})`;
  all.disabled = !rows.length;
  syncActionButtons();
  const activeJobs = jobs.filter((item) => item.state === "running");
  const jobList = $("dm-jobs");
  const expandedJobs = new Set([...jobList.querySelectorAll("details[open]")]
    .map((details) => details.closest("[data-dm-job]").dataset.dmJob));
  const focusedSummaryJob = document.activeElement?.matches("summary")
    ? document.activeElement.closest("[data-dm-job]")?.dataset.dmJob : null;
  const jobScrollTop = jobList.scrollTop;
  const visibleJobs = jobs.filter((task) => task.state !== "stopped" && !(task.state === "done"
    && task.completed === task.total && task.results.every((result) => result.ok)));
  jobList.innerHTML = [...visibleJobs].reverse().map((task) => jobMarkup(task, expandedJobs)).join("");
  jobList.scrollTop = jobScrollTop;
  if (focusedSummaryJob) {
    [...jobList.querySelectorAll("[data-dm-job]")]
      .find((element) => element.dataset.dmJob === focusedSummaryJob)
      ?.querySelector("summary")?.focus({ preventScroll: true });
  }
  if (job) {
    $("dm-progress").max = job.total || 1;
    $("dm-progress").value = job.completed;
    const failures = job.results.filter((item) => !item.ok).length;
    $("dm-status").textContent = activeJobs.length
      ? t("dm.activeJobs").replace("{n}", activeJobs.length)
      : `${t(`dm.action.${job.action}`)} · ${job.completed}/${job.total} · ${t("dm.done")}${failures ? ` · ${failures} ${t("dm.failedCount")}` : ""}`;
    $("dm-eta").textContent = "";
    $("dm-progress").hidden = true;
  }
}

function renderRobotFilter() {
  const found = [...new Set(datasets.map((row) => row.robot || ""))].filter(Boolean).sort();
  if (found.join(" ") === robots.join(" ")) return;
  robots = found;
  if (robotFilter && !robots.includes(robotFilter)) robotFilter = "";
  $("dm-robot").replaceChildren(new Option(t("dm.allRobots"), ""),
    ...robots.map((robot) => new Option(robot, robot)));
  $("dm-robot").value = robotFilter;
}

async function loadTransfers() {
  try {
    const data = await context.api("/api/transfers");
    if (!data.ok) throw new Error(data.error || t("ui.requestFailed"));
    datasets = data.datasets;
    renderRobotFilter();
    acceptSnapshot(data);
    for (const name of selected) {
      if (!datasets.some((row) => row.name === name)) selected.delete(name);
    }
    loaded = true;
    renderTransfers();
    if (running()) schedulePoll();
  } catch (error) {
    $("dm-status").textContent = error.message;
    $("dm-eta").textContent = "ETA --";
  }
}

function schedulePoll() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    try {
      const data = await context.api("/api/transfers/job");
      const wasRunning = running();
      acceptSnapshot(data);
      renderTransfers();
      if (running()) {
        schedulePoll();
      } else {
        if (wasRunning) await context.loadState(false, true);
        await loadTransfers();
      }
    } catch (error) {
      $("dm-status").textContent = t("dm.pollFailed").replace("{error}", error.message);
      $("dm-eta").textContent = "ETA --";
      schedulePoll();
    }
  }, 1000);
}

async function start(action, names) {
  names = names || (sweepAll(action) ? datasets.map((row) => row.name) : [...selected]);
  if (busy() || !names.length) return;
  requesting = true;
  renderTransfers();
  try {
    const data = await context.api("/api/transfers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, datasets: names }),
    });
    if (!data.ok) throw new Error(data.error || t("dm.actionFailed"));
    acceptSnapshot(data);
    renderTransfers();
    schedulePoll();
  } catch (error) {
    $("dm-status").textContent = error.message;
  } finally {
    requesting = false;
    syncActionButtons();
  }
}

export function initDatasetTransfer() {
  const labels = Object.fromEntries(SORT_KEYS.map((key) => [key, t(`dm.sort.${key}`)]));
  $("dm-search").addEventListener("input", renderTransfers);
  $("dm-robot").addEventListener("change", () => {
    robotFilter = $("dm-robot").value;
    renderTransfers();
  });
  $("dm-sort").replaceChildren(...SORT_KEYS.flatMap((key) =>
    [true, false].map((ascending) =>
      new Option(`${labels[key]} ${ascending ? "↑" : "↓"}`, sortOption(key, ascending)))));
  $("dm-sort").addEventListener("change", () => {
    const value = $("dm-sort").value;
    sortAscending = value === "name" || value.endsWith("-asc");
    sortKey = value.replace(/-(asc|desc)$/, "");
    renderTransfers();
  });
  document.querySelector(".dm-table thead").addEventListener("click", (event) => {
    const button = event.target.closest("[data-dm-sort]");
    if (!button) return;
    sortAscending = sortKey === button.dataset.dmSort ? !sortAscending : true;
    sortKey = button.dataset.dmSort;
    renderTransfers();
  });
  $("dm-select-all").addEventListener("click", () => {
    const rows = visibleRows();
    const allSelected = rows.every((row) => selected.has(row.name));
    for (const row of rows) allSelected ? selected.delete(row.name) : selected.add(row.name);
    renderTransfers();
  });
  $("dm-clear").addEventListener("click", () => {
    selected.clear();
    renderTransfers();
  });
  $("dm-rows").addEventListener("change", (event) => {
    const name = event.target.dataset.dmSelect;
    if (!name) return;
    event.target.checked ? selected.add(name) : selected.delete(name);
    renderTransfers();
  });
  $("dm-rows").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-dm-open]");
    if (button) {
      await openSet(button.dataset.dmOpen);
      return;
    }
    if (event.target.closest("input, button, a, select, label") || window.getSelection()?.toString()) {
      return;
    }
    const row = event.target.closest("[data-dm-row]");
    if (!row) return;
    const name = row.dataset.dmRow;
    selected.has(name) ? selected.delete(name) : selected.add(name);
    row.focus({ preventScroll: true });
    renderTransfers();
  });
  $("dm-rows").addEventListener("keydown", (event) => {
    if (!event.target.matches("[data-dm-row]") || ![" ", "Enter"].includes(event.key)) return;
    event.preventDefault();
    const name = event.target.dataset.dmRow;
    selected.has(name) ? selected.delete(name) : selected.add(name);
    renderTransfers();
  });
  document.querySelectorAll("[data-dm-action]").forEach((button) => {
    button.addEventListener("click", () => start(button.dataset.dmAction));
  });
  $("dm-refresh").addEventListener("click", () => start("refresh", datasets.map((row) => row.name)));
  $("dm-jobs").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-dm-stop]");
    if (!button) return;
    const task = jobs.find((item) => item.id === button.dataset.dmStop);
    if (!task) return;
    button.disabled = true;
    try {
      const response = await context.api("/api/transfers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: "stop", job_id: task.id }),
      });
      if (!response.ok) throw new Error(response.error || t("dm.stopFailed"));
      task.stopping = true;
      renderTransfers();
    } catch (error) {
      button.disabled = false;
      $("dm-status").textContent = error.message;
    }
  });
  loadTransfers();
}

async function openSet(name) {
  context.app.batch = name;
  context.app.qcSceneFilter = "";
  context.app.qcTaskFilter = "";
  context.app.qcPage = 0;
  context.app.selectedSlot = "";
  context.app.selected.tasks = "";
  context.app.review = null;
  await context.loadState();
  context.switchTab("tasks");
}
