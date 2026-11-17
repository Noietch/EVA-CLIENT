import { $, S, apiGet, apiPost } from "./core.js";
import { selectCollectionDataset } from "./collect.js";

let datasets = [];
let remote = {};
let job = null;
let jobs = [];
let pollTimer = null;
let requesting = false;
let loaded = false;
const selected = new Set();
const labels = { refresh: "刷新云端并校验", verify: "校验", upload_data: "上传数据",
  download_data: "下载数据", upload_qc: "上传 QC", download_qc: "下载 QC",
  upload_task: "上传任务", download_task: "下载任务" };
const resourceLabels = { local_data: "本地数据", local_qc: "本地 QC", local_task: "本地任务文件",
  remote_data: "云端数据", remote_qc: "云端 QC", remote_task: "云端任务文件" };
const escape = (value) => String(value ?? "").replace(/[&<>"']/g,
  (char) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[char]));
const busy = () => requesting;
const running = () => jobs.some((item) => item.state === "running");
const sortLabels = { name: "名称", collected: "已采条数", total: "目标条数", progress: "采集比例",
  pending: "待采集", accept: "本地通过", fail: "本地失败", unreviewed: "本地未标注",
  cloud_episodes: "云端条数", cloud_tasks: "云端任务数", cloud_accept: "云端通过",
  cloud_fail: "云端失败", cloud_unreviewed: "云端未标注",
  verify_data: "数据校验", verify_task: "任务校验", verify_qc: "QC 校验" };
let sortKey = "name";
let sortAscending = true;
const sortOption = (key, ascending) => key === "name" ? (ascending ? "name" : "name-desc")
  : `${key}${ascending ? "-asc" : ""}`;

function sortValue(row) {
  const cloud = remote[row.name];
  if (sortKey === "name") return row.name;
  if (sortKey === "progress") return row.total > 0 && row.collected != null ? row.collected / row.total : null;
  if (sortKey === "pending") return row.total != null && row.collected != null ? Math.max(0, row.total - row.collected) : null;
  if (sortKey === "cloud_episodes") return cloud?.checked_at ? cloud.episodes : null;
  if (sortKey === "cloud_tasks") return cloud?.checked_at ? cloud.task_count : null;
  if (sortKey.startsWith("cloud_")) return cloud?.qc?.[sortKey.slice(6)] ?? null;
  if (sortKey.startsWith("verify_")) {
    const state = cloud?.verification?.[sortKey.slice(7)]?.state;
    return ({ same: 0, unknown: 1, different: 2 })[state] ?? null;
  }
  return row[sortKey] ?? null;
}
function acceptSnapshot(data) {
  jobs = data.jobs || [];
  job = jobs.at(-1) || null;
  remote = data.remote;
}

function visibleRows() {
  const query = $("dm-search").value.trim().toLowerCase();
  return datasets.filter((row) => `${row.name} ${row.robot || ""}`.toLowerCase().includes(query))
    .sort((a, b) => {
      const av = sortValue(a), bv = sortValue(b);
      if (av == null || bv == null) return av == null && bv == null ? a.name.localeCompare(b.name) : av == null ? 1 : -1;
      const delta = sortKey === "name" ? av.localeCompare(bv) : av - bv;
      return delta * (sortAscending ? 1 : -1) || a.name.localeCompare(b.name);
    });
}

function verification(item) {
  if (!item) return '<span class="dm-muted">未校验</span>';
  const label = { same: "一致", different: "不一致", unknown: "无法确认" }[item.state];
  const details = [
    ["本地缺失", item.missing_local], ["仅本地", item.local_only],
    ["内容不同", item.changed], ["无法确认", item.unknown],
  ].filter(([, values]) => values?.length).map(([title, values]) => `${title}: ${values.join(", ")}`).join("\n");
  return `<span class="dm-check ${item.state === "same" ? "ok" : "err"}" title="${escape(details)}">${label}</span>`;
}

function render() {
  const rows = visibleRows();
  $("dm-sort").value = sortOption(sortKey, sortAscending);
  document.querySelectorAll("[data-dm-sort]").forEach((button) => {
    const active = button.dataset.dmSort === sortKey;
    button.closest("th").setAttribute("aria-sort", active ? (sortAscending ? "ascending" : "descending") : "none");
    button.dataset.direction = active ? (sortAscending ? "asc" : "desc") : "none";
    button.title = `按${sortLabels[button.dataset.dmSort]}${active && sortAscending ? "降序" : "升序"}排列`;
  });
  const results = new Map(jobs.flatMap((task) => task.results.map((item) =>
    [item.dataset, { ...item, action: task.action }])));
  const focused = document.activeElement;
  const focusKey = ["dmSelect", "dmOpen", "dmRow"].find((key) => focused?.dataset?.[key]);
  const focusName = focusKey ? focused.dataset[focusKey] : null;
  const markup = rows.map((row) => {
    const cloud = remote[row.name];
    const result = results.get(row.name);
    const ratio = row.total ? Math.min(100, Math.round(row.collected / row.total * 100)) : 0;
    const name = escape(row.name);
    return `<tr data-dm-row="${name}" tabindex="0" aria-label="选择 ${name}" aria-selected="${selected.has(row.name)}" class="${selected.has(row.name) ? "dm-selected" : ""}">
      <td><input type="checkbox" data-dm-select="${name}" aria-label="选择 ${name}" ${selected.has(row.name) ? "checked" : ""}></td>
      <td><button class="dm-name" data-dm-open="${name}" title="进入采集：${name}">${name}</button><small class="dm-dataset-id">${escape(row.robot || "")}</small>${row.error ? `<small class="dm-fail">${escape(row.error)}</small>` : ""}</td>
      <td>${row.collected ?? "--"}</td><td>${row.total ?? "--"}</td><td>${row.total > 0 ? `${ratio}%` : "--"}<progress max="100" value="${ratio}" aria-label="${name} 采集比例"></progress></td>
      <td class="dm-accept">${row.accept ?? "--"}</td><td class="dm-fail">${row.fail ?? "--"}</td><td class="dm-unreviewed">${row.unreviewed ?? "--"}</td>
      <td data-dm-field="cloud_episodes">${cloud?.checked_at ? cloud.episodes : "--"}</td><td data-dm-field="cloud_tasks">${cloud?.checked_at ? cloud.task_count ?? "--" : "--"}</td>
      <td class="dm-accept" data-dm-field="cloud_accept">${cloud?.qc?.accept ?? "--"}</td><td class="dm-fail" data-dm-field="cloud_fail">${cloud?.qc?.fail ?? "--"}</td><td class="dm-unreviewed" data-dm-field="cloud_unreviewed">${cloud?.qc?.unreviewed ?? "--"}</td>
      <td>${verification(cloud?.verification?.data)}</td>
      <td>${verification(cloud?.verification?.task)}</td>
      <td>${verification(cloud?.verification?.qc)}</td>
      <td><small>${cloud?.checked_at ? escape(new Date(cloud.checked_at * 1000).toLocaleString()) : "未刷新"}${cloud?.stale ? " · 并行写入，待刷新" : ""}</small>
        <span class="${result?.ok === false || cloud?.error ? "dm-fail" : ""}">${escape(result ? (result.ok ? `${labels[result.action]}完成` : result.error) : cloud?.error || "")}</span></td>
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
  $("dm-selected").textContent = `已选 ${selected.size} · 显示 ${rows.length} / ${datasets.length}`;
  $("dm-clear").disabled = !selected.size;
  const all = $("dm-select-all");
  const allSelected = rows.length > 0 && rows.every((row) => selected.has(row.name));
  all.setAttribute("aria-pressed", String(allSelected));
  all.textContent = `${allSelected ? "取消全选" : "全选筛选结果"} (${rows.length})`;
  all.disabled = !rows.length;
  document.querySelectorAll("[data-dm-action]").forEach((button) => { button.disabled = busy() || !selected.size; });
  $("dm-refresh").disabled = busy() || !selected.size;
  const activeJobs = jobs.filter((item) => item.state === "running");
  const jobList = $("dm-jobs");
  const expandedJobs = new Set([...jobList.querySelectorAll("details[open]")]
    .map((details) => details.closest("[data-dm-job]").dataset.dmJob));
  const focusedSummaryJob = document.activeElement?.matches("summary")
    ? document.activeElement.closest("[data-dm-job]")?.dataset.dmJob : null;
  const jobScrollTop = jobList.scrollTop;
  const visibleJobs = jobs.filter((task) => task.state !== "stopped" && !(task.state === "done"
    && task.completed === task.total && task.results.every((result) => result.ok)));
  jobList.innerHTML = [...visibleJobs].reverse().map((task) => {
    const failures = task.results.filter((item) => !item.ok).length;
    const active = task.state === "running";
    const eta = active ? task.waiting ? "等待中" : task.eta == null ? "估算中" : `约 ${Math.ceil(task.eta / 60)} 分钟` : "0";
    return `<div class="dm-job" data-dm-job="${escape(task.id)}">
      <strong>${labels[task.action]} · ${task.completed}/${task.total}</strong>
      <span class="dm-job-detail">${escape(active ? `${task.waiting ? `等待冲突文件：${(task.waiting_resources || []).map((name) => resourceLabels[name] || name).join("、")} · ` : ""}${task.current}${task.detail ? ` · ${task.detail}` : ""}` : task.state === "stopped" ? "已停止" : "完成")}${failures ? ` · ${failures} 失败` : ""}</span>
      <span>ETA ${eta}</span>
      ${active ? `<button class="btn" data-dm-stop="${escape(task.id)}" ${task.stopping || task.stop_requested ? "disabled" : ""}>${task.stopping || task.stop_requested ? "停止中" : "停止后续任务"}</button>` : ""}
      <progress max="${task.total || 1}" value="${task.completed}" aria-label="${labels[task.action]}进度"></progress>
      ${failures ? `<details${expandedJobs.has(task.id) ? " open" : ""}><summary>失败详情 (${failures})</summary>${task.results.filter((item) => !item.ok).map((item) => `<div>${escape(item.dataset)}：${escape(item.error)}</div>`).join("")}</details>` : ""}
    </div>`;
  }).join("");
  jobList.scrollTop = jobScrollTop;
  if (focusedSummaryJob) {
    [...jobList.querySelectorAll("[data-dm-job]")]
      .find((element) => element.dataset.dmJob === focusedSummaryJob)
      ?.querySelector("summary")?.focus({ preventScroll: true });
  }
  if (job) {
    $("dm-progress").max = job.total || 1;
    $("dm-progress").value = job.completed;
    const failures = job.results.filter((r) => !r.ok).length;
    $("dm-status").textContent = activeJobs.length ? `${activeJobs.length} 个批次处理中` : `${labels[job.action]} · ${job.completed}/${job.total} · 完成${failures ? ` · ${failures} 失败` : ""}`;
    $("dm-eta").textContent = "";
    $("dm-progress").hidden = true;
  }
}

export async function loadDatasetManager() {
  try {
    const data = await apiGet("/api/dataset_manager", { timeoutMs: 30000 });
    if (!data.ok) throw new Error(data.error || "加载失败");
    datasets = data.datasets;
    acceptSnapshot(data);
    for (const name of selected) if (!datasets.some((row) => row.name === name)) selected.delete(name);
    loaded = true;
    render();
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
      const data = await apiGet("/api/dataset_manager/job", { timeoutMs: 30000 });
      acceptSnapshot(data);
      render();
      if (running()) schedulePoll();
      else await loadDatasetManager();
    } catch (error) {
      $("dm-status").textContent = `状态刷新失败：${error.message}`;
      $("dm-eta").textContent = "ETA --";
      schedulePoll();
    }
  }, 1000);
}

async function start(action) {
  const names = [...selected];
  if (busy() || !names.length) return;
  requesting = true;
  render();
  try {
    const data = await apiPost("/api/dataset_manager", { action, datasets: names }, { timeoutMs: 30000 });
    if (!data.ok) throw new Error(data.error || "操作失败");
    acceptSnapshot(data);
    render();
    schedulePoll();
  } catch (error) {
    $("dm-status").textContent = error.message;
  } finally {
    requesting = false;
    document.querySelectorAll("[data-dm-action]").forEach((b) => { b.disabled = busy() || !selected.size; });
    $("dm-refresh").disabled = busy() || !selected.size;
  }
}

export function initDatasetManager() {
  $("dm-search").addEventListener("input", render);
  $("dm-sort").replaceChildren(...Object.entries(sortLabels).flatMap(([key, label]) =>
    [true, false].map((ascending) => new Option(`${label} ${ascending ? "↑" : "↓"}`, sortOption(key, ascending)))));
  $("dm-sort").addEventListener("change", () => {
    const value = $("dm-sort").value;
    sortAscending = value === "name" || value.endsWith("-asc");
    sortKey = value.replace(/-(asc|desc)$/, "");
    render();
  });
  document.querySelector(".dm-table thead").addEventListener("click", (event) => {
    const button = event.target.closest("[data-dm-sort]");
    if (!button) return;
    sortAscending = sortKey === button.dataset.dmSort ? !sortAscending : true;
    sortKey = button.dataset.dmSort;
    render();
  });
  $("dm-select-all").addEventListener("click", () => {
    const rows = visibleRows();
    const allSelected = rows.every((row) => selected.has(row.name));
    for (const row of rows) allSelected ? selected.delete(row.name) : selected.add(row.name);
    render();
  });
  $("dm-clear").addEventListener("click", () => { selected.clear(); render(); });
  $("dm-rows").addEventListener("change", (event) => {
    const name = event.target.dataset.dmSelect;
    if (name) { event.target.checked ? selected.add(name) : selected.delete(name); render(); }
  });
  $("dm-rows").addEventListener("click", (event) => {
    const button = event.target.closest("[data-dm-open]");
    if (button) {
      document.querySelector('button[data-tab="collect"]').click();
      selectCollectionDataset(button.dataset.dmOpen);
      return;
    }
    if (event.target.closest("input, button, a, select, label") || window.getSelection()?.toString()) return;
    const row = event.target.closest("[data-dm-row]");
    if (!row) return;
    const name = row.dataset.dmRow;
    selected.has(name) ? selected.delete(name) : selected.add(name);
    row.focus({ preventScroll: true });
    render();
  });
  $("dm-rows").addEventListener("keydown", (event) => {
    if (!event.target.matches("[data-dm-row]") || ![" ", "Enter"].includes(event.key)) return;
    event.preventDefault();
    const name = event.target.dataset.dmRow;
    selected.has(name) ? selected.delete(name) : selected.add(name);
    render();
  });
  document.querySelectorAll("[data-dm-action]").forEach((b) => b.addEventListener("click", () => start(b.dataset.dmAction)));
  $("dm-refresh").addEventListener("click", () => start("refresh"));
  $("dm-jobs").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-dm-stop]");
    if (!button) return;
    const task = jobs.find((item) => item.id === button.dataset.dmStop);
    if (!task) return;
    button.disabled = true;
    try {
      const response = await apiPost("/api/dataset_manager", { action: "stop", job_id: task.id }, { concurrent: true });
      if (!response.ok) throw new Error(response.error || "停止失败");
      task.stopping = true;
      render();
    } catch (error) { button.disabled = false; $("dm-status").textContent = error.message; }
  });
}
