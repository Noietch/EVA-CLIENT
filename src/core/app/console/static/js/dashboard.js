// dashboard.js: stacked raw-only collection and eval operations overview.
import { $, apiGet, apiPost } from "./core.js";

let dashboardData = null;
let dashboardLoading = null;
let dateRangeInitialized = false;
let dateRangeApplied = false;
const selectedTrendDay = {collection: "", eval: ""};
const dashboardUpload = {
  candidateKey: "",
  requestToken: 0,
  jobId: "",
  planId: "",
  state: "idle",
  localFiles: null,
  newFiles: null,
  changedFiles: null,
  sameFiles: null,
  removeFiles: null,
  filesCompleted: 0,
  filesTotal: 0,
  filesDeleted: 0,
  progress: 0,
  currentFile: "",
  error: "",
  posting: false,
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function duration(value) {
  const seconds = Math.max(0, Number(value) || 0);
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(seconds >= 36000 ? 0 : 1)}h`;
  if (seconds >= 60) return `${(seconds / 60).toFixed(seconds >= 600 ? 0 : 1)}m`;
  return `${seconds.toFixed(seconds >= 10 ? 0 : 1)}s`;
}

function percent(value, fallback = "--") {
  if (value == null || !Number.isFinite(Number(value))) return fallback;
  return `${Math.round(Number(value) * 100)}%`;
}

function uploadCandidateKey(candidate) {
  return JSON.stringify([candidate.source_dir, candidate.dataset_format]);
}

function selectedUploadCandidate() {
  const candidates = dashboardData?.upload_candidates || [];
  return candidates.find((candidate) => uploadCandidateKey(candidate) === dashboardUpload.candidateKey) || null;
}

function resetDashboardUploadJob() {
  dashboardUpload.requestToken += 1;
  dashboardUpload.jobId = "";
  dashboardUpload.planId = "";
  dashboardUpload.state = "idle";
  dashboardUpload.localFiles = null;
  dashboardUpload.newFiles = null;
  dashboardUpload.changedFiles = null;
  dashboardUpload.sameFiles = null;
  dashboardUpload.removeFiles = null;
  dashboardUpload.filesCompleted = 0;
  dashboardUpload.filesTotal = 0;
  dashboardUpload.filesDeleted = 0;
  dashboardUpload.progress = 0;
  dashboardUpload.currentFile = "";
  dashboardUpload.error = "";
  dashboardUpload.posting = false;
}

function applyDashboardUploadJob(job) {
  dashboardUpload.jobId = String(job.job_id || dashboardUpload.jobId);
  dashboardUpload.state = String(job.state || dashboardUpload.state);
  dashboardUpload.localFiles = Number(job.local_files) || 0;
  dashboardUpload.newFiles = Number(job.new_files) || 0;
  dashboardUpload.changedFiles = Number(job.changed_files) || 0;
  dashboardUpload.sameFiles = Number(job.files_skipped) || 0;
  dashboardUpload.removeFiles = Number(job.files_to_delete) || 0;
  dashboardUpload.filesCompleted = Number(job.files_completed) || 0;
  dashboardUpload.filesTotal = Number(job.files_total) || 0;
  dashboardUpload.filesDeleted = Number(job.files_deleted) || 0;
  dashboardUpload.progress = Number(job.progress) || 0;
  dashboardUpload.currentFile = String(job.current_file || "");
  dashboardUpload.error = String(job.error || "");
  dashboardUpload.posting = false;
}

function dashboardUploadHasChanges() {
  return Number(dashboardUpload.filesTotal) + Number(dashboardUpload.removeFiles) > 0;
}

function dashboardUploadStatus(candidate, configured) {
  if (!configured) return "UPLOAD TARGET NOT CONFIGURED";
  if (!candidate) return "NO ACCEPTED EXPORT FOUND";
  if (dashboardUpload.error) return dashboardUpload.error.toUpperCase();
  if (dashboardUpload.posting) return "STARTING REMOTE CHECK";
  if (["queued", "scanning"].includes(dashboardUpload.state)) return "SCANNING LOCAL AND REMOTE FILES";
  if (dashboardUpload.state === "ready") {
    if (!dashboardUploadHasChanges()) return `UP TO DATE · ${dashboardUpload.sameFiles || 0} SAME`;
    return `${dashboardUpload.newFiles || 0} NEW · ${dashboardUpload.changedFiles || 0} CHANGED · ${dashboardUpload.sameFiles || 0} SAME · ${dashboardUpload.removeFiles || 0} REMOVE`;
  }
  if (dashboardUpload.state === "running") {
    const detail = dashboardUpload.currentFile ? ` · ${dashboardUpload.currentFile}` : "";
    return `${dashboardUpload.filesCompleted}/${dashboardUpload.filesTotal} SYNCED${detail}`;
  }
  if (dashboardUpload.state === "completed") {
    return `SYNC COMPLETE · ${dashboardUpload.filesCompleted} UPLOADED · ${dashboardUpload.filesDeleted} REMOVED`;
  }
  return `${Number(candidate.accepted_episodes) || 0} ACCEPTED · ${Number(candidate.uploaded_episodes) || 0} UPLOADED`;
}

function renderDashboardUpload() {
  const candidates = dashboardData?.upload_candidates || [];
  const select = $("dashboard-upload-dataset");
  const availableKeys = new Set(candidates.map(uploadCandidateKey));
  if (!availableKeys.has(dashboardUpload.candidateKey)) {
    dashboardUpload.candidateKey = candidates.length ? uploadCandidateKey(candidates[0]) : "";
    resetDashboardUploadJob();
  }
  select.replaceChildren(...candidates.map((candidate) => {
    const option = document.createElement("option");
    option.value = uploadCandidateKey(candidate);
    option.textContent = `${candidate.dataset} · ${candidate.dataset_format} · ${candidate.accepted_episodes} accepted · ${candidate.uploaded_episodes} uploaded`;
    return option;
  }));
  select.value = dashboardUpload.candidateKey;
  const candidate = selectedUploadCandidate();
  const configured = Boolean(dashboardData?.upload?.configured);
  select.disabled = !candidates.length || dashboardUpload.posting || ["queued", "scanning", "running"].includes(dashboardUpload.state);
  $("dashboard-upload-config").textContent = configured
    ? (dashboardData.upload.targets || []).join(" / ")
    : "NOT CONFIGURED";
  $("dashboard-upload-local").textContent = candidate?.accepted_dir || "--";
  $("dashboard-upload-remote").textContent = (candidate?.remote_dirs || []).join(" / ") || "--";

  const scanned = dashboardUpload.localFiles != null;
  $("dashboard-upload-local-files").textContent = scanned ? String(dashboardUpload.localFiles) : "--";
  $("dashboard-upload-new").textContent = scanned ? String(dashboardUpload.newFiles) : "--";
  $("dashboard-upload-changed").textContent = scanned ? String(dashboardUpload.changedFiles) : "--";
  $("dashboard-upload-same").textContent = scanned ? String(dashboardUpload.sameFiles) : "--";
  $("dashboard-upload-remove").textContent = scanned ? String(dashboardUpload.removeFiles) : "--";

  const busy = dashboardUpload.posting || ["queued", "scanning", "running"].includes(dashboardUpload.state);
  const confirm = dashboardUpload.state === "ready" && dashboardUpload.planId && dashboardUploadHasChanges();
  const action = $("dashboard-upload-action");
  action.textContent = busy
    ? (dashboardUpload.state === "running" ? "SYNCING..." : "SCANNING...")
    : (confirm ? "CONFIRM SYNC" : "SCAN TARGET");
  action.disabled = !configured || !candidate || busy;
  const progress = dashboardUpload.state === "ready" || dashboardUpload.state === "completed"
    ? 1 : dashboardUpload.progress;
  $("dashboard-upload-progress").style.width = `${Math.max(0, Math.min(1, progress)) * 100}%`;
  const status = $("dashboard-upload-status");
  status.textContent = dashboardUploadStatus(candidate, configured);
  status.classList.toggle("failed", dashboardUpload.state === "failed" || Boolean(dashboardUpload.error));
}

async function pollDashboardUpload(jobId, requestToken, confirmed) {
  while (dashboardUpload.requestToken === requestToken) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    const job = await apiGet(`/api/dashboard_upload?job_id=${encodeURIComponent(jobId)}`);
    if (!job.ok) throw new Error(job.error || "Upload job is unavailable");
    applyDashboardUploadJob(job);
    renderDashboardUpload();
    if (job.state === "failed") throw new Error(job.error || "Dataset upload failed");
    const terminal = confirmed ? job.state === "completed" : job.state === "ready";
    if (!terminal) continue;
    if (!confirmed && dashboardUploadHasChanges()) dashboardUpload.planId = jobId;
    else dashboardUpload.planId = "";
    renderDashboardUpload();
    await loadDashboard(true);
    return;
  }
}

async function runDashboardUpload() {
  const candidate = selectedUploadCandidate();
  if (!candidate || dashboardUpload.posting) return;
  const confirmed = dashboardUpload.state === "ready"
    && Boolean(dashboardUpload.planId)
    && dashboardUploadHasChanges();
  const requestToken = dashboardUpload.requestToken + 1;
  dashboardUpload.requestToken = requestToken;
  dashboardUpload.posting = true;
  dashboardUpload.error = "";
  renderDashboardUpload();
  try {
    const response = await apiPost("/api/dashboard_upload", {
      source_dir: candidate.source_dir,
      dataset_format: candidate.dataset_format,
      confirmed,
      plan_id: confirmed ? dashboardUpload.planId : "",
    });
    if (!response.ok) throw new Error(response.error || "Dataset upload request failed");
    if (dashboardUpload.requestToken !== requestToken) return;
    if (!confirmed) dashboardUpload.planId = "";
    applyDashboardUploadJob(response);
    renderDashboardUpload();
    await pollDashboardUpload(response.job_id, requestToken, confirmed);
  } catch (error) {
    if (dashboardUpload.requestToken !== requestToken) return;
    dashboardUpload.posting = false;
    dashboardUpload.state = "failed";
    dashboardUpload.error = error.message || "Dataset upload failed";
    renderDashboardUpload();
  }
}

function localTimestamp(value) {
  if (!value) return "--";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  }).format(parsed);
}

function parseDay(value) {
  const parts = String(value || "").split("-").map(Number);
  if (parts.length !== 3 || parts.some((part) => !Number.isFinite(part))) return null;
  return new Date(Date.UTC(parts[0], parts[1] - 1, parts[2]));
}

function addDays(value, count) {
  const next = new Date(value.getTime());
  next.setUTCDate(next.getUTCDate() + count);
  return next;
}

function dayKey(value) {
  return value.toISOString().slice(0, 10);
}

function localToday() {
  const now = new Date();
  return new Date(Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()));
}

function defaultEndDate(data) {
  const availableEnd = parseDay(data?.available_range?.end);
  const today = localToday();
  if (!availableEnd) return dayKey(today);
  return dayKey(new Date(Math.max(availableEnd.getTime(), today.getTime())));
}

function metricView(mode) {
  const view = dashboardData.views[mode];
  const selected = selectedTrendDay[mode];
  if (!selected) return view;
  return (view.daily || []).find((row) => row.date === selected) || {
    episodes: 0,
    frames: 0,
    valid_episodes: 0,
    duration_seconds: 0,
    average_duration_seconds: 0,
    efficiency: null,
    valid_rate: 0,
    eval_success_rate: null,
    uploaded_episodes: 0,
    not_uploaded_episodes: 0,
  };
}

function updateTrendSelection(mode) {
  const selected = selectedTrendDay[mode];
  $(`${mode}-trend`).querySelectorAll(".dashboard-calendar-cell").forEach((cell) => {
    const active = cell.dataset.date === selected;
    cell.classList.toggle("selected", active);
    cell.setAttribute("aria-selected", String(active));
  });
}

function selectTrendDay(mode, date) {
  if (mode !== "collection") return;
  selectedTrendDay[mode] = selectedTrendDay[mode] === date ? "" : date;
  renderModeMetrics(mode);
  updateTrendSelection(mode);
}

function renderTrend(mode, rows, filters = {}) {
  const host = $(`${mode}-trend`);
  const datedRows = rows.filter((row) => parseDay(row.date));
  if (!datedRows.length) {
    host.innerHTML = '<div class="dashboard-trend-empty">No timestamped raw episodes</div>';
    return;
  }

  const values = new Map(datedRows.map((row) => [row.date, row]));
  const rowDates = datedRows.map((row) => parseDay(row.date));
  const filterStart = parseDay(filters.start);
  const filterEnd = parseDay(filters.end);
  const minDate = filterStart || new Date(Math.min(...rowDates.map((value) => value.getTime())));
  const maxDate = filterEnd || new Date(Math.max(
    ...rowDates.map((value) => value.getTime()),
    localToday().getTime(),
  ));
  const firstMonth = new Date(Date.UTC(minDate.getUTCFullYear(), minDate.getUTCMonth(), 1));
  const lastMonth = new Date(Date.UTC(maxDate.getUTCFullYear(), maxDate.getUTCMonth() + 1, 0));
  let start = addDays(firstMonth, -((firstMonth.getUTCDay() + 6) % 7));
  let end = addDays(lastMonth, 6 - ((lastMonth.getUTCDay() + 6) % 7));
  let weeks = Math.round((end - start) / 604800000) + 1;
  let padAtStart = true;
  while (weeks < 24) {
    if (padAtStart) start = addDays(start, -7);
    else end = addDays(end, 7);
    padAtStart = !padAtStart;
    weeks += 1;
  }

  const cellSize = weeks <= 26 ? 17 : 12;
  const pitch = weeks <= 26 ? 22 : 16;
  const x0 = 62;
  const y0 = 32;
  const svgWidth = x0 + (weeks - 1) * pitch + 62;
  const svgHeight = 200;
  const maxEpisodes = Math.max(1, ...datedRows.map((row) => Number(row.episodes) || 0));
  const monthName = new Intl.DateTimeFormat("en", {month: "short", timeZone: "UTC"});
  const monthLabels = [];
  let previousMonth = "";
  for (let week = 0; week < weeks; week += 1) {
    const monday = addDays(start, week * 7);
    const label = `${monthName.format(monday).toUpperCase()} ${monday.getUTCFullYear()}`;
    if (label !== previousMonth) {
      if (!(week === 0 && monday.getUTCDate() > 7)) {
        monthLabels.push(`<text x="${x0 + week * pitch}" y="14" class="dashboard-calendar-month">${label}</text>`);
      }
      previousMonth = label;
    }
  }

  const cells = [];
  let peak = null;
  for (let week = 0; week < weeks; week += 1) {
    for (let weekday = 0; weekday < 7; weekday += 1) {
      const date = addDays(start, week * 7 + weekday);
      const row = values.get(dayKey(date));
      const count = Number(row?.episodes) || 0;
      const x = x0 + week * pitch;
      const y = y0 + weekday * pitch;
      const delay = ((week * 7 + weekday) * 0.004).toFixed(3);
      const ratio = count / maxEpisodes;
      const level = count ? Math.min(4, Math.max(1, Math.ceil(ratio * 4))) : 0;
      const tooltip = count
        ? `${dayKey(date)} - ${count} episodes, ${Number(row.valid_episodes) || 0} valid, ${duration(row.duration_seconds)}`
        : `${dayKey(date)} - 0 episodes`;
      const dateKey = dayKey(date);
      const selected = selectedTrendDay[mode] === dateKey;
      cells.push(`<rect x="${x}" y="${y}" width="${cellSize}" height="${cellSize}" rx="2.5" class="dashboard-calendar-cell level-${level}${selected ? " selected" : ""}" style="--cell-delay:${delay}s" tabindex="0" role="gridcell" aria-selected="${selected}" aria-label="${escapeHtml(tooltip)}" data-date="${dateKey}" data-tip="${escapeHtml(tooltip)}"><title>${escapeHtml(tooltip)}</title></rect>`);
      if (count && (!peak || count > peak.count)) {
        peak = {x: x + cellSize / 2, y: y + cellSize / 2, count, date: dayKey(date)};
      }
    }
  }
  host.innerHTML = `<div class="dashboard-calendar-scroll"><svg class="dashboard-calendar" style="min-width:${svgWidth}px" viewBox="0 0 ${svgWidth} ${svgHeight}" role="img" aria-label="${mode} episodes calendar heatmap">
    ${monthLabels.join("")}
    <text x="50" y="${y0 + 10}" class="dashboard-calendar-weekday">MON</text>
    <text x="50" y="${y0 + pitch * 2 + 10}" class="dashboard-calendar-weekday">WED</text>
    <text x="50" y="${y0 + pitch * 4 + 10}" class="dashboard-calendar-weekday">FRI</text>
    <text x="50" y="${y0 + pitch * 6 + 10}" class="dashboard-calendar-weekday">SUN</text>
    ${cells.join("")}
  </svg></div><div class="dashboard-calendar-legend" aria-label="Episode volume scale">
    <span>LESS</span>
    ${[0, 1, 2, 3, 4].map((level) => `<i class="dashboard-calendar-legend-cell level-${level}" aria-hidden="true"></i>`).join("")}
    <span>MORE</span>
  </div><div class="dashboard-calendar-tooltip" role="status"></div>`;

  const scrollHost = host.querySelector(".dashboard-calendar-scroll");
  const tooltip = host.querySelector(".dashboard-calendar-tooltip");
  const moveTooltip = (event) => {
    const bounds = host.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    const left = Math.min(host.clientWidth - tooltip.offsetWidth - 8, Math.max(8, x + 12));
    const above = y - tooltip.offsetHeight - 12;
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${above >= 8 ? above : y + 12}px`;
  };
  host.querySelectorAll(".dashboard-calendar-cell").forEach((cell) => {
    const show = (event) => {
      tooltip.textContent = cell.dataset.tip;
      tooltip.classList.add("visible");
      const box = cell.getBoundingClientRect();
      moveTooltip(event.clientX == null ? {clientX: box.left + box.width / 2, clientY: box.top} : event);
    };
    cell.addEventListener("pointerenter", show);
    cell.addEventListener("pointermove", moveTooltip);
    cell.addEventListener("pointerleave", () => tooltip.classList.remove("visible"));
    cell.addEventListener("focus", show);
    cell.addEventListener("blur", () => tooltip.classList.remove("visible"));
    if (mode === "collection") {
      cell.addEventListener("click", () => selectTrendDay(mode, cell.dataset.date));
      cell.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        selectTrendDay(mode, cell.dataset.date);
      });
    }
  });
  if (peak) {
    requestAnimationFrame(() => {
      if (scrollHost.scrollWidth <= scrollHost.clientWidth) return;
      const svg = host.querySelector(".dashboard-calendar");
      const scale = svg.getBoundingClientRect().width / svgWidth;
      scrollHost.scrollLeft = Math.max(0, peak.x * scale - scrollHost.clientWidth / 2);
    });
  }
}

function renderRecent(mode, rows) {
  const body = $(`${mode}-recent`);
  $(`${mode}-empty`).style.display = rows.length ? "none" : "block";
  body.innerHTML = rows.slice(0, 20).map((row) => {
    const valid = !!row.valid;
    let state = valid ? "VALID" : "INVALID";
    let stateClass = valid ? "valid" : "invalid";
    if (mode === "eval" && row.result) {
      state = row.result.toUpperCase();
      stateClass = row.result === "success" ? "valid" : "invalid";
    }
    return `<tr>
      <td class="tnum">${escapeHtml(localTimestamp(row.started_at))}</td>
      <td><b>${escapeHtml(row.dataset)}</b><small>${escapeHtml(row.task || "No task")}</small></td>
      <td class="tnum">${escapeHtml(row.robot_id)}</td>
      <td class="tnum">${duration(row.duration_seconds)}</td>
      <td><span class="dashboard-status ${stateClass}">${escapeHtml(state)}</span></td>
    </tr>`;
  }).join("");
}

function renderDemand(tasks) {
  const body = $("collection-demand");
  $("collection-demand-empty").style.display = tasks.length ? "none" : "block";
  body.innerHTML = tasks.map((task) => {
    const requirement = Number(task.required_episodes) || 0;
    const unlimited = requirement === -1;
    const completion = requirement > 0 ? Number(task.episodes) / requirement : 0;
    return `<tr>
      <td><b>${escapeHtml(task.dataset)}</b><small>${escapeHtml(task.task || "No task")}</small></td>
      <td class="tnum">${escapeHtml(task.robot_id)}</td>
      <td class="tnum">${Number(task.episodes) || 0}</td>
      <td class="tnum">${unlimited ? "∞" : (requirement > 0 ? requirement : "--")}</td>
      <td class="tnum">${requirement > 0 ? Math.max(0, requirement - Number(task.episodes)) : "--"}</td>
      <td><div class="dashboard-demand-progress"><i style="width:${Math.min(1, completion) * 100}%"></i></div><small>${unlimited ? "No limit" : (requirement > 0 ? percent(completion) : "Not set")}</small></td>
    </tr>`;
  }).join("");
}

function renderModeMetrics(mode) {
  const view = metricView(mode);
  $(`${mode}-duration`).textContent = duration(view.duration_seconds);
  $(`${mode}-episodes`).textContent = String(view.episodes || 0);
  $(`${mode}-frames`).textContent = Number(view.frames || 0).toLocaleString();
  $(`${mode}-valid-count`).textContent = `${view.valid_episodes || 0} valid`;
  $(`${mode}-average`).textContent = duration(view.average_duration_seconds);
  $(`${mode}-efficiency`).textContent = percent(view.efficiency);
  $(`${mode}-valid-rate`).textContent = mode === "eval"
    ? percent(view.eval_success_rate) : percent(view.valid_rate, "0%");
  if (mode === "collection") {
    $("collection-uploaded").textContent = String(view.uploaded_episodes || 0);
    $("collection-not-uploaded").textContent = String(view.not_uploaded_episodes || 0);
    const scope = $("collection-overview-scope");
    scope.textContent = selectedTrendDay.collection
      ? `${selectedTrendDay.collection}  ×`
      : "ALL DATA";
    scope.disabled = !selectedTrendDay.collection;
    scope.title = selectedTrendDay.collection ? "Clear selected day" : "All collection data";
  }
  const prefix = selectedTrendDay[mode] ? `${selectedTrendDay[mode]} · ` : "";
  $(`${mode}-trend-total`).textContent = `${prefix}${view.episodes || 0} episodes / ${Number(view.frames || 0).toLocaleString()} frames / ${duration(view.duration_seconds)}`;
}

function renderMode(mode) {
  const view = dashboardData.views[mode];
  renderModeMetrics(mode);
  $(`${mode}-health-value`).textContent = percent(view.valid_rate, "0%");
  $(`${mode}-health-fill`).style.width = `${Math.min(100, Math.max(0, Number(view.valid_rate) * 100 || 0))}%`;
  $(`${mode}-active-span`).textContent = duration(view.active_span_seconds);
  $(`${mode}-sources`).textContent = String(dashboardData.sources.filter((source) => source.mode === mode).length);
  renderTrend(mode, view.trend || [], dashboardData.filters || {});
  renderRecent(mode, view.recent || []);
  if (mode === "collection") renderDemand(view.tasks || []);
}

function renderDashboard() {
  if (!dashboardData) return;
  renderMode("collection");
  renderMode("eval");
  renderDashboardUpload();
  $("dash-date-result").textContent = `${dashboardData.views.all.episodes || 0} episodes / ${Number(dashboardData.views.all.frames || 0).toLocaleString()} frames in range`;
}

export function initDashboard() {
  $("collection-overview-scope").addEventListener("click", () => selectTrendDay("collection", ""));
  $("dash-date-apply").addEventListener("click", () => {
    selectedTrendDay.collection = "";
    dateRangeApplied = Boolean($("dash-date-from").value || $("dash-date-to").value);
    loadDashboard(true);
  });
  $("dash-date-reset").addEventListener("click", () => {
    selectedTrendDay.collection = "";
    dateRangeApplied = false;
    loadDashboard(true);
  });
  $("dashboard-upload-dataset").addEventListener("change", (event) => {
    dashboardUpload.candidateKey = event.target.value;
    resetDashboardUploadJob();
    renderDashboardUpload();
  });
  $("dashboard-upload-action").addEventListener("click", runDashboardUpload);
}

export async function loadDashboard(force = false) {
  if (dashboardData && !force) {
    renderDashboard();
    return dashboardData;
  }
  if (dashboardLoading) return dashboardLoading;
  $("dashboard-error").style.display = "none";
  const params = new URLSearchParams();
  if (dateRangeInitialized && dateRangeApplied && $("dash-date-from").value) params.set("from", $("dash-date-from").value);
  if (dateRangeInitialized && dateRangeApplied && $("dash-date-to").value) params.set("to", $("dash-date-to").value);
  const endpoint = `/api/dashboard${params.size ? `?${params.toString()}` : ""}`;
  dashboardLoading = apiGet(endpoint)
    .then((data) => {
      dashboardData = data;
      if (!dateRangeInitialized || !dateRangeApplied) {
        $("dash-date-from").value = data.available_range?.start || "";
        $("dash-date-to").value = defaultEndDate(data);
        dateRangeInitialized = true;
      }
      renderDashboard();
      return data;
    })
    .catch((error) => {
      $("dashboard-error").textContent = error.message || "Dashboard data is unavailable";
      $("dashboard-error").style.display = "block";
      throw error;
    })
    .finally(() => { dashboardLoading = null; });
  return dashboardLoading;
}
