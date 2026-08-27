// dashboard.js: stacked raw-only collection and eval operations overview.
import { $, apiGet } from "./core.js";

let dashboardData = null;
let dashboardLoading = null;
let dateRangeInitialized = false;

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

function renderTrend(mode, rows) {
  const host = $(`${mode}-trend`);
  const datedRows = rows.filter((row) => parseDay(row.date));
  if (!datedRows.length) {
    host.innerHTML = '<div class="dashboard-trend-empty">No timestamped raw episodes</div>';
    return;
  }

  const values = new Map(datedRows.map((row) => [row.date, row]));
  const rowDates = datedRows.map((row) => parseDay(row.date));
  const minDate = new Date(Math.min(...rowDates.map((value) => value.getTime())));
  const maxDate = new Date(Math.max(...rowDates.map((value) => value.getTime())));
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
      cells.push(`<rect x="${x}" y="${y}" width="${cellSize}" height="${cellSize}" rx="2.5" class="dashboard-calendar-cell level-${level}" style="--cell-delay:${delay}s" tabindex="0" role="gridcell" aria-label="${escapeHtml(tooltip)}" data-tip="${escapeHtml(tooltip)}"><title>${escapeHtml(tooltip)}</title></rect>`);
      if (count && (!peak || count > peak.count)) {
        peak = {x: x + cellSize / 2, y: y + cellSize / 2, count, date: dayKey(date)};
      }
    }
  }
  const legendX = svgWidth - 130;
  host.innerHTML = `<div class="dashboard-calendar-scroll"><svg class="dashboard-calendar" style="min-width:${svgWidth}px" viewBox="0 0 ${svgWidth} ${svgHeight}" role="img" aria-label="${mode} episodes calendar heatmap">
    ${monthLabels.join("")}
    <text x="50" y="${y0 + 10}" class="dashboard-calendar-weekday">MON</text>
    <text x="50" y="${y0 + pitch * 2 + 10}" class="dashboard-calendar-weekday">WED</text>
    <text x="50" y="${y0 + pitch * 4 + 10}" class="dashboard-calendar-weekday">FRI</text>
    <text x="50" y="${y0 + pitch * 6 + 10}" class="dashboard-calendar-weekday">SUN</text>
    ${cells.join("")}
    <text x="${legendX - 8}" y="190" text-anchor="end" class="dashboard-calendar-legend">LESS</text>
    ${[0, 1, 2, 3, 4].map((level, index) => `<rect x="${legendX + index * 16}" y="179" width="12" height="12" rx="2.5" class="dashboard-calendar-legend-cell level-${level}"></rect>`).join("")}
    <text x="${legendX + 86}" y="190" class="dashboard-calendar-legend">MORE</text>
  </svg></div><div class="dashboard-calendar-tooltip" role="status"></div>`;

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

function renderMode(mode) {
  const view = dashboardData.views[mode];
  $(`${mode}-duration`).textContent = duration(view.duration_seconds);
  $(`${mode}-episodes`).textContent = String(view.episodes || 0);
  $(`${mode}-valid-count`).textContent = `${view.valid_episodes || 0} valid`;
  $(`${mode}-average`).textContent = duration(view.average_duration_seconds);
  $(`${mode}-efficiency`).textContent = percent(view.efficiency);
  $(`${mode}-valid-rate`).textContent = mode === "eval"
    ? percent(view.eval_success_rate) : percent(view.valid_rate, "0%");
  $(`${mode}-health-value`).textContent = percent(view.valid_rate, "0%");
  $(`${mode}-health-fill`).style.width = `${Math.min(100, Math.max(0, Number(view.valid_rate) * 100 || 0))}%`;
  $(`${mode}-active-span`).textContent = duration(view.active_span_seconds);
  $(`${mode}-sources`).textContent = String(dashboardData.sources.filter((source) => source.mode === mode).length);
  $(`${mode}-trend-total`).textContent = `${view.episodes || 0} episodes / ${duration(view.duration_seconds)}`;
  renderTrend(mode, view.trend || []);
  renderRecent(mode, view.recent || []);
  if (mode === "collection") renderDemand(view.tasks || []);
}

function renderDashboard() {
  if (!dashboardData) return;
  renderMode("collection");
  renderMode("eval");
  $("dash-date-result").textContent = `${dashboardData.views.all.episodes || 0} episodes in range`;
}

export function initDashboard() {
  $("dash-date-apply").addEventListener("click", () => loadDashboard(true));
  $("dash-date-reset").addEventListener("click", () => {
    $("dash-date-from").value = dashboardData?.available_range?.start || "";
    $("dash-date-to").value = dashboardData?.available_range?.end || "";
    loadDashboard(true);
  });
}

export async function loadDashboard(force = false) {
  if (dashboardData && !force) {
    renderDashboard();
    return dashboardData;
  }
  if (dashboardLoading) return dashboardLoading;
  $("dashboard-error").style.display = "none";
  const params = new URLSearchParams();
  if (dateRangeInitialized && $("dash-date-from").value) params.set("from", $("dash-date-from").value);
  if (dateRangeInitialized && $("dash-date-to").value) params.set("to", $("dash-date-to").value);
  const endpoint = `/api/dashboard${params.size ? `?${params.toString()}` : ""}`;
  dashboardLoading = apiGet(endpoint)
    .then((data) => {
      dashboardData = data;
      if (!dateRangeInitialized) {
        $("dash-date-from").value = data.available_range?.start || "";
        $("dash-date-to").value = data.available_range?.end || "";
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
