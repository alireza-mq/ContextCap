/* ================================================================
   Global state
   ================================================================ */

let currentDatasetId = null;
let domainCandidates = [];
let variableProfiles = [];
let targetCandidates = [];
let qualitySummary = null;
let chosenDomain = null;
let rankedInsights = [];
let vizSummary = null;
let lastQA = { question: "", answer: "" };
let datasetPreview = null;

/* ================================================================
   Step navigation
   ================================================================ */

const stepButtons = document.querySelectorAll(".step-btn");
const pageProfile = document.getElementById("page-profile");
const pageInsights = document.getElementById("page-insights");
const pageVisual = document.getElementById("page-visual");

function setStep(step) {
  // Highlight active step pill
  stepButtons.forEach((btn) => {
    const s = btn.dataset.step;
    btn.classList.remove("step-active");
    if (String(step) === s) {
      btn.classList.add("step-active");
    }
  });

  // Show the right page
  pageProfile.classList.toggle("hidden", step !== 1);
  pageInsights.classList.toggle("hidden", step !== 2);
  pageVisual.classList.toggle("hidden", step !== 3);
}

// Only allow forward navigation when the prerequisite state exists
stepButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = Number(btn.dataset.step);
    if (target === 1) {
      setStep(1);
    } else if (target === 2 && currentDatasetId) {
      setStep(2);
    } else if (target === 3 && rankedInsights.length > 0) {
      setStep(3);
    }
  });
});

/* Small helpers for show/hide */
function show(el) {
  if (el) el.classList.remove("hidden");
}
function hide(el) {
  if (el) el.classList.add("hidden");
}

/* ================================================================
   STEP 1 · Upload & Profiling
   ================================================================ */

const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");

// Cards
const domainCard = document.getElementById("domainCard");
const qualityCard = document.getElementById("qualityCard");
const variablesCard = document.getElementById("variablesCard");
const previewCard = document.getElementById("previewCard");
const toInsightsRow = document.getElementById("toInsightsRow");

// Inner containers
const domainList = document.getElementById("domainList");
const qualityBox = document.getElementById("qualityBox");
const qualityApprove = document.getElementById("qualityApprove");
const rangeStartInput = document.getElementById("rangeStart");
const rangeEndInput = document.getElementById("rangeEnd");
const variablesList = document.getElementById("variablesList");
const previewTable = document.getElementById("previewTable");
const toInsightsBtn = document.getElementById("toInsightsBtn");


async function refreshPreviewForRange() {
  if (!currentDatasetId) return;

  try {
    const res = await fetch("/preview_range", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: currentDatasetId,
        date_from: rangeStartInput.value || null,
        date_to: rangeEndInput.value || null,
      }),
    });
    if (!res.ok) {
      throw new Error(`preview_range returned ${res.status}`);
    }
    const data = await res.json();
    datasetPreview = data.preview;
    renderDatasetPreview(datasetPreview);
  } catch (err) {
    console.error("Error refreshing preview:", err);
    previewTable.innerHTML =
      '<p class="placeholder">Error while building preview for this range.</p>';
  }
}

// When the range changes, ask backend for a new preview
rangeStartInput.addEventListener("change", refreshPreviewForRange);
rangeEndInput.addEventListener("change", refreshPreviewForRange);


fileInput.addEventListener("change", () => {
  uploadBtn.disabled = !fileInput.files.length;
});

qualityApprove.addEventListener("change", () => {
  updateStep1Ready();
});


/**
 * Enable / disable "Continue to Insight Generation" and
 * reveal/hide the lower cards depending on user approvals.
 */
function updateStep1Ready() {
  const hasDomain = !!chosenDomain;
  const qualityOk = qualityApprove.checked;
  const ready = currentDatasetId && hasDomain && qualityOk;

  toInsightsBtn.disabled = !ready;

  if (qualityOk) {
    show(variablesCard);
    show(previewCard);
    show(toInsightsRow);
  } else {
    hide(variablesCard);
    hide(previewCard);
    hide(toInsightsRow);
  }
}

/**
 * Upload CSV → /upload_profile → fill all Step 1 cards.
 */
uploadBtn.addEventListener("click", async () => {
  if (!fileInput.files.length) return;

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  // Reset UI
  uploadStatus.textContent = "Profiling dataset...";
  hide(domainCard);
  hide(qualityCard);
  hide(variablesCard);
  hide(previewCard);
  hide(toInsightsRow);

  domainList.innerHTML = '<p class="placeholder">Profiling...</p>';
  qualityBox.innerHTML = '<p class="placeholder">Profiling...</p>';
  variablesList.innerHTML = '<p class="placeholder">Profiling...</p>';
  previewTable.innerHTML = '<p class="placeholder">Profiling...</p>';

  try {
    const res = await fetch("/upload_profile", {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      throw new Error(`Server returned ${res.status}`);
    }
    const data = await res.json();

    currentDatasetId = data.dataset_id;
    domainCandidates = data.domains || [];
    variableProfiles = data.variables || [];
    targetCandidates = data.targets || [];
    qualitySummary = data.quality || {};
    datasetPreview = data.preview || null;
    chosenDomain = null; // user will explicitly select

    // Prefill analysis range with full dataset span
    if (qualitySummary.time_start) {
      rangeStartInput.value = qualitySummary.time_start.slice(0, 10);
    }
    if (qualitySummary.time_end) {
      rangeEndInput.value = qualitySummary.time_end.slice(0, 10);
    }

    uploadStatus.textContent =
      "Profiling complete. Please select the most appropriate domain.";

    renderDomains(domainCandidates, chosenDomain);
    renderQuality(qualitySummary);
    renderVariables(variableProfiles);
    renderDatasetPreview(datasetPreview);

    show(domainCard);
    updateStep1Ready();
  } catch (err) {
    console.error(err);
    uploadStatus.textContent = "Error profiling dataset.";
    domainList.innerHTML =
      '<p class="placeholder">Error while profiling dataset.</p>';
  }
});

/**
 * Domain radio list – user chooses which domain best fits.
 */
function renderDomains(domains, selectedName) {
  if (!domains.length) {
    domainList.innerHTML =
      '<p class="placeholder">No domain candidates available.</p>';
    return;
  }

  domainList.innerHTML = "";
  domains.forEach((d, idx) => {
    const row = document.createElement("div");
    row.className = "domain-option";

    const left = document.createElement("div");
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "domainChoice";
    radio.value = d.name;
    radio.checked = selectedName
      ? d.name === selectedName
      : idx === 0 && !selectedName;

    radio.addEventListener("change", () => {
      chosenDomain = d.name;
      show(qualityCard);
      uploadStatus.textContent =
        "Domain selected. Review data quality and approve the range.";
      updateStep1Ready();
    });

    const label = document.createElement("span");
    label.className = "domain-name";
    label.textContent = d.name;

    left.appendChild(radio);
    left.appendChild(label);

    const right = document.createElement("span");
    right.className = "domain-conf";
    right.textContent = `conf ${d.confidence.toFixed(2)}`;

    row.appendChild(left);
    row.appendChild(right);
    domainList.appendChild(row);

    if (radio.checked) {
      chosenDomain = d.name;
    }
  });

  if (chosenDomain) {
    show(qualityCard);
  }
}

/**
 * Data quality summary card (rows, time range, per-metric stats).
 */
function renderQuality(quality) {
  if (!quality || !quality.per_metric) {
    qualityBox.innerHTML =
      '<p class="placeholder">No quality information available.</p>';
    return;
  }

  const { n_rows, n_cols, time_start, time_end, per_metric } = quality;
  const container = document.createElement("div");

  const grid = document.createElement("div");
  grid.className = "quality-grid";

  const pillRows = document.createElement("div");
  pillRows.className = "quality-pill";
  pillRows.innerHTML = `<strong>ROWS / COLUMNS</strong><span>${n_rows} × ${n_cols}</span>`;

  const pillTime = document.createElement("div");
  pillTime.className = "quality-pill";
  pillTime.innerHTML = `<strong>TIME RANGE</strong><span>${time_start || "?"} → ${
    time_end || "?"
  }</span>`;

  grid.appendChild(pillRows);
  grid.appendChild(pillTime);
  container.appendChild(grid);

  const table = document.createElement("table");
  table.className = "quality-metric-table";
  const thead = document.createElement("thead");
  thead.innerHTML =
    "<tr><th>Metric</th><th>Missing %</th><th>Outlier %</th><th>Range</th></tr>";
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  Object.entries(per_metric).forEach(([name, info]) => {
    const tr = document.createElement("tr");
    const pretty = prettifyName(name);
    const missing = (info.missing_ratio || 0) * 100;
    const outlier = (info.outlier_ratio || 0) * 100;
    const range =
      info.min === null || info.max === null
        ? "-"
        : `${info.min.toFixed(1)} – ${info.max.toFixed(1)}`;
    tr.innerHTML = `<td>${pretty}</td><td>${missing.toFixed(
      1
    )}%</td><td>${outlier.toFixed(1)}%</td><td>${range}</td>`;
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  container.appendChild(table);
  qualityBox.innerHTML = "";
  qualityBox.appendChild(container);
}

/**
 * Variable roles / units / descriptions.
 */
function renderVariables(vars) {
  if (!vars.length) {
    variablesList.innerHTML =
      '<p class="placeholder">No variable information available.</p>';
    return;
  }
  variablesList.innerHTML = "";
  vars.forEach((v) => {
    const row = document.createElement("div");
    row.className = "variable-row";

    const pretty = prettifyName(v.name);

    const col1 = document.createElement("div");
    col1.innerHTML = `<div class="variable-name">${pretty}</div>
      <div class="variable-meta">Original: <code>${v.name}</code></div>`;

    const col2 = document.createElement("div");
    col2.innerHTML = `<div class="variable-meta"><strong>Role:</strong> ${
      v.role
    }</div>
      <div class="variable-meta"><strong>Unit:</strong> ${
        v.unit || "unknown"
      } (conf ${v.unit_confidence?.toFixed?.(2) ?? "0.00"})</div>`;

    const col3 = document.createElement("div");
    col3.innerHTML = `<div class="variable-meta"><strong>Description:</strong> ${
      v.description || ""
    }</div>`;

    row.appendChild(col1);
    row.appendChild(col2);
    row.appendChild(col3);

    variablesList.appendChild(row);
  });
}

/**
 * Dataset preview: up to 20 rows, trying to respect the selected
 * analysis date range. If the sample has no rows in the window,
 * we fall back to the first rows and display a note.
 */
function renderDatasetPreview(preview) {
  if (
    !preview ||
    !preview.columns ||
    !preview.rows ||
    !preview.rows.length
  ) {
    previewTable.innerHTML =
      '<p class="placeholder">No preview rows available for this range.</p>';
    return;
  }

  const { columns, rows } = preview;
  const headRows = rows.slice(0, 10); // safety; backend already does n_rows

  const table = document.createElement("table");
  table.className = "preview-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  columns.forEach((col) => {
    const th = document.createElement("th");
    th.textContent = prettifyName(col);
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  headRows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((col) => {
      const td = document.createElement("td");
      const val = row[col];
      td.textContent =
        val === null || val === undefined ? "" : String(val);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  previewTable.innerHTML = "";
  previewTable.appendChild(table);
}

/* Step 1 → Step 2 transition */
toInsightsBtn.addEventListener("click", () => {
  if (!currentDatasetId || !chosenDomain || !qualityApprove.checked) return;
  setStep(2);
  refreshInsightsContext();
});

/* ================================================================
   STEP 2 · Insight Generation
   ================================================================ */

const insightsContext = document.getElementById("insightsContext");
const runInsightsBtn = document.getElementById("runInsightsBtn");
const insightsLoading = document.getElementById("insightsLoading");
const insightList = document.getElementById("insightList");
const insightDetail = document.getElementById("insightDetail");
const insightSortSelect = document.getElementById("insightSortSelect");
const toExploreBtn = document.getElementById("toExploreBtn");

/**
 * Text summary at the top of Step 2.
 */
function refreshInsightsContext() {
  if (!currentDatasetId) {
    insightsContext.textContent = "No dataset loaded.";
    runInsightsBtn.disabled = true;
    return;
  }
  const targetLabel = targetCandidates?.[0]?.name
    ? prettifyName(targetCandidates[0].name)
    : "traffic volume";

  const rangeStart =
    rangeStartInput.value ||
    (qualitySummary?.time_start
      ? qualitySummary.time_start.slice(0, 10)
      : "?");
  const rangeEnd =
    rangeEndInput.value ||
    (qualitySummary?.time_end ? qualitySummary.time_end.slice(0, 10) : "?");

  insightsContext.textContent =
    `Dataset ready. Domain: ${chosenDomain || "unknown"}. ` +
    `Target candidate: ${targetLabel}. ` +
    `Analysis range: ${rangeStart} → ${rangeEnd}.`;

  runInsightsBtn.disabled = false;
}

/**
 * Kick off backend insight generation.
 */
runInsightsBtn.addEventListener("click", async () => {
  if (!currentDatasetId) return;
  insightsLoading.classList.remove("hidden");
  insightList.innerHTML =
    '<p class="placeholder">Running insight generation...</p>';
  insightDetail.innerHTML =
    '<p class="placeholder">Waiting for insight selection.</p>';

  try {
    const res = await fetch("/generate_insights", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: currentDatasetId,
        chosen_domain: chosenDomain,
        date_from: rangeStartInput.value || null,
        date_to: rangeEndInput.value || null,
      }),
    });

    if (!res.ok) {
      throw new Error(`Server returned ${res.status}`);
    }
    const data = await res.json();
    rankedInsights = data.insights || [];
    vizSummary = data.summary || null;

    renderInsightList(rankedInsights);
    renderSummaryText(vizSummary);
    renderCharts(vizSummary);

    toExploreBtn.disabled = rankedInsights.length === 0;
    document.getElementById("qaAskBtn").disabled = !currentDatasetId;
  } catch (err) {
    console.error(err);
    insightList.innerHTML =
      '<p class="placeholder">Error while generating insights.</p>';
  } finally {
    insightsLoading.classList.add("hidden");
  }
});

insightSortSelect.addEventListener("change", () => {
  if (!rankedInsights.length) return;
  const sorted = sortInsights(rankedInsights, insightSortSelect.value);
  renderInsightList(sorted);
});

function sortInsights(insights, key) {
  const arr = [...insights];
  if (key === "severity") {
    const order = { high: 2, medium: 1, info: 0 };
    arr.sort(
      (a, b) =>
        (order[b.severity] || 0) - (order[a.severity] || 0) ||
        b.confidence - a.confidence
    );
  } else if (key === "confidence") {
    arr.sort((a, b) => b.confidence - a.confidence);
  } else if (key === "type") {
    arr.sort((a, b) =>
      (a.insight_type || "").localeCompare(b.insight_type || "")
    );
  } else {
    arr.sort((a, b) => (b.score || 0) - (a.score || 0));
  }
  return arr;
}

/**
 * Left-hand insight list.
 */
function renderInsightList(insights) {
  if (!insights.length) {
    insightList.innerHTML =
      '<p class="placeholder">No insights returned for this dataset.</p>';
    return;
  }
  insightList.innerHTML = "";
  insights.forEach((ins, idx) => {
    const card = document.createElement("article");
    card.className = "insight-card";
    card.dataset.index = idx;

    const header = document.createElement("div");
    header.className = "insight-header-row";

    const left = document.createElement("div");
    left.className = "insight-type";
    left.textContent = ins.insight_type;

    if (ins.metric) {
      const metricSpan = document.createElement("span");
      metricSpan.textContent = " · " + prettifyName(ins.metric);
      left.appendChild(metricSpan);
    }

    const right = document.createElement("div");
    right.className = "insight-meta";
    right.textContent = `${ins.severity} · conf ${ins.confidence.toFixed(
      2
    )} · score ${
      ins.score !== null && ins.score !== undefined
        ? ins.score.toFixed(2)
        : "n/a"
    }`;

    header.appendChild(left);
    header.appendChild(right);

    const caption = document.createElement("p");
    caption.className = "insight-caption";
    caption.textContent =
      ins.caption || "(no caption generated – check OpenAI key)";

    const timeP = document.createElement("p");
    timeP.className = "insight-time";
    if (Array.isArray(ins.time_window)) {
      timeP.textContent = `${ins.time_window[0]} → ${ins.time_window[1]}`;
    }

    card.appendChild(header);
    card.appendChild(caption);
    card.appendChild(timeP);

    card.addEventListener("click", () => {
      document
        .querySelectorAll(".insight-card")
        .forEach((c) => c.classList.remove("insight-card-selected"));
      card.classList.add("insight-card-selected");
      renderInsightDetail(ins);
    });

    insightList.appendChild(card);
  });

  // Auto-select first insight
  renderInsightDetail(insights[0]);
  const firstCard = insightList.querySelector(".insight-card");
  if (firstCard) firstCard.classList.add("insight-card-selected");
}

function renderInsightDetail(ins) {
  if (!ins) {
    insightDetail.innerHTML =
      '<p class="placeholder">Select an insight to view details.</p>';
    return;
  }
  insightDetail.innerHTML = "";
  const title = document.createElement("h4");
  title.textContent = `${ins.insight_type.toUpperCase()} ${
    ins.metric ? "· " + prettifyName(ins.metric) : ""
  }`;
  const seg = document.createElement("p");
  seg.innerHTML = `<strong>Segment:</strong> ${
    Object.keys(ins.segment || {}).length
      ? JSON.stringify(ins.segment)
      : "none"
  }`;
  const meta = document.createElement("p");
  meta.innerHTML = `<strong>Severity:</strong> ${ins.severity} &nbsp; <strong>Confidence:</strong> ${ins.confidence.toFixed(
    2
  )} &nbsp; <strong>Score:</strong> ${
    ins.score !== null && ins.score !== undefined
      ? ins.score.toFixed(2)
      : "n/a"
  }`;
  const details = document.createElement("pre");
  details.textContent = JSON.stringify(ins.details || {}, null, 2);

  insightDetail.appendChild(title);
  insightDetail.appendChild(seg);
  insightDetail.appendChild(meta);
  insightDetail.appendChild(details);
}

/* Step 2 → Step 3 */
toExploreBtn.addEventListener("click", () => {
  if (!rankedInsights.length) return;
  setStep(3);
});

/* ================================================================
   STEP 3 · Charts & Q/A
   ================================================================ */

const summaryText = document.getElementById("summaryText");
const chartTabs = document.querySelectorAll(".chart-tab");
const chartTargetWrapper = document.getElementById("chart-target-wrapper");
const chartHolidayWrapper = document.getElementById("chart-holiday-wrapper");
const chartWeatherWrapper = document.getElementById("chart-weather-wrapper");

let chartTarget = null;
let chartHoliday = null;
let chartWeather = null;

chartTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const which = tab.dataset.chart;
    chartTabs.forEach((t) => t.classList.remove("chart-tab-active"));
    tab.classList.add("chart-tab-active");
    chartTargetWrapper.classList.toggle("hidden", which !== "target");
    chartHolidayWrapper.classList.toggle("hidden", which !== "holiday");
    chartWeatherWrapper.classList.toggle("hidden", which !== "weather");
  });
});

function renderSummaryText(summary) {
  if (!summary || !summary.target_name) {
    summaryText.textContent =
      "No summary available yet. Make sure insight generation has completed successfully.";
    return;
  }
  const targetPretty = prettifyName(summary.target_name);
  summaryText.textContent = `Summary for target ${targetPretty}: daily averages, holiday vs non-holiday differences, and weather impact are visualised below.`;
}

function renderCharts(summary) {
  if (!summary || !summary.target_name) {
    return;
  }

  const targetPretty = prettifyName(summary.target_name);

  // Destroy existing charts if any
  if (chartTarget) chartTarget.destroy();
  if (chartHoliday) chartHoliday.destroy();
  if (chartWeather) chartWeather.destroy();

  const daily = summary.daily_target || [];
  const labelsDaily = daily.map((d) => d.date);
  const valuesDaily = daily.map((d) => d.value);

  const ctxTarget = document.getElementById("chartTarget").getContext("2d");
  chartTarget = new Chart(ctxTarget, {
    type: "line",
    data: {
      labels: labelsDaily,
      datasets: [
        {
          label: targetPretty,
          data: valuesDaily,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: true },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 6 } },
      },
    },
  });

  const holiday = summary.holiday_summary || [];
  const labelsHoliday = holiday.map((h) => h.holiday);
  const valuesHoliday = holiday.map((h) => h.mean);

  const ctxHoliday = document.getElementById("chartHoliday").getContext("2d");
  chartHoliday = new Chart(ctxHoliday, {
    type: "bar",
    data: {
      labels: labelsHoliday,
      datasets: [
        {
          label: targetPretty,
          data: valuesHoliday,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
    },
  });

  const weather = summary.weather_summary || [];
  const labelsWeather = weather.map((w) => w.weather_main);
  const valuesWeather = weather.map((w) => w.mean);

  const ctxWeather = document.getElementById("chartWeather").getContext("2d");
  chartWeather = new Chart(ctxWeather, {
    type: "bar",
    data: {
      labels: labelsWeather,
      datasets: [
        {
          label: targetPretty,
          data: valuesWeather,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
    },
  });
}

/* Q/A interaction */

const qaInput = document.getElementById("qaInput");
const qaAskBtn = document.getElementById("qaAskBtn");
const qaAnswerBox = document.getElementById("qaAnswerBox");
const qaFeedbackBox = document.getElementById("qaFeedbackBox");
const qaThumbUp = document.getElementById("qaThumbUp");
const qaThumbDown = document.getElementById("qaThumbDown");
const qaCorrectionBox = document.getElementById("qaCorrectionBox");
const qaCorrectionInput = document.getElementById("qaCorrectionInput");
const qaCorrectionSubmit = document.getElementById("qaCorrectionSubmit");

qaAskBtn.addEventListener("click", async () => {
  const question = qaInput.value.trim();
  if (!question || !currentDatasetId) return;

  qaAnswerBox.innerHTML =
    '<p class="placeholder">Thinking about your question...</p>';
  qaFeedbackBox.classList.add("hidden");
  qaCorrectionBox.classList.add("hidden");
  lastQA = { question: "", answer: "" };

  try {
    const res = await fetch("/qa", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: currentDatasetId,
        question,
      }),
    });
    if (!res.ok) {
      throw new Error(`Server returned ${res.status}`);
    }
    const data = await res.json();
    const answer = data.answer || "(no answer)";
    lastQA = { question, answer };
    qaAnswerBox.textContent = answer;
    qaFeedbackBox.classList.remove("hidden");
  } catch (err) {
    console.error(err);
    qaAnswerBox.innerHTML =
      '<p class="placeholder">Error while answering the question.</p>';
  }
});

qaThumbUp.addEventListener("click", async () => {
  if (!lastQA.question || !lastQA.answer) return;
  await sendQAFeedback("up");
  qaFeedbackBox.classList.add("hidden");
});

qaThumbDown.addEventListener("click", () => {
  if (!lastQA.question || !lastQA.answer) return;
  qaCorrectionBox.classList.remove("hidden");
});

qaCorrectionSubmit.addEventListener("click", async () => {
  const corrected = qaCorrectionInput.value.trim();
  if (!corrected) return;
  await sendQAFeedback("down", corrected);
  qaCorrectionInput.value = "";
  qaCorrectionBox.classList.add("hidden");
  qaFeedbackBox.classList.add("hidden");
});

async function sendQAFeedback(feedback, correctedAnswer = "") {
  try {
    await fetch("/qa_feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: currentDatasetId,
        question: lastQA.question,
        answer: lastQA.answer,
        feedback,
        corrected_answer: correctedAnswer,
      }),
    });
  } catch (err) {
    console.error("Error sending QA feedback:", err);
  }
}

/* ================================================================
   Utilities + initial state
   ================================================================ */

function prettifyName(name) {
  if (!name) return "";
  const spaced = String(name).replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

// Start on Step 1
setStep(1);
