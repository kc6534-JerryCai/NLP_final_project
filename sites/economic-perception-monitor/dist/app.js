(() => {
  "use strict";

  const data = window.DASHBOARD_DATA;
  if (!data) {
    document.body.innerHTML = "<p style='padding:2rem'>Dashboard data could not be loaded.</p>";
    return;
  }

  const state = {
    range: 38,
    outcome: "michigan_sentiment",
    model: "economic",
    mediaMetric: "tone",
  };

  const outcomeConfig = {
    michigan_sentiment: { label: "Headline", timelineKey: "sentiment" },
    michigan_current_conditions: { label: "Current conditions", timelineKey: "current" },
    michigan_expectations: { label: "Expectations", timelineKey: "expectations" },
  };

  const mediaMetricConfig = {
    tone: {
      title: "Media tone index",
      subtitle: "Positive share minus negative share, scaled −100 to +100.",
      definition: "Higher values mean more positive language.",
      key: (row) => row.toneIndex,
      color: "#b95348",
      suffix: "",
    },
    frame: {
      title: "Future-risk framing",
      subtitle: "Share of relevant items coded with a future-risk frame.",
      definition: "Framing is multi-label; shares can sum above 100%.",
      key: (row) => row.frameShares.future_risk * 100,
      color: "#dc8a31",
      suffix: "%",
    },
    salience: {
      title: "Cost-of-living salience",
      subtitle: "Share of relevant items with cost of living and income as the primary topic.",
      definition: "Salience measures selection, not audience exposure.",
      key: (row) => row.topicShares.cost_of_living_income * 100,
      color: "#466f8c",
      suffix: "%",
    },
    calibration: {
      title: "Factual-calibration coverage",
      subtitle: "Share of relevant items with an assessable numerical claim.",
      definition: "Low coverage means calibration conclusions must remain narrow.",
      key: (row) => row.calibrationAssessableShare * 100,
      color: "#0d6b57",
      suffix: "%",
    },
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const setText = (selector, value) => {
    const element = $(selector);
    if (element) element.textContent = value;
  };
  const nfmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
  const ifmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  const percent = (value, digits = 0) => value == null ? "—" : `${(value * 100).toFixed(digits)}%`;
  const value = (number, digits = 1) => number == null ? "—" : Number(number).toFixed(digits);
  const signed = (number, digits = 1) => number == null ? "—" : `${number >= 0 ? "+" : ""}${Number(number).toFixed(digits)}`;
  const escapeHtml = (text) => String(text).replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character]));
  const monthLabel = (month, short = false) => {
    const [year, number] = month.split("-").map(Number);
    return new Intl.DateTimeFormat("en-US", { month: short ? "short" : "long", year: "numeric", timeZone: "UTC" }).format(new Date(Date.UTC(year, number - 1, 1)));
  };

  function latestNonNull(rows, key, beforeIndex = rows.length - 1) {
    for (let index = beforeIndex; index >= 0; index -= 1) {
      if (rows[index][key] != null) return { row: rows[index], index };
    }
    return { row: null, index: -1 };
  }

  function initializeKpis() {
    const timeline = data.timeline;
    const latest = timeline[timeline.length - 1];
    const previous = timeline[timeline.length - 2];
    const change = latest.sentiment - previous.sentiment;
    setText("#kpi-sentiment", value(data.latest.sentiment));
    $("#kpi-sentiment-change").innerHTML = `<span class="${change < 0 ? "trend-down" : "trend-up"}">${change < 0 ? "↓" : "↑"} ${Math.abs(change).toFixed(1)} pts</span> vs ${escapeHtml(monthLabel(previous.month, true))}`;
    setText("#kpi-prediction", value(data.latest.econPrediction));
    setText("#kpi-gap", signed(data.latest.gap));
    setText("#kpi-gap-note", `${Math.abs(data.latest.gap).toFixed(1)} points ${data.latest.gap < 0 ? "below" : "above"} model baseline`);
    $("#kpi-gap-range").style.left = `${Math.max(5, Math.min(95, 50 + data.latest.gap * 3.5))}%`;
    setText("#kpi-tone", value(data.latest.toneIndex));
    setText("#kpi-tone-note", `${percent(data.latest.negativeShare)} negative · n=${ifmt.format(data.latest.mediaN)}`);
    $(".tone-spectrum span").style.left = `${Math.max(3, Math.min(97, (data.latest.toneIndex + 100) / 2))}%`;

    setText("#signal-gap", signed(data.latest.gap));
    setText("#signal-low", value(data.latest.predictionLow));
    setText("#signal-high", value(data.latest.predictionHigh));
    setText("#signal-copy", `Observed sentiment is ${value(data.latest.sentiment)} versus an economic-only prediction of ${value(data.latest.econPrediction)}. The observation remains inside the forecast interval.`);
    const marker = ((data.latest.sentiment - data.latest.predictionLow) / (data.latest.predictionHigh - data.latest.predictionLow)) * 100;
    $("#signal-marker").style.left = `${Math.max(0, Math.min(100, marker))}%`;
  }

  function rangeSlice(rows) {
    if (state.range === "all") return rows;
    return rows.slice(-Number(state.range));
  }

  function svgElement(name, attributes = {}) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attributes).forEach(([key, attributeValue]) => element.setAttribute(key, attributeValue));
    return element;
  }

  function pathFromValues(rows, getValue, x, y) {
    let path = "";
    let drawing = false;
    rows.forEach((row, index) => {
      const current = getValue(row);
      if (current == null || Number.isNaN(current)) {
        drawing = false;
        return;
      }
      path += `${drawing ? "L" : "M"}${x(index).toFixed(2)},${y(current).toFixed(2)} `;
      drawing = true;
    });
    return path.trim();
  }

  function areaPath(rows, getLow, getHigh, x, y) {
    const valid = rows.map((row, index) => ({ row, index })).filter(({ row }) => getLow(row) != null && getHigh(row) != null);
    if (!valid.length) return "";
    const upper = valid.map(({ row, index }, i) => `${i ? "L" : "M"}${x(index).toFixed(2)},${y(getHigh(row)).toFixed(2)}`).join(" ");
    const lower = valid.slice().reverse().map(({ row, index }) => `L${x(index).toFixed(2)},${y(getLow(row)).toFixed(2)}`).join(" ");
    return `${upper} ${lower} Z`;
  }

  function renderLineChart(container, rows, series, options = {}) {
    container.innerHTML = "";
    if (!rows.length) return;
    const width = Math.max(360, container.clientWidth || 760);
    const height = options.height || (container.classList.contains("compact-chart") ? 170 : 330);
    const margin = { top: 19, right: 12, bottom: 32, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const values = [];
    series.forEach((item) => rows.forEach((row) => {
      const current = item.get(row);
      if (current != null && !Number.isNaN(current)) values.push(current);
    }));
    if (options.band) {
      rows.forEach((row) => {
        const low = options.band.low(row);
        const high = options.band.high(row);
        if (low != null) values.push(low);
        if (high != null) values.push(high);
      });
    }
    if (options.includeZero) values.push(0);
    let min = options.min != null ? options.min : Math.min(...values);
    let max = options.max != null ? options.max : Math.max(...values);
    if (min === max) { min -= 1; max += 1; }
    const pad = options.fixedDomain ? 0 : (max - min) * 0.09;
    min -= pad;
    max += pad;
    const x = (index) => margin.left + (rows.length === 1 ? innerWidth / 2 : index * innerWidth / (rows.length - 1));
    const y = (current) => margin.top + (max - current) * innerHeight / (max - min);

    const svg = svgElement("svg", { viewBox: `0 0 ${width} ${height}`, role: "presentation" });
    const gridGroup = svgElement("g", { "aria-hidden": "true" });
    for (let tick = 0; tick < 5; tick += 1) {
      const ratio = tick / 4;
      const tickValue = max - ratio * (max - min);
      const tickY = margin.top + ratio * innerHeight;
      gridGroup.append(svgElement("line", { x1: margin.left, x2: width - margin.right, y1: tickY, y2: tickY, stroke: tickValue === 0 ? "#9d9f99" : "#dedbd3", "stroke-width": tickValue === 0 ? 1.1 : .7, "stroke-dasharray": tickValue === 0 ? "3 4" : "" }));
      const label = svgElement("text", { x: margin.left - 8, y: tickY + 3, "text-anchor": "end", fill: "#8f9692", "font-size": 8 });
      label.textContent = options.yFormat ? options.yFormat(tickValue) : nfmt.format(tickValue);
      gridGroup.append(label);
    }
    svg.append(gridGroup);

    const tickCount = Math.min(6, rows.length);
    const usedIndexes = new Set();
    for (let tick = 0; tick < tickCount; tick += 1) {
      const index = Math.round(tick * (rows.length - 1) / Math.max(1, tickCount - 1));
      if (usedIndexes.has(index)) continue;
      usedIndexes.add(index);
      const label = svgElement("text", { x: x(index), y: height - 8, "text-anchor": index === 0 ? "start" : index === rows.length - 1 ? "end" : "middle", fill: "#8f9692", "font-size": 8 });
      label.textContent = monthLabel(rows[index].month, true).replace(" ", " ’").replace("’20", "’");
      svg.append(label);
    }

    if (options.band) {
      const path = areaPath(rows, options.band.low, options.band.high, x, y);
      if (path) svg.append(svgElement("path", { d: path, fill: options.band.fill || "#dceae4", opacity: .7 }));
    }

    series.forEach((item) => {
      const path = pathFromValues(rows, item.get, x, y);
      if (!path) return;
      svg.append(svgElement("path", {
        d: path,
        fill: "none",
        stroke: item.color,
        "stroke-width": item.width || 2,
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
        "stroke-dasharray": item.dash || "",
        opacity: item.opacity || 1,
      }));
      const lastIndex = [...rows.keys()].reverse().find((index) => item.get(rows[index]) != null);
      if (lastIndex != null) svg.append(svgElement("circle", { cx: x(lastIndex), cy: y(item.get(rows[lastIndex])), r: 3.3, fill: item.color, stroke: "#fbfaf6", "stroke-width": 1.5 }));
    });

    const hoverLine = svgElement("line", { y1: margin.top, y2: height - margin.bottom, stroke: "#62706b", "stroke-width": .8, opacity: 0, "stroke-dasharray": "2 3" });
    svg.append(hoverLine);
    const overlay = svgElement("rect", { x: margin.left, y: margin.top, width: innerWidth, height: innerHeight, fill: "transparent", tabindex: 0 });
    svg.append(overlay);
    container.append(svg);
    const tooltip = document.createElement("div");
    tooltip.className = "chart-tooltip";
    tooltip.hidden = true;
    container.append(tooltip);

    const updateTooltip = (event) => {
      const rect = svg.getBoundingClientRect();
      const svgX = (event.clientX - rect.left) * width / rect.width;
      const index = Math.max(0, Math.min(rows.length - 1, Math.round((svgX - margin.left) / innerWidth * (rows.length - 1))));
      const row = rows[index];
      hoverLine.setAttribute("x1", x(index));
      hoverLine.setAttribute("x2", x(index));
      hoverLine.setAttribute("opacity", 1);
      const details = series.map((item) => {
        const current = item.get(row);
        return current == null ? "" : `<div><span>${escapeHtml(item.label)}</span><b>${escapeHtml(item.format ? item.format(current) : nfmt.format(current))}</b></div>`;
      }).join("");
      tooltip.innerHTML = `<strong>${escapeHtml(monthLabel(row.month))}</strong>${details}`;
      tooltip.hidden = false;
      tooltip.style.left = `${Math.max(78, Math.min(rect.width - 78, event.clientX - rect.left))}px`;
      tooltip.style.top = `${Math.max(85, event.clientY - rect.top)}px`;
    };
    overlay.addEventListener("pointermove", updateTooltip);
    overlay.addEventListener("pointerleave", () => { tooltip.hidden = true; hoverLine.setAttribute("opacity", 0); });
  }

  function sentimentRows() {
    const forecastMap = new Map((data.forecasts[state.outcome] || []).map((row) => [row.month, row]));
    const timelineKey = outcomeConfig[state.outcome].timelineKey;
    return rangeSlice(data.timeline).map((row) => ({
      month: row.month,
      observed: row[timelineKey],
      ...(forecastMap.get(row.month) || {}),
    }));
  }

  function renderSentimentChart() {
    const rows = sentimentRows();
    const series = [
      { label: "Observed", color: "#1d2925", width: 2.3, get: (row) => row.observed },
      { label: "Economic baseline", color: "#2e927b", width: state.model === "economic" ? 2.1 : 1.2, dash: state.model === "economic" ? "" : "4 4", opacity: state.model === "economic" ? 1 : .68, get: (row) => row.econ },
    ];
    if (state.model === "media") series.push({ label: "+ Media", color: "#dc8a31", width: 2.2, get: (row) => row.media });
    if (state.model === "full") series.push({ label: "+ Full controls", color: "#466f8c", width: 2.2, get: (row) => row.full });
    renderLineChart($("#sentiment-chart"), rows, series, {
      band: { low: (row) => row.low, high: (row) => row.high, fill: "#dceae4" },
      yFormat: (current) => current.toFixed(0),
    });
  }

  function renderGapChart() {
    const forecastRows = rangeSlice(data.forecasts.michigan_sentiment);
    const container = $("#gap-chart");
    container.innerHTML = "";
    const width = Math.max(360, container.clientWidth || 800);
    const height = 170;
    const margin = { top: 12, right: 8, bottom: 28, left: 38 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const maxAbs = Math.max(1, ...forecastRows.map((row) => Math.abs(row.gap || 0))) * 1.12;
    const xBand = innerWidth / forecastRows.length;
    const y = (current) => margin.top + (maxAbs - current) * innerHeight / (maxAbs * 2);
    const zero = y(0);
    const svg = svgElement("svg", { viewBox: `0 0 ${width} ${height}`, role: "presentation" });
    [-maxAbs, 0, maxAbs].forEach((tick) => {
      const tickY = y(tick);
      svg.append(svgElement("line", { x1: margin.left, x2: width - margin.right, y1: tickY, y2: tickY, stroke: tick === 0 ? "#777e7a" : "#dedbd3", "stroke-width": tick === 0 ? 1 : .7, "stroke-dasharray": tick === 0 ? "3 3" : "" }));
      const label = svgElement("text", { x: margin.left - 7, y: tickY + 3, "text-anchor": "end", fill: "#8f9692", "font-size": 8 });
      label.textContent = tick === 0 ? "0" : signed(tick, 0);
      svg.append(label);
    });
    forecastRows.forEach((row, index) => {
      const barX = margin.left + index * xBand + Math.max(1, xBand * .14);
      const barWidth = Math.max(2, xBand * .72);
      const barY = row.gap >= 0 ? y(row.gap) : zero;
      const barHeight = Math.max(1, Math.abs(y(row.gap) - zero));
      const rect = svgElement("rect", { x: barX, y: barY, width: barWidth, height: barHeight, rx: 1.5, fill: row.gap >= 0 ? "#2e927b" : "#b95348", opacity: .82 });
      const title = svgElement("title");
      title.textContent = `${monthLabel(row.month)}: ${signed(row.gap)} points`;
      rect.append(title);
      svg.append(rect);
    });
    const labelIndexes = [0, Math.floor((forecastRows.length - 1) / 2), forecastRows.length - 1];
    labelIndexes.forEach((index) => {
      const label = svgElement("text", { x: margin.left + (index + .5) * xBand, y: height - 7, "text-anchor": index === 0 ? "start" : index === forecastRows.length - 1 ? "end" : "middle", fill: "#8f9692", "font-size": 8 });
      label.textContent = monthLabel(forecastRows[index].month, true);
      svg.append(label);
    });
    container.append(svg);
  }

  function renderMediaChart() {
    const config = mediaMetricConfig[state.mediaMetric];
    setText("#media-chart-title", config.title);
    setText("#media-chart-subtitle", config.subtitle);
    setText("#media-chart-definition", config.definition);
    const rows = rangeSlice(data.media);
    const total = rows.reduce((sum, row) => sum + row.nRelevant, 0);
    setText("#media-chart-sample", `${ifmt.format(total)} relevant items in displayed window`);
    const options = { includeZero: state.mediaMetric === "tone", yFormat: (current) => `${current.toFixed(0)}${config.suffix}` };
    if (state.mediaMetric !== "tone") { options.min = 0; options.max = Math.max(10, ...rows.map(config.key)) * 1.12; }
    renderLineChart($("#media-chart"), rows, [{ label: config.title, color: config.color, width: 2.25, get: config.key, format: (current) => `${current.toFixed(1)}${config.suffix}` }], options);
  }

  function renderSnapshot() {
    const current = data.media[data.media.length - 1];
    setText("#snapshot-n", `n=${current.nRelevant}`);
    setText("#tone-index-label", value(current.toneIndex));
    const total = Math.max(1, current.negative + current.neutral + current.positive);
    $("#tone-stack").innerHTML = `
      <span class="negative" style="width:${current.negative / total * 100}%" title="Negative: ${current.negative}"></span>
      <span class="neutral" style="width:${current.neutral / total * 100}%" title="Neutral: ${current.neutral}"></span>
      <span class="positive" style="width:${current.positive / total * 100}%" title="Positive: ${current.positive}"></span>`;
    const topicLabels = {
      cost_of_living_income: "Cost of living",
      labor_market: "Labor market",
      household_economy: "Household economy",
      macro_growth: "Macro growth",
      economic_policy: "Economic policy",
    };
    const frameLabels = {
      household_hardship: "Household hardship",
      economic_resilience: "Economic resilience",
      unequal_outcomes: "Unequal outcomes",
      macro_household_disconnect: "Macro–household gap",
      future_risk: "Future risk",
    };
    const barRows = (entries, labels) => entries.sort((a, b) => b[1] - a[1]).map(([key, share]) => `
      <div class="bar-row"><span title="${escapeHtml(labels[key])}">${escapeHtml(labels[key])}</span><span class="bar-track"><i style="width:${Math.max(1, share * 100)}%"></i></span><strong>${percent(share)}</strong></div>`).join("");
    $("#topic-bars").innerHTML = barRows(Object.entries(current.topicShares), topicLabels);
    $("#frame-bars").innerHTML = barRows(Object.entries(current.frameShares), frameLabels);
    setText("#calibration-assessable", percent(current.calibrationAssessableShare, 0));
    setText("#calibration-consistent", current.calibrationConsistentShare == null ? "n/a" : percent(current.calibrationConsistentShare, 0));
  }

  function renderEvents() {
    const sourceEvents = data.events.slice(-3);
    const events = [...sourceEvents, {
      date: data.meta.collectionCutoff,
      type: "cutoff",
      title: "Prospective monitoring cutoff",
      detail: `Latest returned NYT publication: ${String(data.meta.latestPublication).slice(0, 10)}.`,
      url: null,
    }];
    $("#event-timeline").innerHTML = events.map((event) => `
      <article class="event-item ${escapeHtml(event.type)}">
        <div class="event-date">${escapeHtml(event.date)}</div>
        <h4>${escapeHtml(event.title)}</h4>
        <p>${escapeHtml(event.detail)}</p>
        ${event.url ? `<a href="${escapeHtml(event.url)}" target="_blank" rel="noreferrer">Official release ↗</a>` : ""}
      </article>`).join("");
  }

  function renderDiagnostics() {
    const finding = data.findings;
    setText("#diag-lift", `${finding.mediaRmseImprovement.toFixed(2)}%`);
    setText("#diag-rmse-econ", `Econ ${finding.econRmse.toFixed(2)}`);
    setText("#diag-rmse-media", `Media ${finding.mediaRmse.toFixed(2)}`);
    setText("#diag-beta", finding.primaryMedia.estimate.toFixed(3));
    setText("#diag-beta-ci", `95% CI [${finding.primaryMedia.low.toFixed(3)}, ${finding.primaryMedia.high.toFixed(3)}]`);
    setText("#diag-beta-p", `HAC p = ${finding.primaryMedia.p.toFixed(3)}`);
    setText("#diag-coverage", percent(finding.intervalCoverage, 1));
    $("#diag-coverage-bar").style.width = percent(finding.intervalCoverage, 1);
    setText("#diag-iv", `2SLS β = ${finding.ivEstimate.toFixed(3)} · p = ${finding.ivP.toFixed(3)}`);

    const wanted = ["naive", "ols_econ", "ols_media", "ols_full"];
    const labels = { naive: "Naïve last value", ols_econ: "Economic only", ols_media: "Economic + media", ols_full: "Full controls" };
    const metrics = data.forecastMetrics.filter((row) => row.outcome === "michigan_sentiment" && wanted.includes(row.model)).sort((a, b) => wanted.indexOf(a.model) - wanted.indexOf(b.model));
    const max = Math.max(...metrics.map((row) => row.rmse));
    const min = Math.min(...metrics.map((row) => row.rmse));
    $("#model-comparison").innerHTML = metrics.map((row) => {
      const score = 55 + (max - row.rmse) / Math.max(.001, max - min) * 45;
      return `<div class="model-item ${row.rmse === min ? "best" : ""} ${row.model === "ols_media" ? "media-model" : ""}"><div class="model-label"><span>${labels[row.model]}</span><strong>${row.rmse.toFixed(2)}</strong></div><div class="model-bar"><span style="width:${score}%"></span></div></div>`;
    }).join("");
  }

  function renderDataHealth() {
    setText("#corpus-total", ifmt.format(data.corpus.cleanedArticles));
    setText("#ai-topic-accuracy", percent(data.aiPerformance.topicAccuracy, 1));
    setText("#ai-topic-f1", `macro F1 ${data.aiPerformance.topicMacroF1.toFixed(3)}`);
    setText("#ai-tone-accuracy", percent(data.aiPerformance.toneAccuracy, 1));
    setText("#ai-tone-f1", `macro F1 ${data.aiPerformance.toneMacroF1.toFixed(3)}`);
    setText("#ai-tone-agreement", percent(data.aiPerformance.modelToneAgreement, 1));

    const months = data.timeline.slice(-3);
    const series = [
      ["Michigan", "sentiment", "index"],
      ["U-3", "unemployment", "%"],
      ["PCE YoY", "pceInflation", "%"],
      ["Real GDP", "gdpQoq", "% q/q ann."],
      ["Media tone", "toneIndex", "index"],
    ];
    let grid = `<div class="header">Series</div>${months.map((row) => `<div class="header">${escapeHtml(row.month)}</div>`).join("")}`;
    series.forEach(([label, key, unit]) => {
      grid += `<div>${label}</div>`;
      months.forEach((row) => {
        const current = row[key];
        const isGdp = key === "gdpQoq";
        grid += `<div class="${current == null ? "missing" : isGdp ? "quarterly" : "available"}">${current == null ? "Missing" : `${Number(current).toFixed(1)}${unit === "index" ? "" : "%"}${isGdp ? " · Q" : ""}`}</div>`;
      });
    });
    $("#missingness-grid").innerHTML = grid;
    $("#limitations-list").innerHTML = data.limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  }

  function bindControls() {
    $$("[data-range]").forEach((button) => button.addEventListener("click", () => {
      state.range = button.dataset.range === "all" ? "all" : Number(button.dataset.range);
      $$("[data-range]").forEach((item) => item.classList.toggle("active", item === button));
      renderSentimentChart();
      renderGapChart();
      renderMediaChart();
    }));
    $$("[data-outcome]").forEach((button) => button.addEventListener("click", () => {
      state.outcome = button.dataset.outcome;
      $$("[data-outcome]").forEach((item) => item.classList.toggle("active", item === button));
      renderSentimentChart();
    }));
    $$("[data-model]").forEach((button) => button.addEventListener("click", () => {
      if (button.dataset.model === "observed") return;
      state.model = button.dataset.model;
      $$("[data-model]").forEach((item) => {
        if (item.dataset.model !== "observed") item.classList.toggle("active", item === button);
      });
      renderSentimentChart();
    }));
    $$("[data-media-metric]").forEach((button) => button.addEventListener("click", () => {
      state.mediaMetric = button.dataset.mediaMetric;
      $$("[data-media-metric]").forEach((item) => item.classList.toggle("active", item === button));
      renderMediaChart();
    }));
    $$("[data-scroll-to]").forEach((button) => button.addEventListener("click", () => {
      document.getElementById(button.dataset.scrollTo)?.scrollIntoView({ behavior: "smooth" });
    }));

    const dialog = $("#methodology-dialog");
    $("#open-methodology").addEventListener("click", () => dialog.showModal());
    $("#close-methodology").addEventListener("click", () => dialog.close());
    $("#close-methodology-footer").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });

    $("#download-snapshot").addEventListener("click", () => {
      const payload = {
        generated_at: data.meta.generatedAt,
        cutoff: data.meta.collectionCutoff,
        latest: data.latest,
        findings: data.findings,
        limitations: data.limitations,
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "economic-perception-monitor-snapshot.json";
      link.click();
      URL.revokeObjectURL(url);
    });

    const sections = $$("main section[id], main header[id]");
    const navLinks = $$(".section-nav a");
    const observer = new IntersectionObserver((entries) => {
      const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      navLinks.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`));
    }, { rootMargin: "-18% 0px -66%", threshold: [0, .1, .4] });
    sections.forEach((section) => observer.observe(section));
  }

  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      renderSentimentChart();
      renderGapChart();
      renderMediaChart();
    }, 120);
  });

  initializeKpis();
  renderSentimentChart();
  renderGapChart();
  renderMediaChart();
  renderSnapshot();
  renderEvents();
  renderDiagnostics();
  renderDataHealth();
  bindControls();
})();
