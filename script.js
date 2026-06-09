const clusterColors = [
  "#0873e6", "#19bde8", "#0757b8", "#7c83ff", "#35a6ff",
  "#0e9a9a", "#4f69c6", "#58b6d8", "#24508f", "#90a9ff",
];

const datasetInsights = {
  pendigits: {
    takeaway: "Digit shape produces substantial structure, but some written digits share trajectories and merge geometrically.",
  },
  satellite: {
    takeaway: "Three broad spectral groups are strongest internally, while six clusters align more closely with land-cover labels.",
  },
  yeast: {
    takeaway: "Compact distance groups do not reproduce the ten localization classes well; the known biology is not spherical.",
  },
};

const state = {
  artifact: null,
  datasetKey: "pendigits",
  k: 2,
  view: "clusters",
  metric: "silhouette",
  convergenceFrame: 0,
  animationTimer: null,
  renderedPoints: [],
};

const elements = {
  canvas: document.querySelector("#cluster-canvas"),
  tooltip: document.querySelector("#canvas-tooltip"),
  datasetTabs: document.querySelector("#dataset-tabs"),
  kSlider: document.querySelector("#k-slider"),
  kValue: document.querySelector("#k-value"),
  kOutput: document.querySelector("#k-output"),
  kContext: document.querySelector("#k-context"),
  scatterTitle: document.querySelector("#scatter-title"),
  projectionNote: document.querySelector("#projection-note"),
  sampleNote: document.querySelector("#sample-note"),
  qualityChart: document.querySelector("#quality-chart"),
  qualityCaption: document.querySelector("#quality-caption"),
  convergenceChart: document.querySelector("#convergence-chart"),
  iterationValue: document.querySelector("#iteration-value"),
  clusterSizeChart: document.querySelector("#cluster-size-chart"),
};

function currentDataset() {
  return state.artifact.datasets[state.datasetKey];
}

function currentRun() {
  return currentDataset().runs[String(state.k)];
}

function setSliderFill() {
  const fill = (state.k - 2) / 8 * 100;
  elements.kSlider.style.setProperty("--range-fill", `${fill}%`);
}

function formatNumber(value) {
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(3);
}

function renderDatasetTabs() {
  elements.datasetTabs.innerHTML = Object.entries(state.artifact.datasets).map(([key, dataset]) => `
    <button class="dataset-tab" type="button" role="tab" data-dataset="${key}" aria-selected="${key === state.datasetKey}">
      <span><strong>${dataset.title}</strong><small>${dataset.domain}</small></span>
      <span>${dataset.rows.toLocaleString()} x ${dataset.features}</span>
    </button>
  `).join("");
  elements.datasetTabs.querySelectorAll(".dataset-tab").forEach((button) => {
    button.addEventListener("click", () => {
      state.datasetKey = button.dataset.dataset;
      state.k = currentDataset().bestSilhouetteK;
      elements.kSlider.value = state.k;
      renderAll();
    });
  });
}

function resizeCanvas() {
  const rect = elements.canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  elements.canvas.width = Math.round(rect.width * ratio);
  elements.canvas.height = Math.round(rect.height * ratio);
  const context = elements.canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawScatter() {
  resizeCanvas();
  const context = elements.canvas.getContext("2d");
  const rect = elements.canvas.getBoundingClientRect();
  const dataset = currentDataset();
  const run = currentRun();
  const padding = 30;
  const width = rect.width - padding * 2;
  const height = rect.height - padding * 2;
  context.clearRect(0, 0, rect.width, rect.height);
  state.renderedPoints = [];

  dataset.scatter.points.forEach((point, index) => {
    const x = padding + ((point.x + 1) / 2) * width;
    const y = padding + (1 - ((point.y + 1) / 2)) * height;
    const colorIndex = state.view === "clusters"
      ? run.assignments[index]
      : dataset.knownClassLabels.indexOf(point.label);
    context.beginPath();
    context.arc(x, y, state.view === "clusters" ? 3.1 : 2.9, 0, Math.PI * 2);
    context.fillStyle = `${clusterColors[colorIndex % clusterColors.length]}b8`;
    context.fill();
    state.renderedPoints.push({
      x,
      y,
      cluster: run.assignments[index],
      label: point.label,
    });
  });

  if (state.view === "clusters") {
    run.centers.forEach((center, index) => {
      const x = padding + ((center[0] + 1) / 2) * width;
      const y = padding + (1 - ((center[1] + 1) / 2)) * height;
      context.beginPath();
      context.arc(x, y, 9, 0, Math.PI * 2);
      context.fillStyle = "#ffffff";
      context.fill();
      context.lineWidth = 3;
      context.strokeStyle = clusterColors[index % clusterColors.length];
      context.stroke();
      context.beginPath();
      context.moveTo(x - 4, y);
      context.lineTo(x + 4, y);
      context.moveTo(x, y - 4);
      context.lineTo(x, y + 4);
      context.stroke();
    });
  }
}

function renderMetrics() {
  const dataset = currentDataset();
  const run = currentRun();
  elements.kValue.textContent = state.k;
  elements.kOutput.textContent = state.k;
  elements.kContext.textContent = state.k === dataset.bestSilhouetteK
    ? "Strongest silhouette in the tested range"
    : state.k === dataset.knownClassK
      ? `Matches the ${dataset.knownClasses} known classes`
      : `${state.k} candidate clusters`;
  elements.scatterTitle.textContent = `${dataset.title} at k = ${state.k}`;
  elements.projectionNote.textContent = `PCA view retains ${((dataset.pcaExplainedVariance[0] + dataset.pcaExplainedVariance[1]) * 100).toFixed(1)}% of variance`;
  elements.sampleNote.textContent = `${dataset.scatter.sampleSize.toLocaleString()} of ${dataset.rows.toLocaleString()} objects displayed`;
  document.querySelector("#metric-silhouette").textContent = run.silhouette.toFixed(3);
  document.querySelector("#metric-ari").textContent = run.adjustedRand.toFixed(3);
  document.querySelector("#metric-inertia").textContent = formatNumber(run.inertiaPerPoint);
  document.querySelector("#metric-iterations").textContent = run.iterations;
  document.querySelector("#size-total").textContent = `${dataset.rows.toLocaleString()} total`;
  setSliderFill();

  const maxSize = Math.max(...run.clusterSizes);
  elements.clusterSizeChart.innerHTML = run.clusterSizes.map((size, index) => `
    <div class="size-row">
      <span class="size-label">${index + 1}</span>
      <div class="size-track"><div class="size-fill" style="--cluster-color:${clusterColors[index % clusterColors.length]};width:${size / maxSize * 100}%"></div></div>
      <span class="size-value">${size.toLocaleString()}</span>
    </div>
  `).join("");
}

function renderQualityChart() {
  const dataset = currentDataset();
  const values = state.artifact.method.kValues.map((k) => dataset.runs[String(k)][state.metric]);
  const width = 520;
  const height = 300;
  const left = 52;
  const right = 22;
  const top = 24;
  const bottom = 42;
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (state.metric !== "inertiaPerPoint") {
    min = Math.min(0, min - 0.03);
    max = Math.max(max + 0.03, 0.1);
  } else {
    min *= 0.9;
    max *= 1.05;
  }
  const x = (index) => left + index / (values.length - 1) * (width - left - right);
  const y = (value) => top + (max - value) / (max - min || 1) * (height - top - bottom);
  const points = values.map((value, index) => `${x(index)},${y(value)}`).join(" ");
  const area = `${left},${height - bottom} ${points} ${x(values.length - 1)},${height - bottom}`;
  const grids = [0, 0.25, 0.5, 0.75, 1].map((fraction) => {
    const yy = top + fraction * (height - top - bottom);
    const value = max - fraction * (max - min);
    return `<line class="grid-line" x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}"/><text class="chart-label" x="5" y="${yy+4}">${formatNumber(value)}</text>`;
  }).join("");
  const dots = values.map((value, index) => {
    const k = state.artifact.method.kValues[index];
    return `<circle class="chart-dot ${k === state.k ? "selected" : ""}" data-k="${k}" cx="${x(index)}" cy="${y(value)}" r="${k === state.k ? 6 : 4}"><title>k=${k}: ${formatNumber(value)}</title></circle><text class="chart-label" x="${x(index)}" y="${height-17}" text-anchor="middle">${k}</text>`;
  }).join("");
  elements.qualityChart.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${state.metric} across cluster counts">
      <defs><linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#198bff" stop-opacity=".35"/><stop offset="100%" stop-color="#198bff" stop-opacity=".02"/></linearGradient></defs>
      ${grids}
      <line class="axis-line" x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}"/>
      <polygon class="chart-area" points="${area}"/>
      <polyline class="chart-line" points="${points}"/>
      ${dots}
    </svg>`;

  const metricDescriptions = {
    silhouette: `Higher is better. ${dataset.title} reaches its strongest tested separation at k = ${dataset.bestSilhouetteK}.`,
    adjustedRand: `Higher means closer agreement with the known classes. Labels are used only to calculate this diagnostic after clustering.`,
    inertiaPerPoint: "Lower is better, but inertia always decreases as k grows. Look for diminishing returns rather than the absolute minimum.",
  };
  elements.qualityCaption.textContent = metricDescriptions[state.metric];
  elements.qualityChart.querySelectorAll(".chart-dot").forEach((dot) => {
    dot.addEventListener("click", () => {
      state.k = Number(dot.dataset.k);
      elements.kSlider.value = state.k;
      renderAll();
    });
  });
}

function convergenceSvg(frameCount) {
  const history = currentRun().history;
  const shown = history.slice(0, Math.max(1, frameCount));
  const width = 390;
  const height = 300;
  const left = 50;
  const right = 18;
  const top = 25;
  const bottom = 38;
  const min = Math.min(...history) * 0.96;
  const max = Math.max(...history) * 1.02;
  const x = (index) => left + index / Math.max(1, history.length - 1) * (width - left - right);
  const y = (value) => top + (max - value) / (max - min || 1) * (height - top - bottom);
  const points = shown.map((value, index) => `${x(index)},${y(value)}`).join(" ");
  const grids = [0, 0.5, 1].map((fraction) => {
    const yy = top + fraction * (height - top - bottom);
    return `<line class="grid-line" x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}"/>`;
  }).join("");
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Inertia by k-means iteration">
      ${grids}
      <line class="axis-line" x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}"/>
      <polyline class="chart-line" points="${points}"/>
      ${shown.map((value,index) => `<circle class="chart-dot ${index === shown.length-1 ? "selected" : ""}" cx="${x(index)}" cy="${y(value)}" r="${index === shown.length-1 ? 5 : 3}"/>`).join("")}
      <text class="chart-label" x="${left}" y="${height-14}">start</text>
      <text class="chart-label" x="${width-right}" y="${height-14}" text-anchor="end">converged</text>
      <text class="chart-value" x="${width-right}" y="18" text-anchor="end">${formatNumber(shown[shown.length-1])} inertia/object</text>
    </svg>`;
}

function renderConvergence(finalFrame = true) {
  clearInterval(state.animationTimer);
  const history = currentRun().history;
  state.convergenceFrame = finalFrame ? history.length : 1;
  elements.iterationValue.textContent = state.convergenceFrame;
  elements.convergenceChart.innerHTML = convergenceSvg(state.convergenceFrame);
}

function replayConvergence() {
  clearInterval(state.animationTimer);
  state.convergenceFrame = 1;
  elements.iterationValue.textContent = 1;
  elements.convergenceChart.innerHTML = convergenceSvg(1);
  const total = currentRun().history.length;
  state.animationTimer = setInterval(() => {
    state.convergenceFrame += 1;
    elements.iterationValue.textContent = state.convergenceFrame;
    elements.convergenceChart.innerHTML = convergenceSvg(state.convergenceFrame);
    if (state.convergenceFrame >= total) clearInterval(state.animationTimer);
  }, 170);
}

function renderComparison() {
  document.querySelector("#comparison-grid").innerHTML = Object.entries(state.artifact.datasets).map(([key, dataset]) => {
    const best = dataset.runs[String(dataset.bestSilhouetteK)];
    const known = dataset.runs[String(dataset.knownClassK)];
    return `
      <article class="comparison-card">
        <header><span>${dataset.domain}</span><strong>${dataset.features}D</strong></header>
        <h3>${dataset.title}</h3>
        <p>${dataset.shortDescription}</p>
        <div class="comparison-pair">
          <div><span>Best separation</span><strong>k = ${dataset.bestSilhouetteK}</strong><small>silhouette ${best.silhouette.toFixed(3)}</small></div>
          <div><span>Known classes</span><strong>${dataset.knownClasses}</strong><small>ARI ${known.adjustedRand.toFixed(3)}</small></div>
        </div>
        <p class="comparison-takeaway">${datasetInsights[key].takeaway}</p>
      </article>`;
  }).join("");
}

function renderSources() {
  document.querySelector("#dataset-sources").innerHTML = Object.values(state.artifact.datasets).map((dataset) => `
    <a class="dataset-source" href="${dataset.sourceUrl}" target="_blank" rel="noreferrer">
      <span>UCI: ${dataset.title}</span><span>${dataset.sourceDoi} &#8599;</span>
    </a>
  `).join("");
}

function renderAll() {
  renderDatasetTabs();
  renderMetrics();
  drawScatter();
  renderQualityChart();
  renderConvergence();
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.view === state.view));
  });
  document.querySelectorAll("[data-metric]").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.metric === state.metric));
  });
}

function bindInteractions() {
  elements.kSlider.addEventListener("input", () => {
    state.k = Number(elements.kSlider.value);
    renderAll();
  });
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => {
      state.view = button.dataset.view;
      renderAll();
    });
  });
  document.querySelectorAll("[data-metric]").forEach((button) => {
    button.addEventListener("click", () => {
      state.metric = button.dataset.metric;
      renderAll();
    });
  });
  document.querySelector("#replay-convergence").addEventListener("click", replayConvergence);
  window.addEventListener("resize", drawScatter);

  elements.canvas.addEventListener("mousemove", (event) => {
    const rect = elements.canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    let nearest = null;
    let nearestDistance = 70;
    state.renderedPoints.forEach((point) => {
      const distance = (point.x - mouseX) ** 2 + (point.y - mouseY) ** 2;
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearest = point;
      }
    });
    if (!nearest) {
      elements.tooltip.hidden = true;
      return;
    }
    elements.tooltip.hidden = false;
    elements.tooltip.style.left = `${Math.min(mouseX + 12, rect.width - 145)}px`;
    elements.tooltip.style.top = `${Math.max(8, mouseY - 48)}px`;
    elements.tooltip.innerHTML = `Cluster ${nearest.cluster + 1}<br>Known label ${nearest.label}`;
  });
  elements.canvas.addEventListener("mouseleave", () => {
    elements.tooltip.hidden = true;
  });
}

async function initialize() {
  try {
    const response = await fetch("cluster-artifact.json");
    if (!response.ok) throw new Error(`Artifact request failed: ${response.status}`);
    state.artifact = await response.json();
    state.k = currentDataset().bestSilhouetteK;
    elements.kSlider.value = state.k;
    renderComparison();
    renderSources();
    bindInteractions();
    renderAll();
  } catch (error) {
    console.error(error);
    elements.scatterTitle.textContent = "Cluster artifact unavailable";
  }
}

initialize();
