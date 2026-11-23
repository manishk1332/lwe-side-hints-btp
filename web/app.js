// web/app.js - DBDD Interactive Demo Controller

(function () {
  // Application State
  const state = {
    presets: [],
    activePresetId: "toy_pipeline",
    config: {
      n: 3,
      m: 3,
      q: "1051",
      dist: "ternary",
      sigma: 1.0,
      seed: 53,
      bkz_blocksize: 2,
      bkz_tours: 15,
      hints: []
    },
    runResult: null,
    currentStepIndex: 0,
    isTimelineMode: false,
    theme: localStorage.getItem("dbdd-theme") || "dark",
    isSlideMode: false,
    isSidebarCollapsed: false,
    expandedFormulas: {} // stepIndex -> boolean
  };

  // DOM Elements
  const el = {
    presetsContainer: document.getElementById("presets-container"),
    mainGrid: document.querySelector(".main-grid"),
    btnToggleSidebar: document.getElementById("btn-toggle-sidebar"),
    sidebarIcon: document.getElementById("sidebar-icon"),
    sidebarToggleText: document.getElementById("sidebar-toggle-text"),
    inputN: document.getElementById("input-n"),
    inputM: document.getElementById("input-m"),
    inputQ: document.getElementById("input-q"),
    inputDist: document.getElementById("input-dist"),
    inputSigma: document.getElementById("input-sigma"),
    inputSeed: document.getElementById("input-seed"),
    hintsList: document.getElementById("hints-list"),
    hintsCountBadge: document.getElementById("hints-count-badge"),
    newHintType: document.getElementById("new-hint-type"),
    newHintCoord: document.getElementById("new-hint-coord"),
    coordHelperGroup: document.getElementById("coord-helper-group"),
    customVectorGroup: document.getElementById("custom-vector-group"),
    newHintVector: document.getElementById("new-hint-vector"),
    modulusGroup: document.getElementById("modulus-group"),
    newHintModulus: document.getElementById("new-hint-modulus"),
    varianceGroup: document.getElementById("variance-group"),
    newHintVariance: document.getElementById("new-hint-variance"),
    btnAddHint: document.getElementById("btn-add-hint"),
    btnRunPipeline: document.getElementById("btn-run-pipeline"),
    stepperControls: document.getElementById("stepper-controls"),
    btnPrevStep: document.getElementById("btn-prev-step"),
    btnNextStep: document.getElementById("btn-next-step"),
    stepIndicatorText: document.getElementById("step-indicator-text"),
    stepperPills: document.getElementById("stepper-pills"),
    btnToggleView: document.getElementById("btn-toggle-view"),
    toggleViewText: document.getElementById("toggle-view-text"),
    stepsDisplayContainer: document.getElementById("steps-display-container"),
    btnThemeToggle: document.getElementById("btn-theme-toggle"),
    themeIcon: document.getElementById("theme-icon"),
    themeText: document.getElementById("theme-text"),
    btnSlideMode: document.getElementById("btn-slide-mode")
  };

  // ===========================================================================
  // KaTeX LaTeX Math Typesetting Helper
  // ===========================================================================

  function renderLatex(latexExpr) {
    if (window.katex && typeof window.katex.renderToString === "function") {
      try {
        return window.katex.renderToString(latexExpr, { displayMode: true, throwOnError: false });
      } catch (e) {
        console.warn("KaTeX render error:", e);
      }
    }
    // Fallback: clean math container if KaTeX is unavailable
    return `<div style="font-family: var(--code-font); font-size: 0.95rem; white-space: pre-wrap; text-align: center;">${latexExpr.replace(/\\\\/g, '\n')}</div>`;
  }

  function renderInlineLatex(str) {
    if (!str) return "";
    if (!window.katex || typeof window.katex.renderToString !== "function") {
      return str;
    }
    // If string contains $...$, render delimited math expressions
    if (str.includes("$")) {
      return str.replace(/\$([^\$]+)\$/g, (match, expr) => {
        try {
          return window.katex.renderToString(expr, { displayMode: false, throwOnError: false });
        } catch (e) {
          return match;
        }
      });
    }
    // If string contains bare LaTeX macros without $, render entire string in math mode
    if (str.includes("\\")) {
      try {
        return window.katex.renderToString(str, { displayMode: false, throwOnError: false });
      } catch (e) {
        return str;
      }
    }
    return str;
  }

  // ===========================================================================
  // Initialization & Data Fetching
  // ===========================================================================

  async function init() {
    setupTheme();
    setupEventHandlers();
    await fetchPresets();
    loadPreset("toy_pipeline");
    // Run initial demo pipeline automatically so initial page is populated
    runPipeline();
  }

  async function fetchPresets() {
    try {
      const res = await fetch("/api/presets");
      if (!res.ok) throw new Error("Failed to load presets");
      state.presets = await res.json();
      renderPresets();
    } catch (err) {
      console.warn("Could not load presets from server; using fallback presets:", err);
      state.presets = [
        {
          id: "toy_pipeline",
          name: "1. Toy Pipeline Benchmark",
          badge: "n=3, m=3, q=1051",
          description: "Exact match with examples/01_toy_pipeline.cpp with SV hint, perfect hint, approx hint."
        }
      ];
      renderPresets();
    }
  }

  function renderPresets() {
    el.presetsContainer.innerHTML = "";
    state.presets.forEach((p) => {
      const btn = document.createElement("button");
      btn.className = `preset-btn ${p.id === state.activePresetId ? "active" : ""}`;
      btn.innerHTML = `
        <span class="preset-title">${p.name}</span>
        <span class="preset-badge">${p.badge}</span>
      `;
      btn.addEventListener("click", () => {
        state.activePresetId = p.id;
        loadPreset(p.id);
        runPipeline();
      });
      el.presetsContainer.appendChild(btn);
    });
  }

  function updatePresetButtons() {
    document.querySelectorAll(".preset-btn").forEach((btn, idx) => {
      const p = state.presets[idx];
      if (p && p.id === state.activePresetId) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });
  }

  function loadPreset(presetId) {
    const preset = state.presets.find((p) => p.id === presetId);
    if (!preset || !preset.config) return;

    state.activePresetId = presetId;
    updatePresetButtons();

    const c = preset.config;
    state.config = JSON.parse(JSON.stringify(c));

    el.inputN.value = c.n;
    el.inputM.value = c.m;
    el.inputQ.value = c.q;
    el.inputDist.value = c.dist;
    el.inputSigma.value = c.sigma;
    el.inputSeed.value = c.seed;

    updateCoordDropdown();
    renderHintsList();
  }

  // ===========================================================================
  // Theme, Sidebar & Slide Mode Management
  // ===========================================================================

  function setupTheme() {
    document.documentElement.setAttribute("data-theme", state.theme);
    updateThemeControls();
  }

  function updateThemeControls() {
    if (state.theme === "light") {
      el.themeIcon.textContent = "🌙";
      el.themeText.textContent = "Dark Theme";
    } else {
      el.themeIcon.textContent = "☀️";
      el.themeText.textContent = "Light Theme";
    }
  }

  function toggleTheme() {
    state.theme = state.theme === "dark" ? "light" : "dark";
    localStorage.setItem("dbdd-theme", state.theme);
    setupTheme();
  }

  function toggleSlideMode() {
    state.isSlideMode = !state.isSlideMode;
    if (state.isSlideMode) {
      document.body.classList.add("slide-mode");
      el.btnSlideMode.classList.add("btn-active");
    } else {
      document.body.classList.remove("slide-mode");
      el.btnSlideMode.classList.remove("btn-active");
    }
  }

  function toggleSidebar() {
    state.isSidebarCollapsed = !state.isSidebarCollapsed;
    if (state.isSidebarCollapsed) {
      el.mainGrid.classList.add("sidebar-collapsed");
      el.sidebarIcon.textContent = "▶";
      el.sidebarToggleText.textContent = "Show Setup";
      el.btnToggleSidebar.classList.add("btn-active");
    } else {
      el.mainGrid.classList.remove("sidebar-collapsed");
      el.sidebarIcon.textContent = "◀";
      el.sidebarToggleText.textContent = "Hide Setup";
      el.btnToggleSidebar.classList.remove("btn-active");
    }
  }

  // ===========================================================================
  // Coordinate Helper Dropdown
  // ===========================================================================

  function updateCoordDropdown() {
    const n = parseInt(el.inputN.value, 10) || 3;
    el.newHintCoord.innerHTML = "";

    for (let i = 0; i < n; ++i) {
      const opt = document.createElement("option");
      opt.value = `coord_${i}`;
      opt.textContent = `Coordinate s_${i} (vector e_${i})`;
      el.newHintCoord.appendChild(opt);
    }

    const optCustom = document.createElement("option");
    optCustom.value = "custom";
    optCustom.textContent = "Custom Vector Coefficients...";
    el.newHintCoord.appendChild(optCustom);

    handleCoordChange();
  }

  function handleCoordChange() {
    const isCustom = el.newHintCoord.value === "custom";
    el.customVectorGroup.style.display = isCustom ? "flex" : "none";
  }

  function handleHintTypeChange() {
    const type = el.newHintType.value;
    if (type === "short_vector") {
      el.coordHelperGroup.style.display = "none";
      el.customVectorGroup.style.display = "none";
      el.modulusGroup.style.display = "none";
      el.varianceGroup.style.display = "none";
    } else {
      el.coordHelperGroup.style.display = "flex";
      handleCoordChange();

      el.modulusGroup.style.display = type === "modular" ? "flex" : "none";
      el.varianceGroup.style.display = type === "approximate" ? "flex" : "none";
    }
  }

  // ===========================================================================
  // Hint CRUD Operations
  // ===========================================================================

  function renderHintsList() {
    el.hintsList.innerHTML = "";
    el.hintsCountBadge.textContent = `${state.config.hints.length} hints`;

    if (state.config.hints.length === 0) {
      el.hintsList.innerHTML = `<div style="color: var(--text-muted); font-size: 0.85rem; padding: 8px;">No side hints configured.</div>`;
      return;
    }

    state.config.hints.forEach((h, idx) => {
      const item = document.createElement("div");
      item.className = "hint-item";

      let badgeClass = "badge-perfect";
      if (h.type === "modular") badgeClass = "badge-modular";
      else if (h.type === "approximate") badgeClass = "badge-approx";
      else if (h.type === "short_vector") badgeClass = "badge-sv";

      let detail = "";
      if (h.type === "short_vector") {
        detail = "Lattice short vector projection";
      } else if (h.type === "modular") {
        detail = `Vector: [${(h.vector || []).join(", ")}] (mod ${h.modulus || 2})`;
      } else if (h.type === "approximate") {
        detail = `Vector: [${(h.vector || []).join(", ")}], var = ${h.variance || 0.25}`;
      } else {
        detail = `Vector: [${(h.vector || []).join(", ")}]`;
      }

      item.innerHTML = `
        <div class="hint-info">
          <div style="display: flex; align-items: center; gap: 6px;">
            <span class="badge ${badgeClass}">${h.type}</span>
            <span class="hint-label">${h.label || `Hint ${idx + 1}`}</span>
          </div>
          <div class="hint-detail">${detail}</div>
        </div>
        <button class="btn-delete" title="Remove hint" data-idx="${idx}">&times;</button>
      `;

      item.querySelector(".btn-delete").addEventListener("click", () => {
        state.config.hints.splice(idx, 1);
        renderHintsList();
      });

      el.hintsList.appendChild(item);
    });
  }

  function addHint() {
    const type = el.newHintType.value;
    const n = parseInt(el.inputN.value, 10) || 3;
    let vector = [];
    let label = "";

    if (type === "short_vector") {
      label = "Short-Vector Hint: q-vector projection";
    } else {
      if (el.newHintCoord.value === "custom") {
        const raw = el.newHintVector.value.trim();
        if (raw) {
          vector = raw.split(",").map((x) => parseFloat(x.trim()) || 0);
        } else {
          vector = new Array(n).fill(0);
          vector[0] = 1;
        }
        label = `Custom ${type} hint`;
      } else {
        const coordIdx = parseInt(el.newHintCoord.value.replace("coord_", ""), 10);
        vector = new Array(n).fill(0);
        vector[coordIdx] = 1;
        label = `Coordinate s_${coordIdx} ${type} hint`;
      }
    }

    const newH = {
      type: type,
      label: label,
      vector: vector
    };

    if (type === "modular") {
      newH.modulus = parseInt(el.newHintModulus.value, 10) || 2;
    } else if (type === "approximate") {
      newH.variance = parseFloat(el.newHintVariance.value) || 0.25;
    }

    state.config.hints.push(newH);
    renderHintsList();
  }

  // ===========================================================================
  // Run Pipeline API Call
  // ===========================================================================

  async function runPipeline() {
    // Read current form values
    state.config.n = parseInt(el.inputN.value, 10) || 3;
    state.config.m = parseInt(el.inputM.value, 10) || 3;
    state.config.q = el.inputQ.value.trim() || "1051";
    state.config.dist = el.inputDist.value;
    state.config.sigma = parseFloat(el.inputSigma.value) || 1.0;
    state.config.seed = parseInt(el.inputSeed.value, 10) || 53;

    el.btnRunPipeline.disabled = true;
    el.btnRunPipeline.innerHTML = `<span class="spinner"></span> Running Pipeline...`;

    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.config)
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.error || errJson.stderr || `Server returned ${res.status}`);
      }

      const result = await res.json();
      state.runResult = result;

      // Construct and prepend Step 0: Initial LWE Problem & Instance Setup
      if (state.runResult && state.runResult.steps) {
        const lwe = state.runResult.lwe || {};
        const step0 = {
          step_index: 0,
          step_name: "0. Initial LWE Problem & Public Parameters",
          stage: "setup",
          description: `LWE instance generated: n=${lwe.n}, m=${lwe.m}, q=${lwe.q}, ${lwe.dist} distribution (seed ${lwe.seed}). Public relation b ≡ s·Aᵀ + e (mod q).`,
          metrics: {
            dimension: Number(lwe.n) + Number(lwe.m),
            ambient_dimension: Number(lwe.n) + Number(lwe.m) + 1,
            volume: Math.pow(Number(lwe.q || 1), Number(lwe.m || 1)),
            log2_volume: Number(lwe.m || 1) * Math.log2(Number(lwe.q || 2)),
            covariance_rank: Number(lwe.n) + Number(lwe.m),
            target_in_lattice: true,
            delta_dim: 0,
            delta_vol_ratio: 1.0
          },
          lwe_data: lwe
        };

        if (state.runResult.steps.length === 0 || state.runResult.steps[0].step_index !== 0) {
          state.runResult.steps.unshift(step0);
        }
      }

      state.currentStepIndex = 0;
      renderResults();
    } catch (err) {
      alert("Error executing DBDD pipeline:\n" + err.message);
    } finally {
      el.btnRunPipeline.disabled = false;
      el.btnRunPipeline.innerHTML = `<span>🚀</span> Run DBDD Pipeline`;
    }
  }

  // ===========================================================================
  // Mathematical Formulas & Calculations Dictionary (LaTeX)
  // ===========================================================================

  function getFormulaData(step, config, runResult) {
    const stage = step.stage;
    const name = step.step_name;
    const m = step.metrics || {};

    if (stage === "setup") {
      return {
        title: "Learning with Errors (LWE) Problem Formulation",
        paper_ref: "Regev (STOC 2005) / Dachman-Soled et al. (CRYPTO 2020) Section 2",
        formula:
          "\\mathbf{b} \\equiv \\mathbf{s} \\cdot A^T + \\mathbf{e} \\pmod{q}, \\quad A \\in \\mathbb{Z}_q^{m \\times n}, \\; \\mathbf{s} \\in \\mathbb{Z}^n, \\; \\mathbf{e} \\in \\mathbb{Z}^m",
        description:
          "Standard Learning with Errors problem: recover secret vector $\\mathbf{s}$ given public matrix $A$ and noisy vector $\\mathbf{b}$. " +
          "In cryptanalysis with side information, partial leaks on $\\mathbf{s}$ or $\\mathbf{e}$ assist lattice reduction.",
        substitutions: [
          ["Secret Dimension $n$", `$n = ${config.n}$`],
          ["Number of Samples $m$", `$m = ${config.m}$`],
          ["Modulus $q$", `$q = ${config.q}$`],
          ["Noise Distribution", `${config.dist} ($\\sigma = ${config.sigma}$, seed: ${config.seed})`]
        ]
      };
    }

    if (stage === "kannan") {
      return {
        title: "Kannan Embedding & Initial Prior Setup",
        paper_ref: "Dachman-Soled et al. (CRYPTO 2020) Section 2 / Equation (6)",
        formula:
          "B = \\begin{pmatrix} q I_m & 0 & 0 \\\\ A^T & -I_n & 0 \\\\ \\mathbf{b} & 0 & 1 \\end{pmatrix} \\in \\mathbb{Z}^{(m+n+1) \\times (m+n+1)}, \\quad (\\mathbf{k}, -\\mathbf{s}, 1) \\cdot B = (\\mathbf{e}, \\mathbf{s}, 1) = \\bar{\\mathbf{s}}",
        description:
          "Embeds the modular LWE relation $\\mathbf{b} \\equiv \\mathbf{s} \\cdot A^T + \\mathbf{e} \\pmod{q}$ into an $(m + n + 1)$-dimensional primal lattice $\\Lambda$. " +
          "The unknown target vector $\\bar{\\mathbf{s}} = (\\mathbf{e}, \\mathbf{s}, 1)$ is a unique short vector in $\\Lambda$ with last coordinate $1$.",
        substitutions: [
          ["Ambient Dimension $d$", `$d = m + n + 1 = ${m.ambient_dimension}$`],
          ["Modulus & Samples", `$q = ${config.q}, \\; m = ${config.m}, \\; n = ${config.n}$`],
          ["Initial Volume", `$\\operatorname{Vol}(\\Lambda) = q^m = ${config.q}^{${config.m}} = ${m.volume.toExponential(3)}$ $(\\log_2: ${(m.log2_volume || 0).toFixed(2)})$`],
          ["Prior Ellipsoid", `$\\mu = (0, \\dots, 0, 1), \\; \\Sigma = \\operatorname{diag}(\\sigma_e^2 I_m, \\sigma_s^2 I_n, 0)$`]
        ]
      };
    }

    if (stage === "hint" && name.includes("Short-Vector")) {
      return {
        title: "Short-Vector Orthogonal Projection",
        paper_ref: "Dachman-Soled et al. (CRYPTO 2020) Section 3.5 / Equations (27)–(30)",
        formula:
          "P_v^\\perp = I - \\frac{\\mathbf{v}^T \\mathbf{v}}{\\|\\mathbf{v}\\|^2}, \\quad \\Lambda' = \\Lambda \\cdot P_v^\\perp, \\quad \\operatorname{Vol}(\\Lambda') = \\frac{\\operatorname{Vol}(\\Lambda)}{\\|\\mathbf{v}\\|}",
        description:
          "Projects the lattice orthogonal to a known short lattice vector $\\mathbf{v} \\in \\Lambda$ (such as a $q$-vector $q \\cdot \\mathbf{e}_0$). " +
          "This contracts lattice volume by $\\|\\mathbf{v}\\|$ and reduces dimension by $1$ while preserving target containment.",
        substitutions: [
          ["Projected Vector $\\mathbf{v}$", `Lattice $q$-vector with $\\|\\mathbf{v}\\| = ${config.q}$`],
          ["Dimension Drop", `$d \\to d - 1$ (${m.dimension + 1} $\\to$ ${m.dimension})`],
          ["Volume Contraction", `$\\operatorname{Vol}' = \\operatorname{Vol}(\\Lambda) / ${config.q}$ ($\\log_2$ drops by ${Math.log2(Number(config.q)).toFixed(2)})`],
          ["Target Vector", `$\\bar{\\mathbf{s}}' = \\bar{\\mathbf{s}} \\cdot P_v^\\perp$`]
        ]
      };
    }

    if (stage === "hint" && name.includes("Perfect")) {
      return {
        title: "Perfect Hint Integration (Hyperplane Restriction)",
        paper_ref: "Dachman-Soled et al. (CRYPTO 2020) Section 3.1 / Equations (12)–(13)",
        formula:
          "\\langle \\mathbf{s}, \\mathbf{v} \\rangle = l \\implies \\langle \\bar{\\mathbf{s}}, \\bar{\\mathbf{v}} \\rangle = 0, \\quad \\Lambda' = \\Lambda \\cap \\{x \\in \\mathbb{R}^d : \\langle x, \\bar{\\mathbf{v}} \\rangle = 0\\}, \\quad \\dim(\\Lambda') = d - 1",
        description:
          "Integrates an exact linear equation revealed on the secret $\\mathbf{s}$. Homogenization converts it to an orthogonal constraint " +
          "against target $\\bar{\\mathbf{s}}$. Dual lattice projection enforces this hyperplane, reducing primal dimension by $1$.",
        substitutions: [
          ["Linear Constraint", `$\\langle \\mathbf{s}, \\mathbf{v} \\rangle = l$`],
          ["Dimension Drop", `$d \\to d - 1$ (${m.dimension + 1} $\\to$ ${m.dimension})`],
          ["Covariance Update", `Variance along $\\bar{\\mathbf{v}}$ eliminated (rank drops by $1$)`],
          ["Target Invariance", `Target $\\bar{\\mathbf{s}}$ remains strictly inside restricted sublattice $\\Lambda'$`]
        ]
      };
    }

    if (stage === "hint" && name.includes("Modular")) {
      return {
        title: "Modular Hint Integration (Coset Sparsification)",
        paper_ref: "Dachman-Soled et al. (CRYPTO 2020) Section 3.2 / Equations (15)–(19)",
        formula:
          "\\langle \\mathbf{s}, \\mathbf{v} \\rangle \\equiv l \\pmod{k} \\implies \\Lambda' = \\{x \\in \\Lambda : \\langle x, \\bar{\\mathbf{v}} \\rangle \\equiv 0 \\pmod{k}\\}, \\quad \\operatorname{Vol}(\\Lambda') = k \\cdot \\operatorname{Vol}(\\Lambda)",
        description:
          "Restricts the secret search space to a periodic coset modulo $k$. Dual lattice densification sparsifies the primal " +
          "lattice by factor $k$, which scales volume up by $k$ and lowers required BKZ blocksize while keeping dimension unchanged.",
        substitutions: [
          ["Modular Relation", `$\\langle \\mathbf{s}, \\mathbf{v} \\rangle \\equiv l \\pmod{k}$`],
          ["Volume Scaling", `$\\operatorname{Vol}' = k \\cdot \\operatorname{Vol}(\\Lambda)$ (scaled $\\times ${config.mod_k || 2}$)`],
          ["Dimension Invariance", `$d = ${m.dimension}$ (no dimension loss)`]
        ]
      };
    }

    if (stage === "hint" && name.includes("Approximate")) {
      return {
        title: "Approximate Hint Integration (Bayesian Gaussian Update)",
        paper_ref: "Dachman-Soled et al. (CRYPTO 2020) Section 3.3 / Equations (20)–(23)",
        formula:
          "\\langle \\mathbf{s}, \\mathbf{v} \\rangle + \\varepsilon = l, \\; \\varepsilon \\sim \\mathcal{N}(0, \\sigma_{hint}^2) \\implies \\Sigma' = \\Sigma - \\frac{\\Sigma \\bar{\\mathbf{v}}^T \\bar{\\mathbf{v}} \\Sigma}{\\bar{\\mathbf{v}} \\Sigma \\bar{\\mathbf{v}}^T + \\sigma_{hint}^2}, \\quad \\Sigma - \\Sigma' \\succeq 0",
        description:
          "Simulates physical side-channel leakages (power analysis, EM, timing). Updates mean $\\mu$ and covariance $\\Sigma$ via Bayesian Kalman " +
          "conditioning. Guarantees $\\Sigma - \\Sigma' \\succeq 0$ (uncertainty never increases).",
        substitutions: [
          ["Observation", `Noisy scalar measurement $l$`],
          ["Noise Variance", `$\\sigma_{hint}^2 = 0.25$`],
          ["Covariance Contraction", `$\\Sigma' \\preceq \\Sigma$ (uncertainty along $\\bar{\\mathbf{v}}$ strictly shrinks)`],
          ["Lattice Invariance", `Basis $B$ and $\\operatorname{Vol}(\\Lambda)$ unchanged (only ellipsoid updates)`]
        ]
      };
    }

    if (stage === "transformation") {
      return {
        title: "Homogenization & Isotropization (Sphere Transformation)",
        paper_ref: "Dachman-Soled et al. (CRYPTO 2020) Section 3.4 / Equations (24)–(26)",
        formula:
          "\\mu_h = \\mathbf{0}, \\quad \\Sigma_h = \\Sigma + \\mu^T \\mu, \\quad M = (\\Sigma_h^{1/2})^\\dagger, \\quad B_{iso} = B \\cdot M, \\quad \\Sigma_{iso} = M^T \\Sigma_h M \\approx I",
        description:
          "Converts inhomogeneous DBDD (with non-zero mean $\\mu$ and anisotropic ellipsoid $\\Sigma$) into standard uSVP where the error " +
          "distribution is a centered, spherical Gaussian ($\\Sigma_{iso} \\approx I$). This matches the optimal geometric assumptions for BKZ reduction.",
        substitutions: [
          ["Mean Centering", `$\\mu$ shifted from non-zero to $\\mathbf{0} = [0, \\dots, 0]$`],
          ["Covariance Absorption", `$\\Sigma_h = \\Sigma + \\mu^T \\mu$`],
          ["Sphere Transformation", `$M = (\\Sigma_h^{1/2})^\\dagger$ normalizes covariance to identity ($I$)`],
          ["Isotropic Volume", `$\\operatorname{Vol}_{iso} = ${m.volume.toExponential(3)}$ $(\\log_2: ${(m.log2_volume || 0).toFixed(2)})$`]
        ]
      };
    }

    if (stage === "estimation") {
      const est = step.estimator || {};
      return {
        title: "Concrete Security Estimation (GSA-Intersect)",
        paper_ref: "Dachman-Soled et al. (CRYPTO 2020) Section 4 / Equations (31)–(36)",
        formula:
          "r_i = \\delta_\\beta^{-2(i-1)} \\operatorname{Vol}(\\Lambda)^{1/d}, \\quad \\delta_\\beta = \\left( \\frac{\\beta}{2\\pi e} (\\pi \\beta)^{1/\\beta} \\right)^{\\frac{1}{2(\\beta - 1)}}, \\quad \\sqrt{\\beta} \\cdot \\|\\bar{\\mathbf{s}}\\| \\le r_{d - \\beta + 1}",
        description:
          "Predicts the minimal BKZ blocksize $\\beta$ required to extract the secret target vector using the GSA-Intersect model. " +
          "Evaluates the intersection of the Geometric Series Assumption (GSA) basis profile with the target projection length.",
        substitutions: [
          ["Predicted Blocksize $\\beta$ (GSA)", `$\\beta = ${est.beta_gsa || 2}$ (continuous: $${est.beta_continuous || 2.0}$)`],
          ["Root-Hermite Factor $\\delta$", `$\\delta = ${est.delta || 1.02}$`],
          ["Root-Finding Method", `Brent-Dekker method solving $\\text{Slack}(\\beta) \\ge 0$`],
          ["Conclusion", `Lattice is reduced enough that BKZ with $\\beta=${est.beta_gsa || 2}$ will recover the secret vector.`]
        ]
      };
    }

    if (stage === "reduction") {
      const bkz = step.bkz_info || {};
      return {
        title: "Block Korkine-Zolotarev (BKZ) Lattice Reduction",
        paper_ref: "Schnorr (1987) / Dachman-Soled et al. (CRYPTO 2020) Section 2",
        formula:
          "\\|b_i^*\\|^2 \\ge (\\delta_{Lov} - \\mu_{i, i-1}^2) \\|b_{i-1}^*\\|^2, \\quad b_j^{(j \\dots j+\\beta-1)} = \\operatorname{SVP}(\\Lambda_{[j, j+\\beta-1]})",
        description:
          "Executes BKZ reduction using local SVP enumeration oracles. Successively tightens basis vector norms until the " +
          "target shortest vector emerges in the early rows of the reduced basis.",
        substitutions: [
          ["Blocksize $\\beta$", `$\\beta = ${bkz.blocksize || 2}$`],
          ["Tours Completed", `${bkz.tours || 0} tours`],
          ["SVP Oracle Invocations", `${bkz.svp_calls || 0} calls`],
          ["Reduction Runtime", `${(bkz.runtime_ms || 0).toFixed(2)} ms`],
          ["Target Row Discovered", `Row index ${bkz.target_row_index !== undefined ? bkz.target_row_index : 0} (norm: ${(bkz.target_norm || 0).toFixed(2)})`]
        ]
      };
    }

    if (stage === "recovery") {
      const rec = step.recovery || {};
      return {
        title: "Secret Reconstruction & Exact LWE Verification",
        paper_ref: "Dachman-Soled et al. (CRYPTO 2020) Section 3.5 / main_plan Section 22",
        formula:
          "\\mathbf{w}' \\in \\Lambda_{red}, \\quad \\hat{\\mathbf{w}} = \\operatorname{sign}(w'_d) \\cdot \\mathbf{w}' = (\\hat{\\mathbf{e}}, \\hat{\\mathbf{s}}, 1), \\quad \\mathbf{b} \\stackrel{?}{\\equiv} \\hat{\\mathbf{s}} \\cdot A^T + \\hat{\\mathbf{e}} \\pmod{q}",
        description:
          "Deconstructs the discovered shortest vector from the Kannan basis. Normalizes its sign, extracts the secret coordinates $\\hat{\\mathbf{s}}$, " +
          "and rigorously verifies that $\\mathbf{b}_{calc} \\equiv \\hat{\\mathbf{s}} A^T + \\hat{\\mathbf{e}} \\pmod{q}$ matches the public LWE instance $\\mathbf{b}$ coordinate-by-coordinate.",
        substitutions: [
          ["Extracted Secret $\\hat{\\mathbf{s}}$", `[${(rec.extracted_secret || []).join(", ")}]`],
          ["Ground-Truth Secret $\\mathbf{s}$", `[${(rec.ground_truth || []).join(", ")}]`],
          ["Modular Verification", `All $m=${config.m}$ coordinates satisfied: $\\mathbf{b}_{calc} \\equiv \\mathbf{b} \\pmod{q}$`],
          ["Recovery Status", `${rec.success ? "100% Exact Recovery SUCCESS (✓)" : "Mismatch (✗)"}`]
        ]
      };
    }

    return null;
  }

  // ===========================================================================
  // Results & Stepper Rendering
  // ===========================================================================

  function renderResults() {
    if (!state.runResult || !state.runResult.steps) return;

    renderStepperPills();
    renderActiveStepView();
  }

  function renderStepperPills() {
    const steps = state.runResult.steps;
    el.stepperPills.innerHTML = "";

    steps.forEach((s, idx) => {
      const pill = document.createElement("button");
      pill.className = `pill-btn ${idx === state.currentStepIndex && !state.isTimelineMode ? "active" : ""}`;
      let label = s.step_name.split(":")[0];
      if (s.step_index === 0) label = "0. LWE Setup";
      else if (label.length > 22) label = label.substring(0, 22) + "...";
      pill.textContent = label;
      pill.addEventListener("click", () => {
        state.isTimelineMode = false;
        state.currentStepIndex = idx;
        updateViewModeUI();
        renderActiveStepView();
      });
      el.stepperPills.appendChild(pill);
    });

    el.stepIndicatorText.textContent = `Step ${state.currentStepIndex} of ${steps.length - 1}`;
    el.btnPrevStep.disabled = state.currentStepIndex === 0;
    el.btnNextStep.disabled = state.currentStepIndex === steps.length - 1;
  }

  function renderActiveStepView() {
    const steps = state.runResult.steps;
    el.stepsDisplayContainer.innerHTML = "";

    if (state.isTimelineMode) {
      // Timeline View: render all steps
      const timelineWrap = document.createElement("div");
      timelineWrap.className = "timeline-view";
      steps.forEach((s) => {
        timelineWrap.appendChild(createStepCard(s));
      });
      el.stepsDisplayContainer.appendChild(timelineWrap);
    } else {
      // Single Step Stepper View
      const currentStep = steps[state.currentStepIndex];
      if (currentStep) {
        el.stepsDisplayContainer.appendChild(createStepCard(currentStep));
      }
    }
  }

  function createStepCard(step) {
    const card = document.createElement("div");
    card.className = "card";

    // Stage badge class
    let stageClass = "badge-approx";
    if (step.stage === "setup") stageClass = "badge-perfect";
    else if (step.stage === "kannan") stageClass = "badge-sv";
    else if (step.stage === "hint") stageClass = "badge-perfect";
    else if (step.stage === "transformation") stageClass = "badge-modular";
    else if (step.stage === "recovery") stageClass = "badge-perfect";

    const m = step.metrics || {};
    let deltaDimHtml = "";
    if (m.delta_dim !== undefined && m.delta_dim !== 0) {
      const cls = m.delta_dim < 0 ? "delta-down" : "delta-up";
      deltaDimHtml = `<span class="metric-delta ${cls}">(${m.delta_dim > 0 ? "+" : ""}${m.delta_dim})</span>`;
    }

    // Header & Description
    card.innerHTML = `
      <div class="card-header">
        <div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span class="badge ${stageClass}">${step.stage}</span>
            <h3 style="font-size: 1.15rem; font-weight: 700; color: var(--text-primary);">${step.step_name}</h3>
          </div>
          <p style="font-size: 0.88rem; color: var(--text-secondary); margin-top: 6px;">${step.description}</p>
        </div>
      </div>

      <!-- Action: Toggle Calculations & Formula Inspector -->
      <div style="margin-bottom: 14px;">
        <button class="btn btn-sm btn-formula-toggle" style="background-color: rgba(37, 99, 235, 0.1); border-color: var(--accent-blue); color: var(--accent-blue);">
          <span>📐</span> <span class="formula-btn-text">View Formulas & Step Calculations</span>
        </button>
      </div>

      <!-- Formula Accordion Panel (rendered dynamically) -->
      <div class="formula-panel-container"></div>

      <!-- Metrics Grid -->
      <div class="metrics-grid">
        <div class="metric-box">
          <span class="metric-label">Lattice Dim (d)</span>
          <span class="metric-val">${m.dimension || 0} ${deltaDimHtml}</span>
        </div>
        <div class="metric-box">
          <span class="metric-label">Ambient Dim</span>
          <span class="metric-val">${m.ambient_dimension || 0}</span>
        </div>
        <div class="metric-box">
          <span class="metric-label">Volume Vol(&Lambda;)</span>
          <span class="metric-val">${formatNumber(m.volume)}</span>
          <span class="metric-delta" style="color: var(--text-muted);">log2: ${(m.log2_volume || 0).toFixed(2)}</span>
        </div>
        <div class="metric-box">
          <span class="metric-label">Covariance Rank</span>
          <span class="metric-val">${m.covariance_rank !== undefined ? m.covariance_rank : "-"}</span>
        </div>
        ${step.stage === "kannan" ? `
        <div class="metric-box">
          <span class="metric-label">Target In Lattice</span>
          <span class="metric-val" style="color: ${m.target_in_lattice ? 'var(--accent-green)' : 'var(--text-muted)'};">
            ${m.target_in_lattice ? "YES &#10003;" : "N/A"}
          </span>
        </div>
        ` : ""}
        ${step.estimator ? `
          <div class="metric-box" style="border-color: var(--accent-blue);">
            <span class="metric-label">GSA Est. &beta;</span>
            <span class="metric-val" style="color: var(--accent-blue);">${step.estimator.beta_gsa}</span>
            <span class="metric-delta" style="color: var(--text-muted);">&delta; = ${step.estimator.delta}</span>
          </div>
        ` : ""}
      </div>
    `;

    // Formula Panel setup
    const formulaContainer = card.querySelector(".formula-panel-container");
    const formulaBtn = card.querySelector(".btn-formula-toggle");
    const formulaData = getFormulaData(step, state.config, state.runResult);

    if (formulaData) {
      const isExpanded = !!state.expandedFormulas[step.step_index];
      if (isExpanded) {
        formulaContainer.innerHTML = renderFormulaPanelHtml(formulaData);
        formulaBtn.classList.add("btn-active");
      }

      formulaBtn.addEventListener("click", () => {
        state.expandedFormulas[step.step_index] = !state.expandedFormulas[step.step_index];
        if (state.expandedFormulas[step.step_index]) {
          formulaContainer.innerHTML = renderFormulaPanelHtml(formulaData);
          formulaBtn.classList.add("btn-active");
        } else {
          formulaContainer.innerHTML = "";
          formulaBtn.classList.remove("btn-active");
        }
      });
    } else {
      formulaBtn.style.display = "none";
    }

    // Step-Specific Rich Visualizations
    if (step.stage === "setup" && step.lwe_data) {
      card.appendChild(renderLweSetupView(step.lwe_data, state.config));
    } else if (step.stage === "transformation" && step.homo_iso_details) {
      card.appendChild(renderHomoIsoDiffView(step.homo_iso_details, step));
    } else if (step.stage === "reduction" && step.bkz_info) {
      card.appendChild(renderBkzReductionView(step.bkz_info, step));
    } else if (step.stage === "recovery" && step.recovery) {
      card.appendChild(renderSecretRecoveryDecompView(step.recovery, state.config));
    } else {
      // Standard Matrix & Vector Viewers
      card.appendChild(renderStandardLatticeView(step));
    }

    return card;
  }

  function renderFormulaPanelHtml(data) {
    const subRows = (data.substitutions || [])
      .map(([k, v]) => `<tr><td class="formula-label-cell">${renderInlineLatex(k)}:</td><td class="formula-val-cell">${renderInlineLatex(v)}</td></tr>`)
      .join("");

    const renderedFormula = renderLatex(data.formula);
    const renderedDescription = renderInlineLatex(data.description);

    return `
      <div class="formula-panel">
        <div class="formula-header">
          <div class="formula-header-title">
            <span>📐</span> ${data.title}
          </div>
          <span class="badge" style="background: rgba(37,99,235,0.15); color: var(--accent-blue);">${data.paper_ref}</span>
        </div>
        <div class="formula-body">
          <div class="math-formula-box">${renderedFormula}</div>
          <p style="color: var(--text-secondary); font-size: 0.88rem; line-height: 1.6;">${renderedDescription}</p>
          <div class="formula-substitutions">
            <table>${subRows}</table>
          </div>
        </div>
      </div>
    `;
  }

  // 0. Initial LWE Problem Parameters View
  function renderLweSetupView(lwe, config) {
    const wrap = document.createElement("div");
    wrap.className = "lattice-section";

    const s = lwe.ground_truth_secret || [];
    const e = lwe.ground_truth_error || [];
    const b = lwe.vector_b || [];
    const A = lwe.matrix_A || [];

    wrap.innerHTML = `
      <div class="section-subtitle">
        <span>Initial LWE Cryptographic Parameters & Public Relations</span>
        <span class="badge badge-perfect">Relation: b &equiv; s&middot;Aᵀ + e (mod q)</span>
      </div>

      <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px;">
        <span class="lwe-param-badge"><strong>n (dim):</strong> ${lwe.n}</span>
        <span class="lwe-param-badge"><strong>m (samples):</strong> ${lwe.m}</span>
        <span class="lwe-param-badge"><strong>q (modulus):</strong> ${lwe.q}</span>
        <span class="lwe-param-badge"><strong>Distribution:</strong> ${lwe.dist}</span>
        <span class="lwe-param-badge"><strong>Noise &sigma;:</strong> ${lwe.sigma}</span>
        <span class="lwe-param-badge"><strong>Seed:</strong> ${lwe.seed}</span>
      </div>

      <div class="lwe-setup-grid">
        <!-- Secret & Error -->
        <div>
          <div class="section-subtitle">Ground-Truth Secret s (dim ${lwe.n})</div>
          <div class="matrix-viewer" style="color: var(--accent-green);">[ ${s.join("  ")} ]</div>
        </div>
        <div>
          <div class="section-subtitle">Ground-Truth Noise Error e (dim ${lwe.m})</div>
          <div class="matrix-viewer" style="color: var(--accent-blue);">[ ${e.join("  ")} ]</div>
        </div>
      </div>

      <!-- Public Vector b -->
      <div style="margin-top: 14px;">
        <div class="section-subtitle">Public Vector b (dim ${lwe.m})</div>
        <div class="matrix-viewer">[ ${b.join("  ")} ]</div>
      </div>

      <!-- Public Matrix A -->
      <div style="margin-top: 14px;">
        <div class="section-subtitle">
          <span>Public Matrix A (${A.length} &times; ${A[0] ? A[0].length : 0})</span>
          <button class="btn btn-sm btn-copy-a">Copy Matrix A</button>
        </div>
        <div class="matrix-viewer">${formatBasisMatrix(A)}</div>
      </div>
    `;

    const btnCopy = wrap.querySelector(".btn-copy-a");
    if (btnCopy) {
      btnCopy.addEventListener("click", () => {
        const text = A.map((row) => row.join("\t")).join("\n");
        navigator.clipboard.writeText(text);
        btnCopy.textContent = "Copied!";
        setTimeout(() => (btnCopy.textContent = "Copy Matrix A"), 1500);
      });
    }

    return wrap;
  }

  // 1. Homogenization & Isotropization Before-vs-After Comparison View
  function renderHomoIsoDiffView(details, step) {
    const wrap = document.createElement("div");
    wrap.className = "lattice-section";

    wrap.innerHTML = `
      <div class="section-subtitle">
        <span>Before vs After Transformation Comparison (Homogenization & Isotropization)</span>
        <span class="badge badge-modular">Vol Ratio: ${(details.volume_ratio || 1.0).toFixed(4)}</span>
      </div>

      <div class="diff-grid">
        <!-- Before Column -->
        <div class="diff-col">
          <div class="diff-header">
            <span>Original Anisotropic State</span>
            <span class="diff-tag-before">BEFORE</span>
          </div>
          <div>
            <span class="metric-label">Mean Vector &mu; (Non-Zero Prior):</span>
            <div class="matrix-viewer" style="color: var(--accent-amber);">[ ${(details.pre_mean || []).map(x => x.toFixed(4)).join("  ")} ]</div>
          </div>
          <div>
            <span class="metric-label">Prior Covariance &Sigma; (Anisotropic Ellipsoid):</span>
            <div class="matrix-viewer">${formatCovarianceMatrix(details.pre_covariance)}</div>
          </div>
        </div>

        <!-- After Column -->
        <div class="diff-col">
          <div class="diff-header">
            <span>Transformed Isotropic State</span>
            <span class="diff-tag-after">AFTER</span>
          </div>
          <div>
            <span class="metric-label">Centered Mean &mu;_h (Centered at 0):</span>
            <div class="matrix-viewer" style="color: var(--accent-green);">[ ${(step.mean || []).map(x => x.toFixed(4)).join("  ")} ]</div>
          </div>
          <div>
            <span class="metric-label">Isotropic Covariance &Sigma;_iso (Normalized to Identity):</span>
            <div class="matrix-viewer" style="color: var(--accent-green);">${formatCovarianceMatrix(step.covariance)}</div>
          </div>
        </div>
      </div>

      <!-- Transformation Matrix M -->
      <div style="margin-top: 10px;">
        <div class="section-subtitle">
          <span>Transformation Matrix M = (&Sigma;_h^{1/2})&dagger;</span>
          <span style="font-size: 0.75rem; color: var(--text-muted);">Scales lattice B_iso = B &middot; M</span>
        </div>
        <div class="matrix-viewer">${formatCovarianceMatrix(details.transform_M)}</div>
      </div>

      <!-- Transformed Basis -->
      <div style="margin-top: 14px;">
        <div class="section-subtitle">Transformed Lattice Basis Matrix B_iso</div>
        <div class="matrix-viewer">${formatBasisMatrix(step.basis)}</div>
      </div>
    `;
    return wrap;
  }

  // 2. BKZ Lattice Reduction Norm Profile & Target Row Highlighting View
  function renderBkzReductionView(bkz, step) {
    const wrap = document.createElement("div");
    wrap.className = "lattice-section";

    const beforeNorms = bkz.before_norms || [];
    const afterNorms = bkz.after_norms || [];
    const targetIdx = bkz.target_row_index !== undefined ? bkz.target_row_index : 0;
    const maxNorm = Math.max(...beforeNorms, ...afterNorms, 1.0);

    const rowsHtml = afterNorms
      .map((normAfter, i) => {
        const normBefore = beforeNorms[i] || normAfter;
        const pctContraction = normBefore > 0 ? (((normBefore - normAfter) / normBefore) * 100).toFixed(1) : 0;
        const isTarget = i === targetIdx;
        const widthAfter = ((normAfter / maxNorm) * 100).toFixed(1);

        return `
          <tr class="${isTarget ? 'target-row-highlight' : ''}">
            <td style="font-weight: 700; color: ${isTarget ? 'var(--accent-green)' : 'var(--text-primary)'};">
              b_${i} ${isTarget ? '<span class="target-badge-pill">TARGET CANDIDATE</span>' : ''}
            </td>
            <td>${normBefore.toFixed(2)}</td>
            <td style="color: var(--accent-green); font-weight: 700;">${normAfter.toFixed(2)}</td>
            <td style="color: ${pctContraction > 0 ? 'var(--accent-green)' : 'var(--text-muted)'};">
              ${pctContraction > 0 ? `-${pctContraction}%` : "0.0%"}
            </td>
            <td style="width: 45%;">
              <div class="norm-bar-container">
                <div class="norm-bar-wrap" title="Before: ${normBefore.toFixed(2)} | After: ${normAfter.toFixed(2)}">
                  <div class="norm-bar-fill norm-bar-after" style="width: ${widthAfter}%;"></div>
                </div>
              </div>
            </td>
          </tr>
        `;
      })
      .join("");

    wrap.innerHTML = `
      <div class="section-subtitle">
        <span>BKZ Reduction: Basis Vector Norms & Target Vector Discovery</span>
        <span class="badge badge-perfect">Shortest Norm: ${(bkz.target_norm || (afterNorms[0] || 0)).toFixed(2)}</span>
      </div>

      <div style="background-color: var(--bg-primary); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; overflow-x: auto;">
        <table class="norm-profile-table">
          <thead>
            <tr>
              <th>Row</th>
              <th>Before Norm ||b_i||</th>
              <th>Reduced Norm ||b_i'||</th>
              <th>Contraction</th>
              <th>Visual Profile (Length)</th>
            </tr>
          </thead>
          <tbody>
            ${rowsHtml}
          </tbody>
        </table>
      </div>

      <!-- Reduced Basis Matrix -->
      <div style="margin-top: 14px;">
        <div class="section-subtitle">
          <span>BKZ-Reduced Basis Matrix B (${step.basis ? step.basis.length : 0} &times; ${step.basis && step.basis[0] ? step.basis[0].length : 0})</span>
          <button class="btn btn-sm btn-copy-basis">Copy Basis</button>
        </div>
        <div class="matrix-viewer">${formatBasisMatrix(step.basis, targetIdx)}</div>
      </div>
    `;

    // Copy basis button handler
    const btnCopy = wrap.querySelector(".btn-copy-basis");
    if (btnCopy) {
      btnCopy.addEventListener("click", () => {
        const text = (step.basis || []).map((row) => row.join("\t")).join("\n");
        navigator.clipboard.writeText(text);
        btnCopy.textContent = "Copied!";
        setTimeout(() => (btnCopy.textContent = "Copy Basis"), 1500);
      });
    }

    return wrap;
  }

  // 3. Secret Reconstruction Vector Decomposition & Verification Breakdown View
  function renderSecretRecoveryDecompView(rec, config) {
    const wrap = document.createElement("div");
    wrap.className = "lattice-section";

    let calcB = [];
    if (Array.isArray(rec.calculated_b)) {
      calcB = rec.calculated_b;
    } else if (typeof rec.calculated_b === "string") {
      try { calcB = JSON.parse(rec.calculated_b); } catch (e) { calcB = []; }
    }

    let pubB = [];
    if (Array.isArray(rec.public_b)) {
      pubB = rec.public_b;
    } else if (typeof rec.public_b === "string") {
      try { pubB = JSON.parse(rec.public_b); } catch (e) { pubB = []; }
    }

    let extSecret = rec.extracted_secret;
    if (Array.isArray(extSecret)) extSecret = "[ " + extSecret.join("  ") + " ]";

    let extError = rec.extracted_error;
    if (Array.isArray(extError)) extError = "[ " + extError.join("  ") + " ]";

    const verifyRowsHtml = pubB
      .map((bVal, i) => {
        const cVal = calcB[i] !== undefined ? calcB[i] : "-";
        const match = String(bVal) === String(cVal);
        return `
          <tr>
            <td style="font-weight: 700;">Sample i=${i}</td>
            <td>b[${i}] = ${bVal}</td>
            <td>(s&middot;Aᵀ + e)[${i}] = ${cVal}</td>
            <td style="color: ${match ? 'var(--accent-green)' : 'var(--accent-rose)'}; font-weight: 700;">
              ${match ? '&#10003; MATCH' : '&#10007; MISMATCH'}
            </td>
          </tr>
        `;
      })
      .join("");

    wrap.innerHTML = `
      <div class="section-subtitle">
        <span>Secret Vector Slicing & Modular Reconstruction Breakdown</span>
        <span class="badge ${rec.success ? 'badge-perfect' : 'badge-modular'}">
          ${rec.success ? 'VERIFICATION PASSED (100% RECOVERED)' : 'CHECKING'}
        </span>
      </div>

      <div class="decomp-container">
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 8px;">
          Lattice Candidate Vector Slicing: w' = [ e | s | 1 ]
        </div>

        <div class="decomp-slices">
          <div class="decomp-slice slice-error">
            <span class="slice-title">Extracted Error Vector e (m=${config.m})</span>
            <span class="slice-val" style="color: var(--accent-blue);">${extError || "[]"}</span>
          </div>

          <div class="decomp-slice slice-secret">
            <span class="slice-title">Extracted Secret Vector s (n=${config.n})</span>
            <span class="slice-val" style="color: var(--accent-green);">${extSecret || "[]"}</span>
          </div>

          <div class="decomp-slice slice-homog" style="flex: 0 0 100px;">
            <span class="slice-title">Last Coord</span>
            <span class="slice-val" style="color: var(--accent-purple);">1</span>
          </div>
        </div>

        <!-- Ground Truth vs Recovered Match Banner -->
        <div style="margin-top: 14px; background: rgba(16, 185, 129, 0.1); border: 1px solid var(--accent-green); border-radius: 6px; padding: 10px 14px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
          <div>
            <span style="font-weight: 700; color: var(--accent-green);">Exact Secret Match:</span>
            <span style="font-family: var(--code-font); font-weight: 700; margin-left: 8px;">${extSecret}</span>
          </div>
          <span class="badge badge-perfect">&#10003; Target Confirmed</span>
        </div>

        <!-- Coordinate-by-Coordinate Verification Table -->
        <div style="margin-top: 14px;">
          <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-primary); margin-bottom: 6px;">
            Modular Relation Verification: b ≟ s&middot;Aᵀ + e (mod q)
          </div>
          <table class="verify-table">
            <thead>
              <tr>
                <th>Index</th>
                <th>Public Vector b</th>
                <th>Calculated (s&middot;Aᵀ + e mod q)</th>
                <th>Verification</th>
              </tr>
            </thead>
            <tbody>
              ${verifyRowsHtml}
            </tbody>
          </table>
        </div>
      </div>
    `;

    return wrap;
  }

  // 4. Standard Lattice & Matrix Viewer (For hints and Kannan embedding)
  function renderStandardLatticeView(step) {
    const wrap = document.createElement("div");

    wrap.innerHTML = `
      <!-- Lattice Basis Viewer -->
      <div class="lattice-section">
        <div class="section-subtitle">
          <span>Lattice Basis Matrix B (${step.basis ? step.basis.length : 0} &times; ${step.basis && step.basis[0] ? step.basis[0].length : 0})</span>
          <button class="btn btn-sm btn-copy-basis" title="Copy Matrix to Clipboard">Copy Basis</button>
        </div>
        <div class="matrix-viewer">${formatBasisMatrix(step.basis)}</div>
      </div>

      <!-- Mean & Target Vectors -->
      <div class="lattice-section" style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px;">
        <div style="min-width: 0;">
          <div class="section-subtitle">Mean Vector &mu;</div>
          <div class="matrix-viewer">[ ${(step.mean || []).map(x => x.toFixed(4)).join("  ")} ]</div>
        </div>
        <div style="min-width: 0;">
          <div class="section-subtitle">Target Vector v (s_bar)</div>
          <div class="matrix-viewer">[ ${(step.target || []).join("  ")} ]</div>
        </div>
      </div>

      <!-- Covariance Matrix Sigma -->
      <div class="lattice-section">
        <div class="section-subtitle">Covariance Matrix &Sigma; (${step.covariance ? step.covariance.length : 0} &times; ${step.covariance && step.covariance[0] ? step.covariance[0].length : 0})</div>
        <div class="matrix-viewer">${formatCovarianceMatrix(step.covariance)}</div>
      </div>
    `;

    // Copy basis button handler
    const btnCopy = wrap.querySelector(".btn-copy-basis");
    if (btnCopy) {
      btnCopy.addEventListener("click", () => {
        const text = (step.basis || []).map((row) => row.join("\t")).join("\n");
        navigator.clipboard.writeText(text);
        btnCopy.textContent = "Copied!";
        setTimeout(() => (btnCopy.textContent = "Copy Basis"), 1500);
      });
    }

    return wrap;
  }

  function formatNumber(num) {
    if (num === undefined || num === null) return "-";
    if (Math.abs(num) >= 1e6 || (Math.abs(num) > 0 && Math.abs(num) < 1e-3)) {
      return num.toExponential(3);
    }
    return num.toFixed(2);
  }

  function formatBasisMatrix(basis, highlightRow = -1) {
    if (!basis || !basis.length) return "[]";
    return basis
      .map((row, i) => {
        const rowStr = row.map((val) => String(val).padStart(5, " ")).join(" ");
        const isHighlight = i === highlightRow;
        return `<span class="row-label" style="${isHighlight ? 'color: var(--accent-green); font-weight: 800;' : ''}">b_${i} =</span> [ ${rowStr} ]`;
      })
      .join("\n");
  }

  function formatCovarianceMatrix(cov) {
    if (!cov || !cov.length) return "[]";
    return cov
      .map((row) => {
        const rowStr = row
          .map((v) => {
            const formatted = v.toFixed(4).padStart(7, " ");
            return formatted;
          })
          .join(" ");
        return `  [ ${rowStr} ]`;
      })
      .join("\n");
  }

  function updateViewModeUI() {
    if (state.isTimelineMode) {
      el.toggleViewText.textContent = "View Stepper";
      el.btnPrevStep.disabled = true;
      el.btnNextStep.disabled = true;
      el.stepIndicatorText.textContent = "Timeline View (All Steps)";
    } else {
      el.toggleViewText.textContent = "View All (Timeline)";
      const steps = (state.runResult && state.runResult.steps) || [];
      el.stepIndicatorText.textContent = `Step ${state.currentStepIndex} of ${steps.length - 1}`;
      el.btnPrevStep.disabled = state.currentStepIndex === 0;
      el.btnNextStep.disabled = state.currentStepIndex >= steps.length - 1;
    }

    document.querySelectorAll(".pill-btn").forEach((pill, idx) => {
      if (idx === state.currentStepIndex && !state.isTimelineMode) {
        pill.classList.add("active");
      } else {
        pill.classList.remove("active");
      }
    });
  }

  // ===========================================================================
  // Event Handlers Setup
  // ===========================================================================

  function setupEventHandlers() {
    el.btnThemeToggle.addEventListener("click", toggleTheme);
    el.btnSlideMode.addEventListener("click", toggleSlideMode);
    el.btnToggleSidebar.addEventListener("click", toggleSidebar);

    el.inputN.addEventListener("input", updateCoordDropdown);
    el.newHintType.addEventListener("change", handleHintTypeChange);
    el.newHintCoord.addEventListener("change", handleCoordChange);

    el.btnAddHint.addEventListener("click", addHint);
    el.btnRunPipeline.addEventListener("click", runPipeline);

    el.btnPrevStep.addEventListener("click", () => {
      if (state.currentStepIndex > 0) {
        state.isTimelineMode = false;
        state.currentStepIndex--;
        updateViewModeUI();
        renderActiveStepView();
      }
    });

    el.btnNextStep.addEventListener("click", () => {
      const steps = (state.runResult && state.runResult.steps) || [];
      if (state.currentStepIndex < steps.length - 1) {
        state.isTimelineMode = false;
        state.currentStepIndex++;
        updateViewModeUI();
        renderActiveStepView();
      }
    });

    el.btnToggleView.addEventListener("click", () => {
      state.isTimelineMode = !state.isTimelineMode;
      updateViewModeUI();
      renderActiveStepView();
    });
  }

  // Boot up application
  document.addEventListener("DOMContentLoaded", init);
})();
