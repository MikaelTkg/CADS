# Architecture

A code-tour of the CADS package, intended for readers who want to dive
into the implementation rather than just run the demo. For the
high-level method and usage, see the [README](README.md).

## Table of contents

- [Module dependency graph](#module-dependency-graph)
- [Data flow](#data-flow)
- [Module-by-module](#module-by-module)
- [Key abstractions](#key-abstractions)
- [What lives where](#what-lives-where)
- [Design notes](#design-notes)

---

## Module dependency graph

```
                                ┌──────────┐
                                │  config  │   registries + Config
                                └────┬─────┘
                                     │
                ┌────────────────────┼────────────────────┐
                ▼                    ▼                    ▼
         ┌──────────┐         ┌──────────┐          ┌──────────┐
         │   data   │         │ conformal│          │ profiling│
         │ (caches  │         │  (APS)   │          │ (per-exp │
         │ + split) │         │          │          │ stats)   │
         └────┬─────┘         └────┬─────┘          └────┬─────┘
              │                    │                     │
              │                    │   ┌────────────────┘
              │                    │   │
              │                    ▼   ▼
              │              ┌──────────────────┐
              │              │ complementarity  │
              │              │ (Comp(A → B))    │
              │              └────────┬─────────┘
              │                       │
              │                       ▼
              │              ┌──────────────────┐
              ├──────────────►      engine      │  CADSEngine + CADSParams
              │              │  (cascade core)  │
              │              └────────┬─────────┘
              │                       │
              ▼                       ▼
       ┌──────────────┐       ┌──────────────┐
       │  baselines   │       │ optimization │  TPE search per budget
       │ (static refs)│       │              │
       └──────┬───────┘       └──────┬───────┘
              │                      │
              └──────────┬───────────┘
                         ▼
                  ┌──────────────┐
                  │  io_utils    │  JSON + CSV writers
                  └──────────────┘
```

Arrows go from importer to imported. `config` is the only module that
has no internal CADS dependencies; everything else builds on it.

---

## Data flow

A single CADS run threads four data subsets through the pipeline:

```
.npz caches            D_opt              D_cal             D_test
  (on disk)         (49 % default)    (21 % default)    (30 % default)
       │                  │                  │                  │
       │                  │                  │                  │
       ▼                  ▼                  ▼                  ▼
   PredictionCache    profiling +        conformal           final eval
   .load_split()      complementarity    recalibration       (single shot)
       │                  │                  │                  │
       │                  ▼                  ▼                  ▼
       │            ExpertProfile      class_quantiles      accuracy /
       │            Comp(A → B)        global_quantile      GFLOPs /
       │                  │                  │              exit dist
       │                  └──── feeds ───────┤
       │                                     ▼
       └────────────► CADSEngine (calibrate + run) ─► result dict
                                              │
                          Optuna TPE picks θ* on D_opt, then we
                          recalibrate on D_cal, then we evaluate
                          once on D_test.
```

**Critical invariant:** D_test is never touched during search or
calibration. Profiles, complementarity, and TPE search live on D_opt;
the conformal quantile is recalibrated on D_cal; D_test is read once.

---

## Module-by-module

### `cads/config.py`

- **Type:** Registries + run-time config.
- **Reads:** CLI args.
- **Writes:** A `Config` dataclass used by everything else.
- **Public entry points:** `DATASET_CONFIGS`, `EXPERT_GFLOPS`, `Config`.
- **Extension axis:** Add a dataset or an expert here in one line.

### `cads/data.py`

- **Type:** Cache loader + stratified split.
- **Reads:** `.npz` caches under `CADS_CACHE_PATH`.
- **Writes:** Three numpy dicts (D_opt, D_cal, D_test) + their labels.
- **Public:** `PredictionCache.load_all`, `PredictionCache.load_split`.
- **Constraint:** All caches for a given dataset must contain the same
  N samples in the same order, otherwise the split silently pairs
  predictions from different inputs.

### `cads/profiling.py`

- **Type:** Pure-function per-expert statistical profile.
- **Reads:** D_opt (probabilities + labels).
- **Writes:** `{name: ExpertProfile}` mapping with global accuracy,
  per-class accuracy, per-class confidence, efficiency.
- **Public:** `analyze_experts`.
- **Used by:** `engine` (ensemble weights), `baselines` (weighting),
  `optimization` (search).

### `cads/complementarity.py`

- **Type:** Empirical Comp(A → B) at three granularities.
- **Reads:** D_opt.
- **Writes:** Three nested dicts (global, per class, per class-pair).
- **Public:** `ComplementarityAnalyzer.analyze`, `.get_score`.
- **Fallback chain on query:** class-pair → per-class → global → 0.5.

### `cads/conformal.py`

- **Type:** Class-conditional APS predictor.
- **Reads:** Calibration probabilities + labels.
- **Writes:** A global quantile and per-class quantiles (`(n_classes,)`).
- **Public:** `ClassConformalPredictor.calibrate`, `.get_prediction_set`.
- **Reference:** Romano, Sesia, Candès (2020) — see the file header.

### `cads/engine.py`

- **Type:** The cascade itself.
- **Reads:** `Config`, `ExpertProfile` dict, `ComplementarityAnalyzer`,
  plus the appropriate data subset at call time.
- **Writes:** Per-sample prediction + cumulative GFLOPs + stats.
- **Public:** `CADSEngine.calibrate`, `CADSEngine.run`, `CADSParams`.
- **Two halves:**
  1. `calibrate(data, targets, params)` — fits the conformal predictor.
  2. `run(params, data, targets)` — walks each sample through the
     cascade and aggregates metrics.
- **Internal helpers:** `_compute_ensemble` (two-level weights),
  `_select_next_expert` (Comp × efficiency scoring),
  `_infer_one_sample` (the per-sample state machine).

### `cads/optimization.py`

- **Type:** Optuna TPE driver for one budget level.
- **Reads:** All three data subsets + the engine.
- **Writes:** A result dict (per-budget) carrying the best θ*, its
  D_opt metrics, and its D_test metrics after recalibration on D_cal.
- **Public:** `optimize_for_budget`, `process_single_budget`.
- **Search space:** Defined in `_sample_params` and documented inline.

### `cads/baselines.py`

- **Type:** Three families of static comparison points.
- **Reads:** D_test (for the metric) + D_opt profiles (for weights, to
  avoid test leakage in the comparison).
- **Writes:** `{cascade, oracle, expert_points, cumulative_cascade}`.
- **Public:** `compute_baselines`, `get_best_baseline_at_budget`.

### `cads/io_utils.py`

- **Type:** Serialisers.
- **Reads:** Everything assembled above.
- **Writes:** Timestamped JSON + flat CSV under `results/`.
- **Public:** `build_output`, `save_results`, `convert_numpy`.

---

## Key abstractions

| Object              | Lives in                | Role                                                              |
|---------------------|-------------------------|-------------------------------------------------------------------|
| `Config`            | `cads.config`           | All run-time knobs (split ratios, seed, expert list, overhead)    |
| `PredictionCache`   | `cads.data`             | Cache discovery + stratified three-way split                      |
| `ExpertProfile`     | `cads.profiling`        | Static stats per expert on D_opt                                  |
| `ComplementarityAnalyzer` | `cads.complementarity` | Three-level Comp(A → B) lookup with fallbacks               |
| `ClassConformalPredictor` | `cads.conformal`  | APS predictor with marginal + per-class quantiles                 |
| `CADSParams`        | `cads.engine`           | Hyperparameter set θ searched by TPE                              |
| `CADSEngine`        | `cads.engine`           | The cascade (calibrate + run)                                     |

All seven are stateful classes; `Config`, `ExpertProfile`, and
`CADSParams` are simple `@dataclass`es and the rest are plain
classes.

---

## What lives where

| To change…                                  | Edit…                              |
|---------------------------------------------|-------------------------------------|
| Add a dataset                               | `cads/config.py` → `DATASET_CONFIGS` |
| Add an expert                               | `cads/config.py` → `EXPERT_GFLOPS`   |
| The TPE search space                        | `cads/optimization.py` → `_sample_params` |
| Architectural constants (paper, not tuned)  | `cads/engine.py` (top-of-file consts) |
| Default split proportions                   | `cads/config.py` → `Config.val_ratio` / `cal_ratio` |
| Routing overhead value                      | `cads/config.py` → `ROUTING_OVERHEAD_GFLOPS` |
| Conformal predictor (APS variant, score)    | `cads/conformal.py`                  |
| Cache file naming patterns                  | `cads/data.py` → `_find_prediction_file` |
| Output schema (JSON keys, CSV columns)      | `cads/io_utils.py`                   |

---

## Design notes

**Why three subsets and not two?** Two would force conformal calibration
to reuse search data → optimistic coverage. Three keeps each role
distinct: D_opt drives Bayesian search, D_cal calibrates the chosen
configuration, D_test reports a single honest number.

**Why class-conditional quantiles?** Marginal quantiles inflate the
prediction set on easy classes and shrink it on hard ones. The
per-class variant tightens the set on classes the ensemble handles
well, which directly translates into earlier exits in the cascade.

**Why two-level ensemble weights?** A global term `acc^γ` rewards
strong overall experts; a per-class local term emphasises experts who
are strong on the current consensus class. The mixing ratio
(`GLOBAL_LOCAL_GLOBAL_WEIGHT` / `GLOBAL_LOCAL_LOCAL_WEIGHT`) is fixed
to the paper convention and not searched.

**Why complementarity instead of plain cost-sorted cascade?** Cheaper
isn't always smarter: a small CNN and a transformer can fail on
different sample distributions. Comp(A → B) captures that empirically
and lets the cascade prefer experts that patch the current expert's
known weaknesses.

**Why soft penalty in the TPE objective?** A hard budget filter throws
away information from over-budget trials. The
`accuracy − 10 · max(0, gflops − budget)` signal lets TPE see how
"close" infeasible trials are and converge to the feasibility boundary
from above.

**Why is the engine stateful?** Each TPE trial recalibrates the
conformal predictor in place. Parallel budget sweeps work fine because
joblib pickles a fresh copy to each worker.
