"""
cads.io_utils
=============

Result serialisation.

Each CADS run produces two files in the results directory:

  * a JSON dump containing the full configuration, expert profiles
    (on D_opt), baselines (on D_test), the per-budget curve, and the
    total wall-clock time;
  * a flat CSV table with one row per evaluated point, suitable for
    quick spreadsheet inspection or plotting.

The JSON schema is documented inline below.
"""
from __future__ import annotations

import csv
import json
from datetime  import datetime
from pathlib   import Path
from typing    import Any, Dict, List

import numpy as np

from .baselines import compute_baselines  # noqa: F401  (re-export aid)
from .config    import Config, ROUTING_OVERHEAD_GFLOPS
from .profiling import ExpertProfile


# =============================================================================
# JSON SERIALISATION HELPERS
# =============================================================================

def convert_numpy(obj: Any) -> Any:
    """Recursively convert NumPy types to plain Python so ``json``
    can serialise them.
    """
    if isinstance(obj, np.ndarray):           return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):     return bool(obj)
    if isinstance(obj, np.integer):           return int(obj)
    if isinstance(obj, np.floating):          return float(obj)
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj


# =============================================================================
# OUTPUT ASSEMBLY
# =============================================================================

def build_output(
    args_dataset:   str,
    config:         Config,
    budget_levels:  List[float],
    profiles_opt:   Dict[str, ExpertProfile],
    baselines:      Dict[str, Any],
    results:        List[Dict[str, Any]],
    n_opt:          int,
    n_cal:          int,
    n_test:         int,
    elapsed_minutes: float,
) -> Dict[str, Any]:
    """Assemble the JSON-serialisable output dictionary for one run.

    Parameters
    ----------
    args_dataset
        Dataset key (as passed on the command line). Saved as the
        ``dataset`` field of the output.
    config
        Run configuration. The exported config block includes
        ``n_classes``, the expert list and their GFLOPs, the split
        proportions, the seed, and whether routing overhead was charged.
    budget_levels
        All FLOPs budgets that were evaluated, in giga.
    profiles_opt
        Per-expert profiles from :func:`cads.profiling.analyze_experts`,
        computed on D_opt. Only ``accuracy``, ``gflops`` and
        ``efficiency`` are exported.
    baselines
        Baseline dictionary from :func:`cads.baselines.compute_baselines`.
    results
        List of per-budget result dicts from
        :func:`cads.optimization.process_single_budget`.
    n_opt, n_cal, n_test
        Sizes of the three split subsets (saved for traceability).
    elapsed_minutes
        Total wall-clock optimisation time, in minutes.
    """
    return {
        'version':   'CADS_1.0',
        'dataset':   args_dataset,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_classes':                  config.n_classes,
            'n_experts':                  len(config.experts),
            'experts':                    [e['name'] for e in config.experts],
            'expert_gflops':              {e['name']: e['gflops']
                                           for e in config.experts},
            'n_trials_per_budget':        config.n_trials,
            'budget_levels':              budget_levels,
            'routing_overhead_gflops':    (
                ROUTING_OVERHEAD_GFLOPS if config.include_routing_overhead else 0
            ),
            'val_ratio':                  config.val_ratio,
            'cal_ratio':                  config.cal_ratio,
            'seed':                       config.seed,
            'n_opt_samples':              n_opt,
            'n_cal_samples':              n_cal,
            'n_test_samples':             n_test,
        },
        'expert_profiles_opt': {
            name: {
                'accuracy':   p.accuracy,
                'gflops':     p.gflops,
                'efficiency': p.efficiency,
            }
            for name, p in profiles_opt.items()
        },
        'baselines': {
            'full_ensemble': {
                'accuracy': baselines['cascade']['accuracy'],
                'gflops':   baselines['cascade']['gflops'],
            },
            'oracle_accuracy':    baselines['oracle']['accuracy'],
            'individual_experts': baselines['expert_points'],
            'cumulative_cascade': baselines['cumulative_cascade'],
        },
        'budget_curve': [
            {
                'budget':              r['budget'],
                'cads_accuracy':       r['accuracy_test'],
                'cads_gflops':         r['gflops_test'],
                'cads_avg_experts':    r['avg_experts'],
                'baseline_name':       r['baseline_name'],
                'baseline_accuracy':   r['baseline_acc'],
                'baseline_gflops':     r['baseline_gflops'],
                'gain_over_baseline':  r['gain'],
                'feasible':            r['feasible'],
                'test_feasible':       r.get('test_feasible', True),
                'exit_distribution':   r.get('detailed_stats', {}).get(
                                          'exit_distribution', {}),
                'experts_usage':       r.get('detailed_stats', {}).get(
                                          'experts_usage', {}),
            }
            for r in sorted(results, key=lambda x: x['budget'])
        ],
        'optimization_time_minutes': elapsed_minutes,
    }


# =============================================================================
# FILE WRITERS
# =============================================================================

def save_results(
    output:       Dict[str, Any],
    results:      List[Dict[str, Any]],
    baselines:    Dict[str, Any],
    args_dataset: str,
    results_dir:  Path,
) -> tuple[Path, Path]:
    """Write the JSON and CSV files. Returns their paths.

    Parameters
    ----------
    output
        Dictionary built by :func:`build_output`; serialised verbatim
        as the JSON output.
    results
        Per-budget result dicts; iterated to build the flat CSV row
        per evaluated point.
    baselines
        Baseline dictionary from :func:`cads.baselines.compute_baselines`;
        also flattened into CSV rows for individual experts and
        cumulative cascades.
    args_dataset
        Dataset key used to name the output files.
    results_dir
        Destination directory. Created if it does not exist.

    Returns
    -------
    json_path, csv_path
        Paths to the two files written, both timestamped.
    """
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_path = results_dir / f"cads_{args_dataset}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(convert_numpy(output), f, indent=2)

    csv_path = results_dir / f"cads_{args_dataset}_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'method', 'budget', 'accuracy', 'gflops', 'avg_experts',
            'baseline_acc', 'gain', 'feasible',
        ])
        for r in sorted(results, key=lambda x: x['budget']):
            feasible = r['feasible'] and r.get('test_feasible', True)
            w.writerow([
                'CADS', r['budget'], r['accuracy_test'], r['gflops_test'],
                r['avg_experts'], r['baseline_acc'], r['gain'], feasible,
            ])
        for exp in baselines['expert_points']:
            w.writerow([
                f"expert_{exp['name']}", exp['gflops'], exp['accuracy'],
                exp['gflops'], 1, exp['accuracy'], 0, True,
            ])
        for cc in baselines['cumulative_cascade']:
            w.writerow([
                f"cumulative_{cc['n_experts']}", cc['gflops'], cc['accuracy'],
                cc['gflops'], cc['n_experts'], cc['accuracy'], 0, True,
            ])

    return json_path, csv_path
