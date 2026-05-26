#!/usr/bin/env python3
"""
run_cads.py
===========

Command-line entry point: runs CADS on a registered dataset over a
grid of FLOPs budgets and writes JSON + CSV outputs.

Examples
--------
::

    python scripts/run_cads.py --dataset pathmnist --n_trials 200
    python scripts/run_cads.py --dataset cifar100 --parallel
    python scripts/run_cads.py --dataset bloodmnist \\
        --experts mobilenet_tiny,ghostnet,eva02_tiny,maxvit_tiny \\
        --budgets 0.1,0.5,1.0,5.0,10.0 \\
        --n_trials 300

The script orchestrates these stages:

  1. Parse CLI args and build a :class:`cads.Config`.
  2. Load the prediction caches, build the stratified split.
  3. Profile experts and complementarity on D_opt.
  4. Compute baselines on D_test (weights from D_opt).
  5. For each budget level, run Optuna TPE on D_opt, recalibrate on
     D_cal, evaluate on D_test.
  6. Serialise JSON + CSV.
  7. Print a summary block.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing  import Any, Dict, List

import numpy as np

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Allow `python scripts/run_cads.py ...` to find the cads package
# from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cads import (
    Config,
    DATASET_CONFIGS,
    EXPERT_GFLOPS,
    PredictionCache,
    analyze_experts,
    ComplementarityAnalyzer,
    CADSEngine,
    compute_baselines,
    process_single_budget,
    build_output,
    save_results,
)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='CADS — Conformal Adaptive Decision System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_cads.py --dataset pathmnist --n_trials 200
    python scripts/run_cads.py --dataset cifar100 --parallel
    python scripts/run_cads.py --dataset bloodmnist \\
        --experts mobilenet_tiny,ghostnet,eva02_tiny,maxvit_tiny \\
        --budgets 0.1,0.5,1.0,5.0,10.0
        """,
    )
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset key (must exist in DATASET_CONFIGS)')
    parser.add_argument('--n_trials', type=int, default=200,
                        help='Optuna TPE trials per budget level')
    parser.add_argument('--experts', type=str, default=None,
                        help='Comma-separated expert names '
                             '(defaults to a six-expert pool)')
    parser.add_argument('--budgets', type=str, default=None,
                        help='Comma-separated GFLOPs budgets '
                             '(default: adaptive grid)')
    parser.add_argument('--parallel', action='store_true',
                        help='Parallelize budget levels with joblib')
    parser.add_argument('--no_overhead', action='store_true',
                        help='Disable routing overhead accounting')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed (split + TPE)')
    parser.add_argument('--val_ratio', type=float, default=0.7,
                        help='Fraction allocated to D_opt + D_cal (default 0.7)')
    parser.add_argument('--cal_ratio', type=float, default=0.3,
                        help='Fraction of (D_opt + D_cal) for D_cal (default 0.3)')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--no_plot', action='store_true',
                        help='(reserved for compatibility)')
    return parser.parse_args()


# =============================================================================
# BUDGET GRID
# =============================================================================

def build_default_budget_grid(experts: List[Dict[str, Any]]) -> List[float]:
    """Adaptive grid: finer steps at low budgets, coarser at high.

    The lower bound is the cheapest expert; the upper bound is 110 %
    of the sum of all experts (so the unconstrained regime is also
    covered).
    """
    min_g = min(e['gflops'] for e in experts)
    max_g = sum(e['gflops'] for e in experts)

    levels: List[float] = []
    cur = min_g
    while cur <= max_g * 1.1:
        levels.append(round(cur, 3))
        if   cur < 0.5:  cur += 0.05
        elif cur < 2.0:  cur += 0.25
        elif cur < 5.0:  cur += 0.5
        elif cur < 10.0: cur += 1.0
        else:            cur += 2.0
    if levels[-1] < max_g:
        levels.append(round(max_g, 3))
    return levels


# =============================================================================
# MAIN
# =============================================================================

def main() -> Dict[str, Any]:
    args = parse_args()

    print("=" * 80)
    print(f"CADS — {args.dataset.upper()}")
    print("=" * 80)

    config = Config.from_args(args)

    budget_levels = (
        [float(b.strip()) for b in args.budgets.split(',')]
        if args.budgets else
        build_default_budget_grid(config.experts)
    )

    # ----- Print setup ---------------------------------------------------
    print(f"\n  Dataset: {DATASET_CONFIGS[args.dataset]['description']}")
    print(f"  Experts ({len(config.experts)}):")
    for e in config.experts:
        print(f"    - {e['name']:<20} {e['gflops']:.3f} GFLOPs")
    print(f"  Budget levels: {len(budget_levels)} "
          f"({min(budget_levels):.3f} to {max(budget_levels):.3f} GFLOPs)")
    print(f"  Trials per budget : {config.n_trials}")
    print(f"  Routing overhead  : "
          f"{'Yes' if config.include_routing_overhead else 'No'}")
    print(f"  Parallel          : "
          f"{'Yes' if args.parallel and JOBLIB_AVAILABLE else 'No'}")

    # ----- Load and split data ------------------------------------------
    print(f"\n{'='*80}\nLOADING DATA (3-way stratified split)\n{'='*80}")
    cache = PredictionCache(config)
    try:
        opt_data, opt_targets, cal_data, cal_targets, test_data, test_targets = (
            cache.load_split()
        )
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print(f"  Set CADS_CACHE_PATH environment variable to specify "
              f"cache location")
        sys.exit(1)
    print(f"  |D_opt| = {len(opt_targets)}  "
          f"|D_cal| = {len(cal_targets)}  "
          f"|D_test| = {len(test_targets)}")

    # ----- Profile experts on D_opt --------------------------------------
    print(f"\n{'='*80}\nEXPERT ANALYSIS (on D_opt)\n{'='*80}")
    profiles_opt = analyze_experts(opt_data, opt_targets, config)
    print(f"\n  {'Expert':<20} | {'Acc(opt)':>9} | {'GFLOPs':>8} | {'Eff':>10}")
    print("  " + "-" * 58)
    for exp in config.experts:
        p = profiles_opt[exp['name']]
        print(f"  {exp['name']:<20} | {p.accuracy*100:>8.2f}% "
              f"| {p.gflops:>8.3f} | {p.efficiency:>10.1f}")

    # Reference: solo accuracy on D_test (display only — not used in optimisation).
    test_acc = {
        e['name']: float((test_data[e['name']].argmax(1) == test_targets).mean())
        for e in config.experts
    }
    bsn = max(test_acc, key=test_acc.get)
    bsa, bsg = test_acc[bsn], EXPERT_GFLOPS.get(bsn, 0.5)
    print(f"\n  [Test reference] Best solo: {bsn} "
          f"({bsa*100:.2f}% @ {bsg:.3f}G)")

    # ----- Baselines on D_test (weights from D_opt) ---------------------
    print(f"\n{'='*80}\nBASELINES (D_test, weighted by profiles_opt)\n{'='*80}")
    baselines = compute_baselines(test_data, test_targets, config, profiles_opt)
    print(f"  Full Ensemble : {baselines['cascade']['accuracy']*100:.2f}% "
          f"@ {baselines['cascade']['gflops']:.3f}G")
    print(f"  Oracle        : {baselines['oracle']['accuracy']*100:.2f}%")
    print(f"\n  Cumulative cascade:")
    for cc in baselines['cumulative_cascade']:
        print(f"    Top-{cc['n_experts']:>2}: {cc['accuracy']*100:.2f}% "
              f"@ {cc['gflops']:.3f}G")

    # ----- Complementarity on D_opt -------------------------------------
    print(f"\n{'='*80}\nCOMPLEMENTARITY ANALYSIS (on D_opt)\n{'='*80}")
    complementarity = ComplementarityAnalyzer(config)
    complementarity.analyze(opt_data, opt_targets)
    print("  Computed complementarity scores on D_opt")

    engine = CADSEngine(config, profiles_opt, complementarity)

    # ----- Budget loop --------------------------------------------------
    print(f"\n{'='*80}\nBUDGET-CONSTRAINED OPTIMIZATION "
          f"(TPE on D_opt, calib on D_cal, eval on D_test)\n{'='*80}")
    print(f"\n  {'Budget':>8} | {'CADS':>10} | {'GFLOPs':>10} | "
          f"{'Baseline':>10} | {'Gain':>8} | {'AvgExp':>6} | Status")
    print("  " + "-" * 80)

    total_start = time.time()

    common_args = (
        config, engine,
        opt_data, opt_targets,
        cal_data, cal_targets,
        test_data, test_targets,
        baselines,
    )

    if args.parallel and JOBLIB_AVAILABLE:
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(process_single_budget)(budget, *common_args)
            for budget in budget_levels
        )
    else:
        results = [process_single_budget(b, *common_args) for b in budget_levels]

    for r in sorted(results, key=lambda x: x['budget']):
        gain_str = (
            f"+{r['gain']*100:.2f}%" if r['gain'] > 0
            else f"{r['gain']*100:.2f}%"
        )
        status = (
            "OK" if r['feasible'] and r.get('test_feasible', True)
            else "INFEASIBLE"
        )
        print(f"  {r['budget']:>7.3f}G | {r['accuracy_test']*100:>9.2f}% "
              f"| {r['gflops_test']:>9.3f}G | {r['baseline_acc']*100:>9.2f}% "
              f"| {gain_str:>8} | {r['avg_experts']:>6.2f} | {status}")

    total_elapsed = time.time() - total_start
    print(f"\n  Total optimization time: {total_elapsed/60:.1f} min")

    # ----- Save ---------------------------------------------------------
    print(f"\n{'='*80}\nSAVING RESULTS\n{'='*80}")
    results_dir = Path(args.results_dir)

    output = build_output(
        args_dataset    = args.dataset,
        config          = config,
        budget_levels   = budget_levels,
        profiles_opt    = profiles_opt,
        baselines       = baselines,
        results         = results,
        n_opt           = len(opt_targets),
        n_cal           = len(cal_targets),
        n_test          = len(test_targets),
        elapsed_minutes = total_elapsed / 60,
    )
    json_path, csv_path = save_results(
        output, results, baselines, args.dataset, results_dir
    )
    print(f"  JSON: {json_path}")
    print(f"  CSV : {csv_path}")

    # ----- Summary ------------------------------------------------------
    print(f"\n{'='*80}\nSUMMARY — {args.dataset.upper()}\n{'='*80}")
    feasible = [r for r in results
                if r['feasible'] and r.get('test_feasible', True)]
    if feasible:
        avg_gain = float(np.mean([r['gain'] for r in feasible]))
        max_g_r  = max(feasible, key=lambda x: x['gain'])
        best_acc = max(feasible, key=lambda x: x['accuracy_test'])
        print(f"\n  Feasible points : {len(feasible)} / {len(results)}")
        print(f"  Avg gain        : +{avg_gain*100:.2f}%")
        print(f"  Max gain        : +{max_g_r['gain']*100:.2f}% at "
              f"{max_g_r['budget']:.3f}G budget")
        print(f"     -> CADS: {max_g_r['accuracy_test']*100:.2f}% "
              f"@ {max_g_r['gflops_test']:.3f}G")
        print(f"     -> Baseline ({max_g_r['baseline_name']}): "
              f"{max_g_r['baseline_acc']*100:.2f}%")
        print(f"\n  Best accuracy   : {best_acc['accuracy_test']*100:.2f}% "
              f"@ {best_acc['gflops_test']:.3f}G "
              f"(avg {best_acc['avg_experts']:.2f} experts)")
        full_e_acc = baselines['cascade']['accuracy']
        full_e_gf  = baselines['cascade']['gflops']
        similar = [r for r in feasible if r['accuracy_test'] >= full_e_acc * 0.99]
        if similar:
            most_eff = min(similar, key=lambda x: x['gflops_test'])
            savings = (1 - most_eff['gflops_test'] / full_e_gf) * 100
            print(f"\n  At ~Full-Ensemble accuracy ({full_e_acc*100:.1f}%):")
            print(f"     CADS = {most_eff['gflops_test']:.3f}G "
                  f"(vs {full_e_gf:.3f}G full) -> -{savings:.1f}% GFLOPs")
    else:
        print("\n  WARNING: No feasible solutions found!")
    return output


if __name__ == "__main__":
    main()
