#!/usr/bin/env python3
"""
demo_pathmnist.py
=================

End-to-end CADS demonstration on the real PathMNIST experts shipped
with the repository.

This is the recommended entry point for users new to CADS. It uses the
same expert pool, dataset and protocol as the ICIP paper, and runs the
full pipeline (profile, complementarity, budget sweep, evaluation) in
a few minutes on a single workstation.

Expert pool (matching the ICIP paper)
-------------------------------------
Eleven classifiers from four architecture families::

    Lightweight CNNs      mobilenet_tiny, ghostnet, efficientnet_lite,
                          convnextv2_atto
    Mid-range CNN/hybrid  eva02_tiny, mobilevit, convnextv2_tiny
    Vision Transformers   maxvit_tiny, swinv2_tiny
    Heavy classifiers     efficientnetv2, convnextv2_base

Output
------
Two figures saved to ``demo_outputs/figures/``:

  * ``pareto_pathmnist.png``        — accuracy vs GFLOPs (paper style).
  * ``expert_usage_pathmnist.png``  — stacked expert usage by budget.

A JSON summary in ``demo_outputs/pathmnist_summary.json``.

Run::

    python scripts/demo_pathmnist.py
    python scripts/demo_pathmnist.py --n_trials 50    # faster
    python scripts/demo_pathmnist.py --parallel       # all CPU cores
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Allow `python scripts/demo_pathmnist.py` to find the cads package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cads import (
    Config,
    EXPERT_GFLOPS,
    PredictionCache,
    analyze_experts,
    ComplementarityAnalyzer,
    CADSEngine,
    compute_baselines,
    process_single_budget,
    convert_numpy,
)


# =============================================================================
# DEFAULT EXPERT POOL — paper PathMNIST configuration
# =============================================================================

PATHMNIST_PAPER_POOL = [
    'mobilenet_tiny',
    'ghostnet',
    'efficientnet_lite',
    'convnextv2_atto',
    'eva02_tiny',
    'mobilevit',
    'convnextv2_tiny',
    'maxvit_tiny',
    'swinv2_tiny',
    'efficientnetv2',
    'convnextv2_base',
]


def build_expert_list() -> list:
    """Build the ordered expert list (ascending GFLOPs)."""
    experts = [
        {'name': n, 'gflops': EXPERT_GFLOPS[n]} for n in PATHMNIST_PAPER_POOL
    ]
    experts.sort(key=lambda e: e['gflops'])
    return experts


# =============================================================================
# BUDGET GRID — paper-density
# =============================================================================

def default_budget_grid(experts: list) -> list:
    """Budget grid densified to match Figure 1 of the paper.

    The grid is split into three regions:

      * **Low budgets** (0.3G - 0.9G): below the cost of summing the 4
        cheapest experts (~1.1G). With min_experts_difficult=4, most
        of these points are infeasible by construction, but Optuna can
        sometimes find a feasible config that classifies enough samples
        as "singleton" or "binary" to fit in. Six points cover this
        regime.
      * **Variation regime** (1.0G - 7.0G): where CADS exhibits its
        full Pareto curve. Twelve points give the same density as the
        paper figure.
      * **Plateau** (8.0G - full ensemble): two points show that the
        accuracy saturates.

    Total: ~20 budgets, of which 14-17 are typically feasible.
    """
    max_g = sum(e['gflops'] for e in experts)
    return [
        round(b, 3) for b in
        np.unique(np.concatenate([
            np.geomspace(0.3, 0.9, 6),       # 6 pts low end
            np.geomspace(1.0, 7.0, 12),      # 12 pts variation regime
            np.geomspace(8.0, max_g, 3),     # 3 pts plateau + full
        ]))
    ]


# =============================================================================
# PLOTTING — paper style
# =============================================================================

def plot_pareto_paper_style(
    results,
    baselines,
    out_path: Path,
    dataset_label: str = "PATHMNIST",
) -> None:
    """Reproduce the style of Figure 1 (PathMNIST panel) of the paper.

    Series displayed:
      * CADS               — blue solid line with circle markers
      * Full ensemble      — green dotted horizontal line
      * Cumulative cascade — orange dashed line with x markers
      * Individual experts — gray circles with text annotations
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    feasible = [r for r in results
                if r['feasible'] and r.get('test_feasible', True)]
    feasible = sorted(feasible, key=lambda x: x['gflops_test'])

    fig, ax = plt.subplots(figsize=(9, 5.0))

    # Individual experts (gray dots with annotations) -------------------
    expert_pts = sorted(baselines['expert_points'], key=lambda x: x['gflops'])
    xs = [e['gflops']         for e in expert_pts]
    ys = [e['accuracy'] * 100 for e in expert_pts]
    ax.scatter(xs, ys, color='gray', s=40, zorder=3, label='Individual')
    for i, e in enumerate(expert_pts):
        offset = 18 if i % 2 == 0 else -22
        ax.annotate(
            e['name'],
            (e['gflops'], e['accuracy'] * 100),
            xytext=(0, offset), textcoords='offset points',
            fontsize=8.0, color='dimgray', ha='center',
            arrowprops=dict(arrowstyle='-', color='lightgray', lw=0.6),
        )

    # Cumulative cascade -----------------------------------------------
    cc = sorted(baselines['cumulative_cascade'], key=lambda x: x['gflops'])
    ax.plot(
        [c['gflops']         for c in cc],
        [c['accuracy'] * 100 for c in cc],
        marker='x', linestyle='--', color='#ff8c00', linewidth=1.6,
        markersize=8, label='Cumulative',
    )

    # Full ensemble (horizontal line) ----------------------------------
    full_acc = baselines['cascade']['accuracy'] * 100
    ax.axhline(
        full_acc, linestyle=':', color='green', linewidth=1.8, label='Full',
    )

    # CADS curve (headline) --------------------------------------------
    if feasible:
        ax.plot(
            [r['gflops_test']         for r in feasible],
            [r['accuracy_test'] * 100 for r in feasible],
            marker='o', linestyle='-', color='#1f77b4', linewidth=2.5,
            markersize=8, label='CADS', zorder=6,
        )

    # Style -------------------------------------------------------------
    ax.set_xscale('log')
    ax.set_xlabel("Computational Cost (GFLOPs)", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(dataset_label, fontsize=12, fontweight='bold')
    ax.grid(True, which='both', linestyle=':', alpha=0.35)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}%"))

    all_acc = (
        [r['accuracy_test'] * 100 for r in feasible]
        + [e['accuracy']     * 100 for e in expert_pts]
        + [c['accuracy']     * 100 for c in cc]
        + [full_acc]
    )
    ax.set_ylim(min(all_acc) - 2, max(all_acc) + 1.5)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def plot_expert_usage(results, experts, out_path: Path) -> None:
    """Stacked usage of each expert by budget, colored by computational cost.

    Experts are sorted cheapest → most expensive (bottom → top of the
    stack) and colored along a sequential viridis colormap so the cost
    gradient is visible at a glance. Each legend entry shows the
    expert's GFLOPs cost.
    """
    import matplotlib.pyplot as plt

    feasible = [r for r in results
                if r['feasible'] and r.get('test_feasible', True)]
    feasible = sorted(feasible, key=lambda x: x['budget'])
    if not feasible:
        print("  No feasible point — skipping expert usage plot")
        return

    # Sort experts by ascending cost so the stack reads cheapest at the
    # bottom and the colormap aligns with the GFLOPs axis.
    sorted_experts = sorted(experts, key=lambda e: e['gflops'])
    names  = [e['name']   for e in sorted_experts]
    gflops = [e['gflops'] for e in sorted_experts]

    budgets = [r['budget'] for r in feasible]
    matrix = np.zeros((len(names), len(feasible)))
    for j, r in enumerate(feasible):
        usage = r.get('detailed_stats', {}).get('experts_usage', {})
        total = sum(usage.values()) or 1
        for i, name in enumerate(names):
            matrix[i, j] = usage.get(name, 0) / total

    # Color by log10(GFLOPs) so the gradient feels uniform across the
    # several-orders-of-magnitude cost range.
    log_g = np.log10(np.array(gflops))
    norm  = (log_g - log_g.min()) / (log_g.max() - log_g.min() + 1e-9)
    cmap  = plt.colormaps['viridis']
    # Pull slightly off the extremes so labels stay readable.
    colors = [cmap(0.10 + 0.80 * t) for t in norm]

    labels = [f"{n:<18}  {g:>6.2f} G" for n, g in zip(names, gflops)]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.stackplot(budgets, matrix * 100, labels=labels,
                 colors=colors, alpha=0.95, edgecolor='white', linewidth=0.4)

    ax.set_xscale('log')
    ax.set_xlabel("Computational budget (GFLOPs)", fontsize=11)
    ax.set_ylabel("Proportion of expert calls (%)", fontsize=11)
    ax.set_title("CADS on PathMNIST — Expert usage by budget", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xlim(min(budgets), max(budgets))

    # Reverse the legend so the order matches the visual stack
    # (top of the stack = top of the legend).
    handles, lbls = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles[::-1], lbls[::-1],
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        fontsize=9, title="Expert (cost ↑)", title_fontsize=10,
        frameon=False, handlelength=1.8, handleheight=1.2, labelspacing=0.6,
        prop={'family': 'monospace'},
    )
    leg.get_title().set_fontweight('bold')

    ax.grid(True, axis='y', linestyle=':', alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Wrote {out_path}")


# =============================================================================
# PIPELINE
# =============================================================================

def run_demo(
    n_trials:  int,
    parallel:  bool,
    budgets,
    cache_dir: Path,
    out_dir:   Path,
) -> None:
    experts = build_expert_list()
    pool_size = len(experts)

    print("=" * 80)
    print(f"CADS DEMO — PathMNIST ({pool_size}-expert pool)")
    print("=" * 80)

    # Cache presence check ------------------------------------------------
    print("\n[1/5] Checking PathMNIST caches...")
    missing = []
    for expert in experts:
        path = cache_dir / 'pathmnist' / (
            f"pathmnist_{expert['name']}_predictions.npz"
        )
        if not path.exists():
            missing.append(path)
        else:
            print(f"  Found: {path}")
    if missing:
        print("\nMissing caches:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    # Config --------------------------------------------------------------
    print("\n[2/5] Building configuration...")
    config = Config(
        dataset='pathmnist', n_classes=9, n_trials=n_trials,
        experts=experts, include_routing_overhead=True,
        val_ratio=0.7, cal_ratio=0.3, seed=42,
    )
    os.environ['CADS_CACHE_PATH'] = str(cache_dir)
    if budgets is None:
        budgets = default_budget_grid(experts)
    print(f"  Cache dir : {cache_dir}")
    print(f"  Experts   : {pool_size} "
          f"({min(e['gflops'] for e in experts):.3f}G to "
          f"{max(e['gflops'] for e in experts):.3f}G, "
          f"sum={sum(e['gflops'] for e in experts):.3f}G)")
    print(f"  Trials    : {n_trials} per budget level")
    print(f"  Budgets ({len(budgets)}): {budgets}")

    # Load + split --------------------------------------------------------
    print("\n[3/5] Loading and splitting predictions...")
    cache = PredictionCache(config)
    opt_d, opt_t, cal_d, cal_t, test_d, test_t = cache.load_split()
    print(f"  |D_opt|  = {len(opt_t):>5}")
    print(f"  |D_cal|  = {len(cal_t):>5}")
    print(f"  |D_test| = {len(test_t):>5}")

    # Profile + complementarity ------------------------------------------
    print("\n[4/5] Computing expert profiles and complementarity (on D_opt)...")
    profiles = analyze_experts(opt_d, opt_t, config)
    complementarity = ComplementarityAnalyzer(config)
    complementarity.analyze(opt_d, opt_t)

    print(f"\n  {'Expert':<22} | {'Acc(opt)':>9} | {'GFLOPs':>8} | {'Eff':>8}")
    print("  " + "-" * 56)
    for exp in experts:
        p = profiles[exp['name']]
        print(f"  {exp['name']:<22} | {p.accuracy*100:>8.2f}% "
              f"| {p.gflops:>8.3f} | {p.efficiency:>8.1f}")

    # Baselines + budget sweep -------------------------------------------
    print("\n[5/5] Running CADS budget sweep...")
    baselines = compute_baselines(test_d, test_t, config, profiles)
    engine = CADSEngine(config, profiles, complementarity)

    print(f"\n  Full ensemble baseline: {baselines['cascade']['accuracy']*100:.2f}% "
          f"@ {baselines['cascade']['gflops']:.3f}G")
    print(f"  Oracle accuracy       : {baselines['oracle']['accuracy']*100:.2f}%")
    print(f"\n  {'Budget':>8} | {'CADS':>10} | {'GFLOPs':>10} | "
          f"{'Baseline':>10} | {'Gain':>8} | {'AvgExp':>6}")
    print("  " + "-" * 72)

    common = (
        config, engine, opt_d, opt_t, cal_d, cal_t, test_d, test_t, baselines,
    )
    t0 = time.time()
    if parallel and JOBLIB_AVAILABLE:
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(process_single_budget)(b, *common) for b in budgets
        )
    else:
        results = [process_single_budget(b, *common) for b in budgets]

    for r in sorted(results, key=lambda x: x['budget']):
        gain = (f"+{r['gain']*100:.2f}%" if r['gain'] > 0
                else f"{r['gain']*100:.2f}%")
        feasible = r['feasible'] and r.get('test_feasible', True)
        line = (f"  {r['budget']:>7.3f}G | {r['accuracy_test']*100:>9.2f}% "
                f"| {r['gflops_test']:>9.3f}G | {r['baseline_acc']*100:>9.2f}% "
                f"| {gain:>8} | {r['avg_experts']:>6.2f}")
        if not feasible:
            line += "   (infeasible)"
        print(line)
    print(f"\n  Total time: {time.time() - t0:.1f}s")

    # Figures + JSON ------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    print("\nGenerating figures...")
    plot_pareto_paper_style(
        results, baselines, fig_dir / 'pareto_pathmnist.png'
    )
    plot_expert_usage(
        results, experts,
        fig_dir / 'expert_usage_pathmnist.png',
    )

    summary_path = out_dir / 'pathmnist_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(convert_numpy({
            'dataset':  'pathmnist',
            'pool_size': pool_size,
            'experts':  [{'name': e['name'], 'gflops': e['gflops']}
                         for e in experts],
            'baselines': {
                'full_ensemble':      baselines['cascade'],
                'oracle_accuracy':    baselines['oracle']['accuracy'],
                'cumulative_cascade': baselines['cumulative_cascade'],
                'individual_experts': baselines['expert_points'],
            },
            'budget_curve': [
                {
                    'budget':            r['budget'],
                    'cads_accuracy':     r['accuracy_test'],
                    'cads_gflops':       r['gflops_test'],
                    'avg_experts':       r['avg_experts'],
                    'baseline_name':     r['baseline_name'],
                    'baseline_acc':      r['baseline_acc'],
                    'gain':              r['gain'],
                    'feasible':          r['feasible'],
                    'experts_usage':     r.get('detailed_stats', {}).get(
                                            'experts_usage', {}),
                    'exit_distribution': r.get('detailed_stats', {}).get(
                                            'exit_distribution', {}),
                }
                for r in sorted(results, key=lambda x: x['budget'])
            ],
        }), f, indent=2)
    print(f"  Wrote {summary_path}")
    print(f"\nDemo complete. Inspect figures under {fig_dir}/")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CADS demo on the real PathMNIST experts (paper pool)."
    )
    parser.add_argument('--n_trials',  type=int, default=120)
    parser.add_argument('--parallel',  action='store_true')
    parser.add_argument('--budgets',   type=str, default=None,
                        help='Comma-separated GFLOPs budgets '
                             '(default: adaptive grid)')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--out_dir',   type=str, default='demo_outputs')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    budgets = (
        [float(b.strip()) for b in args.budgets.split(',')]
        if args.budgets else None
    )
    run_demo(
        n_trials  = args.n_trials,
        parallel  = args.parallel,
        budgets   = budgets,
        cache_dir = Path(args.cache_dir),
        out_dir   = Path(args.out_dir),
    )