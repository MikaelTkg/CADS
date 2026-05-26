#!/usr/bin/env python3
"""
demo.py
=======

End-to-end demonstration of CADS without requiring any deep learning
framework or a real dataset.

What this script does
---------------------
1. Generates four synthetic "experts" of increasing accuracy and cost
   on a synthetic 10-class problem (5000 samples). Each expert is a
   probabilistic classifier whose accuracy is calibrated to match a
   target value; the four experts also have controlled error
   complementarity so that the cascade is meaningful.

2. Saves their predictions as CADS-compatible ``.npz`` caches in
   ``demo_data/synthetic/``.

3. Registers ``synthetic_demo`` in :data:`cads.config.DATASET_CONFIGS`
   and the four expert names in :data:`cads.config.EXPERT_GFLOPS`, so
   the rest of the pipeline can be used unmodified.

4. Runs the full CADS pipeline (profile, complementarity, budget
   sweep, evaluation).

5. Produces two figures saved to ``demo_data/figures/``:
      * ``pareto.png``  — accuracy vs GFLOPs (CADS plus baselines).
      * ``expert_usage.png`` — stacked usage of each expert per budget.

Run::

    python scripts/demo.py
    python scripts/demo.py --n_trials 50   # faster
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Allow `python scripts/demo.py` to find the cads package.
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
)


# =============================================================================
# DEMO CONFIGURATION
# =============================================================================

DEMO_DATASET   = 'synthetic_demo'
DEMO_N_CLASSES = 10
DEMO_N_SAMPLES = 5000
DEMO_SEED      = 42

# Each entry: (expert_name, target_accuracy, gflops_cost, error_bias)
# - target_accuracy: fraction of samples this expert classifies correctly.
# - gflops_cost: cost charged when this expert is consulted.
# - error_bias: controls correlation of errors with cheaper experts.
#   Higher values mean this expert tends to fail on different samples
#   than the cheaper ones, which is the regime where CADS pays off.
DEMO_EXPERTS = [
    ('demo_scout',      0.62, 0.05, 0.0),
    ('demo_light',      0.74, 0.20, 0.4),
    ('demo_specialist', 0.84, 0.80, 0.5),
    ('demo_oracle',     0.91, 2.50, 0.6),
]


# =============================================================================
# SYNTHETIC EXPERT GENERATOR
# =============================================================================

def generate_synthetic_expert(
    targets:         np.ndarray,
    target_accuracy: float,
    n_classes:       int,
    error_bias:      float,
    cheaper_correct: np.ndarray | None,
    rng:             np.random.Generator,
) -> np.ndarray:
    """Generate softmax predictions calibrated to a target accuracy.

    The construction is deliberately simple:

      * For each sample we draw a Bernoulli to decide whether the
        expert is correct.
      * If correct, the output is a sharp distribution on the true
        class.
      * If wrong, the output is a sharp distribution on a uniformly
        chosen alternative class.
      * The Bernoulli probability is shifted by ``error_bias`` on
        samples the cheaper experts already got right, so that this
        expert mostly contributes new correct answers on the hard
        cases (positive complementarity).

    The output is finally smoothed into a non-degenerate softmax for
    realism.
    """
    n = len(targets)
    if cheaper_correct is None:
        p_correct = np.full(n, target_accuracy)
    else:
        # Boost accuracy on samples cheaper experts already got right
        # (this expert is "redundant" there) and depress it slightly
        # elsewhere; then renormalise to hit the global target.
        p_correct = np.where(
            cheaper_correct,
            target_accuracy + (1.0 - target_accuracy) * error_bias,
            target_accuracy - target_accuracy * error_bias * 0.3,
        )
        # Renormalise to hit the global accuracy target on expectation.
        current = p_correct.mean()
        p_correct = np.clip(p_correct * (target_accuracy / max(current, 1e-9)),
                            0.05, 0.99)

    is_correct = rng.random(n) < p_correct

    probs = np.zeros((n, n_classes), dtype=np.float64)
    for i in range(n):
        if is_correct[i]:
            top = int(targets[i])
        else:
            choices = [c for c in range(n_classes) if c != targets[i]]
            top = int(rng.choice(choices))

        # Soft distribution: most mass on top, rest uniform. The peak
        # is high so that "easy" samples land in singleton/binary
        # categories and the cascade can exit early.
        peak  = rng.uniform(0.85, 0.98)
        rest  = (1.0 - peak) / (n_classes - 1)
        probs[i, :]   = rest
        probs[i, top] = peak

    return probs


def build_demo_caches(out_dir: Path) -> None:
    """Generate the four synthetic .npz caches and the labels.

    The caches are written under ``out_dir/<dataset>/`` using the
    standard naming convention recognised by
    :class:`cads.PredictionCache`.
    """
    rng = np.random.default_rng(DEMO_SEED)
    targets = rng.integers(0, DEMO_N_CLASSES, size=DEMO_N_SAMPLES)

    cumulative_correct: np.ndarray | None = None

    for name, target_acc, _, error_bias in DEMO_EXPERTS:
        probs = generate_synthetic_expert(
            targets, target_acc, DEMO_N_CLASSES, error_bias,
            cumulative_correct, rng,
        )
        # Track which samples have been correctly classified so far,
        # so the next expert is encouraged to disagree.
        preds = probs.argmax(axis=1)
        correct = (preds == targets)
        cumulative_correct = (
            correct if cumulative_correct is None
            else cumulative_correct | correct
        )

        cache_path = out_dir / DEMO_DATASET / (
            f"{DEMO_DATASET}_{name}_predictions.npz"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            probs=probs.astype(np.float32),
            labels=targets.astype(np.int64),
        )
        print(f"  Wrote {cache_path.name}: "
              f"accuracy={correct.mean()*100:.2f}% "
              f"(target {target_acc*100:.0f}%)")


# =============================================================================
# REGISTRATION INTO THE CADS REGISTRIES
# =============================================================================

def register_demo_in_cads() -> None:
    """Inject the demo dataset and experts into the CADS registries.

    Mutating the registries at run time is the same operation a user
    would perform once and for all in :mod:`cads.config` when adding
    a new dataset or expert; doing it here keeps the demo self-
    contained.
    """
    DATASET_CONFIGS[DEMO_DATASET] = {
        'n_classes':   DEMO_N_CLASSES,
        'description': 'Synthetic demo (10 classes, 5000 samples)',
    }
    for name, _, gflops, _ in DEMO_EXPERTS:
        EXPERT_GFLOPS[name] = gflops


# =============================================================================
# PLOTTING
# =============================================================================

def plot_pareto(results, baselines, out_path: Path) -> None:
    """Reproduce Figure 1 of the paper: accuracy vs GFLOPs."""
    import matplotlib.pyplot as plt

    feasible = [r for r in results
                if r['feasible'] and r.get('test_feasible', True)]
    feasible = sorted(feasible, key=lambda x: x['gflops_test'])

    fig, ax = plt.subplots(figsize=(7, 5))

    # CADS curve
    if feasible:
        ax.plot(
            [r['gflops_test'] for r in feasible],
            [r['accuracy_test'] * 100 for r in feasible],
            marker='o', linewidth=2, color='C0', label='CADS', zorder=5,
        )

    # Cumulative cascade
    cc = baselines['cumulative_cascade']
    ax.plot(
        [c['gflops']      for c in cc],
        [c['accuracy']*100 for c in cc],
        marker='s', linestyle='--', color='C1', label='Cumulative cascade',
    )

    # Individual experts
    for exp in baselines['expert_points']:
        ax.scatter(exp['gflops'], exp['accuracy']*100,
                   color='gray', s=40, zorder=3)
        ax.annotate(
            exp['name'],
            (exp['gflops'], exp['accuracy']*100),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, color='gray',
        )

    # Full ensemble accuracy as horizontal reference
    ax.axhline(
        baselines['cascade']['accuracy'] * 100,
        linestyle=':', color='green',
        label=f"Full ensemble ({baselines['cascade']['accuracy']*100:.1f}%)",
    )

    ax.set_xscale('log')
    ax.set_xlabel("Computational Cost (GFLOPs)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CADS — Pareto curve (synthetic demo)")
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def plot_expert_usage(results, expert_names, out_path: Path) -> None:
    """Reproduce Figure 2 of the paper: stacked expert usage by budget."""
    import matplotlib.pyplot as plt

    feasible = [r for r in results
                if r['feasible'] and r.get('test_feasible', True)]
    feasible = sorted(feasible, key=lambda x: x['budget'])
    if not feasible:
        print("  No feasible point — skipping expert usage plot")
        return

    budgets = [r['budget'] for r in feasible]
    # Normalise usage per budget so columns sum to 1.
    matrix = np.zeros((len(expert_names), len(feasible)))
    for j, r in enumerate(feasible):
        usage = r.get('detailed_stats', {}).get('experts_usage', {})
        total = sum(usage.values()) or 1
        for i, name in enumerate(expert_names):
            matrix[i, j] = usage.get(name, 0) / total

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.stackplot(
        budgets, matrix * 100,
        labels=expert_names, alpha=0.85,
    )
    ax.set_xscale('log')
    ax.set_xlabel("Computational Budget (GFLOPs)")
    ax.set_ylabel("Proportion of expert calls (%)")
    ax.set_title("CADS — Expert usage by budget (synthetic demo)")
    ax.set_ylim(0, 100)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, axis='y', linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Wrote {out_path}")


# =============================================================================
# DEMO PIPELINE
# =============================================================================

def run_demo(n_trials: int, demo_dir: Path) -> None:
    """End-to-end demo: cache generation, CADS run, plots."""
    print("=" * 80)
    print("CADS DEMO — synthetic dataset")
    print("=" * 80)

    # ----- Step 1: generate caches --------------------------------------
    print("\n[1/4] Generating synthetic prediction caches...")
    build_demo_caches(demo_dir)

    # ----- Step 2: register in CADS -------------------------------------
    register_demo_in_cads()

    # ----- Step 3: build a Config manually (no CLI here) ----------------
    print("\n[2/4] Configuring CADS for the demo dataset...")
    config = Config(
        dataset=DEMO_DATASET,
        n_classes=DEMO_N_CLASSES,
        n_trials=n_trials,
        experts=[
            {'name': name, 'gflops': gflops}
            for name, _, gflops, _ in DEMO_EXPERTS
        ],
        include_routing_overhead=True,
        val_ratio=0.7,
        cal_ratio=0.3,
        seed=DEMO_SEED,
    )

    # The PredictionCache uses CADS_CACHE_PATH; point it at our demo dir.
    import os
    os.environ['CADS_CACHE_PATH'] = str(demo_dir)

    cache = PredictionCache(config)
    opt_data, opt_targets, cal_data, cal_targets, test_data, test_targets = (
        cache.load_split()
    )
    print(f"  |D_opt|={len(opt_targets)}  "
          f"|D_cal|={len(cal_targets)}  "
          f"|D_test|={len(test_targets)}")

    # ----- Step 4: profile, complementarity, baselines, budget sweep ----
    print("\n[3/4] Running CADS pipeline...")
    profiles = analyze_experts(opt_data, opt_targets, config)
    complementarity = ComplementarityAnalyzer(config)
    complementarity.analyze(opt_data, opt_targets)
    baselines = compute_baselines(test_data, test_targets, config, profiles)

    engine = CADSEngine(config, profiles, complementarity)

    # Budget grid spanning the feasible range of this minimal pool.
    # With only four experts and at least two required on "difficult"
    # samples, the smallest budget where CADS can find a feasible
    # policy is around 0.7-0.8 GFLOPs. Smaller budgets stay
    # infeasible on this synthetic dataset; that is normal behaviour,
    # not a bug.
    budgets = [0.80, 1.20, 1.80, 2.50, 3.55]

    print(f"\n  Budgets: {budgets}")
    print(f"  Trials per budget: {n_trials}")
    print(f"\n  {'Budget':>8} | {'CADS':>10} | {'GFLOPs':>10} | "
          f"{'Baseline':>10} | {'Gain':>8} | {'AvgExp':>6}")
    print("  " + "-" * 72)
    t0 = time.time()
    results = []
    for b in budgets:
        r = process_single_budget(
            b, config, engine,
            opt_data, opt_targets,
            cal_data, cal_targets,
            test_data, test_targets,
            baselines,
        )
        results.append(r)
        gain_str = (f"+{r['gain']*100:.2f}%" if r['gain'] > 0
                    else f"{r['gain']*100:.2f}%")
        print(f"  {b:>7.3f}G | {r['accuracy_test']*100:>9.2f}% "
              f"| {r['gflops_test']:>9.3f}G | {r['baseline_acc']*100:>9.2f}% "
              f"| {gain_str:>8} | {r['avg_experts']:>6.2f}")
    print(f"\n  Total time: {time.time() - t0:.1f}s")

    # ----- Step 5: plot -------------------------------------------------
    print("\n[4/4] Plotting...")
    fig_dir = demo_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    plot_pareto(results, baselines, fig_dir / 'pareto.png')
    plot_expert_usage(
        results,
        [e['name'] for e in config.experts],
        fig_dir / 'expert_usage.png',
    )

    print("\nDemo complete. Inspect:")
    print(f"  - Caches : {demo_dir / DEMO_DATASET}")
    print(f"  - Figures: {fig_dir}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CADS demo script")
    parser.add_argument('--n_trials', type=int, default=80,
                        help='Optuna TPE trials per budget (lower = faster)')
    parser.add_argument('--demo_dir', type=str, default='demo_data',
                        help='Where to write caches and figures')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(n_trials=args.n_trials, demo_dir=Path(args.demo_dir))
