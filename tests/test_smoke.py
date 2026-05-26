"""Smoke tests for CADS, using the shipped PathMNIST caches.

Three focused tests on three real experts from the paper's pool, each
validating one claim. The tests load only the shipped ``.npz`` caches
(no network, no synthetic data) and run in well under a second.

All three tests share the same ``pathmnist_pool`` fixture defined in
:mod:`tests.conftest`. The fixture loads three real PathMNIST experts
(``mobilenet_tiny``, ``eva02_tiny``, ``convnextv2_base`` — chosen to
span the cost spectrum) and returns the triple
``(data, targets, config)``:

* ``data``   — ``{expert_name: probs_(N, n_classes)}``
* ``targets``— integer labels of shape ``(N,)``
* ``config`` — :class:`cads.Config` matching the three experts
"""
from __future__ import annotations

import numpy as np

from cads import (
    CADSEngine,
    CADSParams,
    ClassConformalPredictor,
    ComplementarityAnalyzer,
    analyze_experts,
    compute_baselines,
)


def test_conformal_coverage_guarantee(pathmnist_pool):
    """The APS prediction set covers the true label with probability
    ≥ 1 − α on a held-out split. This is the central mathematical
    property CADS relies on."""
    data, targets, config = pathmnist_pool
    probs = data['convnextv2_base']  # strongest expert
    rng = np.random.default_rng(0)

    perm = rng.permutation(len(targets))
    cal_i, test_i = perm[: len(perm) // 2], perm[len(perm) // 2:]

    alpha = 0.10
    predictor = ClassConformalPredictor(n_classes=config.n_classes)
    predictor.calibrate(probs[cal_i], targets[cal_i], alpha)

    in_set = sum(
        int(targets[i] in predictor.get_prediction_set(probs[i]))
        for i in test_i
    )
    coverage = in_set / len(test_i)
    assert coverage >= (1 - alpha) - 0.03, (
        f"Empirical coverage {coverage:.3f} < target {1 - alpha:.3f}"
    )


def test_cascade_runs_end_to_end(pathmnist_pool):
    """A full calibrate-and-run pass on real PathMNIST predictions
    yields sensible metrics and at least matches the cheapest expert."""
    data, targets, config = pathmnist_pool
    profiles = analyze_experts(data, targets, config)
    analyzer = ComplementarityAnalyzer(config)
    analyzer.analyze(data, targets)

    engine = CADSEngine(config, profiles, analyzer)
    accuracy, avg_gflops, stats = engine.run(
        CADSParams(), data, targets, calibrate=True,
    )

    assert 0.0 <= accuracy <= 1.0
    assert avg_gflops > 0
    assert 1 <= stats['avg_experts_used'] <= len(config.experts)
    cheapest_acc = min(p.accuracy for p in profiles.values())
    assert accuracy >= cheapest_acc - 0.05


def test_oracle_is_an_upper_bound(pathmnist_pool):
    """The oracle (any expert correct) cannot be less accurate than
    the best individual expert — a hard upper bound on what any
    cascade of these experts can achieve."""
    data, targets, config = pathmnist_pool
    profiles = analyze_experts(data, targets, config)
    baselines = compute_baselines(data, targets, config, profiles)

    best_solo = max(e['accuracy'] for e in baselines['expert_points'])
    assert baselines['oracle']['accuracy'] >= best_solo - 1e-9
