"""
cads.optimization
=================

Budget-constrained Bayesian optimisation of the CADS policy.

For each FLOPs budget level, we run an Optuna TPE search over the
:class:`cads.engine.CADSParams` space. The objective evaluates the
candidate on ``D_opt`` (with the engine self-calibrating its conformal
predictor on that same subset for each trial). At the end of the
search:

  1. The best feasible configuration ``theta*`` is selected.
  2. The conformal predictor is recalibrated on a fresh subset
     ``D_cal``, disjoint from ``D_opt``.
  3. The recalibrated policy is evaluated once on ``D_test``.

This three-step protocol (search on D_opt, recalibrate on D_cal,
evaluate on D_test) ensures that the marginal coverage guarantee of
the conformal predictor holds empirically on D_test.

Soft-penalty objective
----------------------
The TPE objective is the trial accuracy minus a soft penalty
``10 * max(0, gflops - budget)``. Infeasible trials are not discarded
outright — they receive a smoothly degraded score — which lets TPE
exploit information from over-budget configurations while still
converging to feasible solutions. See equation (15) of the paper.
"""
from __future__ import annotations

from typing import Any, Dict

import optuna
from optuna.samplers import TPESampler

from .baselines import get_best_baseline_at_budget
from .engine    import CADSEngine, CADSParams
from .config    import Config
from .data      import ExpertDataDict, LabelArray


# =============================================================================
# CONSTANTS
# =============================================================================

#: Multiplier applied to the over-budget excess in the soft-penalty
#: objective (paper eq. 15).
SOFT_PENALTY_WEIGHT: float = 10.0

#: Slack on the budget constraint when checking test-time feasibility.
#: A policy whose measured GFLOPs on D_test is within (1 + this) times
#: the budget is still considered feasible. Paper convention: 5 %.
TEST_FEASIBILITY_SLACK: float = 0.05


# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================
# Ranges searched by Optuna TPE. They reproduce the ICIP paper. Tuning
# these ranges is the typical first thing to try when porting CADS to
# a new domain.
# =============================================================================

def _sample_params(trial: optuna.Trial, n_experts: int) -> CADSParams:
    """Sample a :class:`CADSParams` from the TPE search space."""
    return CADSParams(
        alpha=trial.suggest_float('alpha', 0.05, 0.20),
        use_class_quantiles=trial.suggest_categorical(
            'use_class_q', [True, False]
        ),
        singleton_conf=trial.suggest_float('singleton_conf', 0.80, 0.99),
        binary_conf=trial.suggest_float('binary_conf', 0.65, 0.95),
        difficult_conf=trial.suggest_float('difficult_conf', 0.55, 0.90),
        complementarity_weight=trial.suggest_float('comp_weight', 0.3, 0.9),
        min_experts_singleton=trial.suggest_int('min_exp_single', 1, 2),
        min_experts_binary=trial.suggest_int('min_exp_binary', 1, 3),
        min_experts_difficult=trial.suggest_int(
            'min_exp_diff', 2, min(5, n_experts)
        ),
        weight_power=trial.suggest_float('weight_power', 3.0, 10.0),
        class_weight_power=trial.suggest_float('class_weight_power', 1.5, 7.0),
        conf_boost_per_expert=trial.suggest_float('conf_boost', 0.005, 0.08),
        max_conf_boost=trial.suggest_float('max_boost', 0.03, 0.20),
    )


# =============================================================================
# BUDGET-LEVEL DRIVER
# =============================================================================

def optimize_for_budget(
    config:        Config,
    engine:        CADSEngine,
    opt_data:      ExpertDataDict,
    opt_targets:   LabelArray,
    cal_data:      ExpertDataDict,
    cal_targets:   LabelArray,
    test_data:     ExpertDataDict,
    test_targets:  LabelArray,
    budget:        float,
    n_trials:      int,
) -> Dict[str, Any]:
    """Run TPE search at a single budget, then evaluate on D_test.

    Parameters
    ----------
    config
        Run configuration; provides the seed and the expert list.
    engine
        Pre-built :class:`CADSEngine` (with profiles and complementarity
        already populated on D_opt). Reused across all TPE trials.
    opt_data, opt_targets
        Optimisation subset D_opt — used for self-calibration and as
        the TPE objective signal.
    cal_data, cal_targets
        Calibration subset D_cal — used for the final conformal
        recalibration of the selected ``theta*``.
    test_data, test_targets
        Held-out subset D_test — used once, after recalibration, for
        the final reported metrics.
    budget
        FLOPs budget for this run, in giga. Trials that exceed it are
        penalised by the soft-penalty objective; ``test_feasible`` is
        also checked against this value.
    n_trials
        Number of Optuna TPE trials.

    Returns
    -------
    result
        Dictionary with at least the following keys:

          * ``accuracy_opt``, ``gflops_opt`` — best feasible point
            found by TPE on D_opt.
          * ``params`` — the corresponding :class:`CADSParams`.
          * ``feasible`` — True if any feasible trial was found.
          * ``accuracy_test``, ``gflops_test``, ``avg_experts`` —
            final evaluation on D_test after recalibration on D_cal.
          * ``test_feasible`` — whether D_test GFLOPs is within the
            5 % tolerance band.
          * ``detailed_stats`` — exit and usage distributions from the
            D_test pass.
    """
    best_result: Dict[str, Any] = {
        'accuracy_opt': 0.0,
        'gflops_opt':   budget,
        'params':       None,
        'feasible':     False,
    }

    def objective(trial: optuna.Trial) -> float:
        """Soft-penalty TPE objective evaluated on D_opt.

        Parameters
        ----------
        trial
            Optuna trial whose suggested hyperparameters are sampled
            via :func:`_sample_params`.
        """
        nonlocal best_result
        params = _sample_params(trial, len(config.experts))
        # Trial-internal calibration on D_opt (proxy). The final
        # evaluation uses a fresh D_cal recalibration.
        acc, gflops, _ = engine.run(
            params, opt_data, opt_targets, calibrate=True
        )
        if gflops <= budget:
            if acc > best_result['accuracy_opt']:
                best_result = {
                    'accuracy_opt': acc,
                    'gflops_opt':   gflops,
                    'params':       params,
                    'feasible':     True,
                }
            return acc
        return acc - SOFT_PENALTY_WEIGHT * (gflops - budget)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(
            n_startup_trials=min(30, n_trials // 3),
            seed=config.seed,
            multivariate=True,
        ),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if best_result['params'] is not None:
        # Final calibration on D_cal (fresh, disjoint from D_opt).
        # This is what makes the conformal coverage guarantee hold on
        # D_test.
        engine.calibrate(cal_data, cal_targets, best_result['params'])
        acc_test, gflops_test, stats = engine.run(
            best_result['params'],
            test_data, test_targets,
            collect_detailed_stats=True,
        )
        best_result['accuracy_test']  = acc_test
        best_result['gflops_test']    = gflops_test
        best_result['avg_experts']    = stats['avg_experts_used']
        best_result['test_feasible']  = (
            gflops_test <= budget * (1.0 + TEST_FEASIBILITY_SLACK)
        )
        best_result['detailed_stats'] = stats
    else:
        best_result['accuracy_test']  = 0.0
        best_result['gflops_test']    = budget
        best_result['avg_experts']    = 0.0
        best_result['test_feasible']  = False
        best_result['detailed_stats'] = {}
    return best_result


# =============================================================================
# WRAPPER: SINGLE BUDGET LEVEL + BASELINE COMPARISON
# =============================================================================

def process_single_budget(
    budget:        float,
    config:        Config,
    engine:        CADSEngine,
    opt_data:      ExpertDataDict,
    opt_targets:   LabelArray,
    cal_data:      ExpertDataDict,
    cal_targets:   LabelArray,
    test_data:     ExpertDataDict,
    test_targets:  LabelArray,
    baselines:     Dict[str, Any],
) -> Dict[str, Any]:
    """Run :func:`optimize_for_budget` and attach baseline metrics.

    The baseline at a given budget is the best of (i) any individual
    expert whose cost fits in the budget and (ii) any cumulative
    cascade of the cheapest k experts whose total cost fits in the
    budget.

    Parameters
    ----------
    budget
        FLOPs budget for this point, in giga.
    config, engine
        Run configuration and pre-built cascade engine; see
        :func:`optimize_for_budget`.
    opt_data, opt_targets
        Optimisation subset D_opt.
    cal_data, cal_targets
        Calibration subset D_cal.
    test_data, test_targets
        Held-out test subset D_test.
    baselines
        Pre-computed baseline dictionary (see :func:`compute_baselines`).
        Used to derive the ``baseline_name`` / ``baseline_acc`` / ``gain``
        fields of the returned result.

    Returns
    -------
    result
        The dict returned by :func:`optimize_for_budget`, augmented
        with ``budget``, ``baseline_name``, ``baseline_acc``,
        ``baseline_gflops``, and the ``gain`` of CADS over the
        baseline.
    """
    result = optimize_for_budget(
        config, engine,
        opt_data, opt_targets,
        cal_data, cal_targets,
        test_data, test_targets,
        budget, config.n_trials,
    )
    bl_name, bl_acc, bl_gflops = get_best_baseline_at_budget(baselines, budget)
    result['budget']          = budget
    result['baseline_name']   = bl_name
    result['baseline_acc']    = bl_acc
    result['baseline_gflops'] = bl_gflops
    result['gain'] = (
        (result['accuracy_test'] - bl_acc) if result['feasible'] else 0.0
    )
    return result
