"""
cads.baselines
==============

Baselines for the budget-accuracy trade-off.

CADS is compared to three families of non-adaptive baselines, all
evaluated on the test subset D_test with weights computed once on
D_opt (so no test information leaks into the comparison):

  * **Individual experts** — raw accuracy of each expert on its own.
  * **Cumulative cascades** — the top-k cheapest experts pooled via
    the same accuracy-weighted ensemble used inside CADS.
  * **Full ensemble** — all experts pooled the same way; serves as
    the ceiling of the static ensemble approach.

An oracle accuracy (probability that at least one expert is correct
on a sample) is also reported as an upper bound on what any cascade
of these experts can achieve.

Weighting scheme
----------------
All ensemble baselines use ``probs * accuracy^5`` weights, where the
accuracy comes from the D_opt profiles. The exponent 5 is the same
default used in :class:`cads.engine.CADSEngine` (``weight_power``
default = 6 inside CADS; here a fixed 5 is used to keep the baseline
configuration-independent). This matches the original implementation.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .config    import Config
from .data      import ExpertDataDict, LabelArray
from .profiling import ExpertProfile


# =============================================================================
# CONSTANTS
# =============================================================================

#: Exponent applied to expert accuracy in the baseline ensembles. Kept
#: separate from the CADS ``weight_power`` so baselines do not depend
#: on the searched policy.
BASELINE_WEIGHT_POWER: float = 5.0


# =============================================================================
# BASELINE COMPUTATION
# =============================================================================

def compute_baselines(
    test_data:    ExpertDataDict,
    test_targets: LabelArray,
    config:       Config,
    profiles_opt: Dict[str, ExpertProfile],
) -> Dict[str, Any]:
    """Compute the suite of static baselines on D_test.

    Parameters
    ----------
    test_data, test_targets
        The test subset (probabilities and labels).
    config
        Used for ``n_classes`` and the expert list.
    profiles_opt
        Profiles computed on D_opt; used to derive the ensemble
        weights without touching D_test for that purpose.

    Returns
    -------
    baselines
        Dictionary with the following structure::

            {
                'cascade': {'accuracy': float, 'gflops': float},
                'oracle':  {'accuracy': float},
                'expert_points': [
                    {'name': str, 'accuracy': float, 'gflops': float, ...},
                    ...
                ],
                'cumulative_cascade': [
                    {'n_experts': int, 'accuracy': float, 'gflops': float,
                     'experts': [str], ...},
                    ...
                ],
            }
    """
    n  = len(test_targets)
    nc = config.n_classes
    total_gflops = sum(e['gflops'] for e in config.experts)

    # ----- Full weighted ensemble ---------------------------------------
    # Weights derived from D_opt, evaluated on D_test.
    weighted = np.zeros((n, nc))
    total_w  = 0.0
    for name, profile in profiles_opt.items():
        w = profile.accuracy ** BASELINE_WEIGHT_POWER
        weighted += test_data[name] * w
        total_w  += w
    cascade_acc = float(
        ((weighted / total_w).argmax(axis=1) == test_targets).mean()
    )

    # ----- Oracle accuracy ----------------------------------------------
    any_correct = np.zeros(n, dtype=bool)
    for name in test_data:
        any_correct |= (test_data[name].argmax(axis=1) == test_targets)
    oracle_acc = float(any_correct.mean())

    # ----- Individual experts -------------------------------------------
    # Raw accuracy of each expert, no weighting (so no leakage from
    # D_opt either; pure D_test measurement).
    expert_points = []
    for exp in config.experts:
        name = exp['name']
        acc  = float((test_data[name].argmax(axis=1) == test_targets).mean())
        expert_points.append({
            'name':     name,
            'accuracy': acc,
            'gflops':   exp['gflops'],
            'type':     'individual_expert',
        })

    # ----- Cumulative cascades (top-k cheapest experts) -----------------
    cumulative_points = []
    sorted_experts    = sorted(config.experts, key=lambda x: x['gflops'])
    cumulative_gflops = 0.0
    for i, exp in enumerate(sorted_experts):
        cumulative_gflops += exp['gflops']
        ens = np.zeros((n, nc))
        tw  = 0.0
        for j in range(i + 1):
            name = sorted_experts[j]['name']
            w    = profiles_opt[name].accuracy ** BASELINE_WEIGHT_POWER
            ens += test_data[name] * w
            tw  += w
        ens /= tw
        acc = float((ens.argmax(axis=1) == test_targets).mean())
        cumulative_points.append({
            'n_experts': i + 1,
            'accuracy':  acc,
            'gflops':    cumulative_gflops,
            'experts':   [sorted_experts[j]['name'] for j in range(i + 1)],
            'type':      'cumulative_cascade',
        })

    return {
        'cascade':            {'accuracy': cascade_acc, 'gflops': total_gflops},
        'oracle':             {'accuracy': oracle_acc},
        'expert_points':      expert_points,
        'cumulative_cascade': cumulative_points,
    }


# =============================================================================
# BUDGET LOOKUP
# =============================================================================

def get_best_baseline_at_budget(
    baselines: Dict[str, Any],
    budget:    float,
) -> Tuple[str, float, float]:
    """Return the highest-accuracy baseline that fits within ``budget``.

    Considers both individual experts and cumulative cascades.

    Parameters
    ----------
    baselines
        The dictionary produced by :func:`compute_baselines`. Only the
        ``expert_points`` and ``cumulative_cascade`` lists are read.
    budget
        FLOPs budget, in giga. Any baseline strictly cheaper than this
        is eligible.

    Returns
    -------
    (name, accuracy, gflops)
        The winning baseline's identifier, accuracy and cost. If no
        baseline fits, returns ``("", 0.0, 0.0)``.
    """
    best_acc, best_name, best_gflops = 0.0, "", 0.0

    for exp in baselines['expert_points']:
        if exp['gflops'] <= budget and exp['accuracy'] > best_acc:
            best_acc    = exp['accuracy']
            best_name   = exp['name']
            best_gflops = exp['gflops']

    for cc in baselines['cumulative_cascade']:
        if cc['gflops'] <= budget and cc['accuracy'] > best_acc:
            best_acc    = cc['accuracy']
            best_name   = f"cumulative_top_{cc['n_experts']}"
            best_gflops = cc['gflops']

    return best_name, best_acc, best_gflops
