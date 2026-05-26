"""
cads.profiling
==============

Per-expert statistical profiling.

For each expert, this module computes a static profile capturing its
performance on the optimisation subset ``D_opt``: global accuracy,
per-class accuracy, per-class average confidence, and a cost-normalised
efficiency score. These profiles are consumed downstream by:

  * the ensemble weighting in :class:`cads.engine.CADSEngine` (global
    weights derived from :attr:`ExpertProfile.accuracy`; per-class
    weights derived from :attr:`ExpertProfile.class_accuracy`);
  * the next-expert selection score in :class:`cads.engine.CADSEngine`
    (efficiency component derived from :attr:`ExpertProfile.gflops`);
  * the baselines in :mod:`cads.baselines` (full ensemble and
    cumulative cascade use the same accuracy-based weighting scheme).

All profiles must be computed on ``D_opt`` only. Profiling on the full
cache or on ``D_test`` would leak test information into both the
ensemble weights and the baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .data   import ExpertDataDict, LabelArray


# =============================================================================
# DATACLASS
# =============================================================================

@dataclass
class ExpertProfile:
    """Static statistical profile of a single expert.

    Attributes
    ----------
    name
        Expert identifier (matches :data:`cads.config.EXPERT_GFLOPS` key).
    gflops
        Computational cost in GFLOPs.
    accuracy
        Global classification accuracy on D_opt.
    class_accuracy
        Per-class accuracy on D_opt, shape ``(n_classes,)``.
    class_confidence
        Per-class average top-class probability on D_opt,
        shape ``(n_classes,)``.
    efficiency
        Accuracy / GFLOPs ratio (with a small numerical stabilizer).
    """
    name:             str
    gflops:           float
    accuracy:         float
    class_accuracy:   NDArray[np.float64]
    class_confidence: NDArray[np.float64]
    efficiency:       float = 0.0


# =============================================================================
# PROFILING FUNCTION
# =============================================================================

def analyze_experts(
    data:    ExpertDataDict,
    targets: LabelArray,
    config:  Config,
) -> Dict[str, ExpertProfile]:
    """Compute statistical profiles for every expert in the pool.

    Parameters
    ----------
    data
        Mapping ``{expert_name: probs_(N, n_classes)}``. Must be the
        D_opt subset, not the full cache.
    targets
        Integer labels of shape ``(N,)`` for the same samples.
    config
        Run configuration (uses ``n_classes`` and the expert list).

    Returns
    -------
    profiles
        Mapping ``{expert_name: ExpertProfile}``.

    Notes
    -----
    Must be called on ``D_opt`` only. Calling it on the full cache or
    on ``D_test`` would propagate test information into the downstream
    ensemble weights and baselines.
    """
    profiles: Dict[str, ExpertProfile] = {}
    n_classes = config.n_classes

    for expert in config.experts:
        probs = data[expert['name']]
        preds = probs.argmax(axis=1)
        confs = probs.max(axis=1)

        acc = float((preds == targets).mean())

        class_acc        = np.zeros(n_classes)
        class_confidence = np.zeros(n_classes)
        for c in range(n_classes):
            mask = (targets == c)
            if mask.sum() > 0:
                class_acc[c]        = float((preds[mask] == c).mean())
                class_confidence[c] = float(confs[mask].mean())

        profiles[expert['name']] = ExpertProfile(
            name=expert['name'],
            gflops=expert['gflops'],
            accuracy=acc,
            class_accuracy=class_acc,
            class_confidence=class_confidence,
            efficiency=acc / (expert['gflops'] + 1e-9),
        )

    return profiles
