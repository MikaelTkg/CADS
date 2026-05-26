"""
cads.conformal
==============

Class-conditional Adaptive Prediction Sets (APS).

The conformal predictor wraps a probabilistic classifier and exposes
two operations:

  1. **Calibrate** — fit a global and per-class quantile of the APS
     non-conformity score on a labelled calibration set.
  2. **Predict a set** — for an unseen probability vector, return the
     smallest set of class indices whose cumulative probability
     exceeds the calibrated quantile.

The non-conformity score follows Romano, Sesia & Candès (2020): for a
sample with predicted class ranks ``pi`` and true class at rank ``r``,
the score is the cumulative probability up to rank ``r``:

    s_i = sum_{l=1..r} p_{i, pi(l)}

A high score indicates a poorly calibrated prediction (the true class
sits low in the ranking). The level used for the quantile follows the
finite-sample correction ``ceil((n + 1) * (1 - alpha)) / n``.

Class-conditional vs marginal quantiles
---------------------------------------
By default :meth:`get_prediction_set` uses the per-class quantile of
the top-1 predicted class, which empirically tightens the set on
"easy" classes while loosening it on "hard" ones. Setting
``use_class_specific=False`` falls back to the global marginal quantile.

When the per-class calibration subset is below
:data:`MIN_SAMPLES_CLASS_QUANTILE`, the class-specific entry falls
back to the global quantile to avoid degenerate estimates.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .data import ProbabilityArray, LabelArray


# =============================================================================
# CONSTANTS
# =============================================================================

#: Minimum number of calibration samples per class required to
#: estimate a class-specific quantile. Below this, the global quantile
#: is used instead.
MIN_SAMPLES_CLASS_QUANTILE: int = 20


# =============================================================================
# PREDICTOR
# =============================================================================

class ClassConformalPredictor:
    """APS conformal predictor with marginal and per-class quantiles.

    Attributes
    ----------
    n_classes
        Number of classes.
    global_quantile
        Marginal quantile of the APS score on the calibration set.
    class_quantiles
        Per-class quantiles, shape ``(n_classes,)``.
    class_difficulty
        Per-class mean APS score on the calibration set; used by the
        engine's adaptive exit threshold.
    """

    def __init__(self, n_classes: int) -> None:
        """Initialise an uncalibrated predictor with neutral defaults.

        Parameters
        ----------
        n_classes
            Number of classes in the underlying classification task.
            All quantile arrays are sized to ``n_classes``; call
            :meth:`calibrate` before using :meth:`get_prediction_set`.
        """
        self.n_classes:        int                  = n_classes
        self.global_quantile:  float                = 0.9
        self.class_quantiles:  NDArray[np.float64]  = np.ones(n_classes) * 0.9
        self.class_difficulty: NDArray[np.float64]  = np.ones(n_classes) * 0.5
        self._calibrated:      bool                 = False

    # ----- Calibration -------------------------------------------------------

    def calibrate(
        self,
        probs:   ProbabilityArray,
        targets: LabelArray,
        alpha:   float,
    ) -> float:
        """Fit the global and per-class quantiles on a calibration set.

        Parameters
        ----------
        probs
            Calibration probabilities, shape ``(n, n_classes)``.
        targets
            Ground-truth labels, shape ``(n,)``.
        alpha
            Target non-coverage rate; the quantile level is
            ``min(ceil((n + 1) * (1 - alpha)) / n, 1)``.

        Returns
        -------
        global_quantile
            The fitted marginal quantile.
        """
        n = len(targets)
        scores = np.zeros(n)
        for i in range(n):
            sorted_idx = np.argsort(probs[i])[::-1]
            cumsum     = np.cumsum(probs[i][sorted_idx])
            rank       = int(np.where(sorted_idx == targets[i])[0][0])
            scores[i]  = cumsum[rank]

        level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
        self.global_quantile = float(np.quantile(scores, level, method='higher'))

        for c in range(self.n_classes):
            mask = (targets == c)
            if mask.sum() >= MIN_SAMPLES_CLASS_QUANTILE:
                self.class_quantiles[c] = float(
                    np.quantile(scores[mask], level, method='higher')
                )
                self.class_difficulty[c] = float(scores[mask].mean())
            else:
                self.class_quantiles[c]  = self.global_quantile
                self.class_difficulty[c] = 0.5

        self._calibrated = True
        return self.global_quantile

    # ----- Prediction --------------------------------------------------------

    def get_prediction_set(
        self,
        probs:              NDArray[np.float64],
        use_class_specific: bool = True,
    ) -> NDArray[np.int64]:
        """Return the APS prediction set for a single probability vector.

        Parameters
        ----------
        probs
            Probability vector, shape ``(n_classes,)``.
        use_class_specific
            If True, use the quantile of the top-1 predicted class.
            Else use the global marginal quantile.

        Returns
        -------
        set_indices
            Array of class indices (sorted by decreasing probability)
            forming the prediction set.
        """
        sorted_idx = np.argsort(probs)[::-1]
        cumsum     = np.cumsum(probs[sorted_idx])
        q = (
            self.class_quantiles[sorted_idx[0]]
            if use_class_specific else self.global_quantile
        )
        set_size = int(np.searchsorted(cumsum, q, side='right')) + 1
        return sorted_idx[:min(set_size, len(probs))]

    def get_difficulty(self, pred_class: int) -> float:
        """Return the mean calibration APS score of the predicted class."""
        return float(self.class_difficulty[pred_class])
