"""
cads.engine
===========

The core CADS inference engine.

This module defines two objects:

  * :class:`CADSParams` — the hyperparameter set ``theta`` searched by
    TPE in :mod:`cads.optimization`. Defaults match the ICIP paper.
  * :class:`CADSEngine` — the sequential cascade. Given a calibrated
    conformal predictor, expert profiles, and complementarity scores,
    it walks each test sample through the expert pool one at a time,
    builds a weighted ensemble at every step, and decides whether to
    halt or to query another expert.

Inference loop overview
-----------------------
For each sample::

    used = {first_expert}                # cheapest expert is always called first
    while not exit_condition:
        compute ensemble probs from used experts
        compute APS prediction set
        category = singleton | binary | difficult     (set on first iteration)
        compute adaptive exit threshold (consensus boost + class difficulty)
        if min_experts_used and ensemble_conf >= threshold and last_two_agree:
            exit
        else:
            pick next expert by complementarity-and-efficiency score

The "consensus boost" lowers the threshold when many consulted experts
agree; the "class difficulty adjustment" raises it for historically
hard classes. Both are documented in equations (12)-(13) of the paper.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .complementarity import ComplementarityAnalyzer
from .config          import Config, ROUTING_OVERHEAD_GFLOPS
from .conformal       import ClassConformalPredictor
from .data            import ExpertDataDict, LabelArray
from .profiling       import ExpertProfile


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

@dataclass
class CADSParams:
    """The hyperparameter set ``theta`` of the policy.

    Defaults reproduce the ICIP paper. The ranges searched by Optuna
    are defined in :mod:`cads.optimization`.

    Attributes
    ----------
    alpha
        Target non-coverage rate of the conformal predictor.
    use_class_quantiles
        If True, the APS predictor uses the per-class quantile of the
        top-1 predicted class. Else the marginal quantile.
    singleton_conf, binary_conf, difficult_conf
        Base ensemble-confidence thresholds for the three difficulty
        categories defined by the conformal set size.
    min_experts_singleton, min_experts_binary, min_experts_difficult
        Minimum number of experts required before the cascade may
        exit, per category.
    complementarity_weight
        ``w`` in the next-expert selection score
        ``w * Comp + (1 - w) * efficiency``.
    weight_power
        Exponent ``gamma`` in the global ensemble weight ``acc^gamma``.
    class_weight_power
        Exponent ``beta`` in the local ensemble weight
        ``(class_acc + epsilon)^beta``.
    conf_boost_per_expert, max_conf_boost
        Parameters of the consensus boost mechanism: each additional
        agreeing expert lowers the exit threshold by
        ``conf_boost_per_expert``, up to a maximum of ``max_conf_boost``.
    """
    alpha:                  float = 0.10
    use_class_quantiles:    bool  = True
    singleton_conf:         float = 0.90
    binary_conf:            float = 0.80
    difficult_conf:         float = 0.70
    min_experts_singleton:  int   = 1
    min_experts_binary:     int   = 2
    min_experts_difficult:  int   = 4
    complementarity_weight: float = 0.6
    weight_power:           float = 6.0
    class_weight_power:     float = 4.0
    conf_boost_per_expert:  float = 0.02
    max_conf_boost:         float = 0.10


# =============================================================================
# FIXED ARCHITECTURAL CONSTANTS
# =============================================================================
# These reproduce the paper as-is. They are intentionally not exposed
# to Optuna search; changing them changes the architecture rather than
# tuning the policy.
# =============================================================================

#: Mixing weight of the global term in the two-level ensemble (eq. 10).
GLOBAL_LOCAL_GLOBAL_WEIGHT: float = 0.6

#: Mixing weight of the local term in the two-level ensemble (eq. 10).
GLOBAL_LOCAL_LOCAL_WEIGHT:  float = 0.4

#: Probability gap below which the top-2 classes are treated as a
#: confused pair when scoring candidate next experts.
CONFUSED_PAIR_GAP: float = 0.15

#: Fraction of consulted experts that must agree on the current
#: ensemble prediction to trigger the consensus boost on the exit
#: threshold.
CONSENSUS_AGREEMENT_RATIO: float = 0.8

#: Number of most recently consulted experts whose agreement is
#: additionally required to commit to an early exit (eq. just below
#: eq. 13 in the paper: "the two most recently consulted experts
#: agree on the predicted class").
RECENT_AGREEMENT_WINDOW: int = 2

#: Coefficient of the class-difficulty adjustment of the exit
#: threshold (eq. 13).
CLASS_DIFFICULTY_COEFF: float = 0.1

#: Hard upper bound on the adjusted exit threshold.
EXIT_THRESHOLD_CAP: float = 0.98

#: Lower bound on the consensus-boosted threshold (the boost cannot
#: drop the threshold below this value).
EXIT_THRESHOLD_FLOOR: float = 0.5

#: Numerical stabiliser used in the local (per-class) weight to avoid
#: zero values when a class is never correctly predicted.
LOCAL_WEIGHT_EPSILON: float = 0.01


# =============================================================================
# ENGINE
# =============================================================================

class CADSEngine:
    """Sequential CADS inference engine.

    Holds references to the static analysis (profiles, complementarity)
    and to its own conformal predictor; mutates the predictor and the
    per-class weights on calibration.

    Notes
    -----
    The engine is stateful: each call to :meth:`calibrate` overwrites
    ``conformal`` and ``class_weights``. When optimising multiple
    budget levels in parallel, each worker receives its own pickled
    copy and there is no cross-worker contention.
    """

    def __init__(
        self,
        config:          Config,
        profiles:        Dict[str, ExpertProfile],
        complementarity: ComplementarityAnalyzer,
    ) -> None:
        """Build a cascade engine bound to a config and its static analyses.

        Parameters
        ----------
        config
            Run configuration. Drives ``n_classes``, the expert list
            (and their order by cost), and whether the per-routing
            overhead is charged.
        profiles
            Per-expert profiles built on ``D_opt`` by
            :func:`cads.profiling.analyze_experts`. Must be computed
            on ``D_opt`` only.
        complementarity
            Empirical complementarity scores from
            :class:`cads.complementarity.ComplementarityAnalyzer`, also
            estimated on ``D_opt``.
        """
        self.config          = config
        self.profiles        = profiles  # must be profiles built on D_opt
        self.complementarity = complementarity
        self.n_classes       = config.n_classes
        self.include_overhead = config.include_routing_overhead

        self.expert_names  = [e['name'] for e in config.experts]
        self.expert_gflops = {e['name']: e['gflops'] for e in config.experts}
        self.first_expert  = min(self.expert_names, key=lambda n: self.expert_gflops[n])

        self.conformal     = ClassConformalPredictor(config.n_classes)
        self.class_weights: Dict[int, Dict[str, float]] = {}

    # =========================================================================
    # CALIBRATION
    # =========================================================================

    def _compute_class_weights(self, power: float) -> None:
        """Precompute normalised per-class ensemble weights.

        Stored as ``self.class_weights[c][name]`` for fast lookup
        during inference.
        """
        self.class_weights = {}
        for c in range(self.n_classes):
            raw = {
                name: (self.profiles[name].class_accuracy[c] + LOCAL_WEIGHT_EPSILON)
                       ** power
                for name in self.expert_names
            }
            total = sum(raw.values())
            self.class_weights[c] = {k: v / total for k, v in raw.items()}

    def calibrate(
        self,
        data:    ExpertDataDict,
        targets: LabelArray,
        params:  CADSParams,
    ) -> None:
        """Fit the conformal predictor on ``(data, targets)``.

        Builds the same accuracy-weighted ensemble used at inference,
        then calibrates the APS quantiles. Must be called on the
        appropriate subset:

          * during TPE search, the engine self-calibrates on D_opt;
          * once theta* is selected, the final reported policy is
            recalibrated on a fresh subset D_cal (see
            :func:`cads.optimization.optimize_for_budget`).

        Parameters
        ----------
        data
            Per-expert probabilities of shape ``(N, n_classes)`` on the
            subset used for calibration (D_opt during search, D_cal
            after).
        targets
            Integer labels of shape ``(N,)`` aligned with ``data``.
        params
            Hyperparameter set θ. Only ``alpha``, ``weight_power`` and
            ``class_weight_power`` are read here; the rest drive
            inference, not calibration.
        """
        self._compute_class_weights(params.class_weight_power)
        n = len(targets)
        ensemble = np.zeros((n, self.n_classes))
        total_w  = 0.0
        for name in self.expert_names:
            w = self.profiles[name].accuracy ** params.weight_power
            ensemble += data[name] * w
            total_w  += w
        ensemble /= total_w
        self.conformal.calibrate(ensemble, targets, params.alpha)

    # =========================================================================
    # NEXT-EXPERT SELECTION
    # =========================================================================

    def _select_next_expert(
        self,
        used:            Set[str],
        last_expert:     str,
        ensemble_probs:  NDArray[np.float64],
        params:          CADSParams,
    ) -> Optional[str]:
        """Pick the next expert to consult.

        Score: ``w * Comp(last, candidate | context) + (1 - w) * efficiency``,
        where ``Comp`` is the most specific complementarity score
        available (class-pair > per-class > global).
        """
        available = set(self.expert_names) - used
        if not available:
            return None

        pred_class = int(np.argmax(ensemble_probs))

        # Detect a "confused pair" when the top-2 probabilities are close.
        sorted_p = np.sort(ensemble_probs)[::-1]
        confused_pair: Optional[Tuple[int, int]] = None
        if len(sorted_p) >= 2 and sorted_p[0] - sorted_p[1] < CONFUSED_PAIR_GAP:
            top2 = np.argsort(ensemble_probs)[-2:]
            confused_pair = (int(top2[0]), int(top2[1]))

        max_gflops = max(self.expert_gflops.values())
        scores: Dict[str, float] = {}
        for candidate in available:
            comp = self.complementarity.get_score(
                last_expert, candidate, pred_class, confused_pair
            )
            efficiency = 1.0 - (self.expert_gflops[candidate] / max_gflops)
            w = params.complementarity_weight
            scores[candidate] = w * comp + (1.0 - w) * efficiency

        return max(scores.items(), key=lambda x: x[1])[0]

    # =========================================================================
    # TWO-LEVEL WEIGHTED ENSEMBLE (eq. 10-11 of the paper)
    # =========================================================================

    def _compute_ensemble(
        self,
        expert_probs: Dict[str, NDArray[np.float64]],
        pred_class:   int,
        params:       CADSParams,
    ) -> NDArray[np.float64]:
        """Aggregate the predictions of the currently consulted experts.

        Combines a global term ``acc^gamma`` with a per-class term
        derived from :attr:`class_weights`, in the fixed ratio
        :data:`GLOBAL_LOCAL_GLOBAL_WEIGHT` / :data:`GLOBAL_LOCAL_LOCAL_WEIGHT`
        (paper convention, not searched).
        """
        ensemble = np.zeros(self.n_classes)
        total_w  = 0.0
        class_w  = self.class_weights.get(pred_class, {})
        for name, probs in expert_probs.items():
            global_w = self.profiles[name].accuracy ** params.weight_power
            local_w  = class_w.get(name, 1.0 / len(expert_probs))
            w = (
                GLOBAL_LOCAL_GLOBAL_WEIGHT * global_w
                + GLOBAL_LOCAL_LOCAL_WEIGHT * local_w
            )
            ensemble += probs * w
            total_w  += w
        return ensemble / total_w if total_w > 0 else ensemble

    # =========================================================================
    # INFERENCE LOOP
    # =========================================================================

    def run(
        self,
        params:                  CADSParams,
        data:                    ExpertDataDict,
        targets:                 LabelArray,
        calibrate:               bool = False,
        collect_detailed_stats:  bool = False,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Run the cascade on ``(data, targets)`` and return aggregate metrics.

        Parameters
        ----------
        params
            Hyperparameter set theta.
        data
            Expert prediction subset.
        targets
            Ground-truth labels.
        calibrate
            If True, calibrate the conformal predictor on this subset
            before running. Used by the TPE objective to self-calibrate
            on D_opt.
        collect_detailed_stats
            If True, also collect per-sample diagnostics (number of
            experts used, FLOPs, correctness) and aggregate exit /
            usage distributions. Used for the final reporting pass on
            D_test.

        Returns
        -------
        accuracy
            Fraction of correctly classified samples.
        avg_gflops
            Average GFLOPs per sample (including routing overhead if
            enabled).
        stats
            Dict containing at least ``avg_experts_used``, and the
            detailed fields when requested.
        """
        if calibrate:
            self.calibrate(data, targets, params)

        n_samples = len(targets)
        n_experts = len(self.expert_names)
        total_gflops = 0.0
        correct      = 0

        stats: Dict[str, Any] = defaultdict(int)
        stats['avg_experts_used'] = 0.0
        if collect_detailed_stats:
            stats['per_sample_experts'] = []
            stats['per_sample_gflops']  = []
            stats['per_sample_correct'] = []
            stats['exit_distribution']  = defaultdict(int)
            stats['experts_usage']      = defaultdict(int)

        for idx in range(n_samples):
            final_pred, sample_gflops, n_used = self._infer_one_sample(
                idx, data, params, stats, collect_detailed_stats
            )
            total_gflops += sample_gflops

            stats['avg_experts_used'] += n_used
            if collect_detailed_stats:
                stats['per_sample_experts'].append(n_used)
                stats['per_sample_gflops'].append(sample_gflops)
                stats['per_sample_correct'].append(final_pred == targets[idx])

            if final_pred == targets[idx]:
                correct += 1

        stats['avg_experts_used'] /= n_samples
        if collect_detailed_stats:
            stats['exit_distribution'] = dict(stats['exit_distribution'])
            stats['experts_usage']     = dict(stats['experts_usage'])

        return correct / n_samples, total_gflops / n_samples, dict(stats)

    # ----- Per-sample helper -------------------------------------------------

    def _infer_one_sample(
        self,
        idx:                    int,
        data:                   ExpertDataDict,
        params:                 CADSParams,
        stats:                  Dict[str, Any],
        collect_detailed_stats: bool,
    ) -> Tuple[int, float, int]:
        """Walk a single sample through the cascade.

        Returns
        -------
        final_pred
            Predicted class index.
        sample_gflops
            Cumulative GFLOPs charged on this sample (including overhead).
        n_used
            Number of experts consulted on this sample.

        Side effects on ``stats``: increments exit / category / usage
        counters used by :meth:`run` for the final report.
        """
        n_experts     = len(self.expert_names)
        used:         Set[str]                            = set()
        expert_probs: Dict[str, NDArray[np.float64]]      = {}
        predictions:  Dict[str, int]                      = {}
        final_pred:   Optional[int]                       = None
        current_expert: Optional[str]                     = self.first_expert
        category:     Optional[str]                       = None

        min_experts    = 2
        conf_threshold = params.difficult_conf
        sample_gflops  = 0.0
        ens_pred       = 0

        while current_expert is not None and len(used) < n_experts:
            used.add(current_expert)
            if collect_detailed_stats:
                stats['experts_usage'][current_expert] += 1

            expert_cost = self.expert_gflops[current_expert]
            sample_gflops += expert_cost
            if self.include_overhead and len(used) > 1:
                sample_gflops += ROUTING_OVERHEAD_GFLOPS

            probs = data[current_expert][idx]
            pred  = int(np.argmax(probs))
            expert_probs[current_expert] = probs
            predictions[current_expert]  = pred

            # Consensus class of votes so far, used to look up local
            # ensemble weights.
            consensus = max(
                set(predictions.values()),
                key=list(predictions.values()).count,
            )
            ensemble = self._compute_ensemble(expert_probs, consensus, params)
            ens_pred = int(np.argmax(ensemble))
            ens_conf = float(ensemble[ens_pred])

            pred_set = self.conformal.get_prediction_set(
                ensemble, params.use_class_quantiles
            )
            set_size = len(pred_set)

            # ----- Set category and base threshold on first iteration ----
            if category is None:
                if set_size == 1:
                    category       = 'singleton'
                    min_experts    = params.min_experts_singleton
                    conf_threshold = params.singleton_conf
                elif set_size == 2:
                    category       = 'binary'
                    min_experts    = params.min_experts_binary
                    conf_threshold = params.binary_conf
                else:
                    category       = 'difficult'
                    min_experts    = params.min_experts_difficult
                    conf_threshold = params.difficult_conf

            # ----- Consensus boost (eq. 12) ------------------------------
            if len(predictions) >= 2:
                agreeing = sum(1 for p in predictions.values() if p == ens_pred)
                agreement = agreeing / len(predictions)
                if agreement >= CONSENSUS_AGREEMENT_RATIO:
                    boost = min(
                        params.conf_boost_per_expert * (len(used) - 1),
                        params.max_conf_boost,
                    )
                    conf_threshold = max(
                        conf_threshold - boost, EXIT_THRESHOLD_FLOOR
                    )

            # ----- Hard cap: all experts consulted -----------------------
            if len(used) >= n_experts:
                final_pred = ens_pred
                stats['all_experts'] += 1
                if collect_detailed_stats:
                    stats['exit_distribution']['all_experts'] += 1
                break

            # ----- Exit test ---------------------------------------------
            if len(used) >= min_experts:
                difficulty   = self.conformal.get_difficulty(ens_pred)
                adjusted_thr = min(
                    conf_threshold + (difficulty - 0.5) * CLASS_DIFFICULTY_COEFF,
                    EXIT_THRESHOLD_CAP,
                )
                if ens_conf >= adjusted_thr:
                    recent = list(predictions.values())[
                        -min(RECENT_AGREEMENT_WINDOW, len(predictions)):
                    ]
                    if all(p == ens_pred for p in recent):
                        final_pred = ens_pred
                        stats[f'{category}_exit'] += 1
                        if collect_detailed_stats:
                            stats['exit_distribution'][
                                f'{category}_exit_{len(used)}'
                            ] += 1
                        break

            # ----- Continue: pick next expert ----------------------------
            current_expert = self._select_next_expert(
                used, current_expert, ensemble, params
            )

        if final_pred is None:
            final_pred = ens_pred

        return final_pred, sample_gflops, len(used)
