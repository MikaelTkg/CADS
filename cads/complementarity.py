"""
cads.complementarity
====================

Empirical inter-expert complementarity scores.

For an ordered pair of experts (A, B), the complementarity of B with
respect to A is the empirical probability that B is correct on the
samples where A is wrong:

    Comp(A, B) = P[ y_hat(B) = y  |  y_hat(A) != y ]

CADS uses these scores to drive the next-expert selection: after expert
A has been consulted, the candidate B that best "patches" A's known
failure modes is preferred over the cheapest unused expert.

Three levels of granularity are computed, each conditioned on
increasingly specific failure modes:

  1. **Global** — averaged over every sample on which A is wrong.
  2. **Per predicted class** — restricted to samples where A's wrong
     prediction was a specific class ``c``.
  3. **Per confused class pair** — restricted to samples where A
     confused two specific classes ``(c1, c2)``.

The class-pair score is the strongest signal but suffers from sparsity
on rare classes; missing entries fall back to coarser estimates.

All scores must be computed on ``D_opt`` only. Computing them on the
full cache leaks test information into the routing decisions.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from .config import Config
from .data   import ExpertDataDict, LabelArray


# =============================================================================
# CONSTANTS
# =============================================================================

#: Minimum number of samples required to estimate a class-conditional
#: complementarity score. Below this, the estimator falls back to the
#: global score for the same expert pair.
MIN_SAMPLES_CLASS_COMP: int = 5

#: Minimum number of samples required to estimate a class-pair
#: complementarity score. Below this, the estimator falls back to the
#: average of the two class-conditional scores.
MIN_SAMPLES_PAIR_COMP: int = 3


# =============================================================================
# ANALYZER
# =============================================================================

class ComplementarityAnalyzer:
    """Compute and serve empirical complementarity scores.

    Three nested dictionaries are populated by :meth:`analyze`:

      * ``global_complementarity[(a, b)]`` — float in [0, 1]
      * ``class_complementarity[(a, b, c)]`` — float in [0, 1]
      * ``pair_complementarity[(a, b, c1, c2)]`` — float in [0, 1],
        with ``c1 < c2``.

    All scores are queried through :meth:`get_score`, which automatically
    selects the finest level available given the query and falls back
    to coarser levels when no data is available.
    """

    def __init__(self, config: Config) -> None:
        """Prepare an empty analyser bound to a run-time configuration.

        Parameters
        ----------
        config
            Run configuration. Only ``n_classes`` and the expert list
            are used; the per-pair score dictionaries are populated
            later by :meth:`analyze`.
        """
        self.config       = config
        self.n_classes    = config.n_classes
        self.expert_names = [e['name'] for e in config.experts]

        self.global_complementarity: Dict[Tuple[str, str], float]              = {}
        self.class_complementarity:  Dict[Tuple[str, str, int], float]         = {}
        self.pair_complementarity:   Dict[Tuple[str, str, int, int], float]    = {}

    # ----- Estimation --------------------------------------------------------

    def analyze(self, data: ExpertDataDict, targets: LabelArray) -> None:
        """Estimate all three levels of complementarity on the given subset.

        Must be called on ``D_opt`` only. Calling on the full cache or
        on ``D_test`` leaks test information into the routing
        decisions.

        Parameters
        ----------
        data
            Mapping ``{expert_name: probs_(N, n_classes)}`` restricted
            to the optimisation subset ``D_opt``.
        targets
            Ground-truth labels of shape ``(N,)`` aligned with ``data``.
        """
        predictions = {n: data[n].argmax(axis=1) for n in self.expert_names}
        correct     = {n: predictions[n] == targets for n in self.expert_names}

        for a in self.expert_names:
            a_wrong = ~correct[a]

            for b in self.expert_names:
                if a == b:
                    continue

                # ----- Level 1: global -----------------------------------
                self.global_complementarity[a, b] = (
                    float(correct[b][a_wrong].mean()) if a_wrong.sum() else 0.5
                )

                # ----- Level 2 & 3: class- and pair-conditional ----------
                for c in range(self.n_classes):
                    # Class-conditional: where A predicted c (wrongly)
                    mask_c = (targets == c) & a_wrong
                    if mask_c.sum() >= MIN_SAMPLES_CLASS_COMP:
                        self.class_complementarity[a, b, c] = float(
                            correct[b][mask_c].mean()
                        )
                    else:
                        self.class_complementarity[a, b, c] = (
                            self.global_complementarity[a, b]
                        )

                    # Pair-conditional: A confused c and c2
                    for c2 in range(c + 1, self.n_classes):
                        mask_1  = (targets == c)  & (predictions[a] == c2)
                        mask_2  = (targets == c2) & (predictions[a] == c)
                        mask_12 = mask_1 | mask_2
                        if mask_12.sum() >= MIN_SAMPLES_PAIR_COMP:
                            self.pair_complementarity[a, b, c, c2] = float(
                                correct[b][mask_12].mean()
                            )
                        else:
                            fallback_c  = self.class_complementarity.get(
                                (a, b, c),  self.global_complementarity[a, b]
                            )
                            fallback_c2 = self.class_complementarity.get(
                                (a, b, c2), self.global_complementarity[a, b]
                            )
                            self.pair_complementarity[a, b, c, c2] = (
                                0.5 * (fallback_c + fallback_c2)
                            )

    # ----- Query -------------------------------------------------------------

    def get_score(
        self,
        from_exp:      str,
        to_exp:        str,
        pred_class:    Optional[int] = None,
        confused_pair: Optional[Tuple[int, int]] = None,
    ) -> float:
        """Return Comp(from_exp -> to_exp), finest level available.

        The lookup order is: class-pair, then per-class, then global,
        then a neutral fallback of 0.5.

        Parameters
        ----------
        from_exp, to_exp
            Expert names.
        pred_class
            Class currently predicted by ``from_exp``. Used to select
            the per-class score when no confused pair is provided.
        confused_pair
            Tuple of two class indices identifying a known confusion
            of ``from_exp``. Triggers the per-pair lookup. Order is
            irrelevant: the analyzer stores pairs canonically with
            ``c1 < c2``.

        Returns
        -------
        score
            Estimated probability in [0, 1] that ``to_exp`` is correct
            on the failure modes of ``from_exp``.
        """
        if from_exp == to_exp:
            return 0.0

        if confused_pair is not None:
            c1, c2 = min(confused_pair), max(confused_pair)
            return self.pair_complementarity.get(
                (from_exp, to_exp, c1, c2),
                self.global_complementarity.get((from_exp, to_exp), 0.5),
            )

        if pred_class is not None:
            return self.class_complementarity.get(
                (from_exp, to_exp, pred_class),
                self.global_complementarity.get((from_exp, to_exp), 0.5),
            )

        return self.global_complementarity.get((from_exp, to_exp), 0.5)
