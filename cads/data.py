"""
cads.data
=========

Prediction-cache discovery, loading, and stratified three-way split.

CADS operates on cached expert predictions stored as NumPy ``.npz``
archives, one per (dataset, expert) pair. This module reads those
caches and partitions them into the three disjoint subsets used by
the rest of the pipeline (D_opt, D_cal, D_test).

Expected ``.npz`` schema
------------------------
Each archive must contain at least:

  * ``probs`` (or ``probabilities``): float array of shape
    ``(N, n_classes)``. Softmax outputs on the dataset's held-out test
    split. Rows must sum to 1.0 within numerical tolerance.
  * ``labels``: integer array of shape ``(N,)``.
    Ground-truth class indices in ``[0, n_classes)``.

All caches for a given dataset must be evaluated on the **same**
samples in the **same** order. Otherwise the stratified split will
silently pair predictions from different inputs.

File naming
-----------
The cache loader tries the following patterns, in order, under both
``CADS_CACHE_PATH`` and ``CADS_CACHE_PATH/<dataset>``:

  * ``<dataset>_<expert>_predictions.npz``
  * ``<expert>_<dataset>_predictions.npz``
  * ``<expert>_predictions.npz``
  * ``<dataset>_<expert>.npz``
  * ``<expert>_<dataset>.npz``

Cache location
--------------
Read from the ``CADS_CACHE_PATH`` environment variable (default
``./cache``). To use a custom layout, set this variable before running
or override the path manually when instantiating ``PredictionCache``.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .config import Config, MIN_CAL_SIZE

# =============================================================================
# TYPE ALIASES
# =============================================================================

ProbabilityArray = NDArray[np.float64]
LabelArray       = NDArray[np.int64]
ExpertDataDict   = Dict[str, ProbabilityArray]


# =============================================================================
# PREDICTION CACHE
# =============================================================================

class PredictionCache:
    """Loads expert prediction caches and produces a stratified split.

    The cache directory is read from the ``CADS_CACHE_PATH``
    environment variable, defaulting to ``./cache``. Files may live
    directly in that directory or in a ``<dataset>/`` subdirectory.
    """

    def __init__(self, config: Config) -> None:
        """Bind the cache to a run-time configuration.

        Parameters
        ----------
        config
            Run configuration. Only ``config.dataset`` and the expert
            list are used here; the cache directory is read from the
            ``CADS_CACHE_PATH`` environment variable (default ``./cache``).
        """
        cache_path = os.getenv("CADS_CACHE_PATH", "./cache")
        self.cache_dir: Path = Path(cache_path)
        self.config: Config  = config
        self.dataset: str    = config.dataset

    # ----- File discovery ----------------------------------------------------

    def _find_prediction_file(self, expert_name: str) -> Optional[Path]:
        """Locate the .npz file for a given expert, trying common
        naming patterns under several candidate directories.
        """
        patterns = [
            f"{self.dataset}_{expert_name}_predictions.npz",
            f"{expert_name}_{self.dataset}_predictions.npz",
            f"{expert_name}_predictions.npz",
            f"{self.dataset}_{expert_name}.npz",
            f"{expert_name}_{self.dataset}.npz",
        ]
        search_dirs = [
            self.cache_dir,
            self.cache_dir / self.dataset,
            Path("./cache"),
            Path("./cache") / self.dataset,
        ]
        for directory in search_dirs:
            for pattern in patterns:
                candidate = directory / pattern
                if candidate.exists():
                    return candidate
        return None

    # ----- Loading -----------------------------------------------------------

    def load_all(self) -> Tuple[ExpertDataDict, LabelArray]:
        """Load all expert caches and return their predictions plus
        the shared label vector.

        Returns
        -------
        data
            ``{expert_name: probs_(N, n_classes)}``.
        targets
            Integer label array of shape ``(N,)``.

        Raises
        ------
        FileNotFoundError
            If any expert's cache cannot be located.
        ValueError
            If no data could be loaded (empty expert list).
        """
        data: ExpertDataDict = {}
        targets: Optional[LabelArray] = None
        for expert in self.config.experts:
            path = self._find_prediction_file(expert['name'])
            if path is None:
                raise FileNotFoundError(
                    f"Cache missing for {expert['name']} on {self.dataset}. "
                    f"Searched in: {self.cache_dir}"
                )
            npz = np.load(path)
            key = "probs" if "probs" in npz else "probabilities"
            data[expert['name']] = npz[key].astype(np.float64)
            targets = npz['labels'].astype(np.int64)
        if targets is None:
            raise ValueError("No data loaded")
        return data, targets

    # ----- Stratified three-way split ---------------------------------------

    def load_split(self) -> Tuple[
        ExpertDataDict, LabelArray,    # D_opt
        ExpertDataDict, LabelArray,    # D_cal
        ExpertDataDict, LabelArray,    # D_test
    ]:
        """Build the stratified three-way split.

        With the default ``val_ratio=0.7`` and ``cal_ratio=0.3``, the
        resulting proportions are:

          * D_opt  = (1 - cal_ratio) * val_ratio   ~ 49 %
          * D_cal  = cal_ratio * val_ratio         ~ 21 %
          * D_test = 1 - val_ratio                 = 30 %

        Stratification is performed per class with a fixed RNG seed
        (``config.seed``) to avoid missing-class issues in datasets
        with high class imbalance.

        Returns
        -------
        opt_data, opt_targets, cal_data, cal_targets, test_data, test_targets
        """
        data, targets = self.load_all()
        rng = np.random.RandomState(self.config.seed)

        opt_indices:  list = []
        cal_indices:  list = []
        test_indices: list = []

        for class_id in np.unique(targets):
            idx = np.where(targets == class_id)[0]
            rng.shuffle(idx)
            n_val = int(self.config.val_ratio * len(idx))
            n_cal = int(self.config.cal_ratio * n_val)
            test_indices.extend(idx[n_val:])
            cal_indices.extend(idx[:n_val][:n_cal])
            opt_indices.extend(idx[:n_val][n_cal:])

        opt_i  = np.array(opt_indices)
        cal_i  = np.array(cal_indices)
        test_i = np.array(test_indices)

        if len(cal_i) < MIN_CAL_SIZE:
            warnings.warn(
                f"|D_cal|={len(cal_i)} < {MIN_CAL_SIZE}; conformal coverage "
                f"approximate (slack approx +/- {100.0/(len(cal_i)+1):.1f}%)."
            )

        return (
            {n: p[opt_i]  for n, p in data.items()}, targets[opt_i],
            {n: p[cal_i]  for n, p in data.items()}, targets[cal_i],
            {n: p[test_i] for n, p in data.items()}, targets[test_i],
        )
