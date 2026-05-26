"""Shared fixtures and import-path setup for the test suite."""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so ``from cads import ...`` works
# without requiring an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from cads import Config


@pytest.fixture
def pathmnist_pool():
    """Three real PathMNIST experts spanning the cost spectrum.

    Loaded from the shipped ``cache/pathmnist/`` directory — no synthetic
    data, no network access. Picks one cheap, one mid-range, and one
    expensive expert so the cascade has a meaningful cost gradient.

    Returns
    -------
    data    : dict[str, np.ndarray]    ``{name: probs_(N, 9)}``
    targets : np.ndarray                integer labels, shape ``(N,)``
    config  : Config                    matching three-expert config
    """
    pool = [
        ('mobilenet_tiny',  0.024),
        ('eva02_tiny',      1.700),
        ('convnextv2_base', 15.380),
    ]
    cache_dir = REPO_ROOT / 'cache' / 'pathmnist'

    data: dict[str, np.ndarray] = {}
    targets = None
    for name, _ in pool:
        z = np.load(cache_dir / f'pathmnist_{name}_predictions.npz')
        data[name] = z['probs'].astype(np.float64)
        targets    = z['labels'].astype(np.int64)

    config = Config(
        dataset='pathmnist', n_classes=9, n_trials=10,
        experts=[{'name': n, 'gflops': g} for n, g in pool],
        include_routing_overhead=True,
        val_ratio=0.7, cal_ratio=0.3, seed=0,
    )
    return data, targets, config
