#!/usr/bin/env python3
"""
generate_cache.py
=================

Utility to build a CADS-compatible prediction cache from a trained
classifier.

CADS does not train models — it consumes precomputed softmax
predictions stored as ``.npz`` files. This script is a reference
implementation showing exactly what shape, dtype, and metadata the
``.npz`` files must contain. Adapt it to your training framework: the
only contract that matters is the resulting ``.npz`` schema, not the
code that produces it.

Cache schema (mandatory)
------------------------
::

    probs   : float32 or float64, shape (N, n_classes), softmax outputs
              on the dataset's held-out test split. Rows must sum to 1
              within numerical tolerance.
    labels  : int32 or int64, shape (N,), ground-truth labels in
              [0, n_classes).

Critical invariants across experts
----------------------------------
For a given dataset, all expert caches must share:

  * the **same** N samples,
  * in the **same** order.

If you generate caches with different DataLoaders, double-check that
``shuffle=False`` is set everywhere.

Usage
-----
This file is intentionally minimal. The function
:func:`save_prediction_cache` takes precomputed arrays and writes the
``.npz`` in the expected format. The accompanying example
(:func:`example_pytorch_pipeline`) shows the typical PyTorch loop that
produces those arrays.
"""
from __future__ import annotations

from pathlib import Path
from typing  import Optional

import numpy as np


# =============================================================================
# CORE: WRITE THE CACHE FILE
# =============================================================================

def save_prediction_cache(
    probs:     np.ndarray,
    labels:    np.ndarray,
    out_path:  str | Path,
    n_classes: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    """Validate and write a CADS-compatible ``.npz`` prediction cache.

    Parameters
    ----------
    probs
        Float array of shape ``(N, n_classes)`` containing per-sample
        softmax distributions. Rows must sum to 1 within ``1e-4`` and
        all entries must be non-negative.
    labels
        Integer array of shape ``(N,)`` with ground-truth class indices.
    out_path
        Destination path. Parent directories are created if needed.
    n_classes
        Optional explicit class count for an extra sanity check
        against ``probs.shape[1]``.
    overwrite
        If False (default), refuses to overwrite an existing file.

    Returns
    -------
    out_path
        The resolved Path object.

    Raises
    ------
    ValueError
        If the input arrays violate the cache schema.
    FileExistsError
        If the target exists and ``overwrite`` is False.
    """
    probs  = np.asarray(probs)
    labels = np.asarray(labels)

    # ----- Shape and dtype --------------------------------------------------
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape {probs.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D, got shape {labels.shape}")
    if probs.shape[0] != labels.shape[0]:
        raise ValueError(
            f"probs and labels disagree on N: "
            f"{probs.shape[0]} vs {labels.shape[0]}"
        )
    if not np.issubdtype(probs.dtype, np.floating):
        raise ValueError(f"probs dtype must be float, got {probs.dtype}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels dtype must be int, got {labels.dtype}")

    nc_from_probs = probs.shape[1]
    if n_classes is not None and nc_from_probs != n_classes:
        raise ValueError(
            f"probs has {nc_from_probs} classes but n_classes={n_classes}"
        )

    # ----- Numerical sanity -------------------------------------------------
    if (probs < 0).any():
        raise ValueError("probs contains negative entries")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        max_dev = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(
            f"probs rows do not sum to 1 (max deviation {max_dev:.3e}). "
            f"Did you apply a softmax?"
        )

    # ----- Labels in range --------------------------------------------------
    if labels.min() < 0:
        raise ValueError(f"labels has negative value: {labels.min()}")
    if labels.max() >= nc_from_probs:
        raise ValueError(
            f"labels.max() = {labels.max()} >= n_classes = {nc_from_probs}"
        )

    # ----- IO ---------------------------------------------------------------
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists; pass overwrite=True to replace it"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, probs=probs.astype(np.float32), labels=labels.astype(np.int64))

    print(f"Wrote {out_path}")
    print(f"  shape   = {probs.shape}")
    print(f"  classes = {nc_from_probs}")
    print(f"  accuracy on this cache = {(probs.argmax(axis=1) == labels).mean()*100:.2f}%")
    return out_path


# =============================================================================
# EXAMPLE: PYTORCH INFERENCE LOOP
# =============================================================================
# The function below is illustrative. It is not imported anywhere else
# and is meant to be copy-pasted into your own training script and
# adapted to your model and dataset.
# =============================================================================

def example_pytorch_pipeline() -> None:  # pragma: no cover
    """Reference PyTorch loop that produces a CADS-compatible cache.

    Replace the dataset, the model, and the device with your own.
    The critical detail is ``shuffle=False`` in the DataLoader: every
    expert must see the test set in the same order.
    """
    # The imports below are guarded inside the function so the rest of
    # this script remains import-light. Importing torch is not needed
    # if you only want to call save_prediction_cache from your own
    # framework.
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        raise SystemExit(
            "example_pytorch_pipeline requires torch. "
            "Install it or adapt this code to your framework."
        )

    # >>> Replace these three placeholders >>>
    model       = ...   # your trained nn.Module, set to .eval()
    test_set    = ...   # your test torch.utils.data.Dataset
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_path    = "cache/pathmnist/pathmnist_my_expert_predictions.npz"
    n_classes   = 9
    # <<< Replace these three placeholders <<<

    if isinstance(model, type(Ellipsis)) or isinstance(test_set, type(Ellipsis)):
        raise SystemExit(
            "Edit example_pytorch_pipeline() to plug in your model and dataset."
        )

    model.eval()
    all_probs, all_labels = [], []

    loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    probs  = np.concatenate(all_probs,  axis=0).astype(np.float32)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)

    save_prediction_cache(probs, labels, out_path, n_classes=n_classes)


# =============================================================================
# CLI: VALIDATE OR INSPECT AN EXISTING CACHE
# =============================================================================

def _cli() -> None:
    """Minimal CLI: inspect or validate an existing .npz cache."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Inspect or validate a CADS prediction cache."
    )
    parser.add_argument('cache_path', type=str,
                        help='Path to a .npz cache file')
    args = parser.parse_args()

    npz = np.load(args.cache_path)
    keys = list(npz.keys())
    print(f"Keys: {keys}")

    probs_key = 'probs' if 'probs' in npz else 'probabilities'
    if probs_key not in npz or 'labels' not in npz:
        raise SystemExit(
            f"Cache must contain 'probs' (or 'probabilities') and 'labels'; "
            f"got {keys}"
        )

    probs  = npz[probs_key]
    labels = npz['labels']

    print(f"\n{probs_key:10s}: shape={probs.shape}  dtype={probs.dtype}")
    print(f"labels    : shape={labels.shape}  dtype={labels.dtype}")
    print(f"\nrow sums  : min={probs.sum(axis=1).min():.6f} "
          f"max={probs.sum(axis=1).max():.6f}")
    print(f"label range: [{labels.min()}, {labels.max()}]")
    print(f"accuracy  : {(probs.argmax(axis=1) == labels).mean()*100:.2f}%")


if __name__ == "__main__":
    _cli()
