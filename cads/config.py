"""
cads.config
===========

Static registries and the run-time configuration dataclass.

This module is the **single point of customization** for two of the three
extension axes of CADS:

  * adding a new dataset
  * adding a new expert architecture

Both are one-line edits in the dictionaries below. The third axis
(adding a custom dataset for which CADS_CONFIGS is not enough — e.g.,
non-image classification) requires changing the cache loader in
``cads.data`` instead.

Adding a new dataset
--------------------
Append an entry to ``DATASET_CONFIGS`` with the number of classes::

    DATASET_CONFIGS['my_dataset'] = {
        'n_classes': 7,
        'description': 'My custom 7-class problem',
    }

The dataset string passed to ``--dataset`` must match the key. The
prediction cache files for this dataset must follow the naming
conventions documented in ``cads.data``.

Adding a new expert
-------------------
Append an entry to ``EXPERT_GFLOPS`` with the model's FLOPs cost in
giga (multiply-accumulates at native input resolution)::

    EXPERT_GFLOPS['my_model'] = 2.3

The expert name is then usable in ``--experts``. CADS uses the GFLOPs
value only for cost accounting and budget enforcement; any standard
profiler (fvcore, thop, ptflops) at the model's native input
resolution will give an acceptable value.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List

# =============================================================================
# EXTENSION POINT 1: REGISTERED DATASETS
# =============================================================================
# Each entry maps a dataset key (as passed to --dataset) to:
#   - n_classes:   integer number of classes
#   - description: human-readable description (printed at run start)
# To add a new dataset, append an entry here and provide the matching
# .npz prediction caches in the cache directory.
# =============================================================================

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'tissuemnist':  {'n_classes': 8,   'description': 'Tissue MNIST (8 classes)'},
    'pathmnist':    {'n_classes': 9,   'description': 'Pathology MNIST (9 classes)'},
    'bloodmnist':   {'n_classes': 8,   'description': 'Blood Cell MNIST (8 classes)'},
    'dermamnist':   {'n_classes': 7,   'description': 'Dermatoscopy MNIST (7 classes)'},
    'organamnist':  {'n_classes': 11,  'description': 'Organ-A MNIST (11 classes)'},
    'retinamnist':  {'n_classes': 5,   'description': 'Retina MNIST (5 classes)'},
    'cifar100':     {'n_classes': 100, 'description': 'CIFAR-100 (100 classes)'},
    'cifar10':      {'n_classes': 10,  'description': 'CIFAR-10 (10 classes)'},
}


# =============================================================================
# EXTENSION POINT 2: REGISTERED EXPERT MODELS
# =============================================================================
# Each entry maps an expert name (as passed to --experts) to its
# computational cost in GFLOPs (multiply-accumulates at native input
# resolution).
#
# The grouping below is informational only; CADS treats all experts
# uniformly and orders them by GFLOPs at run time.
# =============================================================================

EXPERT_GFLOPS: Dict[str, float] = {
    # Ultra-light (< 0.2 GFLOPs)
    'mobilenet_tiny':       0.024,
    'ghostnet':             0.142,

    # Light (0.2 - 1.0 GFLOPs)
    'efficientnet_lite':    0.390,
    'convnextv2_atto':      0.553,

    # Medium (1.0 - 5.0 GFLOPs)
    'eva02_tiny':           1.700,
    'mobilevit':            1.850,
    'convnextv2_tiny':      4.470,

    # Large (5.0 - 10.0 GFLOPs)
    'maxvit_tiny':          5.600,
    'swinv2_tiny':          5.960,
    'efficientnetv2':       8.420,

    # Very large (>= 10 GFLOPs)
    'convnextv2_base':      15.380,

    # Legacy entries kept for backward compatibility
    'mobilenet':            0.057,
    'yolov8n':              0.100,
    'efficientnet':         0.130,
    'resnet18':             0.148,
    'yolov8s':              0.500,
    'yolov8m':              1.200,
    'swin':                 4.508,
    'convnext':             4.470,
}


# =============================================================================
# SYSTEM CONSTANTS
# =============================================================================

#: Per-routing-step overhead, in GFLOPs, charged each time the cascade
#: queries an additional expert. Negligible relative to model FLOPs but
#: tracked for honest cost accounting.
ROUTING_OVERHEAD_GFLOPS: float = 0.0001

#: Minimum size of the conformal calibration subset (D_cal) below which
#: the marginal coverage guarantee becomes practically loose. A warning
#: is emitted when |D_cal| drops below this threshold.
MIN_CAL_SIZE: int = 20


# =============================================================================
# RUN-TIME CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Run-time configuration for one CADS invocation.

    Built from CLI arguments via :meth:`from_args`. The defaults match
    the ICIP paper protocol.

    Parameters
    ----------
    dataset
        One of the keys of :data:`DATASET_CONFIGS`.
    n_classes
        Number of classes; resolved from ``DATASET_CONFIGS[dataset]``.
    n_trials
        Number of Optuna TPE trials per budget level.
    experts
        List of dicts ``{'name': str, 'gflops': float}``, sorted by
        increasing cost. Built by :meth:`from_args`.
    include_routing_overhead
        If True, charges :data:`ROUTING_OVERHEAD_GFLOPS` per additional
        expert query beyond the first one.
    val_ratio
        Fraction of the cache allocated to D_opt + D_cal (the
        "calibration block"). The remaining 1 - val_ratio forms D_test.
    cal_ratio
        Fraction of D_opt + D_cal allocated to D_cal (the conformal
        calibration subset). D_opt gets the complement.
    seed
        RNG seed for the stratified split and TPE.
    """
    dataset: str = 'tissuemnist'
    n_classes: int = 8
    n_trials: int = 200
    experts: List[Dict[str, Any]] = field(default_factory=list)
    include_routing_overhead: bool = True
    val_ratio: float = 0.7
    cal_ratio: float = 0.3
    seed: int = 42

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Build a Config from parsed CLI arguments.

        Resolves the expert list against :data:`EXPERT_GFLOPS`, warning
        on unknown names (which default to 0.5 GFLOPs), and sorts
        experts by increasing cost so the cascade starts from the
        cheapest model.
        """
        if args.dataset not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset: {args.dataset}. "
                f"Available: {list(DATASET_CONFIGS.keys())}"
            )
        n_classes = DATASET_CONFIGS[args.dataset]['n_classes']

        if args.experts:
            expert_names = [e.strip() for e in args.experts.split(',')]
        else:
            # Default pool: six experts spanning the cost spectrum used
            # in the ICIP paper for the headline experiments.
            expert_names = [
                'mobilenet_tiny', 'ghostnet', 'efficientnet_lite',
                'eva02_tiny', 'maxvit_tiny', 'swinv2_tiny',
            ]

        experts: List[Dict[str, Any]] = []
        for name in expert_names:
            if name not in EXPERT_GFLOPS:
                print(f"  Warning: Unknown expert '{name}', defaulting to 0.5 GFLOPs")
            experts.append({
                'name':   name,
                'gflops': EXPERT_GFLOPS.get(name, 0.5),
            })
        experts.sort(key=lambda x: x['gflops'])

        return cls(
            dataset=args.dataset,
            n_classes=n_classes,
            n_trials=args.n_trials,
            experts=experts,
            include_routing_overhead=not args.no_overhead,
            val_ratio=args.val_ratio,
            cal_ratio=args.cal_ratio,
            seed=getattr(args, 'seed', 42),
        )
