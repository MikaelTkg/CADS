"""
CADS: Conformal Adaptive Decision System.

A cost-aware multi-expert cascade for image classification with
conformal-prediction-based exit decisions.

Reference
---------
Turkoglu, M., Bary, T., Thielens, V., Dausort, M., Macq, B. (2026).
CADS: Conformal Adaptive Decision System for Cost-Efficient Image
Classification. IEEE ICIP.

Public API
----------
Pipeline primitives (typical user-facing imports):

    from cads import (
        Config,
        PredictionCache,
        analyze_experts,
        ComplementarityAnalyzer,
        CADSEngine,
        CADSParams,
        process_single_budget,
        compute_baselines,
    )

Registries to extend for new datasets or new experts:

    from cads.config import DATASET_CONFIGS, EXPERT_GFLOPS
"""
from .config          import (
    Config,
    DATASET_CONFIGS,
    EXPERT_GFLOPS,
    MIN_CAL_SIZE,
    ROUTING_OVERHEAD_GFLOPS,
)
from .data            import (
    PredictionCache,
    ExpertDataDict,
    LabelArray,
    ProbabilityArray,
)
from .profiling       import ExpertProfile, analyze_experts
from .complementarity import ComplementarityAnalyzer
from .conformal       import ClassConformalPredictor
from .engine          import CADSEngine, CADSParams
from .optimization    import optimize_for_budget, process_single_budget
from .baselines       import compute_baselines, get_best_baseline_at_budget
from .io_utils        import build_output, save_results, convert_numpy

__all__ = [
    # Configuration
    'Config',
    'DATASET_CONFIGS',
    'EXPERT_GFLOPS',
    'MIN_CAL_SIZE',
    'ROUTING_OVERHEAD_GFLOPS',
    # Data
    'PredictionCache',
    'ExpertDataDict',
    'LabelArray',
    'ProbabilityArray',
    # Profiling
    'ExpertProfile',
    'analyze_experts',
    # Complementarity
    'ComplementarityAnalyzer',
    # Conformal
    'ClassConformalPredictor',
    # Engine
    'CADSEngine',
    'CADSParams',
    # Optimisation
    'optimize_for_budget',
    'process_single_budget',
    # Baselines
    'compute_baselines',
    'get_best_baseline_at_budget',
    # IO
    'build_output',
    'save_results',
    'convert_numpy',
]

__version__ = '1.0.0'
