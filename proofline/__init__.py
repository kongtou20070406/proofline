"""Proofline: Zero-GPU, formally verified hardware latency lower bounds for PyTorch."""

from __future__ import annotations

from .autodiff_bounds import (
    ComputeDAG,
    OpNode,
    dag_from_op,
    dag_from_torch_module,
    evaluate_dag_bounds,
)
from .devices import GPU, GPU_PRESETS, Cost, DeviceSpec, bound_by, dtype_bytes, score
from .graph_rewrites import RewriteGroup, RewriteVariant, enumerate_rewrite_candidates
from .latency_bounds import phase_report, print_phase_report, trace_stages
from .meter import analyze, compare_models, print_compare
from .precision import PrecisionSpec, precision_sweep, print_precision_sweep

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "GPU",
    "DeviceSpec",
    "GPU_PRESETS",
    "Cost",
    "score",
    "bound_by",
    "dtype_bytes",
    "PrecisionSpec",
    "precision_sweep",
    "print_precision_sweep",
    "RewriteGroup",
    "RewriteVariant",
    "enumerate_rewrite_candidates",
    "analyze",
    "compare_models",
    "print_compare",
    "phase_report",
    "print_phase_report",
    "trace_stages",
    "ComputeDAG",
    "OpNode",
    "dag_from_op",
    "dag_from_torch_module",
    "evaluate_dag_bounds",
]

try:
    from .formal_prover import (
        derive_symbolic_bounds,
        emit_lean4_certificate,
        emit_rewrite_lean4_certificate,
        prove_lower_bound_z3,
        prove_rewrite_family_z3,
    )

    __all__.extend(
        [
            "prove_lower_bound_z3",
            "prove_rewrite_family_z3",
            "derive_symbolic_bounds",
            "emit_lean4_certificate",
            "emit_rewrite_lean4_certificate",
        ]
    )
except ImportError:
    pass
