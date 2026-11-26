"""
precision -- Multi-precision quantization, 2:4 structured sparsity, and mixed-precision guards.

Models:
  - FP32, TF32, BF16, FP16, FP8, INT8, INT4 (W4A16) with optional 2:4 structured sparsity
  - High-precision guard: LayerNorm, RMSNorm, Softmax, BatchNorm, GroupNorm, and residual adds
    stay at >= BF16/FP32 dense so numerical stability is preserved
  - Group-quantization scale metadata overhead (2 bytes per `group_size` weight elements)
  - 2:4 sparsity weight compression (0.5x weight + 2-bit/4-elem index metadata = +0.125 B/elem)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .autodiff_bounds import ComputeDAG, OpNode, dag_from_torch_module
from .devices import GPU, _dtype_key, dtype_bytes

_SENSITIVE_OPS = (
    "layer_norm",
    "native_layer_norm",
    "rms_norm",
    "batch_norm",
    "native_batch_norm",
    "group_norm",
    "native_group_norm",
    "instance_norm",
    "softmax",
    "_softmax",
    "_safe_softmax",
    "log_softmax",
    "add",
    "add_",
    "embedding",
)

_QUANT_ELIGIBLE_KINDS = ("gemm", "conv")


def is_sensitive_op(name: str, kind: str) -> bool:
    """Return True if op must stay at high precision (>= bf16/fp32) and dense."""
    low = name.lower()
    if any(tok in low for tok in ("norm", "softmax", "embedding")):
        return True
    if low in _SENSITIVE_OPS:
        return True
    if kind == "reduce" and low in ("sum", "mean", "var", "std"):
        return True
    return False


@dataclass(frozen=True)
class PrecisionSpec:
    fmt: str = "bf16"
    sparse_24: bool = False
    group_size: int = 128
    keep_norms_high_precision: bool = True
    norm_dtype: str = "bf16"

    @property
    def key(self) -> str:
        return _dtype_key(self.fmt)

    @property
    def label(self) -> str:
        tag = self.key.upper()
        if self.key in ("fp8", "int8", "int4") and self.group_size > 0:
            tag += f"(g{self.group_size})"
        if self.sparse_24:
            tag += "+2:4"
        return tag

    def effective_norm_dtype(self) -> str:
        if self.key in ("fp32", "fp64"):
            return self.key
        return _dtype_key(self.norm_dtype) if _dtype_key(self.norm_dtype) in ("fp32", "bf16", "fp16") else "bf16"

    def weight_bytes_per_elem(self) -> float:
        k = self.key
        base = float(dtype_bytes(k))
        if self.sparse_24 and k not in ("fp32", "fp64"):
            # 2:4 keeps 50% weights + 2-bit index per nonzero (0.125 B per original elem)
            base = 0.5 * base + 0.125
        if k in ("fp8", "int8", "int4") and self.group_size > 0:
            # FP16/BF16 group scale = 2 bytes per group_size elements
            base += 2.0 / float(self.group_size)
        return base

    def act_bytes_per_elem(self) -> float:
        k = self.key
        if k == "int4":
            # W4A16: activations stay in BF16/FP16
            return 2.0
        return float(dtype_bytes(k))


DEFAULT_PRECISION_FORMATS: tuple[PrecisionSpec, ...] = (
    PrecisionSpec("fp32"),
    PrecisionSpec("tf32"),
    PrecisionSpec("bf16"),
    PrecisionSpec("fp8"),
    PrecisionSpec("int8"),
    PrecisionSpec("int4", group_size=128),
    PrecisionSpec("bf16", sparse_24=True),
    PrecisionSpec("fp8", sparse_24=True),
    PrecisionSpec("int4", sparse_24=True, group_size=128),
)


def _node_precision_cost(
    node: OpNode,
    spec: PrecisionSpec,
    gpu: GPU,
) -> tuple[float, float, float, bool]:
    """Return (t_node_s, q_node_bytes, w_node_flops, is_high_prec_guarded) for one OpNode."""
    base_b = 4.0  # DAG is traced in fp32
    sensitive = is_sensitive_op(node.name, node.kind) and spec.keep_norms_high_precision

    if sensitive:
        nd = spec.effective_norm_dtype()
        f_op = gpu.peak_flops(nd, sparse_24=False)
        b_scale = float(dtype_bytes(nd)) / base_b
        q_op = max(node.Q_fwd, node.Q_hk_fwd) * b_scale
        w_op = node.W_fwd
        t_op = max(w_op / f_op, q_op / gpu.BW, node.tau_fwd * gpu.tau_graph)
        return t_op, q_op, w_op, True

    if node.kind in _QUANT_ELIGIBLE_KINDS:
        f_op = gpu.peak_flops(spec.key, sparse_24=spec.sparse_24)
        act_in = float(node.attrs.get("act_in_bytes", 0.0))
        param_in = float(node.attrs.get("param_bytes", 0.0))
        out_b = float(node.attrs.get("out_bytes", 0.0))
        if act_in + param_in + out_b > 0.0:
            q_act = (act_in + out_b) * (spec.act_bytes_per_elem() / base_b)
            q_w = param_in * (spec.weight_bytes_per_elem() / base_b)
            q_compulsory = q_act + q_w
            hk_ratio = max(node.Q_hk_fwd / max(node.Q_fwd, 1.0), 1.0)
            q_op = q_compulsory * hk_ratio
        else:
            q_op = max(node.Q_fwd, node.Q_hk_fwd) * (spec.weight_bytes_per_elem() / base_b)
        w_op = node.W_fwd
        t_op = max(w_op / f_op, q_op / gpu.BW, node.tau_fwd * gpu.tau_graph)
        return t_op, q_op, w_op, False

    # Other ops (e.g., SDPA, activations): use dense format >= bf16 if sub-byte/int
    op_fmt = spec.key if spec.key in ("fp32", "tf32", "bf16", "fp16", "fp8") else "bf16"
    f_op = gpu.peak_flops(op_fmt, sparse_24=False)
    b_scale = float(dtype_bytes(op_fmt)) / base_b
    q_op = max(node.Q_fwd, node.Q_hk_fwd) * b_scale
    w_op = node.W_fwd
    t_op = max(w_op / f_op, q_op / gpu.BW, node.tau_fwd * gpu.tau_graph)
    return t_op, q_op, w_op, False


def _evaluate_precision_spec(dag: ComputeDAG, spec: PrecisionSpec, gpu: GPU) -> dict[str, Any]:
    t_stage_s = 0.0
    q_tot_bytes = 0.0
    w_tot_flops = 0.0
    t_comp_sum = 0.0
    t_mem_sum = 0.0
    t_ser_sum = 0.0
    high_prec_ops = 0
    quantized_ops = 0

    for node in dag.nodes:
        t_op, q_op, w_op, guarded = _node_precision_cost(node, spec, gpu)
        t_stage_s += t_op
        q_tot_bytes += q_op
        w_tot_flops += w_op
        if guarded:
            high_prec_ops += 1
        elif node.kind in _QUANT_ELIGIBLE_KINDS:
            quantized_ops += 1
        # Track aggregate component contributions
        if guarded:
            f_op = gpu.peak_flops(spec.effective_norm_dtype(), sparse_24=False)
        elif node.kind in _QUANT_ELIGIBLE_KINDS:
            f_op = gpu.peak_flops(spec.key, sparse_24=spec.sparse_24)
        else:
            op_fmt = spec.key if spec.key in ("fp32", "tf32", "bf16", "fp16", "fp8") else "bf16"
            f_op = gpu.peak_flops(op_fmt, sparse_24=False)
        t_comp_sum += w_op / f_op
        t_mem_sum += q_op / gpu.BW
        t_ser_sum += node.tau_fwd * gpu.tau_graph

    # Compulsory input/output + parameter floor
    act_scale = spec.act_bytes_per_elem() / 4.0
    w_scale = spec.weight_bytes_per_elem() / 4.0
    q_min_io = (dag.input_bytes + dag.output_bytes) * act_scale + dag.param_bytes * w_scale
    t_abs_s = max(t_comp_sum, max(q_min_io, 0.0) / gpu.BW, t_ser_sum if dag.nodes else 0.0)
    t_stage_s = max(t_stage_s, t_abs_s)

    binding = max(
        {"compute": t_comp_sum, "memory": t_mem_sum, "serial": t_ser_sum}.items(),
        key=lambda kv: kv[1],
    )[0]

    return {
        "format": spec.key,
        "label": spec.label,
        "sparse_24": spec.sparse_24,
        "group_size": spec.group_size,
        "norm_dtype": spec.effective_norm_dtype() if spec.keep_norms_high_precision else spec.key,
        "T_fwd_us": t_stage_s * 1e6,
        "T_abs_us": t_abs_s * 1e6,
        "W_GFLOP": w_tot_flops / 1e9,
        "Q_MB": q_tot_bytes / 1e6,
        "binding": binding,
        "high_prec_ops": high_prec_ops,
        "quantized_ops": quantized_ops,
    }


def precision_sweep(
    model: torch.nn.Module,
    sample_input: Any,
    gpu: GPU,
    formats: Sequence[PrecisionSpec | str] | None = None,
) -> list[dict[str, Any]]:
    """Sweep precision & 2:4 sparsity configurations on a PyTorch model (zero GPU memory)."""
    specs: list[PrecisionSpec] = []
    for item in (formats if formats is not None else DEFAULT_PRECISION_FORMATS):
        if isinstance(item, PrecisionSpec):
            specs.append(item)
        else:
            specs.append(PrecisionSpec(str(item)))

    dag = dag_from_torch_module(model, sample_input, dtype="fp32", gpu=gpu)
    results = [_evaluate_precision_spec(dag, sp, gpu) for sp in specs]

    # Compute speedup relative to fp32 baseline (or first entry if fp32 not in list)
    fp32_entry = next((r for r in results if r["format"] == "fp32" and not r["sparse_24"]), None)
    if fp32_entry is None:
        fp32_entry = _evaluate_precision_spec(dag, PrecisionSpec("fp32"), gpu)
    base_us = max(fp32_entry["T_fwd_us"], 1e-12)
    for r in results:
        r["speedup_vs_fp32"] = base_us / max(r["T_fwd_us"], 1e-12)
    return results


def print_precision_sweep(results: Sequence[dict[str, Any]], title: str | None = None) -> None:
    """Pretty-print precision sweep table."""
    if title:
        print(title)
    print(
        "%-14s %-7s %10s %10s %9s %8s %-8s %s"
        % ("Format", "2:4", "T_fwd(us)", "T_abs(us)", "Q(MB)", "vs_FP32", "Binding", "HighPrecOps")
    )
    print("-" * 82)
    for r in results:
        print(
            "%-14s %-7s %10.2f %10.2f %9.3f %7.2fx %-8s %d (norm=%s)"
            % (
                r["label"],
                "yes" if r["sparse_24"] else "no",
                r["T_fwd_us"],
                r["T_abs_us"],
                r["Q_MB"],
                r["speedup_vs_fp32"],
                r["binding"],
                r["high_prec_ops"],
                r["norm_dtype"],
            )
        )
