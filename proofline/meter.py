"""
Trace a real nn.Module under FakeTensorMode (zero GPU memory, supports dynamic
control flow and dtype selection) and compute its hardware latency bounds.

    from .devices import GPU_PRESETS
    from .meter import analyze, compare_models, print_compare

    g = GPU_PRESETS["rtx5090_measured"]
    r = analyze(model, (1, 32, 256, 256), g, dtype="bf16")
    r["T_stage_fwd_us"]    stage-fused hardware lower bound (forward)
    r["T_stage_train_us"]  stage-fused hardware lower bound (fwd + bwd)
    r["T_abs_floor_fwd_us"] cross-layer persistent-kernel floor
    r["T_eager_us"]        per-kernel floor under eager PyTorch dispatch
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode


def _parse_dtype(dtype: Any) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        return torch.float32
    s = str(dtype).lower().replace("torch.", "").strip()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp64": torch.float64,
        "float64": torch.float64,
        "double": torch.float64,
    }
    if s not in mapping:
        raise ValueError(f"Unsupported dtype={dtype!r}")
    return mapping[s]


def _short_op_name(func_name: str) -> str:
    if "." in func_name:
        parts = func_name.split(".")
        return parts[1] if len(parts) > 1 else func_name
    return func_name


_VIEW_OPS = {
    "view", "_unsafe_view", "reshape", "_reshape_alias", "permute", "transpose",
    "t", "slice", "select", "narrow", "expand", "squeeze", "unsqueeze",
    "split", "split_with_sizes", "chunk", "unbind", "detach", "alias",
    "as_strided", "unfold", "flatten", "unflatten", "movedim", "swapaxes",
    "contiguous", "lift_fresh", "sym_size",
}

_SIMPLE_PW_OPS = {
    "add", "sub", "mul", "div", "rsub", "true_divide", "floor_divide",
    "neg", "abs", "relu", "leaky_relu", "hardtanh", "clamp", "clamp_min",
    "clamp_max", "where", "maximum", "minimum", "eq", "ne", "lt", "le",
    "gt", "ge", "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "logical_and", "logical_or", "logical_not", "isnan", "isinf",
}

_FMA_PW_OPS = {"lerp", "addcmul", "addcdiv"}

_TRANSCENDENTAL_PW_OPS = {
    "gelu", "silu", "sigmoid", "tanh", "exp", "exp2", "expm1", "log", "log2",
    "log10", "log1p", "rsqrt", "sqrt", "pow", "reciprocal", "softplus",
    "sin", "cos", "tan", "erf", "square",
}

_MEMORY_ONLY_BARRIER_OPS = {
    "constant_pad_nd", "reflection_pad1d", "reflection_pad2d", "reflection_pad3d",
    "replication_pad1d", "replication_pad2d", "replication_pad3d",
}

_ZERO_FLOP_PW_OPS = {
    "clone", "_to_copy", "copy_", "fill_", "zero_", "zeros_like", "ones_like",
    "full_like", "empty_like", "to", "type_as", "dropout", "native_dropout",
    "cat", "stack",
}

_REDUCE_OPS = {
    "sum", "mean", "amax", "amin", "argmax", "argmin", "prod", "cumsum",
    "cumprod", "softmax", "_softmax", "_safe_softmax", "_log_softmax",
    "native_layer_norm", "native_group_norm", "native_batch_norm",
    "rms_norm", "linalg_vector_norm", "var_mean", "std_mean", "norm",
}

_POOL_INTERP_OPS = {
    "upsample_bilinear2d", "upsample_nearest2d", "upsample_bicubic2d",
    "adaptive_avg_pool2d", "adaptive_max_pool2d", "avg_pool2d", "max_pool2d",
    "grid_sampler_2d",
}


def _classify_op(func_name: str) -> str:
    if func_name.startswith("prim."):
        return "prim"
    short = _short_op_name(func_name)
    if short in _VIEW_OPS:
        return "view"
    if "convolution" in short or short.startswith("conv"):
        return "conv"
    if short.startswith("_scaled_dot_product") or "flash_attention" in short or "efficient_attention" in short:
        return "sdpa"
    if short in ("mm", "mv", "bmm", "addmm", "baddbmm", "matmul", "linear", "dot", "_weight_int4pack_mm"):
        return "gemm"
    if short in _POOL_INTERP_OPS or "upsample" in short or "pool" in short:
        return "pool_interp"
    if short in _REDUCE_OPS or "norm" in short or "softmax" in short:
        return "reduce"
    if short in _MEMORY_ONLY_BARRIER_OPS:
        return "spatial_mem"
    if (
        short in _SIMPLE_PW_OPS
        or short in _FMA_PW_OPS
        or short in _TRANSCENDENTAL_PW_OPS
        or short in _ZERO_FLOP_PW_OPS
    ):
        return "pointwise"
    return "other"


def _iter_tensors(obj: Any):
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_tensors(item)
    elif isinstance(obj, dict):
        for val in obj.values():
            yield from _iter_tensors(val)


def _tensor_nbytes(t: torch.Tensor) -> int:
    try:
        return int(t.numel()) * int(t.element_size())
    except Exception:
        return 0


def _prod(shape: Any) -> int:
    p = 1
    for v in shape:
        p *= int(v)
    return p


def _op_flops(func_name: str, kind: str, args: tuple[Any, ...], kwargs: dict[str, Any], out: Any) -> int:
    short = _short_op_name(func_name)
    tensors_in = [x for x in _iter_tensors(args)]
    tensors_out = [x for x in _iter_tensors(out)]
    if not tensors_out:
        return 0
    out_t = tensors_out[0]
    out_el = int(out_t.numel())

    try:
        if kind == "gemm":
            if short == "mv" and len(tensors_in) >= 2:
                return int(2 * tensors_in[0].numel())
            if short in ("mm", "matmul") and len(tensors_in) >= 2:
                a0, a1 = tensors_in[0], tensors_in[1]
                return int(2 * a0.numel() * a1.shape[-1])
            if short == "bmm" and len(tensors_in) >= 2:
                a0, a1 = tensors_in[0], tensors_in[1]
                return int(2 * a0.shape[0] * a0.shape[1] * a0.shape[2] * a1.shape[2])
            if short == "addmm" and len(tensors_in) >= 3:
                mat1, mat2 = tensors_in[1], tensors_in[2]
                return int(2 * mat1.shape[0] * mat1.shape[1] * mat2.shape[1] + out_el)
            if short == "baddbmm" and len(tensors_in) >= 3:
                b1, b2 = tensors_in[1], tensors_in[2]
                return int(2 * b1.shape[0] * b1.shape[1] * b1.shape[2] * b2.shape[2] + out_el)
            if short == "linear" and len(tensors_in) >= 2:
                x, w = tensors_in[0], tensors_in[1]
                return int(2 * (x.numel() // max(1, x.shape[-1])) * x.shape[-1] * w.shape[0])

        if kind == "conv" and len(tensors_in) >= 2:
            w = tensors_in[1]
            ks = _prod(w.shape[2:]) if w.dim() >= 3 else 1
            # w.shape[1] already equals in_channels // groups in PyTorch aten.convolution
            flops = 2 * out_el * int(w.shape[1]) * ks
            return int(flops)

        if kind == "sdpa" and len(tensors_in) >= 2:
            q, k = tensors_in[0], tensors_in[1]
            b_heads = _prod(q.shape[:-2]) if q.dim() >= 2 else 1
            n_q = int(q.shape[-2])
            d = int(q.shape[-1])
            n_kv = int(k.shape[-2])
            is_causal = bool(kwargs.get("is_causal", False))
            if not is_causal and len(args) > 5 and isinstance(args[5], bool):
                is_causal = args[5]
            factor = 2 if is_causal else 4
            return int(factor * b_heads * n_q * n_kv * d)

        if kind == "pool_interp" and tensors_in:
            in_el = int(tensors_in[0].numel())
            if "upsample_bilinear" in short or "upsample_bicubic" in short:
                return int(8 * out_el)
            if "upsample_nearest" in short:
                return 0
            return int(in_el)

        if kind == "reduce" and tensors_in:
            in_el = int(tensors_in[0].numel())
            if "softmax" in short:
                return int(5 * in_el)
            if "norm" in short and "linalg_vector_norm" not in short:
                return int(6 * in_el)
            if short in ("var_mean", "std_mean", "linalg_vector_norm"):
                return int(2 * in_el)
            return int(in_el)

        if kind == "pointwise":
            if short in _ZERO_FLOP_PW_OPS:
                return 0
            if short in _FMA_PW_OPS:
                return int(2 * out_el)
            if short in _TRANSCENDENTAL_PW_OPS:
                return int(4 * out_el)
            return int(out_el)
    except Exception:
        return 0
    return 0


class CostMeter(TorchDispatchMode):
    """Dynamic operator + fused-stage HBM traffic tracer on Fake/Meta tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.rec: dict[str, dict[str, int]] = {}
        self.errors: list[tuple[str, str]] = []
        self.n_ops = 0
        self.n_kernels = 0
        self.n_view_ops = 0

        # Keep live references so Python never reuses tensor object IDs during trace.
        self._live_tensors: list[torch.Tensor] = []
        # id(tensor) -> (root_id, stage_id, nbytes)
        self._tensor_info: dict[int, tuple[int, int | None, int]] = {}
        self._root_bytes: dict[int, int] = {}

        # Per-stage stats
        self.stage_flops: list[float] = []
        self.stage_ext_read_bytes: list[int] = []
        self.stage_kinds: list[str] = []
        self._stage_read_roots: list[dict[int, int]] = []
        self._stage_escaped: list[dict[int, int]] = []
        self.external_inputs_seen: dict[int, int] = {}

    def _register_external(self, t: torch.Tensor) -> tuple[int, int | None, int]:
        tid = id(t)
        info = self._tensor_info.get(tid)
        if info is not None:
            return info
        self._live_tensors.append(t)
        nb = _tensor_nbytes(t)
        info = (tid, None, nb)
        self._tensor_info[tid] = info
        self._root_bytes[tid] = nb
        return info

    def _record_external_read(self, sid: int, root_id: int, p_sid: int | None, nb: int) -> None:
        if nb <= 0 or p_sid == sid:
            return
        max_nb = self._root_bytes.get(root_id, nb)
        already = self._stage_read_roots[sid].get(root_id, 0)
        new_total = min(max_nb, already + nb)
        delta = new_total - already
        if delta > 0:
            self._stage_read_roots[sid][root_id] = new_total
            self.stage_ext_read_bytes[sid] += delta
        if p_sid is None:
            prev_ext = self.external_inputs_seen.get(root_id, 0)
            self.external_inputs_seen[root_id] = max(prev_ext, min(max_nb, already + nb))
        else:
            # Mark producer stage output tensor as escaped so src stage counts HBM write once.
            prev_w = self._stage_escaped[p_sid].get(root_id, 0)
            self._stage_escaped[p_sid][root_id] = max(prev_w, min(max_nb, nb))

    def mark_final_outputs(self, outputs: Any) -> int:
        final_bytes = 0
        seen_roots: dict[int, int] = {}
        for t in _iter_tensors(outputs):
            info = self._tensor_info.get(id(t))
            if info is None:
                nb = _tensor_nbytes(t)
                final_bytes += nb
                continue
            root_id, p_sid, nb = info
            max_nb = self._root_bytes.get(root_id, nb)
            prev_f = seen_roots.get(root_id, 0)
            new_f = min(max_nb, prev_f + nb)
            final_bytes += max(0, new_f - prev_f)
            seen_roots[root_id] = new_f
            if p_sid is not None:
                prev_w = self._stage_escaped[p_sid].get(root_id, 0)
                self._stage_escaped[p_sid][root_id] = max(prev_w, new_f)
        return final_bytes

    def stages(self) -> list[tuple[float, int]]:
        out: list[tuple[float, int]] = []
        for sid in range(len(self.stage_flops)):
            w = self.stage_flops[sid]
            q_read = self.stage_ext_read_bytes[sid]
            q_write = sum(self._stage_escaped[sid].values())
            out.append((w, q_read + q_write))
        return out

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = str(func)
        kind = _classify_op(name)
        if kind == "prim":
            return out

        self.n_ops += 1
        in_tensors = list(_iter_tensors((args, kwargs)))
        out_tensors = list(_iter_tensors(out))
        in_infos = [self._register_external(t) for t in in_tensors]

        if kind == "view":
            self.n_view_ops += 1
            r = self.rec.setdefault(name, {"count": 0, "flops": 0, "bytes": 0})
            r["count"] += 1
            if in_infos:
                base_root, base_sid, _ = in_infos[0]
                base_max = self._root_bytes.get(base_root, 0)
                for ot in out_tensors:
                    oid = id(ot)
                    self._live_tensors.append(ot)
                    nb = _tensor_nbytes(ot)
                    # Same storage root lets stage fusion deduplicate overlapping stencil slices
                    # while capping split slices at base tensor bytes.
                    self._tensor_info[oid] = (base_root, base_sid, min(nb, base_max) if base_max else nb)
            return out

        self.n_kernels += 1
        flops = _op_flops(name, kind, args, kwargs, out)
        unfused_bytes = sum(_tensor_nbytes(t) for t in in_tensors) + sum(_tensor_nbytes(t) for t in out_tensors)
        r = self.rec.setdefault(name, {"count": 0, "flops": 0, "bytes": 0})
        r["count"] += 1
        r["flops"] += flops
        r["bytes"] += unfused_bytes

        producer_sids = [p_sid for _, p_sid, _ in in_infos if p_sid is not None]
        if kind == "pointwise" and producer_sids:
            # Topological max producer prevents fused cycles across residual skip connections.
            sid = max(producer_sids)
        else:
            sid = len(self.stage_flops)
            self.stage_flops.append(0.0)
            self.stage_ext_read_bytes.append(0)
            self.stage_kinds.append(kind)
            self._stage_read_roots.append({})
            self._stage_escaped.append({})

        self.stage_flops[sid] += float(flops)
        for root_id, p_sid, nb in in_infos:
            self._record_external_read(sid, root_id, p_sid, nb)

        # Update tracking on outputs (for in-place ops like copy_, tag target as mutated stage output)
        for ot in out_tensors:
            oid = id(ot)
            self._live_tensors.append(ot)
            nb = _tensor_nbytes(ot)
            self._tensor_info[oid] = (oid, sid, nb)
            self._root_bytes[oid] = nb

        return out


def analyze(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    gpu: Any,
    warmup: bool = True,
    quiet: bool = True,
    dtype: Any = torch.float32,
) -> dict[str, Any]:
    """Trace model on FakeTensorMode/meta without mutating caller model."""
    del warmup
    target_dtype = _parse_dtype(dtype)
    try:
        m_trace = copy.deepcopy(model).eval().to(device="meta", dtype=target_dtype)
    except Exception:
        m_trace = copy.deepcopy(model).eval()

    fwd_error = None
    cm = CostMeter()
    out_val: Any = None
    with FakeTensorMode(allow_non_fake_inputs=True):
        x = torch.empty(*input_shape, device="meta", dtype=target_dtype)
        with cm, torch.no_grad():
            try:
                out_val = m_trace(x)
            except Exception as exc:
                fwd_error = repr(exc)[:240]

    if out_val is None and fwd_error is not None:
        # Clean fallback in case custom module rejects FakeTensorMode nesting.
        cm = CostMeter()
        fwd_error = None
        x_meta = torch.empty(*input_shape, device="meta", dtype=target_dtype)
        with cm, torch.no_grad():
            try:
                out_val = m_trace(x_meta)
            except Exception as exc:
                fwd_error = repr(exc)[:240]

    final_out_bytes = cm.mark_final_outputs(out_val)
    stages = cm.stages()
    per = cm.rec
    n_ops = cm.n_ops
    n_kernels = cm.n_kernels
    n_stages = max(1, len(stages)) if n_kernels > 0 else 0

    W_total = float(sum(w for w, _ in stages))
    Q_fused = float(sum(q for _, q in stages))
    Q_unfused = float(sum(v["bytes"] for v in per.values()))
    Q_min_io = float(sum(cm.external_inputs_seen.values()) + final_out_bytes)

    f_eff = float(gpu.peak_flops(target_dtype)) if hasattr(gpu, "peak_flops") else float(gpu.F)
    bw = float(gpu.BW)
    tau_g = float(getattr(gpu, "tau_graph", 0.93e-6))
    tau_e = float(getattr(gpu, "tau_eager", gpu.t))

    t_stage_fwd = sum(max(w / f_eff, q / bw, tau_g) for w, q in stages) if stages else 0.0
    # Lower bound on backward: ~2x compute, ~2x traffic, at least same serial stage count.
    t_stage_bwd = sum(max(2.0 * w / f_eff, 2.0 * q / bw, tau_g) for w, q in stages) if stages else 0.0
    t_stage_train = t_stage_fwd + t_stage_bwd

    t_abs_compute = W_total / f_eff
    t_abs_mem = Q_min_io / bw
    t_abs_floor_fwd = max(t_abs_compute, t_abs_mem, tau_g if stages else 0.0)

    t_compute = W_total / f_eff
    t_memory = Q_fused / bw
    t_serial = n_stages * tau_g
    t_no_fusion = max(t_compute, Q_unfused / bw, n_kernels * tau_g)
    t_eager = t_stage_fwd + n_kernels * tau_e

    # Stage breakdown determines overall binding regime cleanly.
    stage_comp_tot = sum(w / f_eff for w, _ in stages)
    stage_mem_tot = sum(q / bw for _, q in stages)
    stage_ser_tot = n_stages * tau_g
    binding = max(
        {"compute": stage_comp_tot, "memory": stage_mem_tot, "serial": stage_ser_tot}.items(),
        key=lambda kv: kv[1],
    )[0]
    ai = W_total / max(Q_fused, 1.0)

    top = sorted(per.items(), key=lambda kv: -kv[1]["count"])[:12]
    out: dict[str, Any] = {
        "ok": fwd_error is None,
        "error": fwd_error,
        "dtype": str(target_dtype).replace("torch.", ""),
        "n_ops": n_ops,
        "n_kernels": n_kernels,
        "n_view_ops": cm.n_view_ops,
        "n_stages": n_stages,
        "W_GFLOP": W_total / 1e9,
        "Q_MB": Q_fused / 1e6,
        "Q_fused_MB": Q_fused / 1e6,
        "Q_unfused_MB": Q_unfused / 1e6,
        "Q_min_io_MB": Q_min_io / 1e6,
        "arithmetic_intensity": ai,
        "binding": binding,
        "T_compute_us": t_compute * 1e6,
        "T_memory_us": t_memory * 1e6,
        "T_serial_us": t_serial * 1e6,
        "T_stage_fwd_us": t_stage_fwd * 1e6,
        "T_stage_train_us": t_stage_train * 1e6,
        "T_abs_floor_fwd_us": t_abs_floor_fwd * 1e6,
        "T_no_fusion_us": t_no_fusion * 1e6,
        "T_fused_us": t_stage_fwd * 1e6,
        "T_eager_us": t_eager * 1e6,
        "fusion_gain_x": t_eager / max(t_stage_fwd, 1e-18),
        "top_ops": [(_short_op_name(k), v["count"]) for k, v in top],
        "rec": {k: dict(v) for k, v in per.items()},
        "stages": stages,
    }
    if not quiet:
        print(
            "ok=%s dtype=%s kernels=%d (views=%d) stages=%d W=%.3f GFLOP Q_fused=%.2f MB (unfused=%.2f MB) AI=%.1f FLOP/B"
            % (
                out["ok"],
                out["dtype"],
                n_kernels,
                cm.n_view_ops,
                n_stages,
                out["W_GFLOP"],
                out["Q_fused_MB"],
                out["Q_unfused_MB"],
                ai,
            )
        )
        if fwd_error:
            print("  trace error:", fwd_error)
        print(
            "  T_stage_fwd %.1f us | T_stage_train %.1f us | T_abs_floor %.1f us | T_eager %.1f us | binding=%s"
            % (
                out["T_stage_fwd_us"],
                out["T_stage_train_us"],
                out["T_abs_floor_fwd_us"],
                out["T_eager_us"],
                binding,
            )
        )
    return out


# ---------------------------------------------------------------- ranking
def compare_models(
    models: dict[str, torch.nn.Module] | list[torch.nn.Module] | tuple[torch.nn.Module, ...],
    input_shape: tuple[int, ...],
    gpu: Any,
    names: list[str] | None = None,
    use: str = "stage_fwd",
    dtype: Any = torch.float32,
) -> list[dict[str, Any]]:
    """Trace multiple models and rank by requested floor key."""
    if isinstance(models, (list, tuple)):
        names = names or [f"m{i}" for i in range(len(models))]
        models = dict(zip(names, models))

    key_map = {
        "eager": "T_eager_us",
        "fused": "T_stage_fwd_us",
        "stage_fwd": "T_stage_fwd_us",
        "stage_train": "T_stage_train_us",
        "abs_floor": "T_abs_floor_fwd_us",
    }
    sort_key = key_map.get(use, "T_stage_fwd_us")
    rows: list[dict[str, Any]] = []
    for name, m in models.items():
        r = analyze(m, input_shape, gpu, quiet=True, dtype=dtype)
        r_row = dict(r)
        r_row["name"] = name
        r_row["sort_val"] = float(r[sort_key])
        rows.append(r_row)

    rows.sort(key=lambda x: x["sort_val"])
    best = max(rows[0]["sort_val"], 1e-12) if rows else 1.0
    for r in rows:
        r["vs_best"] = r["sort_val"] / best
    return rows


def print_compare(
    models: dict[str, torch.nn.Module] | list[torch.nn.Module] | tuple[torch.nn.Module, ...],
    input_shape: tuple[int, ...],
    gpu: Any,
    names: list[str] | None = None,
    use: str = "stage_fwd",
    dtype: Any = torch.float32,
    phase: str = "both",
) -> list[dict[str, Any]]:
    rows = compare_models(models, input_shape, gpu, names=names, use=use, dtype=dtype)
    dt_str = str(_parse_dtype(dtype)).replace("torch.", "")
    print(gpu)
    print("input %s | dtype=%s | ranked by T_%s (lower = faster floor)" % (input_shape, dt_str, use))
    print()
    if phase == "forward":
        hdr = "%-16s %10s %10s %7s %7s %8s %9s %7s %8s"
        print(hdr % ("model", "T_fwd(us)", "T_eag(us)", "vsBest", "stages", "W(GF)", "Q_f(MB)", "AI", "binding"))
        print("-" * 92)
        for r in rows:
            mark = "" if r["ok"] else " *"
            print(
                "%-16s %10.1f %10.1f %6.2fx %7d %8.3f %9.2f %7.1f %8s%s"
                % (
                    r["name"][:16],
                    r["T_stage_fwd_us"],
                    r["T_eager_us"],
                    r["vs_best"],
                    r["n_stages"],
                    r["W_GFLOP"],
                    r["Q_fused_MB"],
                    r["arithmetic_intensity"],
                    r["binding"],
                    mark,
                )
            )
    else:
        hdr = "%-16s %10s %10s %10s %7s %7s %8s %9s %8s"
        print(hdr % ("model", "T_fwd(us)", "T_trn(us)", "T_eag(us)", "vsBest", "stages", "W(GF)", "Q_f(MB)", "binding"))
        print("-" * 96)
        for r in rows:
            mark = "" if r["ok"] else " *"
            print(
                "%-16s %10.1f %10.1f %10.1f %6.2fx %7d %8.3f %9.2f %8s%s"
                % (
                    r["name"][:16],
                    r["T_stage_fwd_us"],
                    r["T_stage_train_us"],
                    r["T_eager_us"],
                    r["vs_best"],
                    r["n_stages"],
                    r["W_GFLOP"],
                    r["Q_fused_MB"],
                    r["binding"],
                    mark,
                )
            )
    if any(not r["ok"] for r in rows):
        print("\n* trace partially failed (unsupported op during meta/FakeTensor trace)")
    return rows
