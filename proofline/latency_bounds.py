"""
理论时延下界，前向与训练都算。

支持两级追踪：
1. 对静态结构优先走 `torch.fx` 融合阶段追踪（已修复 tie-breaking、参数字节与去重流出统计）；
2. 对含动态分支 / checkpoint / 跨尺度控制流的模型（如 MG-NSI、VIVNet），自动无缝回退到
   `torch_meter.analyze`（基于 FakeTensorMode，零显存开销、不修改调用方模型）。

三个层次的界：
    T_stage    每阶段理想融合 + 硬件峰谷 + 最小核延迟 = Σ_s max(W_s/F, Q_s/B, τ_graph)
    T_abs      跨层持久核绝对下限 = max(W_tot/F, Q_min_io/B, τ_graph)
    T_eager    按实际 GPU kernel 发射的未融合评估上参考 = T_stage + n_kernels * τ_eager
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from .devices import GPU_PRESETS
from . import meter as torch_meter

MIN_KERNEL = 0.93e-6
DISPATCH_EAGER = 10.32e-6

POINTWISE = {
    "relu", "gelu", "silu", "sigmoid", "tanh", "exp", "log", "sqrt", "rsqrt",
    "neg", "abs", "pow", "clamp", "where", "masked_fill", "softmax",
    "dropout", "contiguous", "softplus", "square", "reciprocal", "lerp",
    "addcmul", "addcdiv", "maximum", "minimum", "rsub", "add", "sub", "mul", "div",
}
VIEW_OPS = {
    "permute", "transpose", "view", "reshape", "clone", "expand", "flatten",
    "unflatten", "narrow", "slice", "chunk", "getitem", "split", "unbind",
    "squeeze", "unsqueeze", "select", "detach", "_to_copy", "to", "type_as",
    "t", "alias", "as_strided", "unfold",
}


def _numel(meta: Any) -> int:
    if meta is None or not hasattr(meta, "shape"):
        return 0
    n = 1
    for v in meta.shape:
        n *= int(v)
    return n


def _nbytes(meta: Any) -> int:
    if meta is None or not hasattr(meta, "shape"):
        return 0
    try:
        itemsize = meta.dtype.itemsize if hasattr(meta.dtype, "itemsize") else 4
        return _numel(meta) * int(itemsize)
    except Exception:
        return 0


def _module_kind(mod: torch.nn.Module) -> str:
    import torch.nn as nn
    if isinstance(mod, nn.MultiheadAttention):
        return "gemm"
    if isinstance(mod, (nn.Linear, nn.Bilinear)):
        return "gemm"
    if isinstance(mod, nn.modules.conv._ConvNd):
        return "conv"
    if isinstance(
        mod,
        (
            nn.LayerNorm,
            nn.GroupNorm,
            nn.modules.batchnorm._BatchNorm,
            nn.Softmax,
        ),
    ):
        return "reduce"
    if isinstance(
        mod,
        (
            nn.modules.pooling._AdaptiveAvgPoolNd,
            nn.modules.pooling._AvgPoolNd,
            nn.modules.pooling._MaxPoolNd,
            nn.Upsample,
        ),
    ):
        return "pool_interp"
    if isinstance(mod, (nn.Dropout, nn.Identity, nn.Flatten, nn.Unflatten)):
        return "view"
    if len(list(mod.parameters(recurse=False))) == 0:
        return "pointwise"
    return "other"


def _node_name(target: Any) -> str:
    if hasattr(target, "__name__"):
        return str(target.__name__).lower()
    s = str(target).lower()
    if "." in s:
        s = s.split(".")[0]
    return s


def _kind(node: Any, mod: torch.nn.Module | None = None) -> str:
    if mod is not None:
        return _module_kind(mod)
    s = _node_name(node.target)
    if s in VIEW_OPS:
        return "view"
    if any(k in s for k in ("conv", "_convolution")):
        return "conv"
    if s in ("mm", "mv", "bmm", "addmm", "baddbmm", "linear", "matmul") or "scaled_dot" in s or "sdpa" in s:
        return "gemm"
    if any(k in s for k in ("upsample", "interpolate", "pool", "adaptive")):
        return "pool_interp"
    if any(k in s for k in ("sum", "mean", "amax", "amin", "argmax", "cumsum", "prod", "softmax", "norm")):
        return "reduce"
    if any(k in s for k in ("pad",)):
        return "spatial_mem"
    if s in POINTWISE or s in ("cat", "stack") or any(k in s for k in ("relu", "gelu", "silu", "sigmoid", "tanh", "softplus")):
        return "pointwise"
    return "other"


def _module_flops(node: Any, mod: torch.nn.Module) -> float:
    import torch.nn as nn
    ins = [p.meta.get("tensor_meta") for p in node.all_input_nodes]
    ins = [x for x in ins if x is not None and hasattr(x, "shape")]
    m = node.meta.get("tensor_meta")
    out_el = _numel(m)
    if not ins:
        return 0.0
    x = ins[0]
    tokens = _numel(x) // max(1, x.shape[-1]) if len(x.shape) else 0
    if isinstance(mod, nn.MultiheadAttention):
        d = mod.embed_dim
        n = x.shape[1] if getattr(mod, "batch_first", False) else x.shape[0]
        return float(8 * tokens * d * d + 4 * tokens * n * d)
    if isinstance(mod, nn.Linear):
        return float(2 * tokens * mod.in_features * mod.out_features)
    if isinstance(mod, nn.modules.conv._ConvNd):
        ks = 1
        for v in mod.kernel_size:
            ks *= int(v)
        return float(2 * out_el * ks * (mod.in_channels // mod.groups))
    if isinstance(mod, (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm, nn.Softmax)):
        return float(6 * _numel(x))
    if isinstance(mod, (nn.modules.pooling._AdaptiveAvgPoolNd, nn.modules.pooling._AvgPoolNd, nn.modules.pooling._MaxPoolNd)):
        return float(_numel(x))
    if isinstance(mod, nn.Upsample):
        return float(8 * out_el) if out_el else 0.0
    if isinstance(mod, (nn.ReLU, nn.LeakyReLU)):
        return float(out_el)
    if isinstance(mod, (nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.Softplus)):
        return float(4 * out_el)
    return 0.0


def _fn_flops(node: Any) -> float:
    m = node.meta.get("tensor_meta", None)
    out_el = _numel(m)
    if out_el == 0:
        return 0.0
    s = _node_name(node.target)
    k = _kind(node)
    ins = [p.meta.get("tensor_meta") for p in node.all_input_nodes]
    ins = [x for x in ins if x is not None and hasattr(x, "shape")]
    try:
        if k == "gemm" and len(ins) >= 2:
            return float(2 * out_el * ins[0].shape[-1])
        if k == "conv" and len(ins) >= 2 and len(ins[1].shape) >= 3:
            ks = 1
            for v in ins[1].shape[2:]:
                ks *= int(v)
            return float(2 * out_el * ks * ins[1].shape[1])
        if k == "pool_interp" and ins:
            return float(8 * out_el) if ("upsample" in s or "interpolate" in s) else float(_numel(ins[0]))
        if k == "reduce" and ins:
            in_el = _numel(ins[0])
            return float(5 * in_el) if "softmax" in s else float(in_el)
        if k == "pointwise":
            if any(t in s for t in ("gelu", "silu", "sigmoid", "tanh", "exp", "log", "rsqrt", "sqrt", "pow", "softplus")):
                return float(4 * out_el)
            return float(out_el)
    except Exception:
        return 0.0
    return 0.0


def _trace_stages_fx(model: torch.nn.Module, example_input: torch.Tensor) -> list[tuple[float, float]]:
    import torch.fx as fx
    from torch.fx.passes.shape_prop import ShapeProp

    m_work = copy.deepcopy(model).eval()
    target_dtype = example_input.dtype if torch.is_tensor(example_input) else torch.float32
    try:
        m_work = m_work.to(device="meta", dtype=target_dtype)
        ex_meta = torch.empty(example_input.shape, device="meta", dtype=target_dtype)
    except Exception:
        ex_meta = example_input.detach().cpu()

    graph = fx.symbolic_trace(m_work)
    ShapeProp(graph).propagate(ex_meta)

    node_root: dict[Any, Any] = {}
    stage_of: dict[Any, int | None] = {}
    W: list[float] = []
    read_bytes: list[int] = []
    read_seen: list[set[Any]] = []
    escaped_outputs: list[set[Any]] = []
    modules = dict(graph.named_modules())

    output_roots: set[Any] = set()
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "get_attr"):
            node_root[node] = node
            stage_of[node] = None
        elif node.op == "output":
            for p in node.all_input_nodes:
                output_roots.add(node_root.get(p, p))

    for node in graph.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        mod = modules.get(node.target) if node.op == "call_module" else None
        k = _kind(node, mod)
        if k == "view":
            ins = node.all_input_nodes
            if ins:
                node_root[node] = node_root.get(ins[0], ins[0])
                stage_of[node] = stage_of.get(ins[0])
            else:
                node_root[node] = node
                stage_of[node] = None
            continue

        prods = [stage_of[p] for p in node.all_input_nodes if stage_of.get(p) is not None]
        if k == "pointwise" and prods:
            sid = max(prods)
        else:
            sid = len(W)
            W.append(0.0)
            read_bytes.append(0)
            read_seen.append(set())
            escaped_outputs.append(set())

        stage_of[node] = sid
        node_root[node] = node
        W[sid] += _module_flops(node, mod) if mod is not None else _fn_flops(node)

        # Include module parameter reads once per stage execution
        if mod is not None:
            for p in mod.parameters(recurse=False):
                read_bytes[sid] += int(p.numel()) * int(p.element_size())

        for p in node.all_input_nodes:
            r = node_root.get(p, p)
            p_sid = stage_of.get(r)
            if p_sid != sid and r not in read_seen[sid]:
                read_seen[sid].add(r)
                read_bytes[sid] += _nbytes(r.meta.get("tensor_meta"))
                if p_sid is not None:
                    escaped_outputs[p_sid].add(r)

    for r in output_roots:
        p_sid = stage_of.get(r)
        if p_sid is not None:
            escaped_outputs[p_sid].add(r)

    stages: list[tuple[float, float]] = []
    for sid in range(len(W)):
        q_w = sum(_nbytes(r.meta.get("tensor_meta")) for r in escaped_outputs[sid])
        stages.append((W[sid], float(read_bytes[sid] + q_w)))
    return stages


def trace_stages(model: torch.nn.Module, example_input: Any) -> list[tuple[float, float]]:
    """返回 [(W, Q), ...]，每个融合阶段的 (FLOPs, HBM 字节)。"""
    ex_shape = tuple(example_input.shape) if torch.is_tensor(example_input) else tuple(example_input)
    ex_dtype = example_input.dtype if torch.is_tensor(example_input) else torch.float32
    ex_tensor = example_input if torch.is_tensor(example_input) else torch.empty(ex_shape, device="meta", dtype=ex_dtype)
    try:
        return _trace_stages_fx(model, ex_tensor)
    except Exception:
        res = torch_meter.analyze(model, ex_shape, GPU_PRESETS["rtx5090_measured"], dtype=ex_dtype, quiet=True)
        return [(float(w), float(q)) for w, q in res["stages"]]


def _resolve_machine(
    example_input: Any,
    peak_flops: float | None,
    peak_bandwidth: float | None,
    gpu: Any = None,
    dtype: Any = None,
    min_kernel: float = MIN_KERNEL,
) -> tuple[float, float, float, torch.dtype, tuple[int, ...]]:
    ex_shape = tuple(example_input.shape) if torch.is_tensor(example_input) else tuple(example_input)
    if dtype is not None:
        dt = torch_meter._parse_dtype(dtype)
    elif torch.is_tensor(example_input):
        dt = example_input.dtype
    else:
        dt = torch.float32

    if peak_flops is not None and peak_bandwidth is not None:
        return float(peak_flops), float(peak_bandwidth), float(min_kernel), dt, ex_shape

    g = gpu or GPU_PRESETS["rtx5090_measured"]
    f_eff = float(peak_flops) if peak_flops is not None else float(g.peak_flops(dt))
    bw = float(peak_bandwidth) if peak_bandwidth is not None else float(g.BW)
    tau_g = float(getattr(g, "tau_graph", min_kernel))
    return f_eff, bw, tau_g, dt, ex_shape


def theoretical_min(
    model: torch.nn.Module,
    example_input: Any,
    peak_flops: float | None = None,
    peak_bandwidth: float | None = None,
    phase: str = "forward",
    min_kernel: float = MIN_KERNEL,
    gpu: Any = None,
    dtype: Any = None,
) -> tuple[float, int]:
    """理论最短时延（秒）。phase: 'forward' | 'train'."""
    f_eff, bw, tau_g, dt, ex_shape = _resolve_machine(
        example_input, peak_flops, peak_bandwidth, gpu=gpu, dtype=dtype, min_kernel=min_kernel
    )
    stages = trace_stages(model, torch.empty(ex_shape, device="meta", dtype=dt))
    fwd = sum(max(w / f_eff, q / bw, tau_g) for w, q in stages) if stages else 0.0
    if phase == "forward":
        return fwd, len(stages)
    bwd = sum(max(2.0 * w / f_eff, 2.0 * q / bw, tau_g) for w, q in stages) if stages else 0.0
    return fwd + bwd, len(stages)


def no_fusion_min(
    n_ops: int,
    peak_flops: float = 195.7e12,
    peak_bandwidth: float = 1508e9,
    W: float = 0.0,
    Q: float = 0.0,
    per_kernel: float = MIN_KERNEL,
) -> float:
    """不融合时的最短时延：每个算子各占一次 kernel。"""
    return max(max(W / peak_flops, Q / peak_bandwidth), n_ops * per_kernel)


def count_ops(model: torch.nn.Module, example_input: Any) -> int:
    """数出真实运行的 aten 算子数量（含动态分支自动回退）。"""
    ex_shape = tuple(example_input.shape) if torch.is_tensor(example_input) else tuple(example_input)
    dt = example_input.dtype if torch.is_tensor(example_input) else torch.float32
    res = torch_meter.analyze(model, ex_shape, GPU_PRESETS["rtx5090_measured"], dtype=dt, quiet=True)
    return int(res["n_kernels"])


# ---------------------------------------------------------------- both phases
def phase_report(
    model: torch.nn.Module,
    example_input: Any,
    peak_flops: float | None = None,
    peak_bandwidth: float | None = None,
    gpu: Any = None,
    dtype: Any = None,
) -> dict[str, Any]:
    """报告前向与训练一步的理论最短时延、瓶颈与算术强度。

    同时兼容两种调用：
      1) phase_report(model, x, peak_flops, peak_bandwidth)
      2) phase_report(model, x, gpu=GPU_PRESETS["rtx5090_measured"], dtype="bf16")
    """
    f_eff, bw, tau_g, dt, ex_shape = _resolve_machine(
        example_input, peak_flops, peak_bandwidth, gpu=gpu, dtype=dtype
    )

    trace_path = "fx"
    ex_tensor = torch.empty(ex_shape, device="meta", dtype=dt)
    try:
        stages = _trace_stages_fx(model, ex_tensor)
        q_min_io = 0.0
    except Exception:
        trace_path = "fake_tensor"
        res = torch_meter.analyze(model, ex_shape, GPU_PRESETS["rtx5090_measured"], dtype=dt, quiet=True)
        stages = [(float(w), float(q)) for w, q in res["stages"]]
        q_min_io = float(res["Q_min_io_MB"]) * 1e6

    fwd = sum(max(w / f_eff, q / bw, tau_g) for w, q in stages) if stages else 0.0
    bwd = sum(max(2.0 * w / f_eff, 2.0 * q / bw, tau_g) for w, q in stages) if stages else 0.0
    train = fwd + bwd

    W_tot = sum(w for w, _ in stages)
    Q_tot = sum(q for _, q in stages)
    if q_min_io <= 0.0:
        in_out = 2.0 * ex_tensor.numel() * ex_tensor.element_size()
        param_bytes = sum(int(p.numel()) * int(p.element_size()) for p in model.parameters())
        q_min_io = in_out + param_bytes
    t_abs = max(W_tot / f_eff, q_min_io / bw, tau_g if stages else 0.0)

    t_comp_sum = W_tot / f_eff
    t_mem_sum = Q_tot / bw
    t_ser_sum = len(stages) * tau_g
    binding = max(
        {"compute": t_comp_sum, "memory": t_mem_sum, "serial": t_ser_sum}.items(),
        key=lambda kv: kv[1],
    )[0]
    ai = W_tot / max(Q_tot, 1.0)

    return {
        "理论最短_前向_秒": fwd,
        "理论最短_训练一步_秒": train,
        "绝对极限_前向_秒": t_abs,
        "融合阶段数": len(stages),
        "反向增加的倍数": (train - fwd) / fwd if fwd else float("inf"),
        "前向占比": fwd / train if train else 0.0,
        "约束瓶颈": binding,
        "算术强度_FLOP_per_B": ai,
        "计算量_GFLOP": W_tot / 1e9,
        "融合访存_MB": Q_tot / 1e6,
        "跟踪路径": trace_path,
        "dtype": str(dt).replace("torch.", ""),
    }


def print_phase_report(report: dict[str, Any]) -> None:
    print("理论最短时延（任何实现都不可能更快，%s，追踪=%s）" % (report.get("dtype", "fp32"), report.get("跟踪路径", "fx")))
    print("  前向推理      : %10.1f 微秒" % (report["理论最短_前向_秒"] * 1e6))
    print("  训练一步      : %10.1f 微秒" % (report["理论最短_训练一步_秒"] * 1e6))
    if "绝对极限_前向_秒" in report:
        print("  跨层持久核极限: %10.1f 微秒" % (report["绝对极限_前向_秒"] * 1e6))
    print(
        "  （反向约为前向 %.1f 倍 | 融合阶段 %d | 瓶颈 %s | %.2f GFLOP / %.1f MB | AI=%.1f FLOP/B）"
        % (
            report["反向增加的倍数"],
            report["融合阶段数"],
            report.get("约束瓶颈", "-"),
            report.get("计算量_GFLOP", 0.0),
            report.get("融合访存_MB", 0.0),
            report.get("算术强度_FLOP_per_B", 0.0),
        )
    )


def exact_phase_report(
    model: torch.nn.Module,
    example_input: Any,
    gpu: Any = None,
    dtype: Any = None,
    phase: str = "train_store",
    input_requires_grad: bool = False,
    prove_z3: bool = True,
) -> dict[str, Any]:
    """基于精确反向传播 DAG（dX 关键路径 vs dW 旁路、Hong-Kung SRAM I/O）与 Z3 形式化证明的严格极限报告。"""
    from . import autodiff_bounds
    from . import formal_prover

    g = gpu or GPU_PRESETS["rtx5090_measured"]
    dt = torch_meter._parse_dtype(dtype if dtype is not None else (example_input.dtype if torch.is_tensor(example_input) else torch.float32))
    dag = autodiff_bounds.dag_from_torch_module(
        model, example_input, dtype=dt, gpu=g, input_requires_grad=input_requires_grad
    )
    ev = autodiff_bounds.evaluate_dag_bounds(dag, g, phase=phase)
    z3_cert = formal_prover.prove_lower_bound_z3(dag, g, phase=phase) if prove_z3 else None
    return {"eval": ev, "z3": z3_cert, "dag": dag}
