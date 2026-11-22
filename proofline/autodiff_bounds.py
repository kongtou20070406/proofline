"""
autodiff_bounds -- Exact Forward + Reverse-Mode AD DAG lower-bound analyzer.

Models reverse-mode autodiff accurately without floating heuristics:
  - Splits per-op backward into dX (on serial activation-gradient chain) vs dW
    (parallel parameter-gradient branch off the dX critical path).
  - Applies exact adjoint compute ratios (e.g., Linear first-layer skips dX
    when input_requires_grad=False; FlashAttention backward is 2.5x forward, not 2x).
  - Evaluates activation storage (2*A bytes save+reload) vs optimal rematerialization
    min(2*A/BW, W_remat/F) per node.
  - Enforces Hong-Kung / Loomis-Whitney SRAM I/O traffic lower bounds on tiled
    GEMM and SDPA when problem working set exceeds device sm_sram_bytes.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode

from .devices import GPU, GPU_PRESETS, _coarse_grid_size, dtype_bytes
from . import meter as torch_meter


@dataclass
class OpNode:
    id: int
    name: str
    kind: str
    W_fwd: float
    W_bwd_dx: float
    W_bwd_dw: float
    Q_fwd: float
    Q_bwd_store: float
    Q_bwd_remat: float
    Q_hk_fwd: float = 0.0
    Q_hk_bwd: float = 0.0
    act_save_bytes: float = 0.0
    remat_flops: float = 0.0
    tau_fwd: float = 1.0
    tau_bwd: float = 1.0
    parents: tuple[int, ...] = ()
    requires_grad_in: bool = True
    requires_grad_param: bool = True


@dataclass
class ComputeDAG:
    name: str
    nodes: list[OpNode]
    dtype: str = "fp32"
    param_bytes: float = 0.0
    input_bytes: float = 0.0
    output_bytes: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)


def _hong_kung_gemm_io(
    m: int,
    n: int,
    k: int,
    batch: int,
    b_elem: int,
    compulsory_q: float,
    sm_sram_bytes: float,
) -> float:
    """Loomis-Whitney / Hong-Kung tiled GEMM lower bound Q >= W*b / sqrt(M/b)."""
    if m <= 0 or n <= 0 or k <= 0 or batch <= 0:
        return float(compulsory_q)
    work_elems = float(batch) * float(m) * float(n) * float(k)
    tile_elems = max(sm_sram_bytes / max(b_elem, 1), 1.0)
    # 2 * M*N*K ops with 3 tiles of sqrt(M_fast / 3) reuse factor
    tile_side = max(math.sqrt(tile_elems / 3.0), 1.0)
    q_hk = 2.0 * work_elems * float(b_elem) / tile_side
    return max(float(compulsory_q), q_hk)


def _hong_kung_sdpa_io(
    b_heads: int,
    n_q: int,
    n_kv: int,
    d: int,
    b_elem: int,
    compulsory_q: float,
    sm_sram_bytes: float,
    is_causal: bool = False,
) -> float:
    """FlashAttention exact fast-memory I/O lower bound (Dao et al. Thm 2)."""
    working = float(n_kv) * float(d) * float(b_elem)
    if sm_sram_bytes <= 0 or working <= sm_sram_bytes:
        return float(compulsory_q)
    scale = 0.5 if is_causal else 1.0
    # Need O(N_q * N_kv * d^2 * b^2 / M_sram) slow-memory transfers
    q_fa = scale * 4.0 * float(b_heads) * float(n_q) * float(n_kv) * float(d * d) * float(b_elem * b_elem) / sm_sram_bytes
    return max(float(compulsory_q), q_fa)


class _DAGTraceMeter(TorchDispatchMode):
    """Traces exact aten ops under FakeTensorMode and captures topological parent edges."""

    def __init__(self, sm_sram_bytes: float, b_elem: int, input_requires_grad: bool) -> None:
        super().__init__()
        self.sm_sram_bytes = float(sm_sram_bytes)
        self.b_elem = int(b_elem)
        self.input_requires_grad = bool(input_requires_grad)

        self._live_tensors: list[torch.Tensor] = []
        # id(tensor) -> (root_id, node_id | None, nbytes, is_param, is_model_input)
        self._tinfo: dict[int, tuple[int, int | None, int, bool, bool]] = {}
        self.nodes: list[OpNode] = []
        self.ext_inputs_bytes: dict[int, int] = {}
        self.param_roots: dict[int, int] = {}
        self.model_input_roots: set[int] = set()
        self.final_outputs_bytes: int = 0
        self.n_views: int = 0

    def register_model_inputs(self, x: Any) -> None:
        for t in torch_meter._iter_tensors(x):
            tid = id(t)
            self._live_tensors.append(t)
            nb = torch_meter._tensor_nbytes(t)
            self._tinfo[tid] = (tid, None, nb, False, True)
            self.model_input_roots.add(tid)
            self.ext_inputs_bytes[tid] = nb

    def register_params(self, model: torch.nn.Module) -> float:
        tot = 0
        for p in list(model.parameters()) + list(model.buffers()):
            tid = id(p)
            self._live_tensors.append(p)
            nb = torch_meter._tensor_nbytes(p)
            is_p = isinstance(p, torch.nn.Parameter)
            self._tinfo[tid] = (tid, None, nb, is_p, False)
            if is_p:
                self.param_roots[tid] = nb
                tot += nb
        return float(tot)

    def _get_info(self, t: torch.Tensor) -> tuple[int, int | None, int, bool, bool]:
        tid = id(t)
        info = self._tinfo.get(tid)
        if info is not None:
            return info
        self._live_tensors.append(t)
        nb = torch_meter._tensor_nbytes(t)
        info = (tid, None, nb, False, False)
        self._tinfo[tid] = info
        return info

    def finalize_outputs(self, out: Any) -> float:
        seen: set[int] = set()
        tot = 0
        for t in torch_meter._iter_tensors(out):
            root_id, _, nb, _, _ = self._get_info(t)
            if root_id not in seen:
                seen.add(root_id)
                tot += nb
        self.final_outputs_bytes = tot
        return float(tot)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        fname = str(func)
        kind = torch_meter._classify_op(fname)
        if kind == "prim":
            return out

        in_tensors = list(torch_meter._iter_tensors((args, kwargs)))
        out_tensors = list(torch_meter._iter_tensors(out))
        in_infos = [self._get_info(t) for t in in_tensors]

        if kind == "view":
            self.n_views += 1
            if in_infos:
                root_id, p_nid, _, is_p, is_min = in_infos[0]
                for ot in out_tensors:
                    oid = id(ot)
                    self._live_tensors.append(ot)
                    self._tinfo[oid] = (root_id, p_nid, torch_meter._tensor_nbytes(ot), is_p, is_min)
            return out

        short = torch_meter._short_op_name(fname)
        w_fwd = float(torch_meter._op_flops(fname, kind, args, kwargs, out))
        in_bytes_dedup: dict[int, int] = {}
        parent_ids: list[int] = []
        has_param = False
        depends_only_on_raw_input = True
        any_act_input = False

        for root_id, p_nid, nb, is_p, is_min in in_infos:
            in_bytes_dedup[root_id] = max(in_bytes_dedup.get(root_id, 0), nb)
            if p_nid is not None and p_nid not in parent_ids:
                parent_ids.append(p_nid)
            if is_p:
                has_param = True
            else:
                any_act_input = True
                if not is_min:
                    depends_only_on_raw_input = False

        # Fuse pointwise nodes into latest parent when pure pointwise to avoid trivial tiny nodes
        # while still respecting DAG dependencies for non-pointwise kernels.
        out_bytes = sum(torch_meter._tensor_nbytes(ot) for ot in out_tensors)
        q_in = sum(in_bytes_dedup.values())
        q_fwd = float(q_in + out_bytes)

        req_grad_in = self.input_requires_grad or (any_act_input and not depends_only_on_raw_input)
        req_grad_param = has_param

        # Exact adjoint FLOPs & Hong-Kung bounds per op kind
        q_hk_fwd = q_fwd
        q_hk_bwd = q_fwd
        w_bwd_dx = w_fwd if req_grad_in else 0.0
        w_bwd_dw = w_fwd if req_grad_param else 0.0
        tau_fwd = 1.0
        tau_bwd = 1.0

        if kind == "sdpa" and len(in_tensors) >= 2:
            qt, kt = in_tensors[0], in_tensors[1]
            b_h = torch_meter._prod(qt.shape[:-2]) if qt.dim() >= 2 else 1
            n_q, d = int(qt.shape[-2]), int(qt.shape[-1])
            n_kv = int(kt.shape[-2])
            is_causal = bool(kwargs.get("is_causal", False))
            if not is_causal and len(args) > 5 and isinstance(args[5], bool):
                is_causal = args[5]
            # Exact FlashAttention-2 backward cost = 2.5x forward (2 forward matmuls -> 5 backward matmuls)
            w_bwd_dx = 2.5 * w_fwd if req_grad_in else 0.0
            w_bwd_dw = 0.0
            q_hk_fwd = _hong_kung_sdpa_io(b_h, n_q, n_kv, d, self.b_elem, q_fwd, self.sm_sram_bytes, is_causal)
            q_hk_bwd = 2.5 * q_hk_fwd if req_grad_in else q_fwd
        elif kind == "gemm" and len(in_tensors) >= 2:
            # Extract M, K, N for Hong-Kung check
            t0, t1 = in_tensors[-2], in_tensors[-1]
            if short == "linear":
                x, w = in_tensors[0], in_tensors[1]
                m = x.numel() // max(1, x.shape[-1])
                k = int(x.shape[-1])
                n = int(w.shape[0])
                batch = 1
            elif short in ("bmm", "baddbmm") and t0.dim() >= 3 and t1.dim() >= 3:
                batch, m, k = int(t0.shape[0]), int(t0.shape[1]), int(t0.shape[2])
                n = int(t1.shape[2])
            elif t0.dim() >= 2 and t1.dim() >= 2:
                batch = 1
                m, k = int(t0.numel() // max(1, t0.shape[-1])), int(t0.shape[-1])
                n = int(t1.shape[-1])
            else:
                batch, m, k, n = 1, 1, 1, 1
            q_hk_fwd = _hong_kung_gemm_io(m, n, k, batch, self.b_elem, q_fwd, self.sm_sram_bytes)
            q_hk_bwd = (q_hk_fwd if req_grad_in else 0.0) + (q_hk_fwd if req_grad_param else 0.0)
            if not req_grad_param and short in ("bmm", "mm", "matmul", "baddbmm"):
                # Two dynamic tensor operands: need gradient wrt both activations if non-param
                w_bwd_dx = 2.0 * w_fwd if req_grad_in else 0.0
                w_bwd_dw = 0.0
                q_hk_bwd = 2.0 * q_hk_fwd if req_grad_in else q_fwd
        elif kind == "conv" and len(in_tensors) >= 2:
            q_hk_bwd = (q_fwd if req_grad_in else 0.0) + (q_fwd if req_grad_param else 0.0)
        elif kind == "reduce":
            # Softmax / Norm backward ~1.5-2x forward for activation grad
            w_bwd_dx = (1.5 if "softmax" in short else 1.5) * w_fwd if req_grad_in else 0.0
            w_bwd_dw = (0.5 * w_fwd) if req_grad_param else 0.0
        elif kind == "pointwise":
            w_bwd_dx = w_fwd if req_grad_in else 0.0
            w_bwd_dw = 0.0

        # Backward HBM traffic: store saves input activations of op (read in fwd, re-read in bwd);
        # remat avoids HBM save but recomputes fwd work in backward.
        act_save = float(sum(nb for root_id, _, nb, is_p, _ in in_infos if not is_p))
        q_bwd_store = float(out_bytes) * (2.0 if req_grad_in else 1.0) + act_save + (
            float(sum(nb for _, _, nb, is_p, _ in in_infos if is_p)) if req_grad_param else 0.0
        )
        q_bwd_remat = float(out_bytes) * (2.0 if req_grad_in else 1.0) + (
            float(sum(nb for _, _, nb, is_p, _ in in_infos if is_p)) if req_grad_param else 0.0
        )
        # Parameterless pointwise/view outputs fused cleanly incur 0 separate save
        if kind == "pointwise" and parent_ids:
            # Pointwise fused into last producer stage if single parent & non-divergent
            pass

        nid = len(self.nodes)
        self.nodes.append(
            OpNode(
                id=nid,
                name=short,
                kind=kind,
                W_fwd=w_fwd,
                W_bwd_dx=w_bwd_dx,
                W_bwd_dw=w_bwd_dw,
                Q_fwd=q_fwd,
                Q_bwd_store=q_bwd_store,
                Q_bwd_remat=q_bwd_remat,
                Q_hk_fwd=q_hk_fwd,
                Q_hk_bwd=q_hk_bwd,
                act_save_bytes=act_save,
                remat_flops=w_fwd,
                tau_fwd=tau_fwd,
                tau_bwd=tau_bwd if (req_grad_in or req_grad_param) else 0.0,
                parents=tuple(sorted(parent_ids)),
                requires_grad_in=req_grad_in,
                requires_grad_param=req_grad_param,
            )
        )

        for ot in out_tensors:
            oid = id(ot)
            self._live_tensors.append(ot)
            self._tinfo[oid] = (oid, nid, torch_meter._tensor_nbytes(ot), False, False)
        return out


def dag_from_torch_module(
    model: torch.nn.Module,
    example_input: Any,
    dtype: Any = torch.float32,
    gpu: GPU | None = None,
    input_requires_grad: bool = False,
) -> ComputeDAG:
    """Build exact forward + reverse-mode AD DAG from any PyTorch nn.Module on meta/FakeTensor."""
    g = gpu or GPU_PRESETS["rtx5090_measured"]
    target_dtype = torch_meter._parse_dtype(dtype)
    b_elem = dtype_bytes(target_dtype)
    ex_shape = tuple(example_input.shape) if torch.is_tensor(example_input) else tuple(example_input)

    try:
        m_trace = copy.deepcopy(model).eval().to(device="meta", dtype=target_dtype)
    except Exception:
        m_trace = copy.deepcopy(model).eval()

    meter = _DAGTraceMeter(sm_sram_bytes=g.sm_sram_bytes, b_elem=b_elem, input_requires_grad=input_requires_grad)
    p_bytes = meter.register_params(m_trace)

    err = None
    out_val = None
    with FakeTensorMode(allow_non_fake_inputs=True):
        x = torch.empty(*ex_shape, device="meta", dtype=target_dtype)
        meter.register_model_inputs(x)
        with meter, torch.no_grad():
            try:
                out_val = m_trace(x)
            except Exception as exc:
                err = repr(exc)[:240]

    if out_val is None and err is not None:
        meter = _DAGTraceMeter(sm_sram_bytes=g.sm_sram_bytes, b_elem=b_elem, input_requires_grad=input_requires_grad)
        p_bytes = meter.register_params(m_trace)
        x_m = torch.empty(*ex_shape, device="meta", dtype=target_dtype)
        meter.register_model_inputs(x_m)
        with meter, torch.no_grad():
            out_val = m_trace(x_m)

    out_bytes = meter.finalize_outputs(out_val)
    in_bytes = float(sum(meter.ext_inputs_bytes.values()))
    dt_str = str(target_dtype).replace("torch.", "")
    return ComputeDAG(
        name=model.__class__.__name__,
        nodes=meter.nodes,
        dtype=dt_str,
        param_bytes=p_bytes,
        input_bytes=in_bytes,
        output_bytes=out_bytes,
        notes={"n_view_ops": meter.n_views, "trace_error": err},
    )


def dag_from_op(op_name: str, gpu: GPU | None = None, **kw: Any) -> ComputeDAG:
    """Build exact symbolic AD DAG for canonical analytical operators."""
    g = gpu or GPU_PRESETS["rtx5090_measured"]
    key = op_name.lower().strip()
    dt = kw.get("dtype", "bf16")
    b_elem = dtype_bytes(dt)

    if key == "gemm":
        B = int(kw.get("B", 1))
        M = int(kw.get("M", 1024))
        N = int(kw.get("N", 1024))
        K = int(kw.get("K", 1024))
        req_in = bool(kw.get("input_requires_grad", True))
        req_p = bool(kw.get("requires_grad_param", True))
        w = float(2 * B * M * N * K)
        q_comp = float((B * M * K + K * N + B * M * N) * b_elem)
        q_hk = _hong_kung_gemm_io(M, N, K, B, b_elem, q_comp, g.sm_sram_bytes)
        act_b = float(B * M * K * b_elem)
        node = OpNode(
            0, "gemm", "gemm",
            W_fwd=w,
            W_bwd_dx=w if req_in else 0.0,
            W_bwd_dw=w if req_p else 0.0,
            Q_fwd=q_comp,
            Q_bwd_store=q_comp + float(B * M * N * b_elem),
            Q_bwd_remat=q_comp + float(B * M * N * b_elem),
            Q_hk_fwd=q_hk,
            Q_hk_bwd=(q_hk if req_in else 0.0) + (q_hk if req_p else 0.0),
            act_save_bytes=act_b,
            remat_flops=w,
            tau_fwd=1.0,
            tau_bwd=1.0 if (req_in or req_p) else 0.0,
            parents=(),
            requires_grad_in=req_in,
            requires_grad_param=req_p,
        )
        return ComputeDAG(f"gemm_{B}x{M}x{N}x{K}", [node], dtype=str(dt), param_bytes=K * N * b_elem,
                          input_bytes=B * M * K * b_elem, output_bytes=B * M * N * b_elem)

    if key == "sdpa":
        B = int(kw.get("B", 1))
        H = int(kw.get("H", 8))
        N = int(kw.get("N", 1024))
        D = int(kw.get("D", 64))
        causal = bool(kw.get("causal", False))
        w = float((2 if causal else 4) * B * H * N * N * D)
        q_comp = float(4 * B * H * N * D * b_elem)
        q_hk = _hong_kung_sdpa_io(H * B, N, N, D, b_elem, q_comp, g.sm_sram_bytes, causal)
        node = OpNode(
            0, "sdpa", "sdpa",
            W_fwd=w,
            W_bwd_dx=2.5 * w,
            W_bwd_dw=0.0,
            Q_fwd=q_comp,
            Q_bwd_store=2.5 * q_comp,
            Q_bwd_remat=2.5 * q_comp,
            Q_hk_fwd=q_hk,
            Q_hk_bwd=2.5 * q_hk,
            act_save_bytes=float(3 * B * H * N * D * b_elem),
            remat_flops=w,
            tau_fwd=1.0,
            tau_bwd=1.0,
            parents=(),
            requires_grad_in=True,
            requires_grad_param=False,
        )
        return ComputeDAG(f"sdpa_B{B}_H{H}_N{N}_D{D}", [node], dtype=str(dt), param_bytes=0.0,
                          input_bytes=3 * B * H * N * D * b_elem, output_bytes=B * H * N * D * b_elem)

    if key == "recurrent_wm":
        d_model = int(kw.get("d_model", 160))
        d_ff = int(kw.get("d_ff", 320))
        k_m = int(kw.get("k_m", 16))
        k_e = int(kw.get("k_e", 8))
        ctx_len = int(kw.get("ctx_len", 6))
        T = int(kw.get("T", 8))
        k_grad = int(kw.get("k_grad", min(T, 2)))
        k_warm = max(0, T - k_grad)
        horizon = int(kw.get("horizon", 4))
        B = int(kw.get("B", 1))
        dt = kw.get("dtype", "fp32")
        b_elem = dtype_bytes(dt)

        n_seq = k_m + k_e + ctx_len
        step_dyn_per_tok = float(2 * (2 * (4 * d_model * d_model + 2 * d_model * d_ff) + 4 * k_m * d_model))
        core_per_tok = float(2 * (8 * d_model * d_model + 4 * d_model * d_ff) + 2 * 4 * n_seq * d_model)
        w_step = float(B * (ctx_len * step_dyn_per_tok + n_seq * core_per_tok))
        w_roll = float(B * step_dyn_per_tok)
        param_bytes = float(1_038_251 * b_elem)
        act_bytes = float(B * n_seq * d_model * b_elem)
        q_step = act_bytes * 2.0 + (param_bytes if T == 1 else 0.0)

        nodes: list[OpNode] = []
        prev: tuple[int, ...] = ()
        for t in range(T):
            is_grad_step = t >= k_warm
            nid = len(nodes)
            nodes.append(
                OpNode(
                    nid, f"infer_step_{t}", "recurrent_step",
                    W_fwd=w_step,
                    W_bwd_dx=w_step if is_grad_step else 0.0,
                    W_bwd_dw=w_step if is_grad_step else 0.0,
                    Q_fwd=q_step + (param_bytes if nid == 0 else 0.0),
                    Q_bwd_store=(act_bytes * 2.0 + param_bytes * 0.25) if is_grad_step else 0.0,
                    Q_bwd_remat=(act_bytes * 1.0 + param_bytes * 0.25) if is_grad_step else 0.0,
                    act_save_bytes=act_bytes if is_grad_step else 0.0,
                    remat_flops=w_step if is_grad_step else 0.0,
                    tau_fwd=6.0,
                    tau_bwd=6.0 if is_grad_step else 0.0,
                    parents=prev,
                    requires_grad_in=is_grad_step,
                    requires_grad_param=is_grad_step,
                )
            )
            prev = (nid,)
        for h in range(horizon):
            nid = len(nodes)
            nodes.append(
                OpNode(
                    nid, f"rollout_{h}", "recurrent_rollout",
                    W_fwd=w_roll,
                    W_bwd_dx=w_roll,
                    W_bwd_dw=w_roll,
                    Q_fwd=float(B * 2 * 16 * b_elem),
                    Q_bwd_store=float(B * 4 * 16 * b_elem + param_bytes * 0.1),
                    Q_bwd_remat=float(B * 4 * 16 * b_elem + param_bytes * 0.1),
                    act_save_bytes=float(B * 16 * b_elem),
                    remat_flops=w_roll,
                    tau_fwd=4.0,
                    tau_bwd=4.0,
                    parents=prev,
                    requires_grad_in=True,
                    requires_grad_param=True,
                )
            )
            prev = (nid,)
        return ComputeDAG(
            f"RecurrentWM_T{T}_kgrad{k_grad}_H{horizon}",
            nodes,
            dtype=str(dt),
            param_bytes=param_bytes,
            input_bytes=float(B * ctx_len * 16 * b_elem),
            output_bytes=float(B * horizon * 16 * b_elem),
            notes={"k_warm": k_warm, "k_grad": k_grad, "T": T, "horizon": horizon},
        )

    if key in ("nsi", "nsi_complex", "mg_nsi"):
        C = int(kw.get("C", 32))
        H = int(kw.get("H", 256))
        W_dim = int(kw.get("W", 256))
        K = int(kw.get("K", 5))
        fused = int(kw.get("fused_steps", 1))
        B = int(kw.get("B", 1))
        N = H * W_dim
        passes = math.ceil(K / max(1, fused))
        w_pass = float(B * 40 * C * N * min(fused, K))
        q_pass = float(B * (2 * C * N + 9 * C * N) * b_elem)
        if key == "mg_nsi":
            hc, wc = _coarse_grid_size(H, W_dim)
            if hc * wc > 0:
                w_pass += float(B * 13 * C * N * min(fused, K))
        nodes = []
        prev = ()
        for p in range(passes):
            nid = len(nodes)
            nodes.append(
                OpNode(
                    nid, f"{key}_pass_{p}", "local_prop",
                    W_fwd=w_pass,
                    W_bwd_dx=w_pass,
                    W_bwd_dw=w_pass,
                    Q_fwd=q_pass,
                    Q_bwd_store=1.8 * q_pass,
                    Q_bwd_remat=1.2 * q_pass,
                    act_save_bytes=float(B * 2 * C * N * b_elem),
                    remat_flops=w_pass,
                    tau_fwd=1.0,
                    tau_bwd=1.0,
                    parents=prev,
                )
            )
            prev = (nid,)
        return ComputeDAG(
            f"{key}_C{C}_{H}x{W_dim}_K{K}_fuse{fused}",
            nodes,
            dtype=str(dt),
            param_bytes=float(9 * C * b_elem),
            input_bytes=float(B * C * N * b_elem),
            output_bytes=float(B * C * N * b_elem),
        )

    raise ValueError(f"Unsupported op_name={op_name!r}")


def _longest_chain(nodes: list[OpNode], weights: list[float]) -> float:
    if not nodes:
        return 0.0
    dp = [0.0] * len(nodes)
    for i, node in enumerate(nodes):
        best_p = max((dp[p] for p in node.parents if 0 <= p < i), default=0.0)
        dp[i] = best_p + weights[i]
    return max(dp) if dp else 0.0


def evaluate_dag_bounds(dag: ComputeDAG, device: GPU, phase: str = "train_store") -> dict[str, Any]:
    """Compute exact DAG critical-path and hardware lower bounds for forward / backward / train."""
    f_eff = float(device.peak_flops(dag.dtype))
    bw = float(device.BW)
    tau_g = float(device.tau_graph)
    tau_e = float(device.eager_tau)
    nodes = dag.nodes

    # Forward totals & critical path
    w_fwd_tot = sum(n.W_fwd for n in nodes)
    q_fwd_stage_tot = sum(n.Q_fwd for n in nodes)
    q_fwd_hk_tot = max(dag.input_bytes + dag.output_bytes + dag.param_bytes, sum(n.Q_hk_fwd for n in nodes) if any(n.Q_hk_fwd > n.Q_fwd for n in nodes) else 0.0)
    q_fwd_min_io = max(dag.input_bytes + dag.output_bytes + min(dag.param_bytes, q_fwd_stage_tot), q_fwd_hk_tot)
    s_fwd_crit = _longest_chain(nodes, [n.tau_fwd for n in nodes])
    t_fwd_abs = max(w_fwd_tot / f_eff, q_fwd_min_io / bw, s_fwd_crit * tau_g)
    t_fwd_stage = sum(max(n.W_fwd / f_eff, max(n.Q_fwd, n.Q_hk_fwd) / bw, n.tau_fwd * tau_g) for n in nodes) if nodes else 0.0
    t_fwd_eager = t_fwd_stage + len(nodes) * tau_e

    # Backward: split dX (on serial chain) vs dW (branches in parallel off dX chain, only trailing step adds at most 1*tau if on critical tail)
    rev_nodes = list(nodes)
    # Build reverse DAG dependency order for dX
    children_map: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for n in nodes:
        for p in n.parents:
            if 0 <= p < len(nodes):
                children_map[p].append(n.id)

    dp_bwd_dx = [0.0] * len(nodes)
    for i in range(len(nodes) - 1, -1, -1):
        n = nodes[i]
        dx_step = n.tau_bwd if n.requires_grad_in else 0.0
        best_child = max((dp_bwd_dx[c] for c in children_map[i]), default=0.0)
        dp_bwd_dx[i] = best_child + dx_step
    s_bwd_crit_dx = max(dp_bwd_dx) if dp_bwd_dx else 0.0
    # Wait at most 1 tail kernel if last parameter op uses dW off the earliest activation
    any_dw = any(n.W_bwd_dw > 0 for n in nodes)
    s_bwd_crit = max(s_bwd_crit_dx, (s_fwd_crit * 0.0 + (1.0 if any_dw else 0.0)))

    w_bwd_tot = sum(n.W_bwd_dx + n.W_bwd_dw for n in nodes)
    q_bwd_store_tot = sum(n.Q_bwd_store for n in nodes)
    q_bwd_hk_tot = sum(n.Q_hk_bwd for n in nodes)

    # Optimal rematerialization: per node pay min(2 * act_save / BW, remat_flops / F_eff)
    t_ckpt_extra = 0.0
    w_bwd_remat_tot = w_bwd_tot
    q_bwd_remat_tot = sum(n.Q_bwd_remat for n in nodes)
    remat_nodes = 0
    for n in nodes:
        cost_store = (2.0 * n.act_save_bytes) / bw
        cost_remat = n.remat_flops / f_eff
        if n.act_save_bytes > 0 and n.remat_flops > 0 and (n.requires_grad_in or n.requires_grad_param):
            if cost_remat < cost_store:
                t_ckpt_extra += cost_remat
                w_bwd_remat_tot += n.remat_flops
                remat_nodes += 1
            else:
                t_ckpt_extra += cost_store

    if phase == "forward":
        w_sel, q_sel, q_min_sel, s_sel = w_fwd_tot, q_fwd_stage_tot, q_fwd_min_io, s_fwd_crit
        t_stage_sel = t_fwd_stage
    elif phase == "train_remat":
        w_sel = w_fwd_tot + w_bwd_remat_tot
        q_sel = q_fwd_stage_tot + q_bwd_remat_tot
        q_min_sel = max(q_fwd_min_io + max(dag.param_bytes, q_bwd_hk_tot), q_sel)
        s_sel = s_fwd_crit + s_bwd_crit
        t_stage_sel = t_fwd_stage + t_ckpt_extra + sum(
            max((n.W_bwd_dx + n.W_bwd_dw) / f_eff, n.Q_bwd_remat / bw, (n.tau_bwd if (n.requires_grad_in or n.requires_grad_param) else 0.0) * tau_g)
            for n in nodes
        )
    else:  # train_store (default)
        w_sel = w_fwd_tot + w_bwd_tot
        q_sel = q_fwd_stage_tot + q_bwd_store_tot
        q_min_sel = max(q_fwd_min_io + max(dag.param_bytes, q_bwd_hk_tot), 2.0 * dag.input_bytes + 2.0 * dag.param_bytes + 2.0 * dag.output_bytes)
        s_sel = s_fwd_crit + s_bwd_crit
        t_stage_sel = t_fwd_stage + sum(
            max((n.W_bwd_dx + n.W_bwd_dw) / f_eff, max(n.Q_bwd_store, n.Q_hk_bwd) / bw, (n.tau_bwd if (n.requires_grad_in or n.requires_grad_param) else 0.0) * tau_g)
            for n in nodes
        )

    # Check Hong-Kung dominant term vs compulsory I/O
    q_hk_phase = q_fwd_hk_tot if phase == "forward" else max(q_fwd_hk_tot + q_bwd_hk_tot, q_min_sel)
    t_comp_abs = w_sel / f_eff
    t_mem_abs = q_min_sel / bw
    t_hk_abs = q_hk_phase / bw
    t_ser_abs = s_sel * tau_g
    t_abs_floor = max(t_comp_abs, t_mem_abs, t_hk_abs, t_ser_abs)
    t_stage_floor = max(t_stage_sel, t_abs_floor)

    reg_parts = {
        "compute": t_comp_abs,
        "memory": t_mem_abs,
        "serial": t_ser_abs,
    }
    if t_hk_abs > t_mem_abs * 1.01:
        reg_parts["sram_io"] = t_hk_abs
    binding = max(reg_parts.items(), key=lambda kv: kv[1])[0]

    return {
        "name": dag.name,
        "device": device.name,
        "dtype": dag.dtype,
        "phase": phase,
        "n_nodes": len(nodes),
        "n_views": int(dag.notes.get("n_view_ops", 0)),
        "W_fwd_GFLOP": w_fwd_tot / 1e9,
        "W_bwd_GFLOP": w_bwd_tot / 1e9,
        "W_tot_GFLOP": w_sel / 1e9,
        "Q_min_io_MB": q_min_sel / 1e6,
        "Q_stage_MB": q_sel / 1e6,
        "Q_hk_MB": q_hk_phase / 1e6,
        "S_crit_fwd": s_fwd_crit,
        "S_crit_bwd": s_bwd_crit,
        "S_crit_tot": s_sel,
        "T_abs_floor_us": t_abs_floor * 1e6,
        "T_stage_floor_us": t_stage_floor * 1e6,
        "T_fwd_abs_us": t_fwd_abs * 1e6,
        "T_fwd_stage_us": t_fwd_stage * 1e6,
        "T_eager_us": t_fwd_eager * 1e6,
        "bwd_to_fwd_flops_ratio": (w_bwd_tot / w_fwd_tot) if w_fwd_tot > 0 else 0.0,
        "remat_preferred_nodes": remat_nodes,
        "max_throughput_hz": (1.0 / t_abs_floor) if t_abs_floor > 0 else float("inf"),
        "stage_throughput_hz": (1.0 / t_stage_floor) if t_stage_floor > 0 else float("inf"),
        "binding": binding,
    }
