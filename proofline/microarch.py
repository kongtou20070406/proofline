"""
microarch -- Microarchitecture-Aware Multi-Hardware Formal Roofline (Proofline v4).

Models four physical hardware walls on a unified stage-fused ComputeDAG:
  Wall 1 (SM Wave Quantization): CTA grid count N_cta bounded by SMEM+TMEM/RegFile,
      wave efficiency eta_wave = N_cta / (ceil(N_cta / N_sm) * N_sm).
  Wall 2 (Three-Unit Execution Roofs): Splits FLOPs into Tensor Core (W_tc / F_tc),
      FP32 ALU (W_alu / F_alu), and SFU (W_sfu / F_sfu).
  Wall 3 (Three-Level Memory + SM Occupancy Tiling): On-chip SMEM+TMEM tile capacity,
      L2 residency check (working_set <= l2_bytes), and HBM / L2 bandwidth roofs.
  Wall 4 (Inter-SM Grid Sync vs. Cluster DSM Ceiling): Stages exceeding
      onchip_capacity * cluster_max_sms pay tau_grid_sync_us; cluster-resident
      stages pay tau_cluster_sync_us (when cluster_max_sms > 1); CTA-resident pay 0.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from .autodiff_bounds import ComputeDAG, OpNode, dag_from_torch_module, evaluate_dag_bounds
from .devices import GPU, GPU_PRESETS, _dtype_key, dtype_bytes

_POINTWISE_FUSIBLE_OPS = {
    "relu", "relu_", "leaky_relu", "gelu", "silu", "sigmoid", "tanh", "exp",
    "log", "sqrt", "rsqrt", "neg", "abs", "pow", "clamp", "where", "masked_fill",
    "dropout", "contiguous", "softplus", "square", "reciprocal", "lerp",
    "addcmul", "addcdiv", "maximum", "minimum", "rsub", "add", "add_", "sub",
    "sub_", "mul", "mul_", "div", "div_", "sin", "cos", "hardtanh", "hardsigmoid",
    "hardswish", "mish", "elu", "celu", "selu",
}

_NONLINEAR_OPS = {
    "relu", "relu_", "leaky_relu", "gelu", "silu", "sigmoid", "tanh", "exp",
    "log", "sqrt", "rsqrt", "softplus", "sin", "cos", "hardtanh", "hardsigmoid",
    "hardswish", "mish", "elu", "celu", "selu", "pow", "reciprocal",
}

_SFU_PURE_OPS = {
    "exp", "rsqrt", "sqrt", "sin", "cos", "tanh", "sigmoid", "softplus",
    "log", "pow", "reciprocal",
}

_SFU_MIXED_OPS = {"gelu", "silu", "mish", "hardswish"}


def _raw_unit_flops(node: OpNode) -> tuple[float, float, float]:
    """Split a single raw OpNode's W_fwd into (W_tensor, W_alu, W_sfu)."""
    if "unit_flops" in node.attrs:
        u = node.attrs["unit_flops"]
        return float(u.get("tensor", 0.0)), float(u.get("alu", 0.0)), float(u.get("sfu", 0.0))

    w = float(node.W_fwd)
    if w <= 0.0:
        return 0.0, 0.0, 0.0

    low = node.name.lower()
    if node.kind in ("gemm", "conv"):
        if "sdpa" in low or "scaled_dot" in low:
            # SDPA: QK^T + PV on Tensor Cores, internal softmax on SFU+ALU
            m = float(node.attrs.get("m", 0.0))
            n = float(node.attrs.get("n", 0.0))
            b_h = float(node.attrs.get("b_heads", 1.0))
            if m > 0.0 and n > 0.0:
                w_sm = min(w * 0.25, 5.0 * b_h * m * n)
                return max(0.0, w - w_sm), 0.4 * w_sm, 0.6 * w_sm
            return 0.95 * w, 0.02 * w, 0.03 * w
        return w, 0.0, 0.0

    if "softmax" in low:
        return 0.0, 0.4 * w, 0.6 * w
    if any(tok in low for tok in _SFU_PURE_OPS):
        return 0.0, 0.0, w
    if any(tok in low for tok in _SFU_MIXED_OPS):
        return 0.0, 0.25 * w, 0.75 * w
    if node.kind == "reduce" and "norm" in low:
        return 0.0, 0.8 * w, 0.2 * w
    return 0.0, w, 0.0


def _is_fusible_pointwise(node: OpNode) -> bool:
    if node.kind == "pointwise":
        return True
    low = node.name.lower()
    return low in _POINTWISE_FUSIBLE_OPS and node.kind not in ("gemm", "conv", "reduce")


def fuse_stage_dag(dag: ComputeDAG) -> ComputeDAG:
    """Collapse fusible pointwise/residual chains into producer stages.

    Matches `latency_bounds.phase_report` fusion semantics so tiny elementwise ops
    do not inflate kernel launch counts or Z3/precision/rewrite stage counts.
    """
    if dag.notes.get("is_stage_fused"):
        return dag
    if not dag.nodes:
        return ComputeDAG(
            name=dag.name,
            nodes=[],
            dtype=dag.dtype,
            param_bytes=dag.param_bytes,
            input_bytes=dag.input_bytes,
            output_bytes=dag.output_bytes,
            notes={**dag.notes, "is_stage_fused": True, "raw_nodes": 0},
        )

    stage_of: dict[int, int] = {}
    stages: list[dict[str, Any]] = []

    for node in dag.nodes:
        prod_stages = [stage_of[p] for p in node.parents if p in stage_of]
        w_tc, w_alu, w_sfu = _raw_unit_flops(node)

        if _is_fusible_pointwise(node) and prod_stages:
            sid = max(prod_stages)
            st = stages[sid]
            st["W_fwd"] += node.W_fwd
            st["W_bwd_dx"] += node.W_bwd_dx
            st["W_bwd_dw"] += node.W_bwd_dw
            st["remat_flops"] += node.remat_flops
            st["unit_tc"] += w_tc
            st["unit_alu"] += w_alu
            st["unit_sfu"] += w_sfu
            st["fused_ops"].append(node.name)

            if node.name.lower() in _NONLINEAR_OPS:
                st["has_fused_nonlinearity"] = True

            # Only external inputs from *other* stages contribute extra HBM read traffic
            ext_parents = [stage_of[p] for p in node.parents if p in stage_of and stage_of[p] != sid]
            for psid in ext_parents:
                st["parents"].add(psid)
            if ext_parents:
                st["has_fused_ext_merge"] = True
                ext_read = max(0.0, float(node.attrs.get("act_in_bytes", 0.0)) - st["out_bytes"])
                st["Q_fwd"] += ext_read
                st["Q_bwd_store"] += ext_read
                st["Q_bwd_remat"] += ext_read
                st["act_in_bytes"] += ext_read

            out_b = float(node.attrs.get("out_bytes", 0.0))
            if out_b > 0.0:
                st["out_bytes"] = out_b
                st["act_save_bytes"] = max(st["act_save_bytes"], node.act_save_bytes)

            st["requires_grad_in"] = st["requires_grad_in"] or node.requires_grad_in
            st["requires_grad_param"] = st["requires_grad_param"] or node.requires_grad_param
            stage_of[node.id] = sid
        else:
            sid = len(stages)
            attrs = dict(node.attrs)
            stages.append(
                {
                    "id": sid,
                    "name": node.name,
                    "kind": node.kind,
                    "W_fwd": float(node.W_fwd),
                    "W_bwd_dx": float(node.W_bwd_dx),
                    "W_bwd_dw": float(node.W_bwd_dw),
                    "Q_fwd": float(node.Q_fwd),
                    "Q_bwd_store": float(node.Q_bwd_store),
                    "Q_bwd_remat": float(node.Q_bwd_remat),
                    "Q_hk_fwd": float(node.Q_hk_fwd),
                    "Q_hk_bwd": float(node.Q_hk_bwd),
                    "act_save_bytes": float(node.act_save_bytes),
                    "remat_flops": float(node.remat_flops),
                    "tau_fwd": float(node.tau_fwd),
                    "tau_bwd": float(node.tau_bwd),
                    "parents": set(prod_stages),
                    "requires_grad_in": bool(node.requires_grad_in),
                    "requires_grad_param": bool(node.requires_grad_param),
                    "attrs": attrs,
                    "unit_tc": w_tc,
                    "unit_alu": w_alu,
                    "unit_sfu": w_sfu,
                    "fused_ops": [node.name],
                    "has_fused_nonlinearity": node.name.lower() in _NONLINEAR_OPS,
                    "has_fused_ext_merge": False,
                    "act_in_bytes": float(attrs.get("act_in_bytes", 0.0)),
                    "out_bytes": float(attrs.get("out_bytes", 0.0)),
                }
            )
            stage_of[node.id] = sid

    fused_nodes: list[OpNode] = []
    for st in stages:
        attrs = dict(st["attrs"])
        attrs["unit_flops"] = {
            "tensor": st["unit_tc"],
            "alu": st["unit_alu"],
            "sfu": st["unit_sfu"],
        }
        attrs["fused_ops"] = tuple(st["fused_ops"])
        if st["has_fused_nonlinearity"]:
            attrs["has_fused_nonlinearity"] = True
        if st["has_fused_ext_merge"]:
            attrs["has_fused_ext_merge"] = True
        attrs["act_in_bytes"] = st["act_in_bytes"]
        attrs["out_bytes"] = st["out_bytes"]

        fused_nodes.append(
            OpNode(
                id=st["id"],
                name=st["name"],
                kind=st["kind"],
                W_fwd=st["W_fwd"],
                W_bwd_dx=st["W_bwd_dx"],
                W_bwd_dw=st["W_bwd_dw"],
                Q_fwd=st["Q_fwd"],
                Q_bwd_store=st["Q_bwd_store"],
                Q_bwd_remat=st["Q_bwd_remat"],
                Q_hk_fwd=st["Q_hk_fwd"],
                Q_hk_bwd=st["Q_hk_bwd"],
                act_save_bytes=st["act_save_bytes"],
                remat_flops=st["remat_flops"],
                tau_fwd=st["tau_fwd"],
                tau_bwd=st["tau_bwd"],
                parents=tuple(sorted(p for p in st["parents"] if p < st["id"])),
                requires_grad_in=st["requires_grad_in"],
                requires_grad_param=st["requires_grad_param"],
                attrs=attrs,
            )
        )

    return ComputeDAG(
        name=dag.name,
        nodes=fused_nodes,
        dtype=dag.dtype,
        param_bytes=dag.param_bytes,
        input_bytes=dag.input_bytes,
        output_bytes=dag.output_bytes,
        notes={
            **dag.notes,
            "is_stage_fused": True,
            "raw_nodes": len(dag.nodes),
            "fused_stages": len(fused_nodes),
        },
    )


def _estimate_stage_ctas(node: OpNode, device: GPU, b_elem: int) -> int:
    """Estimate CTA grid count N_cta bounded by SMEM+TMEM tile capacity."""
    onchip_sm = max((float(device.smem_per_sm_kb) + float(device.tmem_per_sm_kb)) * 1024.0, 16384.0)
    max_tile_elems = max(onchip_sm / (2.0 * max(1, b_elem)), 64.0)
    raw_side = int(2 ** math.floor(math.log2(math.sqrt(max_tile_elems))))
    tile_side = min(128, max(16, raw_side))

    a = node.attrs
    if node.kind == "gemm" and "m" in a and "n" in a:
        batch = max(1, int(a.get("batch", 1)))
        m = max(1, int(a["m"]))
        n = max(1, int(a["n"]))
        return max(1, batch * math.ceil(m / tile_side) * math.ceil(n / tile_side))
    if node.kind == "conv" and all(k in a for k in ("batch", "h_out", "w_out", "c_out")):
        m_conv = max(1, int(a["batch"]) * int(a["h_out"]) * int(a["w_out"]))
        n_conv = max(1, int(a["c_out"]))
        return max(1, math.ceil(m_conv / tile_side) * math.ceil(n_conv / tile_side))

    # Fallback for SDPA or untyped GEMM/Conv
    if node.kind in ("gemm", "conv"):
        return max(1, int(device.sm_count))
    return 1


def _stage_matrix_dims(node: OpNode) -> tuple[int, int] | None:
    """Return (M, N) matrix dimensions for GEMM/Conv nodes if available."""
    a = node.attrs
    if node.kind == "gemm" and "m" in a and "n" in a:
        batch = max(1, int(a.get("batch", 1)))
        return batch * max(1, int(a["m"])), max(1, int(a["n"]))
    if node.kind == "conv" and all(k in a for k in ("batch", "h_out", "w_out", "c_out")):
        m_conv = max(1, int(a["batch"]) * int(a["h_out"]) * int(a["w_out"]))
        return m_conv, max(1, int(a["c_out"]))
    return None


def _stage_wave_or_mxu_efficiency(node: OpNode, device: GPU, b_elem: int) -> float:
    """Return SM wave quantization efficiency (GPU) or MXU systolic padding efficiency (TPU)."""
    mxu = int(getattr(device, "mxu_dim", 0))
    if mxu > 0:
        dims = _stage_matrix_dims(node)
        if dims is None:
            return 1.0
        m_dim, n_dim = dims
        m_pad = math.ceil(m_dim / mxu) * mxu
        n_pad = math.ceil(n_dim / mxu) * mxu
        return (float(m_dim) / float(m_pad)) * (float(n_dim) / float(n_pad))

    n_sm = max(1, int(device.sm_count))
    n_cta = _estimate_stage_ctas(node, device, b_elem)
    n_waves = math.ceil(n_cta / n_sm)
    return float(n_cta) / float(n_waves * n_sm)


def _effective_tc_dtype(device: GPU, dtype: str) -> str:
    k = _dtype_key(dtype)
    if k in ("bf16", "fp16") and not device.supports_bf16:
        return "fp32"
    if k == "fp8" and not device.supports_fp8:
        return "bf16" if device.supports_bf16 else "fp32"
    if k == "int4" and not device.supports_fp4:
        return "bf16" if device.supports_bf16 else "fp32"
    return k


def evaluate_microarch_dag(
    dag: ComputeDAG,
    device: GPU,
    phase: str = "fwd",
    dtype: str | None = None,
    kv_cache_gb: float = 0.0,
) -> dict[str, Any]:
    """Evaluate all five physical microarchitecture walls on a ComputeDAG."""
    fused = fuse_stage_dag(dag)
    dt = _dtype_key(dtype if dtype is not None else fused.dtype)
    eff_dt = _effective_tc_dtype(device, dt)
    b_elem = max(1, int(math.ceil(dtype_bytes(eff_dt))))
    scale = 1.0 if phase in ("fwd", "forward") else 3.0

    f_tc = float(device.peak_flops(eff_dt))
    f_alu = float(device.peak_flops("fp32"))
    f_sfu = float(device.sfu_flops())
    bw = float(device.BW)
    tau_g_us = float(device.tau_graph) * 1e6

    n_sm = max(1, int(device.sm_count))
    onchip_sm_bytes = max((float(device.smem_per_sm_kb) + float(device.tmem_per_sm_kb)) * 1024.0, 1024.0)
    cluster_cap_bytes = onchip_sm_bytes * max(1, int(device.cluster_max_sms))
    l2_bytes = float(device.l2_bytes)

    w_tc_tot = 0.0
    w_alu_tot = 0.0
    w_sfu_tot = 0.0
    t_wave_compute_us = 0.0
    weighted_eta_num = 0.0
    weighted_eta_den = 0.0

    grid_sync_stages = 0
    cluster_dsm_stages = 0
    t_sync_us = 0.0
    total_param_bytes = 0.0

    for node in fused.nodes:
        u = node.attrs.get("unit_flops")
        if u is not None:
            w_tc = float(u.get("tensor", 0.0)) * scale
            w_alu = float(u.get("alu", 0.0)) * scale
            w_sfu = float(u.get("sfu", 0.0)) * scale
        else:
            wt, wa, ws = _raw_unit_flops(node)
            w_tc, w_alu, w_sfu = wt * scale, wa * scale, ws * scale

        w_tc_tot += w_tc
        w_alu_tot += w_alu
        w_sfu_tot += w_sfu

        # Wall 1: SM Wave Quantization (GPU) or MXU Systolic Padding (TPU)
        if node.kind in ("gemm", "conv") and w_tc > 0.0:
            eta_wave = _stage_wave_or_mxu_efficiency(node, device, b_elem)
            t_wave_compute_us += (w_tc / f_tc) * 1e6 / eta_wave
            weighted_eta_num += eta_wave * w_tc
            weighted_eta_den += w_tc
        elif w_tc > 0.0:
            t_wave_compute_us += (w_tc / f_tc) * 1e6

        # Wall 4: Inter-SM Grid Sync vs. Cluster DSM
        p_b = float(node.attrs.get("param_bytes", 0.0))
        total_param_bytes += p_b
        stage_ws = (
            float(node.attrs.get("act_in_bytes", 0.0))
            + float(node.attrs.get("out_bytes", 0.0))
            + p_b
        )
        if stage_ws <= 0.0:
            stage_ws = max(float(node.Q_fwd), float(node.Q_hk_fwd))

        if stage_ws > cluster_cap_bytes or node.kind == "reduce":
            grid_sync_stages += 1
            t_sync_us += float(device.tau_grid_sync_us) * (1.0 if scale == 1.0 else 2.0)
        elif stage_ws > onchip_sm_bytes and device.cluster_max_sms > 1:
            cluster_dsm_stages += 1
            t_sync_us += float(device.tau_cluster_sync_us) * (1.0 if scale == 1.0 else 2.0)

    wave_efficiency = (weighted_eta_num / weighted_eta_den) if weighted_eta_den > 0.0 else 1.0

    # Wall 2: Three-Unit Execution Roofs
    t_tc_us = (w_tc_tot / f_tc) * 1e6
    t_alu_us = (w_alu_tot / f_alu) * 1e6
    t_sfu_us = (w_sfu_tot / f_sfu) * 1e6
    t_three_unit_us = max(t_tc_us, t_alu_us, t_sfu_us)

    # Wall 3: Three-Level Memory + L2 Residency
    dag_phase = "forward" if phase in ("fwd", "forward") else "train_store"
    ev = evaluate_dag_bounds(fused, device, phase=dag_phase)
    q_hbm_bytes = max(float(ev["Q_min_io_MB"]), float(ev["Q_hk_MB"])) * 1e6
    t_mem_hbm_us = (q_hbm_bytes / bw) * 1e6

    unique_param_bytes = float(fused.param_bytes)
    total_param_bytes = max(total_param_bytes, unique_param_bytes)
    param_reuse_factor = total_param_bytes / max(unique_param_bytes, 1.0)
    working_set_bytes = unique_param_bytes + float(fused.input_bytes) + float(fused.output_bytes)
    l2_resident = (l2_bytes > 0.0) and (working_set_bytes <= l2_bytes)
    if l2_resident and device.l2_bw_gbs:
        t_mem_l2_us = (q_hbm_bytes / (float(device.l2_bw_gbs) * 1e9)) * 1e6
    else:
        t_mem_l2_us = t_mem_hbm_us

    # Serial launch floor from fused DAG critical path
    t_serial_us = float(ev["S_crit_tot"]) * tau_g_us
    t_sync_and_serial_us = t_serial_us + t_sync_us

    # Wall 5: Large-Model VRAM Capacity Overflow, Host PCIe Offload & Multi-Chip Scale-Out
    peak_act_bytes = max(
        float(fused.input_bytes) + float(fused.output_bytes),
        max((float(n.act_save_bytes) for n in fused.nodes), default=0.0),
    )
    kv_bytes = max(0.0, float(kv_cache_gb)) * 1e9
    if phase in ("fwd", "forward"):
        required_mem_bytes = unique_param_bytes + kv_bytes + peak_act_bytes
    else:
        required_mem_bytes = unique_param_bytes * 4.0 + kv_bytes + peak_act_bytes

    usable_vram_bytes = max(float(device.vram_gb) * 1e9 * 0.90, 1.0)
    if required_mem_bytes <= usable_vram_bytes:
        fits_in_vram = True
        spill_bytes = 0.0
        t_offload_us = 0.0
        min_chips_no_offload = 1
        t_multichip_comm_us = 0.0
    else:
        fits_in_vram = False
        spill_bytes = required_mem_bytes - usable_vram_bytes
        transfer_bytes = spill_bytes if phase in ("fwd", "forward") else 2.0 * spill_bytes
        host_bw_bps = max(float(device.host_bw_gbs) * 1e9, 1.0)
        t_offload_us = (transfer_bytes / host_bw_bps) * 1e6
        min_chips_no_offload = int(math.ceil(required_mem_bytes / usable_vram_bytes))
        ic_bw_bps = max(
            float(device.interconnect_bw_gbs if device.interconnect_bw_gbs > 0 else device.host_bw_gbs) * 1e9,
            1.0,
        )
        comm_bytes = (
            2.0
            * float(min_chips_no_offload - 1)
            / float(min_chips_no_offload)
            * max(float(fused.output_bytes), peak_act_bytes)
        )
        t_multichip_comm_us = (comm_bytes / ic_bw_bps) * 1e6

    walls = {
        "wave_compute": t_wave_compute_us,
        "tensor_core": t_tc_us,
        "alu": t_alu_us,
        "sfu": t_sfu_us,
        "memory": t_mem_hbm_us,
        "grid_sync": t_sync_and_serial_us,
        "pcie_offload": t_offload_us,
    }
    dominant_wall = max(walls, key=walls.get)
    t_microarch_floor_us = max(walls.values())

    advice: list[str] = []
    if not fits_in_vram:
        advice.append(
            f"VRAM overflow ({required_mem_bytes/1e9:.2f} GB > {usable_vram_bytes/1e9:.2f} GB usable on "
            f"{device.vram_gb:.1f} GB); host offload adds {t_offload_us/1e3:.2f} ms/step or shard across "
            f">={min_chips_no_offload} chips (interconnect comm ~{t_multichip_comm_us:.1f} us)."
        )
    if wave_efficiency < 0.75:
        advice.append(
            f"SM wave quantization efficiency={wave_efficiency:.2f} on {n_sm} SMs; "
            "pad/increase batch or hidden dim to fill full SM waves."
        )
    if t_sfu_us >= 0.5 * t_microarch_floor_us and t_sfu_us > 0.0:
        advice.append("SFU-bound (exp/rsqrt/softmax/gelu); fuse or approximate activations.")
    if grid_sync_stages > 0:
        advice.append(
            f"{grid_sync_stages} stage(s) exceed cluster DSM ({cluster_cap_bytes/1024:.0f} KB) "
            f"and require cross-SM grid sync ({device.tau_grid_sync_us:.2f} us/stage)."
        )
    if l2_resident:
        advice.append(
            f"Working set ({working_set_bytes/1e6:.2f} MB) fits in L2 ({l2_bytes/1e6:.1f} MB); "
            "persistent L2-resident kernels can avoid HBM traffic."
        )
    if dt in ("bf16", "fp16") and not device.supports_bf16:
        advice.append(f"{device.name} ({device.arch}) lacks native BF16 tensor cores; falls back to FP32/FP16.")
    if device.supports_fp4:
        advice.append(f"{device.name} ({device.arch}) supports FP4 tensor cores for further weight/compute compression.")

    return {
        "model": fused.name,
        "device": device.name,
        "arch": device.arch,
        "dtype": dt,
        "effective_tc_dtype": eff_dt,
        "phase": phase,
        "n_stages": len(fused.nodes),
        "raw_ops": int(fused.notes.get("raw_nodes", len(fused.nodes))),
        "T_microarch_floor_us": t_microarch_floor_us,
        "T_abs_roofline_us": float(ev["T_abs_floor_us"]),
        "T_wave_compute_us": t_wave_compute_us,
        "T_three_unit_us": t_three_unit_us,
        "T_tc_us": t_tc_us,
        "T_alu_us": t_alu_us,
        "T_sfu_us": t_sfu_us,
        "T_mem_us": t_mem_hbm_us,
        "T_mem_l2_us": t_mem_l2_us,
        "T_serial_us": t_serial_us,
        "T_sync_us": t_sync_us,
        "T_sync_and_serial_us": t_sync_and_serial_us,
        "T_offload_us": t_offload_us,
        "T_multichip_comm_us": t_multichip_comm_us,
        "vram_gb": float(device.vram_gb),
        "usable_vram_gb": usable_vram_bytes / 1e9,
        "required_mem_gb": required_mem_bytes / 1e9,
        "fits_in_vram": fits_in_vram,
        "spill_gb": spill_bytes / 1e9,
        "min_chips_no_offload": min_chips_no_offload,
        "dominant_wall": dominant_wall,
        "wave_efficiency": wave_efficiency,
        "sm_count": n_sm,
        "onchip_sm_kb": onchip_sm_bytes / 1024.0,
        "l2_mb": l2_bytes / 1e6,
        "l2_resident": l2_resident,
        "working_set_mb": working_set_bytes / 1e6,
        "unique_param_bytes": unique_param_bytes,
        "total_param_bytes": total_param_bytes,
        "param_reuse_factor": param_reuse_factor,
        "grid_sync_stages": grid_sync_stages,
        "cluster_dsm_stages": cluster_dsm_stages,
        "W_tc_GFLOP": w_tc_tot / 1e9,
        "W_alu_GFLOP": w_alu_tot / 1e9,
        "W_sfu_GFLOP": w_sfu_tot / 1e9,
        "advice": advice,
    }


def microarch_report(
    model_or_dag: torch.nn.Module | ComputeDAG,
    sample_input: Any = None,
    gpu: GPU | None = None,
    dtype: str = "bf16",
    phase: str = "fwd",
    kv_cache_gb: float = 0.0,
) -> dict[str, Any]:
    """Compute microarchitecture-aware 5-wall latency report for a model or ComputeDAG."""
    dev = gpu or GPU_PRESETS["rtx5090_measured"]
    if isinstance(model_or_dag, ComputeDAG):
        dag = model_or_dag
    else:
        if sample_input is None:
            raise ValueError("sample_input is required when passing a torch.nn.Module")
        dag = dag_from_torch_module(model_or_dag, sample_input, dtype=dtype, gpu=dev)
    return evaluate_microarch_dag(dag, dev, phase=phase, dtype=dtype, kv_cache_gb=kv_cache_gb)


def compare_hardware(
    model: torch.nn.Module,
    sample_input: Any,
    gpus: Sequence[str | GPU] | None = None,
    dtype: str = "bf16",
    phase: str = "fwd",
    kv_cache_gb: float = 0.0,
) -> list[dict[str, Any]]:
    """Compare microarchitecture-aware lower bounds across multiple hardware targets."""
    gpu_list: list[GPU] = []
    if gpus is None:
        gpu_list = [
            GPU_PRESETS[k]
            for k in ("b200", "h100", "rtx5090_measured", "a100", "l4", "t4")
            if k in GPU_PRESETS
        ]
    else:
        for item in gpus:
            if isinstance(item, GPU):
                gpu_list.append(item)
            else:
                key = str(item).strip()
                if key not in GPU_PRESETS:
                    raise ValueError(f"Unknown GPU preset {key!r}")
                gpu_list.append(GPU_PRESETS[key])

    rows: list[dict[str, Any]] = []
    for dev in gpu_list:
        dag = dag_from_torch_module(model, sample_input, dtype=dtype, gpu=dev)
        rows.append(evaluate_microarch_dag(dag, dev, phase=phase, dtype=dtype, kv_cache_gb=kv_cache_gb))
    return rows


def print_hardware_compare(
    rows: Sequence[dict[str, Any]],
    title: str | None = None,
    show_advice: bool = False,
) -> None:
    """Pretty-print cross-hardware microarchitecture comparison table."""
    if title:
        print(title)
    print(
        "%-17s %-7s %4s %6s %4s %11s %10s %10s %10s %7s %4s %5s %-13s"
        % (
            "Device",
            "Arch",
            "SMs",
            "VRAM",
            "Fit?",
            "Offload(ms)",
            "T_uArch(us)",
            "T_wave(us)",
            "T_mem(us)",
            "WaveEff",
            "L2?",
            "Chips",
            "DominantWall",
        )
    )
    print("-" * 120)
    for r in rows:
        print(
            "%-17s %-7s %4d %5.0fG %-4s %11.2f %10.2f %10.2f %10.2f %6.1f%% %-4s %5d %-13s"
            % (
                r["device"],
                r["arch"],
                r["sm_count"],
                r["vram_gb"],
                "yes" if r["fits_in_vram"] else "NO",
                r["T_offload_us"] / 1e3,
                r["T_microarch_floor_us"],
                r["T_wave_compute_us"],
                r["T_mem_us"],
                r["wave_efficiency"] * 100.0,
                "yes" if r["l2_resident"] else "no",
                r["min_chips_no_offload"],
                r["dominant_wall"],
            )
        )
        if show_advice and r.get("advice"):
            for adv in r["advice"]:
                print(f"    -> {adv}")
