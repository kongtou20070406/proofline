"""CPU-only verification suite for Proofline (zero GPU allocation required)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.quickstart_models import conv_block, mlp_block, transformer_block
from proofline import (
    GPU_PRESETS,
    analyze,
    dag_from_op,
    dag_from_torch_module,
    derive_symbolic_bounds,
    emit_lean4_certificate,
    evaluate_dag_bounds,
    phase_report,
    prove_lower_bound_z3,
)


class _ViewHeavy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        v = x.view(b, c // 2, 2, h, w).reshape(b, c // 2, 2, h * w).permute(0, 1, 3, 2)
        p = v.unfold(2, 4, 2)
        a, _ = p.chunk(2, dim=1)
        return a.transpose(1, 3)


def test_caller_model_immutability() -> None:
    gpu = GPU_PRESETS["rtx5090_measured"]
    mod = conv_block( ) if False else conv_block()
    w_before = mod.conv.weight.detach().clone()
    r_fp32 = analyze(mod, (1, 64, 32, 32), gpu, dtype="fp32")
    r_bf16 = analyze(mod, (1, 64, 32, 32), gpu, dtype="bf16")
    assert r_fp32["ok"] and r_bf16["ok"]
    assert mod.conv.weight.device.type == "cpu"
    assert mod.conv.weight.dtype == torch.float32
    assert torch.equal(mod.conv.weight, w_before)


def test_zero_copy_view_ops_and_stage_fusion() -> None:
    gpu = GPU_PRESETS["rtx5090_measured"]
    rv = analyze(_ViewHeavy(), (2, 16, 32, 32), gpu, dtype="fp32")
    assert rv["ok"]
    assert rv["n_kernels"] == 0
    assert rv["n_view_ops"] >= 5
    assert rv["W_GFLOP"] == 0.0
    assert rv["Q_fused_MB"] == 0.0

    fused = analyze(conv_block(), (1, 64, 32, 32), gpu, dtype="fp32")
    assert fused["n_stages"] == 1
    assert fused["Q_fused_MB"] < fused["Q_unfused_MB"]


def test_fp32_vs_bf16_peak_switching() -> None:
    gpu = GPU_PRESETS["h100"]
    mod = mlp_block()
    r32 = analyze(mod, (1, 256, 512), gpu, dtype="fp32")
    r16 = analyze(mod, (1, 256, 512), gpu, dtype="bf16")
    assert r32["ok"] and r16["ok"]
    assert r32["T_abs_floor_fwd_us"] > r16["T_abs_floor_fwd_us"]


def test_phase_report_fx_tracing() -> None:
    gpu = GPU_PRESETS["rtx5090_measured"]
    rep = phase_report(transformer_block(), torch.empty(1, 256, 512), gpu=gpu, dtype="bf16")
    assert rep["理论最短_前向_秒"] >= rep["绝对极限_前向_秒"] > 0.0
    assert rep["约束瓶颈"] in ("compute", "memory", "serial")


def test_z3_and_symbolic_and_lean_emission(tmp_path: Path) -> None:
    pytest.importorskip("z3")
    pytest.importorskip("sympy")
    gpu = GPU_PRESETS["rtx5090_measured"]
    dag = dag_from_torch_module(mlp_block(), (1, 256, 512), dtype="bf16", gpu=gpu)
    cert = prove_lower_bound_z3(dag, gpu, phase="train_store")
    assert cert["proven"] is True
    assert cert["T_abs_floor_us"] > 0.0
    assert len(cert["unsat_core"]) >= 1

    sym = derive_symbolic_bounds("gemm")
    assert sym["op"] == "gemm"

    lean_path = tmp_path / "CertTest.lean"
    lres = emit_lean4_certificate(dag, gpu, str(lean_path), phase="train_store", verify=False)
    assert lean_path.is_file()
    assert lres["certified_ns"] > 0
    assert "theorem certified_speed_limit" in lean_path.read_text(encoding="utf-8")


def test_l2_hong_kung_and_fusion_floor_consistency() -> None:
    pytest.importorskip("z3")

    class _LargeTransformerBlock(nn.Module):
        def __init__(self, d_model: int = 1024, heads: int = 16) -> None:
            super().__init__()
            self.h = heads
            self.dh = d_model // heads
            self.ln1 = nn.LayerNorm(d_model)
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.proj = nn.Linear(d_model, d_model, bias=False)
            self.ln2 = nn.LayerNorm(d_model)
            self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
            self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, s, d = x.shape
            qkv = self.qkv(self.ln1(x)).view(b, s, 3, self.h, self.dh)
            q, k, v = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2)
            att = F.softmax((q @ k.transpose(-2, -1)) * (self.dh ** -0.5), dim=-1)
            h = x + self.proj((att @ v).transpose(1, 2).reshape(b, s, d))
            return h + self.fc2(F.gelu(self.fc1(self.ln2(h))))

    gpu = GPU_PRESETS["rtx5090_measured"]
    mod = _LargeTransformerBlock(1024, 16)
    res = analyze(mod, (2, 1024, 1024), gpu, dtype="bf16")
    dag = dag_from_torch_module(mod, (2, 1024, 1024), dtype="bf16", gpu=gpu)
    cert = prove_lower_bound_z3(dag, gpu, phase="forward")
    assert cert["proven"] is True
    assert cert["T_abs_floor_us"] <= cert["T_stage_floor_us"]
    assert abs(cert["T_abs_floor_us"] - res["T_abs_floor_fwd_us"]) / res["T_abs_floor_fwd_us"] < 0.05

    gemm_dag = dag_from_op("gemm", gpu=gpu, B=1, M=16384, N=16384, K=16384, dtype="bf16")
    gemm_ev = evaluate_dag_bounds(gemm_dag, gpu, phase="forward")
    assert gemm_ev["Q_hk_MB"] > (3 * 16384 * 16384 * 2) / 1e6
