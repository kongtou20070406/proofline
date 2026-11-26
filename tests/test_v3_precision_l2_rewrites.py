"""Tests for Proofline v3: multi-precision & 2:4 sparsity, L2 residency, and R_exact Z3 rewrites."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.quickstart_models import conv_block, mlp_block, transformer_block
from proofline import (
    GPU_PRESETS,
    PrecisionSpec,
    dag_from_torch_module,
    emit_rewrite_lean4_certificate,
    enumerate_rewrite_candidates,
    phase_report,
    precision_sweep,
    prove_rewrite_family_z3,
)


def test_device_presets_v3_fields() -> None:
    for key in ("rtx5090_measured", "rtx5090", "h100", "a100", "tpu_v5p", "cpu_avx512"):
        g = GPU_PRESETS[key]
        assert g.F_tf32 is not None and g.F_tf32 > 0
        assert g.F_fp8 is not None and g.F_fp8 > 0
        assert g.F_int8 is not None and g.F_int8 > 0
        assert g.F_fp4 is not None and g.F_fp4 > 0
        assert g.l2_bw_gbs is not None and g.l2_bw_gbs > 0
        assert g.l2_bytes > 0
    assert GPU_PRESETS["rtx5090"].peak_flops("fp8", sparse_24=True) > GPU_PRESETS["rtx5090"].peak_flops("fp8", sparse_24=False)
    # FP32 does not get 2:4 tensor-core speedup
    assert GPU_PRESETS["rtx5090"].peak_flops("fp32", sparse_24=True) == GPU_PRESETS["rtx5090"].peak_flops("fp32", sparse_24=False)


def test_precision_sweep_and_high_precision_norm_guard() -> None:
    gpu = GPU_PRESETS["h100"]
    mod = transformer_block()
    specs = [
        PrecisionSpec("fp32"),
        PrecisionSpec("bf16"),
        PrecisionSpec("fp8", group_size=128),
        PrecisionSpec("int4", group_size=128),
        PrecisionSpec("fp8", sparse_24=True, group_size=128),
    ]
    res = precision_sweep(mod, (2, 256, 512), gpu, formats=specs)
    assert len(res) == len(specs)
    by_lbl = {r["label"]: r for r in res}
    assert by_lbl["FP8(g128)"]["T_fwd_us"] < by_lbl["BF16"]["T_fwd_us"] < by_lbl["FP32"]["T_fwd_us"]
    assert by_lbl["FP8(g128)+2:4"]["T_fwd_us"] <= by_lbl["FP8(g128)"]["T_fwd_us"]
    # High-precision guard keeps LayerNorm/Softmax/adds at >= bf16
    assert by_lbl["INT4(g128)"]["high_prec_ops"] >= 2
    assert by_lbl["INT4(g128)"]["norm_dtype"] == "bf16"
    # Group-quant scale metadata adds >0 bytes over pure 0.5 B/elem
    assert PrecisionSpec("int4", group_size=64).weight_bytes_per_elem() > PrecisionSpec("int4", group_size=128).weight_bytes_per_elem()


def test_l2_cache_resident_bound() -> None:
    gpu = GPU_PRESETS["rtx5090_measured"]  # L2 = 96 MB, l2_bw_gbs = 6000
    small_mlp = mlp_block()  # ~2.1M params -> ~4.2 MB in bf16 (fits in 96 MB L2)
    rep_small = phase_report(small_mlp, (1, 128, 512), gpu=gpu, dtype="bf16")
    assert rep_small["l2_resident"] is True
    assert rep_small["working_set_mb"] < rep_small["l2_capacity_mb"]
    assert rep_small["T_l2_resident_us"] <= rep_small["理论最短_前向_秒"] * 1e6

    # Huge model exceeding 96 MB L2 should not be marked l2_resident
    big_mlp = nn.Sequential(nn.Linear(4096, 8192, bias=False), nn.Linear(8192, 4096, bias=False))
    rep_big = phase_report(big_mlp, (1, 512, 4096), gpu=gpu, dtype="bf16")
    assert rep_big["l2_resident"] is False
    assert rep_big["working_set_mb"] > rep_big["l2_capacity_mb"]


def test_exact_rewrite_rules_and_z3_proof(tmp_path: Path) -> None:
    pytest.importorskip("z3")
    gpu = GPU_PRESETS["rtx5090_measured"]

    # (a) matmul_chain_assoc: (16 x 512) @ (512 x 512) @ (512 x 16) -> assoc_opt collapses W1@W2 or optimal DP
    class _MatMulChain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = nn.Linear(512, 512, bias=False)
            self.l2 = nn.Linear(512, 16, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.l2(self.l1(x))

    dag_chain = dag_from_torch_module(_MatMulChain(), (16, 512), dtype="bf16", gpu=gpu)
    groups_chain = enumerate_rewrite_candidates(dag_chain, gpu, phase="fwd")
    assert any(g.rule == "matmul_chain_assoc" for g in groups_chain)
    cert_chain = prove_rewrite_family_z3(dag_chain, gpu, phase="fwd")
    assert cert_chain["proven"] is True
    assert cert_chain["T_rewrite_floor_us"] <= cert_chain["T_baseline_us"] + 1e-9

    # (b) fuse_shared_input_linears: Q, K, V sharing input
    class _SharedQKV(nn.Module):
        def __init__(self, d: int = 512) -> None:
            super().__init__()
            self.q = nn.Linear(d, d, bias=False)
            self.k = nn.Linear(d, d, bias=False)
            self.v = nn.Linear(d, d, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.q(x) + self.k(x) + self.v(x)

    dag_qkv = dag_from_torch_module(_SharedQKV(), (2, 128, 512), dtype="bf16", gpu=gpu)
    groups_qkv = enumerate_rewrite_candidates(dag_qkv, gpu, phase="fwd")
    assert any(g.rule == "fuse_shared_input_linears" for g in groups_qkv)
    cert_qkv = prove_rewrite_family_z3(dag_qkv, gpu, phase="fwd")
    assert cert_qkv["proven"] is True
    assert any(p["rule"] == "fuse_shared_input_linears" for p in cert_qkv["prescriptions"])

    # (c) conv3x3_winograd_fft: 3x3 stride-1 Winograd + 5x5 stride-1 FFT
    class _ConvPair(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.c3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
            self.c5 = nn.Conv2d(64, 64, kernel_size=5, padding=2, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.c5(F.relu(self.c3(x)))

    dag_conv = dag_from_torch_module(_ConvPair(), (1, 64, 64, 64), dtype="bf16", gpu=gpu)
    groups_conv = enumerate_rewrite_candidates(dag_conv, gpu, phase="fwd")
    conv_variants = {c.variant for g in groups_conv if g.rule == "conv3x3_winograd_fft" for c in g.candidates}
    assert "winograd_f2x2_3x3" in conv_variants
    assert "fft_conv" in conv_variants
    cert_conv = prove_rewrite_family_z3(dag_conv, gpu, phase="fwd")
    assert cert_conv["proven"] is True
    assert cert_conv["T_rewrite_floor_us"] < cert_conv["T_baseline_us"]

    # (d) remat_tradeoff in training + Lean 4 certificate emission
    dag_train = dag_from_torch_module(conv_block(), (1, 64, 32, 32), dtype="bf16", gpu=gpu)
    groups_train = enumerate_rewrite_candidates(dag_train, gpu, phase="train")
    assert any(g.rule == "remat_tradeoff" for g in groups_train)
    cert_train = prove_rewrite_family_z3(dag_train, gpu, phase="train")
    assert cert_train["proven"] is True

    lean_path = tmp_path / "RewriteCert.lean"
    lres = emit_rewrite_lean4_certificate(dag_conv, gpu, str(lean_path), phase="fwd", verify=False)
    assert lean_path.is_file()
    assert "theorem certified_rewrite_speed_limit" in lean_path.read_text(encoding="utf-8")
    assert lres["certified_ns"] > 0


def test_cli_v3_flags() -> None:
    pytest.importorskip("z3")
    proc = subprocess.run(
        [
            sys.executable,
            "-E",
            "-s",
            "-X",
            "utf8",
            "-B",
            "-m",
            "proofline.cli",
            "--model",
            "examples.quickstart_models:transformer_block",
            "--input",
            "1,256,512",
            "--gpu",
            "rtx5090_measured",
            "--dtype",
            "bf16",
            "--precision-sweep",
            "--rewrite-opt",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[Precision Sweep]" in proc.stdout
    assert "[RewriteOpt]" in proc.stdout
    assert "proven=True" in proc.stdout
