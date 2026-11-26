"""Tests for Proofline v4: microarchitecture-aware multi-hardware formal roofline."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.quickstart_models import mlp_block, transformer_block
from proofline import (
    GPU_PRESETS,
    compare_hardware,
    dag_from_torch_module,
    fuse_stage_dag,
    microarch_report,
    prove_microarch_bound_z3,
    prove_rewrite_family_z3,
)


def test_v4_hardware_presets_and_sfu() -> None:
    expected = (
        "b200", "rtx5090_measured", "rtx5090", "h100", "a100", "l4", "t4", "rtx4090",
        "mi300x", "mi325x", "tpu_v5p", "tpu_v6e", "m4_max", "m3_ultra", "cpu_avx512",
    )
    for k in expected:
        g = GPU_PRESETS[k]
        assert g.sm_count > 0
        assert g.smem_per_sm_kb > 0
        assert g.sfu_flops() > 0

    assert GPU_PRESETS["b200"].tmem_per_sm_kb == 256.0
    assert GPU_PRESETS["b200"].supports_fp4 is True
    assert GPU_PRESETS["a100"].supports_fp8 is False
    assert GPU_PRESETS["t4"].supports_bf16 is False
    assert GPU_PRESETS["mi300x"].vendor == "amd"
    assert GPU_PRESETS["mi300x"].l2_bytes == 256e6
    assert GPU_PRESETS["tpu_v6e"].vendor == "google"
    assert GPU_PRESETS["tpu_v6e"].mxu_dim == 256
    assert GPU_PRESETS["m4_max"].vendor == "apple"
    assert GPU_PRESETS["m4_max"].supports_fp8 is False
    assert GPU_PRESETS["m3_ultra"].l2_bytes == 96e6


def test_unified_stage_fused_dag_and_grouped_prescriptions() -> None:
    pytest.importorskip("z3")
    gpu = GPU_PRESETS["rtx5090_measured"]

    class _LoopSharedQKV(nn.Module):
        def __init__(self, d: int = 256, steps: int = 4) -> None:
            super().__init__()
            self.q = nn.Linear(d, d, bias=False)
            self.k = nn.Linear(d, d, bias=False)
            self.v = nn.Linear(d, d, bias=False)
            self.steps = steps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = x
            for _ in range(self.steps):
                q = F.gelu(self.q(h))
                k = F.silu(self.k(h))
                v = self.v(h)
                h = h + q + k + v
            return h

    mod = _LoopSharedQKV()
    dag = dag_from_torch_module(mod, (1, 64, 256), dtype="bf16", gpu=gpu)
    fused = fuse_stage_dag(dag)
    assert len(fused.nodes) < len(dag.nodes)
    # Idempotent
    assert fuse_stage_dag(fused) is fused

    # Weight-shared parameter reuse > 1 across loop steps
    rep = microarch_report(fused, gpu=gpu, dtype="bf16", phase="fwd")
    assert rep["total_param_bytes"] > rep["unique_param_bytes"]
    assert rep["param_reuse_factor"] >= 3.5

    # Module-grouped prescriptions aggregate repeated loop ops
    rcert = prove_rewrite_family_z3(dag, gpu, phase="fwd")
    assert rcert["proven"] is True
    assert len(rcert["prescriptions"]) >= 1
    assert any(p["count"] >= 2 and p["total_delta_us"] >= p["delta_us"] for p in rcert["prescriptions"])


def test_four_walls_amd_tpu_apple_and_z3() -> None:
    pytest.importorskip("z3")

    # 1) Wall 3: AMD MI300X 256 MB Infinity Cache vs H100 50 MB L2
    big_mlp = nn.Sequential(nn.Linear(4096, 8192, bias=False), nn.Linear(8192, 4096, bias=False))
    # ~67M params * 2 bytes = ~134 MB working set -> fits MI300X (256 MB), exceeds H100 (50 MB)
    rep_mi = microarch_report(big_mlp, (1, 128, 4096), gpu=GPU_PRESETS["mi300x"], dtype="bf16")
    rep_h100 = microarch_report(big_mlp, (1, 128, 4096), gpu=GPU_PRESETS["h100"], dtype="bf16")
    assert rep_mi["l2_resident"] is True
    assert rep_h100["l2_resident"] is False

    # 2) Wall 1 & Wall 4: TPU v6e 256x256 MXU padding + low XLA launch overhead
    small_lin = nn.Sequential(nn.Linear(128, 128, bias=False), nn.ReLU(), nn.Linear(128, 128, bias=False))
    rep_tpu_unaligned = microarch_report(small_lin, (1, 64, 128), gpu=GPU_PRESETS["tpu_v6e"], dtype="bf16")
    rep_tpu_aligned = microarch_report(small_lin, (1, 256, 256)[:2] + (128,), gpu=GPU_PRESETS["tpu_v6e"], dtype="bf16")
    assert rep_tpu_unaligned["wave_efficiency"] < 0.5
    assert rep_tpu_aligned["wave_efficiency"] >= rep_tpu_unaligned["wave_efficiency"]

    # 3) Wall 2: SFU-heavy op triggers SFU roof
    class _SfuHeavy(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.exp(torch.tanh(torch.sigmoid(x)))

    rep_sfu = microarch_report(_SfuHeavy(), (1, 1024, 1024), gpu=GPU_PRESETS["h100"], dtype="bf16")
    assert rep_sfu["T_sfu_us"] > rep_sfu["T_tc_us"]

    # 4) Z3 4-wall certification
    mod = transformer_block()
    dag = dag_from_torch_module(mod, (1, 256, 512), dtype="bf16", gpu=GPU_PRESETS["b200"])
    z3_cert = prove_microarch_bound_z3(dag, GPU_PRESETS["b200"], phase="fwd")
    assert z3_cert["proven"] is True
    assert z3_cert["z3_refutation_unsat"] is True
    assert z3_cert["z3_tight_witness_sat"] is True
    assert len(z3_cert["unsat_core"]) >= 1

    # 5) Cross-hardware comparison including Apple Silicon
    rows = compare_hardware(
        mlp_block(),
        (1, 128, 512),
        gpus=("b200", "mi300x", "tpu_v6e", "m4_max", "m3_ultra"),
        dtype="bf16",
    )
    assert len(rows) == 5
    by_dev = {r["device"]: r for r in rows}
    assert by_dev["B200"]["T_microarch_floor_us"] < by_dev["M4-Max"]["T_microarch_floor_us"]


def test_cli_v4_compare_gpus_and_microarch() -> None:
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
            "examples/quickstart_models.py:transformer_block",
            "--input",
            "1,256,512",
            "--compare-gpus",
            "b200,h100,mi300x,tpu_v6e,m3_ultra",
            "--dtype",
            "bf16",
            "--microarch",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "[Microarch Hardware Compare]" in proc.stdout
    assert "B200" in proc.stdout
    assert "MI300X" in proc.stdout
    assert "TPU-v6e" in proc.stdout
    assert "M3-Ultra" in proc.stdout


def test_wall5_meta_multibillion_vram_offload_and_z3() -> None:
    pytest.importorskip("z3")

    # ~26.8B params in BF16 (~53.7 GB) created on meta device with zero host RAM
    with torch.device("meta"):
        giant = nn.Sequential(
            *[nn.Linear(16384, 16384, bias=False) for _ in range(100)]
        )

    # RTX 5090 (32 GB) overflows -> pcie_offload dominates & Z3 core includes offload_pigeonhole_ceiling
    dag_5090 = dag_from_torch_module(giant, (1, 128, 16384), dtype="bf16", gpu=GPU_PRESETS["rtx5090_measured"])
    cert_5090 = prove_microarch_bound_z3(dag_5090, GPU_PRESETS["rtx5090_measured"], phase="fwd")
    assert cert_5090["proven"] is True
    assert cert_5090["fits_in_vram"] is False
    assert cert_5090["dominant_wall"] == "pcie_offload"
    assert cert_5090["min_chips_no_offload"] >= 2
    assert "offload_pigeonhole_ceiling" in cert_5090["unsat_core"]

    # B200 (192 GB), MI300X (192 GB), and M3 Ultra (512 GB unified) fit without offload
    for key in ("b200", "mi300x", "m3_ultra"):
        rep = microarch_report(giant, (1, 128, 16384), gpu=GPU_PRESETS[key], dtype="bf16", phase="fwd")
        assert rep["fits_in_vram"] is True
        assert rep["T_offload_us"] == 0.0
        assert rep["min_chips_no_offload"] == 1


def test_measure_machine_cpu_json_helper(capsys) -> None:
    import json
    from proofline.measure_machine import detect_backend, main as mm_main, measure

    assert detect_backend() in ("cuda", "mps", "cpu")
    res = measure(dev="cpu", n=128)
    for k in (
        "device_name", "backend", "vram_gb", "F_fp32", "F_bf16",
        "F_sfu", "BW", "host_bw_gbs", "tau_eager", "tau_graph",
    ):
        assert k in res
    assert res["backend"] == "cpu"
    assert res["F_fp32"] > 0 and res["F_sfu"] > 0 and res["host_bw_gbs"] > 0

    mm_main(["--device", "cpu", "--n", "128", "--json"])
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["backend"] == "cpu"
