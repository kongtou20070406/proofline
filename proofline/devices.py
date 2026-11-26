"""
perf-ruler -- Theoretical minimum latency floor for neural operators and models.

    score(op, gpu) = max( W / F_eff , Q / BW , S * tau )      [seconds]

Inputs are properties of the algorithm, not of any particular kernel:
  W  FLOPs counted from the operator definition
  Q  bytes that MUST move through HBM (input reads + output writes under ideal stage fusion)
  S  inherent serial stages that cannot execute concurrently

Machine constants (calibrated per GPU / accelerator / CPU):
  F / F_fp32     peak low-precision (BF16/FP16) and FP32 compute throughput [FLOP/s]
  BW             peak HBM / DRAM memory bandwidth [bytes/s]
  tau_graph      minimum dependent kernel step floor under CUDA Graph replay [s]
  tau_eager      per-op launch + Python dispatch floor under eager PyTorch [s]
  sram_bytes     aggregate on-chip L2 / shared SRAM pool [bytes]
  sm_sram_bytes  per-core / per-SM fast-memory tile capacity [bytes] for Hong-Kung I/O bounds
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def _dtype_key(dtype: Any) -> str:
    if dtype is None:
        return "bf16"
    s = str(dtype).lower().replace("torch.", "").strip()
    if s in ("fp32", "float32", "float"):
        return "fp32"
    if s in ("tf32", "tfloat32"):
        return "tf32"
    if s in ("fp64", "float64", "double"):
        return "fp64"
    if s in ("fp16", "float16", "half"):
        return "fp16"
    if s in ("bf16", "bfloat16"):
        return "bf16"
    if s in ("fp8", "float8", "float8_e4m3fn", "float8_e5m2", "fp8_e4m3", "fp8_e5m2"):
        return "fp8"
    if s in ("int8", "qint8", "uint8"):
        return "int8"
    if s in ("int4", "uint4", "fp4", "nf4"):
        return "int4"
    return s


def dtype_bytes(dtype: Any) -> float:
    k = _dtype_key(dtype)
    if k == "fp64":
        return 8
    if k in ("fp32", "tf32"):
        return 4
    if k in ("fp8", "int8"):
        return 1
    if k == "int4":
        return 0.5
    return 2


@dataclass(frozen=True)
class GPU:
    name: str
    F: float
    BW: float
    t: float
    F_fp32: float | None = None
    tau_graph: float = 0.93e-6
    tau_eager: float | None = None
    sram_bytes: float = 0.0
    sm_sram_bytes: float = 128.0 * 1024.0
    F_tf32: float | None = None
    F_fp8: float | None = None
    F_int8: float | None = None
    F_fp4: float | None = None
    sparse_24_multiplier: float = 2.0
    l2_bw_gbs: float | None = None
    arch: str = "generic"
    sm_count: int = 1
    smem_per_sm_kb: float = 128.0
    regfile_per_sm_kb: float = 256.0
    tmem_per_sm_kb: float = 0.0
    cluster_max_sms: int = 1
    F_sfu: float | None = None
    tau_grid_sync_us: float = 2.0
    tau_cluster_sync_us: float = 0.25
    supports_bf16: bool = True
    supports_fp8: bool = True
    supports_fp4: bool = False
    vendor: str = "nvidia"
    mxu_dim: int = 0
    vram_gb: float = 80.0
    host_bw_gbs: float = 56.0
    interconnect_bw_gbs: float = 0.0
    unified_memory: bool = False

    @property
    def l2_bytes(self) -> float:
        return float(self.sram_bytes)

    def sfu_flops(self) -> float:
        """Return Special Function Unit (exp/rsqrt/sin/cos/tanh) peak FLOP/s."""
        if self.F_sfu is not None:
            return float(self.F_sfu)
        fp32_peak = self.F_fp32 if self.F_fp32 is not None else self.F / 2.0
        if self.arch in ("sm89", "sm90", "sm100", "sm120"):
            return float(fp32_peak * 4.0 / 128.0)
        if self.arch in ("sm75", "sm80"):
            return float(fp32_peak * 4.0 / 64.0)
        return float(fp32_peak / 16.0)

    def peak_flops(self, dtype: Any = None, sparse_24: bool = False) -> float:
        """Return effective peak FLOP/s for the target precision and optional 2:4 sparsity."""
        k = _dtype_key(dtype)
        fp32_peak = self.F_fp32 if self.F_fp32 is not None else self.F / 2.0
        if k == "fp32":
            base = fp32_peak
        elif k == "fp64":
            base = fp32_peak / 2.0
        elif k == "tf32":
            base = self.F_tf32 if self.F_tf32 is not None else fp32_peak
        elif k == "fp8":
            base = self.F_fp8 if self.F_fp8 is not None else self.F * 2.0
        elif k == "int8":
            base = self.F_int8 if self.F_int8 is not None else (self.F_fp8 if self.F_fp8 is not None else self.F * 2.0)
        elif k == "int4":
            base = self.F_fp4 if self.F_fp4 is not None else (self.F_fp8 * 2.0 if self.F_fp8 is not None else self.F * 4.0)
        else:
            base = self.F
        if sparse_24 and k not in ("fp32", "fp64"):
            base *= max(1.0, float(self.sparse_24_multiplier))
        return float(base)

    @property
    def eager_tau(self) -> float:
        return self.tau_eager if self.tau_eager is not None else self.t

    def __str__(self) -> str:
        sram_mb = self.sram_bytes / 1e6 if self.sram_bytes > 0 else 0.0
        return "%s: %.1f TF(bf16) / %.1f TF(fp32), %.0f GB/s, L2 %.0f MB, %.2f us graph / %.2f us eager" % (
            self.name,
            self.F / 1e12,
            self.peak_flops("fp32") / 1e12,
            self.BW / 1e9,
            sram_mb,
            self.tau_graph * 1e6,
            self.eager_tau * 1e6,
        )


DeviceSpec = GPU


def custom_device(
    name: str,
    peak_flops_bf16: float,
    peak_flops_fp32: float | None = None,
    bandwidth_bps: float = 1000e9,
    tau_graph: float = 1.0e-6,
    tau_eager: float = 10.0e-6,
    sram_mb: float = 40.0,
    sm_sram_kb: float = 128.0,
) -> GPU:
    """Create any GPU, CPU, TPU, NPU, or custom ASIC specification."""
    return GPU(
        name=name,
        F=float(peak_flops_bf16),
        BW=float(bandwidth_bps),
        t=float(tau_eager),
        F_fp32=float(peak_flops_fp32) if peak_flops_fp32 is not None else float(peak_flops_bf16) / 2.0,
        tau_graph=float(tau_graph),
        tau_eager=float(tau_eager),
        sram_bytes=float(sram_mb) * 1e6,
        sm_sram_bytes=float(sm_sram_kb) * 1024.0,
    )


GPU_PRESETS = {
    "t4": GPU(
        "T4", 65.0e12, 320e9, 10.0e-6, F_fp32=8.1e12,
        tau_graph=1.50e-6, tau_eager=10.0e-6, sram_bytes=4e6, sm_sram_bytes=64 * 1024.0,
        F_tf32=8.1e12, F_fp8=65.0e12, F_int8=130.0e12, F_fp4=260.0e12,
        sparse_24_multiplier=1.0, l2_bw_gbs=1200.0,
        arch="sm75", sm_count=40, smem_per_sm_kb=64.0, regfile_per_sm_kb=256.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=1,
        supports_bf16=False, supports_fp8=False, supports_fp4=False,
        vram_gb=16.0, host_bw_gbs=12.5, interconnect_bw_gbs=12.5,
    ),
    "l4": GPU(
        "L4", 121.0e12, 300e9, 10.0e-6, F_fp32=30.3e12,
        tau_graph=1.20e-6, tau_eager=10.0e-6, sram_bytes=48e6, sm_sram_bytes=128 * 1024.0,
        F_tf32=60.5e12, F_fp8=242.5e12, F_int8=242.5e12, F_fp4=242.5e12,
        sparse_24_multiplier=2.0, l2_bw_gbs=1800.0,
        arch="sm89", sm_count=58, smem_per_sm_kb=128.0, regfile_per_sm_kb=256.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=1,
        supports_bf16=True, supports_fp8=True, supports_fp4=False,
        vram_gb=24.0, host_bw_gbs=26.0, interconnect_bw_gbs=26.0,
    ),
    "rtx4090": GPU(
        "RTX4090", 165.2e12, 1008e9, 10.0e-6, F_fp32=82.6e12,
        tau_graph=1.00e-6, tau_eager=10.0e-6, sram_bytes=72e6, sm_sram_bytes=128 * 1024.0,
        F_tf32=82.6e12, F_fp8=330.3e12, F_int8=330.3e12, F_fp4=330.3e12,
        sparse_24_multiplier=2.0, l2_bw_gbs=4500.0,
        arch="sm89", sm_count=128, smem_per_sm_kb=128.0, regfile_per_sm_kb=256.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=1,
        supports_bf16=True, supports_fp8=True, supports_fp4=False,
        vram_gb=24.0, host_bw_gbs=26.0, interconnect_bw_gbs=26.0,
    ),
    "rtx5090": GPU(
        "RTX5090", 209.5e12, 1792e9, 10.32e-6, F_fp32=104.8e12,
        tau_graph=0.93e-6, tau_eager=10.32e-6, sram_bytes=96e6, sm_sram_bytes=228 * 1024.0,
        F_tf32=104.8e12, F_fp8=419.0e12, F_int8=419.0e12, F_fp4=838.0e12,
        sparse_24_multiplier=2.0, l2_bw_gbs=6500.0,
        arch="sm120", sm_count=170, smem_per_sm_kb=228.0, regfile_per_sm_kb=256.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=4,
        supports_bf16=True, supports_fp8=True, supports_fp4=True,
        vram_gb=32.0, host_bw_gbs=56.0, interconnect_bw_gbs=56.0,
    ),
    "rtx5090_measured": GPU(
        "RTX5090-measured", 195.7e12, 1508e9, 10.32e-6, F_fp32=95.0e12,
        tau_graph=0.93e-6, tau_eager=10.32e-6, sram_bytes=96e6, sm_sram_bytes=228 * 1024.0,
        F_tf32=95.0e12, F_fp8=391.4e12, F_int8=391.4e12, F_fp4=782.8e12,
        sparse_24_multiplier=2.0, l2_bw_gbs=6000.0,
        arch="sm120", sm_count=170, smem_per_sm_kb=228.0, regfile_per_sm_kb=256.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=4,
        supports_bf16=True, supports_fp8=True, supports_fp4=True,
        vram_gb=32.0, host_bw_gbs=56.0, interconnect_bw_gbs=56.0,
    ),
    "a100": GPU(
        "A100-80G", 312.0e12, 2039e9, 10.0e-6, F_fp32=19.5e12,
        tau_graph=1.20e-6, tau_eager=10.0e-6, sram_bytes=40e6, sm_sram_bytes=164 * 1024.0,
        F_tf32=156.0e12, F_fp8=312.0e12, F_int8=624.0e12, F_fp4=624.0e12,
        sparse_24_multiplier=2.0, l2_bw_gbs=5000.0,
        arch="sm80", sm_count=108, smem_per_sm_kb=164.0, regfile_per_sm_kb=256.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=1,
        supports_bf16=True, supports_fp8=False, supports_fp4=False,
        vram_gb=80.0, host_bw_gbs=26.0, interconnect_bw_gbs=300.0,
    ),
    "h100": GPU(
        "H100", 989.4e12, 3350e9, 10.0e-6, F_fp32=67.0e12,
        tau_graph=1.00e-6, tau_eager=10.0e-6, sram_bytes=50e6, sm_sram_bytes=228 * 1024.0,
        F_tf32=494.7e12, F_fp8=1978.9e12, F_int8=1978.9e12, F_fp4=1978.9e12,
        sparse_24_multiplier=2.0, l2_bw_gbs=12000.0,
        arch="sm90", sm_count=132, smem_per_sm_kb=228.0, regfile_per_sm_kb=256.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=8,
        supports_bf16=True, supports_fp8=True, supports_fp4=False,
        vram_gb=80.0, host_bw_gbs=56.0, interconnect_bw_gbs=450.0,
    ),
    "b200": GPU(
        "B200", 2250.0e12, 8000e9, 10.0e-6, F_fp32=80.0e12,
        tau_graph=0.80e-6, tau_eager=10.0e-6, sram_bytes=128e6, sm_sram_bytes=228 * 1024.0,
        F_tf32=1125.0e12, F_fp8=4500.0e12, F_int8=4500.0e12, F_fp4=9000.0e12,
        sparse_24_multiplier=2.0, l2_bw_gbs=10000.0,
        arch="sm100", sm_count=148, smem_per_sm_kb=228.0, regfile_per_sm_kb=256.0,
        tmem_per_sm_kb=256.0, cluster_max_sms=8,
        supports_bf16=True, supports_fp8=True, supports_fp4=True,
        vram_gb=192.0, host_bw_gbs=56.0, interconnect_bw_gbs=900.0,
    ),
    "tpu_v5p": GPU(
        "TPU-v5p", 459.0e12, 2765e9, 2.0e-6, F_fp32=229.5e12,
        tau_graph=0.15e-6, tau_eager=2.0e-6, sram_bytes=128e6, sm_sram_bytes=16 * 1024.0 * 1024.0,
        F_tf32=229.5e12, F_fp8=459.0e12, F_int8=918.0e12, F_fp4=918.0e12,
        sparse_24_multiplier=1.0, l2_bw_gbs=12000.0,
        arch="tpu_v5p", sm_count=8, smem_per_sm_kb=16384.0, regfile_per_sm_kb=1024.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=1, tau_grid_sync_us=0.20,
        supports_bf16=True, supports_fp8=True, supports_fp4=False,
        vendor="google", mxu_dim=128,
        vram_gb=95.0, host_bw_gbs=32.0, interconnect_bw_gbs=600.0,
    ),
    "tpu_v6e": GPU(
        "TPU-v6e", 918.0e12, 1638e9, 2.0e-6, F_fp32=459.0e12,
        tau_graph=0.15e-6, tau_eager=2.0e-6, sram_bytes=64e6, sm_sram_bytes=16 * 1024.0 * 1024.0,
        F_tf32=459.0e12, F_fp8=1836.0e12, F_int8=1836.0e12, F_fp4=1836.0e12,
        sparse_24_multiplier=1.0, l2_bw_gbs=10000.0,
        arch="tpu_v6e", sm_count=2, smem_per_sm_kb=16384.0, regfile_per_sm_kb=1024.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=1, tau_grid_sync_us=0.20,
        supports_bf16=True, supports_fp8=True, supports_fp4=False,
        vendor="google", mxu_dim=256,
        vram_gb=31.25, host_bw_gbs=32.0, interconnect_bw_gbs=400.0,
    ),
    "mi300x": GPU(
        "MI300X", 1307.4e12, 5300e9, 12.0e-6, F_fp32=163.4e12,
        tau_graph=1.20e-6, tau_eager=12.0e-6, sram_bytes=256e6, sm_sram_bytes=64 * 1024.0,
        F_tf32=653.7e12, F_fp8=2614.9e12, F_int8=2614.9e12, F_fp4=2614.9e12,
        sparse_24_multiplier=2.0, l2_bw_gbs=12000.0,
        arch="cdna3", sm_count=304, smem_per_sm_kb=64.0, regfile_per_sm_kb=256.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=1, tau_grid_sync_us=2.5,
        supports_bf16=True, supports_fp8=True, supports_fp4=False,
        vendor="amd", mxu_dim=0,
        vram_gb=192.0, host_bw_gbs=56.0, interconnect_bw_gbs=448.0,
    ),
    "mi325x": GPU(
        "MI325X", 1307.4e12, 6000e9, 12.0e-6, F_fp32=163.4e12,
        tau_graph=1.20e-6, tau_eager=12.0e-6, sram_bytes=256e6, sm_sram_bytes=64 * 1024.0,
        F_tf32=653.7e12, F_fp8=2614.9e12, F_int8=2614.9e12, F_fp4=2614.9e12,
        sparse_24_multiplier=2.0, l2_bw_gbs=12000.0,
        arch="cdna3", sm_count=304, smem_per_sm_kb=64.0, regfile_per_sm_kb=256.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=1, tau_grid_sync_us=2.5,
        supports_bf16=True, supports_fp8=True, supports_fp4=False,
        vendor="amd", mxu_dim=0,
        vram_gb=256.0, host_bw_gbs=56.0, interconnect_bw_gbs=448.0,
    ),
    "m4_max": GPU(
        "M4-Max", 34.0e12, 546e9, 15.0e-6, F_fp32=17.0e12,
        tau_graph=2.50e-6, tau_eager=15.0e-6, sram_bytes=48e6, sm_sram_bytes=32 * 1024.0,
        F_tf32=17.0e12, F_fp8=34.0e12, F_int8=68.0e12, F_fp4=68.0e12,
        sparse_24_multiplier=1.0, l2_bw_gbs=1800.0,
        arch="apple_m4", sm_count=40, smem_per_sm_kb=32.0, regfile_per_sm_kb=256.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=1, tau_grid_sync_us=3.0,
        supports_bf16=True, supports_fp8=False, supports_fp4=False,
        vendor="apple", mxu_dim=0,
        vram_gb=128.0, host_bw_gbs=546.0, interconnect_bw_gbs=546.0, unified_memory=True,
    ),
    "m3_ultra": GPU(
        "M3-Ultra", 64.0e12, 819e9, 15.0e-6, F_fp32=32.0e12,
        tau_graph=2.50e-6, tau_eager=15.0e-6, sram_bytes=96e6, sm_sram_bytes=32 * 1024.0,
        F_tf32=32.0e12, F_fp8=64.0e12, F_int8=128.0e12, F_fp4=128.0e12,
        sparse_24_multiplier=1.0, l2_bw_gbs=2600.0,
        arch="apple_m3", sm_count=80, smem_per_sm_kb=32.0, regfile_per_sm_kb=256.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=1, tau_grid_sync_us=3.0,
        supports_bf16=True, supports_fp8=False, supports_fp4=False,
        vendor="apple", mxu_dim=0,
        vram_gb=512.0, host_bw_gbs=819.0, interconnect_bw_gbs=819.0, unified_memory=True,
    ),
    "cpu_avx512": GPU(
        "CPU-EPYC-9654", 5.5e12, 460e9, 15.0e-6, F_fp32=2.75e12,
        tau_graph=2.0e-6, tau_eager=15.0e-6, sram_bytes=384e6, sm_sram_bytes=1024 * 1024.0,
        F_tf32=2.75e12, F_fp8=5.5e12, F_int8=11.0e12, F_fp4=11.0e12,
        sparse_24_multiplier=1.0, l2_bw_gbs=2000.0,
        arch="avx512", sm_count=96, smem_per_sm_kb=1024.0, regfile_per_sm_kb=64.0,
        tmem_per_sm_kb=0.0, cluster_max_sms=1,
        supports_bf16=True, supports_fp8=False, supports_fp4=False,
        vendor="cpu", mxu_dim=0,
        vram_gb=256.0, host_bw_gbs=460.0, interconnect_bw_gbs=460.0, unified_memory=True,
    ),
}


@dataclass(frozen=True)
class Cost:
    name: str
    W: float
    Q: float
    S: int
    dtype: str = "bf16"


def score(c: Cost, g: GPU, use_graph_tau: bool = False) -> float:
    """Single-number hardware floor in seconds. Smaller = theoretically faster."""
    f_eff = g.peak_flops(getattr(c, "dtype", None))
    tau = g.tau_graph if use_graph_tau else g.t
    return max(c.W / f_eff, c.Q / g.BW, c.S * tau)


def bound_by(c: Cost, g: GPU, use_graph_tau: bool = False) -> str:
    """Return dominant bottleneck: 'compute' | 'memory' | 'serial'."""
    f_eff = g.peak_flops(getattr(c, "dtype", None))
    tau = g.tau_graph if use_graph_tau else g.t
    parts = {"compute": c.W / f_eff, "memory": c.Q / g.BW, "serial": c.S * tau}
    return max(parts, key=parts.get)


def rank(cands: dict[str, Cost], g: GPU, use_graph_tau: bool = False):
    """cands: {name: Cost} -> [(name, score, binding, relative_to_best)] sorted."""
    sc = [(n, score(c, g, use_graph_tau=use_graph_tau), bound_by(c, g, use_graph_tau=use_graph_tau))
          for n, c in cands.items()]
    sc.sort(key=lambda x: x[1])
    best = sc[0][1]
    return [(n, s, b, s / best) for n, s, b in sc]


# ------------------------------------------------------------------ operators
def attention(N: int, D: int, H: int = 1, B: int = 1, causal: bool = True, dtype_bytes: int = 2) -> Cost:
    w = (2 if causal else 4) * B * N * N * H * D
    q = B * (4 * N * D * H * dtype_bytes)
    dt = "fp32" if dtype_bytes == 4 else "bf16"
    return Cost("attention", w, q, 1, dtype=dt)


def linear_attention(N: int, D: int, H: int = 1, B: int = 1, chunk: int = 512, dtype_bytes: int = 2) -> Cost:
    steps = -(-N // chunk)
    w = B * (4 * N * H * D * D + 2 * N * chunk * H * D)
    q = B * (4 * N * D * H * dtype_bytes + steps * H * D * D * dtype_bytes)
    dt = "fp32" if dtype_bytes == 4 else "bf16"
    return Cost("linear_attention", w, q, steps, dtype=dt)


def depthwise_conv(C: int, N: int, k: int = 3, B: int = 1, dtype_bytes: int = 2) -> Cost:
    w = B * N * C * k * k * 2
    q = B * 2 * C * N * dtype_bytes
    dt = "fp32" if dtype_bytes == 4 else "bf16"
    return Cost("dwconv", w, q, 1, dtype=dt)


def local_propagation(C: int, N: int, K: int, B: int = 1, dtype_bytes: int = 2) -> Cost:
    w = B * N * 9 * C * K * 2
    q = B * 2 * 2 * C * N * K * (dtype_bytes / 2.0)
    dt = "fp32" if dtype_bytes == 4 else "bf16"
    return Cost("local_prop", w, q, K, dtype=dt)


def nsi_complex(
    C: int,
    N: int,
    K: int = 5,
    fused_steps: int = 1,
    dtype_bytes: int = 2,
    halo_factor: float = 1.0,
    B: int = 1,
) -> Cost:
    """Complex 2x2 paired 9-neighbor propagation (NSI) with explicit state + coeff traffic."""
    passes = math.ceil(K / max(1, fused_steps))
    w = B * 40 * C * N * K
    state_bytes = 2 * C * N * dtype_bytes * passes
    coeff_bytes = 9 * C * N * dtype_bytes * passes * halo_factor
    q = B * (state_bytes + coeff_bytes)
    s = passes + 1
    dt = "fp32" if dtype_bytes == 4 else "bf16"
    name = f"nsi_K{K}_fuse{fused_steps}"
    return Cost(name, w, q, s, dtype=dt)


def _coarse_grid_size(H: int, W: int, short_side: int = 8, max_long_side: int = 16, min_enable_side: int = 16) -> tuple[int, int]:
    if min(H, W) <= min_enable_side:
        return (0, 0)
    if H <= W:
        hc = short_side
        wc = max(short_side, int(round(short_side * (W / float(H)))))
        if max_long_side is not None:
            wc = min(wc, max_long_side)
    else:
        wc = short_side
        hc = max(short_side, int(round(short_side * (H / float(W)))))
        if max_long_side is not None:
            hc = min(hc, max_long_side)
    return hc, wc


def mg_nsi_scheme_a(
    C: int,
    H: int,
    W: int,
    K: int = 5,
    short_side: int = 8,
    max_long_side: int = 16,
    fused_steps: int = 1,
    dtype_bytes: int = 2,
    halo_factor: float = 1.0,
    B: int = 1,
) -> Cost:
    """Unidirectional Gauss-Seidel MG-NSI (coarse summary grid + fine SRAM rollout)."""
    n_fine = H * W
    base = nsi_complex(C, n_fine, K=K, fused_steps=fused_steps, dtype_bytes=dtype_bytes, halo_factor=halo_factor, B=B)
    hc, wc = _coarse_grid_size(H, W, short_side=short_side, max_long_side=max_long_side)
    n_coarse = hc * wc
    if n_coarse == 0:
        return Cost(f"mg_nsi_K{K}_fuse{fused_steps}", base.W, base.Q, base.S, dtype=base.dtype)

    passes = math.ceil(K / max(1, fused_steps))
    m_sub = max(1, math.ceil((max(hc, wc) - 1) / float(K)))
    k_coarse = K * m_sub

    w_extra_fine = B * (4 * C * n_fine * K + C * n_fine + 8 * C * n_fine * K)
    w_coarse = B * 40 * C * n_coarse * k_coarse

    if fused_steps >= K:
        q_cross = B * (K * C * n_coarse * dtype_bytes)
        s_coarse = 1
    else:
        q_cross = B * (2 * C * n_fine * dtype_bytes * passes + K * C * n_coarse * dtype_bytes)
        s_coarse = k_coarse
    q_coarse = B * (2 * C * n_coarse * dtype_bytes * k_coarse + 9 * C * n_coarse * dtype_bytes)

    dt = "fp32" if dtype_bytes == 4 else "bf16"
    return Cost(
        f"mg_nsi_K{K}_fuse{fused_steps}",
        base.W + w_extra_fine + w_coarse,
        base.Q + q_cross + q_coarse,
        base.S + s_coarse,
        dtype=dt,
    )


def mamba3_mimo(
    N: int,
    C: int,
    d_state: int = 64,
    rank: int = 4,
    chunk: int = 256,
    dtype_bytes: int = 2,
    B: int = 1,
) -> Cost:
    """Mamba-3 MIMO rank-R state space cost (arXiv:2603.15569 style)."""
    steps = math.ceil(N / max(1, chunk))
    w = B * (4 * N * C * d_state * rank + 2 * N * chunk * d_state * rank)
    q = B * (2 * N * C * dtype_bytes * (1 + rank) + 2 * steps * C * d_state * dtype_bytes)
    s = steps + 4
    dt = "fp32" if dtype_bytes == 4 else "bf16"
    return Cost(f"mamba3_mimo_r{rank}", w, q, s, dtype=dt)


def mlp(C: int, N: int, ratio: int = 4, B: int = 1, dtype_bytes: int = 2) -> Cost:
    w = B * N * (C * ratio * C + ratio * C * C) * 2
    q = B * 2 * (C * N + ratio * C * N) * dtype_bytes
    dt = "fp32" if dtype_bytes == 4 else "bf16"
    return Cost("mlp", w, q, 1, dtype=dt)


def window_attention(N: int, D: int, win: int, H: int = 1, B: int = 1, dtype_bytes: int = 2) -> Cost:
    """Local window attention: work is O(N*win*D), one stage."""
    w = B * 2 * N * win * D * H
    q = B * (4 * N * D * H * dtype_bytes)
    dt = "fp32" if dtype_bytes == 4 else "bf16"
    return Cost("window_attn", w, q, 1, dtype=dt)


def recurrent_wm_cost(
    d_model: int = 160,
    n_heads: int = 4,
    d_ff: int = 320,
    k_m: int = 16,
    k_e: int = 8,
    ctx_len: int = 6,
    T: int = 8,
    horizon: int = 4,
    B: int = 1,
    dtype_bytes: int = 4,
) -> Cost:
    """Analytical forward floor matching loop_agi.reference.RecurrentWorldModel."""
    del n_heads
    n_state = k_m + k_e
    n_seq = n_state + ctx_len
    # Per inference step: step_dyn on ctx_len tokens (2 cross blocks) + core (2 self-attn blocks)
    step_dyn_per_tok = 2 * (2 * (4 * d_model * d_model + 2 * d_model * d_ff) + 4 * k_m * d_model)
    core_per_tok = 2 * (8 * d_model * d_model + 4 * d_model * d_ff) + 2 * 4 * n_seq * d_model
    w_infer = B * max(1, T) * (ctx_len * step_dyn_per_tok + n_seq * core_per_tok)
    w_roll = B * max(1, horizon) * step_dyn_per_tok
    w_tot = w_infer + w_roll
    # Model params ~1.04M -> 4 bytes * 1.04M resident in L2/registers across steps under fusion
    param_elems = 1_038_251
    state_bytes = B * n_seq * d_model * dtype_bytes
    q_tot = param_elems * dtype_bytes + (T + horizon) * 2 * state_bytes
    s_tot = max(1, T) * 6 + max(1, horizon) * 4
    dt = "fp32" if dtype_bytes == 4 else "bf16"
    return Cost(f"recurrent_wm_T{T}_H{horizon}", float(w_tot), float(q_tot), int(s_tot), dtype=dt)


# ------------------------------------------------------------------ cli
def table(cands: dict[str, Cost], gpu: GPU, title: str = "", use_graph_tau: bool = False) -> None:
    if title:
        print(title)
    print(gpu)
    print()
    print("%-22s %14s %10s %10s" % ("candidate", "theoretical floor", "binding", "vs best"))
    print("-" * 62)
    for name, s, b, rel in rank(cands, gpu, use_graph_tau=use_graph_tau):
        unit = "ms" if s >= 1e-3 else "us"
        val = s * 1e3 if s >= 1e-3 else s * 1e6
        print("%-22s %11.1f %2s %10s %9.2fx" % (name, val, unit, b, rel))


# ------------------------------------------------------------------ regime law
def crit_work(g: GPU, dtype: Any = None, use_graph_tau: bool = False) -> float:
    tau = g.tau_graph if use_graph_tau else g.t
    return tau * g.peak_flops(dtype)


def crit_traffic(g: GPU, use_graph_tau: bool = False) -> float:
    tau = g.tau_graph if use_graph_tau else g.t
    return tau * g.BW


def accelerability(c: Cost, g: GPU, use_graph_tau: bool = False) -> float:
    f_eff = g.peak_flops(getattr(c, "dtype", None))
    tau = g.tau_graph if use_graph_tau else g.t
    return max(c.W / f_eff, c.Q / g.BW) / max(c.S * tau, 1e-18)


def regime_report(cands: dict[str, Cost], g: GPU, use_graph_tau: bool = False) -> None:
    cw = crit_work(g, use_graph_tau=use_graph_tau)
    cq = crit_traffic(g, use_graph_tau=use_graph_tau)
    print("machine thresholds (bf16):  W_crit = %.2e FLOPs   Q_crit = %.2e bytes" % (cw, cq))
    print()
    print("%-22s %14s %12s %10s %14s" % ("operator", "W/W_crit", "Q/Q_crit", "A", "regime"))
    print("-" * 78)
    for n, c in cands.items():
        f_eff = g.peak_flops(getattr(c, "dtype", None))
        tau = g.tau_graph if use_graph_tau else g.t
        cw_c = tau * f_eff
        A = accelerability(c, g, use_graph_tau=use_graph_tau)
        reg = bound_by(c, g, use_graph_tau=use_graph_tau)
        print("%-22s %14.3e %12.3e %10.4f %14s" % (n, c.W / cw_c, c.Q / cq, A, reg))


def attention_critical_N(D: int, H: int, g: GPU, dtype: Any = None) -> float:
    """N above which global attention becomes compute-bound instead of serial-bound."""
    return math.sqrt(crit_work(g, dtype=dtype) / (2 * D * H))


def local_prop_critical_CN(g: GPU, k_taps: int = 9, dtype: Any = None) -> float:
    """C*N above which K-step local propagation becomes compute-bound."""
    return crit_work(g, dtype=dtype) / k_taps
