"""
Rank token mixers at a DEPLOYMENT scale, by theoretical minimum latency.

    python scaling.py                     # default N = 1,000,000 (bf16)
    python scaling.py --N 65536 --C 32    # 256x256 image restoration stage
    python scaling.py --N 32400 --dtype fp32
"""

from __future__ import annotations

import argparse
import math
import os
import sys


from .devices import (
    GPU_PRESETS,
    Cost,
    dtype_bytes,
    mamba3_mimo,
    mg_nsi_scheme_a,
    nsi_complex,
)


def candidates(N: int, C: int = 256, D: int = 64, H: int = 8, w: int = 16, chunk: int = 512, K: int = 5, dtype: str = "bf16") -> dict[str, Cost]:
    """Return candidate operators as Cost(name, W, Q, S, dtype)."""
    b = dtype_bytes(dtype)
    steps = -(-N // chunk)
    ds = 16
    side_h = max(1, int(math.isqrt(N)))
    side_w = max(1, -(-N // side_h))

    c_nsi_unfused = nsi_complex(C, N, K=K, fused_steps=1, dtype_bytes=b)
    c_nsi_fused5 = nsi_complex(C, N, K=K, fused_steps=K, dtype_bytes=b)
    c_mg_unfused = mg_nsi_scheme_a(C, side_h, side_w, K=K, fused_steps=1, dtype_bytes=b)
    c_mg_fused5 = mg_nsi_scheme_a(C, side_h, side_w, K=K, fused_steps=K, dtype_bytes=b)
    c_mamba3 = mamba3_mimo(N, C, d_state=64, rank=4, chunk=chunk, dtype_bytes=b)

    return {
        "depthwise_conv": Cost("depthwise_conv", 9 * C * N * 2, 2 * C * N * b, 1, dtype),
        "mamba2_ssd": Cost("mamba2_ssd", 6 * N * C, 2 * C * N * b, steps + 4, dtype),
        "selective_scan_d16": Cost("selective_scan_d16", 6 * N * C * ds, (2 * N * C + 3 * steps * C * ds) * b, steps + 4, dtype),
        "mamba3_mimo_r4": c_mamba3,
        "sliding_window_attn": Cost("sliding_window_attn", 2 * N * w * H * D, 4 * N * H * D * b, 3, dtype),
        "gla_gated_linear": Cost("gla_gated_linear", 4 * N * H * D * D + 2 * N * chunk * H * D, (4 * N * H * D + steps * H * D * D * 2) * b, 2 * steps + 2, dtype),
        "deltanet_kda": Cost("deltanet_kda", 6 * N * H * D * D + 2 * N * chunk * H * D, (4 * N * H * D + steps * H * D * D * 2) * b, 3 * steps + 2, dtype),
        "nsi_complex_unfused": c_nsi_unfused,
        "nsi_complex_fused5": c_nsi_fused5,
        "mg_nsi_unfused": c_mg_unfused,
        "mg_nsi_fused5": c_mg_fused5,
        "global_attention": Cost("global_attention", 2 * N * N * H * D, 4 * N * H * D * b, 3, dtype),
    }


def minimum(work: float, qbytes: float, stages: int, f_eff: float, bw: float, tau: float) -> float:
    return max(work / f_eff, qbytes / bw, stages * tau)


def bound_by(work: float, qbytes: float, stages: int, f_eff: float, bw: float, tau: float) -> str:
    parts = {"compute": work / f_eff, "memory": qbytes / bw, "serial": stages * tau}
    return max(parts, key=parts.get)


def report(N: int, C: int = 256, D: int = 64, H: int = 8, dtype: str = "bf16", gpu_key: str = "rtx5090_measured"):
    gpu = GPU_PRESETS[gpu_key]
    f_eff = gpu.peak_flops(dtype)
    bw = gpu.BW
    tau = gpu.tau_graph

    rows = []
    for name, c in candidates(N, C, D, H, dtype=dtype).items():
        t = minimum(c.W, c.Q, c.S, f_eff, bw, tau)
        bnd = bound_by(c.W, c.Q, c.S, f_eff, bw, tau)
        rows.append((name, c.W, c.Q, c.S, t, bnd))

    rows.sort(key=lambda r: r[4])
    best = rows[0][4]
    print("scale: N = %d tokens, C = %d, D = %d, H = %d, dtype = %s" % (N, C, D, H, dtype))
    print("machine (%s): %.1f TFLOP/s (%s), %.0f GB/s, %.2f us/kernel (graph)" % (
        gpu.name, f_eff / 1e12, dtype, bw / 1e9, tau * 1e6
    ))
    print()
    print("%-22s %12s %11s %7s %13s %9s %8s" % (
        "operator", "work(GFLOP)", "bytes(MB)", "stages", "min latency", "vs best", "binding"
    ))
    print("-" * 90)
    for name, W, Q, S, t, b in rows:
        print("%-22s %12.2f %11.1f %7d %11.1f %s %8.1fx %9s" % (
            name, W / 1e9, Q / 1e6, S,
            t * 1e6 if t < 1e-3 else t * 1e3,
            "us" if t < 1e-3 else "ms",
            t / best,
            b,
        ))

    if N >= 1000:
        b = dtype_bytes(dtype)
        K = 5
        c_un = nsi_complex(C, N, K=K, fused_steps=1, dtype_bytes=b)
        c_fu = nsi_complex(C, N, K=K, fused_steps=K, dtype_bytes=b)
        t_un = minimum(c_un.W, c_un.Q, c_un.S, f_eff, bw, tau)
        t_fu = minimum(c_fu.W, c_fu.Q, c_fu.S, f_eff, bw, tau)
        print()
        print("NSI / MG-NSI breakdown at this scale (K=%d, %s, %d bytes/elem):" % (K, dtype, b))
        print("  unfused work = 40*C*N*K = %.2f GFLOP | HBM bytes = (2+9)*b*C*N*K = %.1f MB -> %.3f ms (%s)" % (
            c_un.W / 1e9, c_un.Q / 1e6, t_un * 1e3, bound_by(c_un.W, c_un.Q, c_un.S, f_eff, bw, tau)
        ))
        print("  5-step fused HBM bytes = (2+9)*b*C*N = %.1f MB -> %.3f ms (%.1fx lower memory floor)" % (
            c_fu.Q / 1e6, t_fu * 1e3, t_un / max(t_fu, 1e-18)
        ))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1_000_000, help="sequence length; default 1e6")
    ap.add_argument("--C", type=int, default=256)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--gpu", choices=tuple(GPU_PRESETS), default="rtx5090_measured")
    a = ap.parse_args()
    report(a.N, a.C, a.D, a.H, dtype=a.dtype, gpu_key=a.gpu)
