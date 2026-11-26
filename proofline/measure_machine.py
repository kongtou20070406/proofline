"""Calibrate (F_bf16, F_fp32, F_sfu, BW, host_bw_gbs, tau_eager, tau_graph) on CUDA / ROCm / MPS / CPU."""

from __future__ import annotations

import argparse
import json
import platform
import time

import torch


def detect_backend() -> str:
    """Auto-detect compute backend: 'cuda' (NVIDIA/AMD ROCm), 'mps' (Apple Silicon), or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _sync(backend: str) -> None:
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _best(fn, backend: str = "cpu", it: int = 20, rep: int = 3) -> float:
    for _ in range(3):
        fn()
    _sync(backend)
    best = 1e9
    for _ in range(rep):
        t0 = time.perf_counter()
        for _ in range(it):
            fn()
        _sync(backend)
        best = min(best, (time.perf_counter() - t0) / max(1, it))
    return max(best, 1e-12)


def measure(dev: str | None = None, n: int | None = None) -> dict:
    """Run a fast hardware calibration microbenchmark on CUDA, MPS, or CPU."""
    backend = dev if dev is not None else detect_backend()
    if backend == "cpu":
        n_gemm = int(n) if n is not None else 512
        n_vec = 1024 * 1024
        it_fast, it_gemm, it_bw = 200, 5, 5
        device_name = platform.processor() or "CPU"
        vram_gb = 0.0
    elif backend == "mps":
        n_gemm = int(n) if n is not None else 4096
        n_vec = 32 * 1024 * 1024
        it_fast, it_gemm, it_bw = 500, 10, 10
        device_name = f"Apple-MPS ({platform.machine()})"
        vram_gb = 0.0
    else:
        n_gemm = int(n) if n is not None else 8192
        n_vec = 128 * 1024 * 1024
        it_fast, it_gemm, it_bw = 5000, 10, 10
        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = float(props.total_memory) / 1e9

    x = torch.randn(1024, device=backend, dtype=torch.float32)

    # Eager dispatch latency on short op
    tau_eager = _best(lambda: x + 1.0, backend=backend, it=it_fast, rep=3)

    # Graph / compiled chain floor per kernel
    if backend == "cuda":
        g = torch.cuda.CUDAGraph()
        static_x = torch.ones(1024, device=backend, dtype=torch.float32)
        static_y = torch.empty_like(static_x)
        chain_len = 64
        with torch.cuda.graph(g):
            cur = static_x
            for _ in range(chain_len):
                cur = cur + 1.0
            static_y.copy_(cur)
        tau_graph = _best(lambda: g.replay(), backend=backend, it=500, rep=3) / float(chain_len)
        del static_x, static_y, g
    else:
        tau_graph = tau_eager * 0.25

    # Tensor / BF16 GEMM peak (fallback to float16 if bf16 unsupported)
    dt_low = torch.bfloat16
    try:
        a16 = torch.randn(n_gemm, n_gemm, device=backend, dtype=dt_low)
        b16 = torch.randn(n_gemm, n_gemm, device=backend, dtype=dt_low)
        _ = a16 @ b16
    except Exception:
        dt_low = torch.float16
        a16 = torch.randn(n_gemm, n_gemm, device=backend, dtype=dt_low)
        b16 = torch.randn(n_gemm, n_gemm, device=backend, dtype=dt_low)
    f_bf16 = (2.0 * (n_gemm ** 3)) / _best(lambda: a16 @ b16, backend=backend, it=it_gemm, rep=3)
    del a16, b16

    # Pure FP32 GEMM peak (disable TF32 on CUDA)
    n32 = max(128, n_gemm // 2)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if backend == "cuda" else None
    if backend == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        a32 = torch.randn(n32, n32, device=backend, dtype=torch.float32)
        b32 = torch.randn(n32, n32, device=backend, dtype=torch.float32)
        f_fp32 = (2.0 * (n32 ** 3)) / _best(lambda: a32 @ b32, backend=backend, it=it_gemm, rep=3)
        del a32, b32
    finally:
        if backend == "cuda" and prev_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    # SFU throughput (exp + rsqrt)
    sfu_n = min(n_vec, 16 * 1024 * 1024)
    s_in = torch.rand(sfu_n, device=backend, dtype=torch.float32) + 0.1
    t_sfu = _best(lambda: torch.rsqrt(torch.exp(s_in)), backend=backend, it=it_bw, rep=3)
    f_sfu = (2.0 * sfu_n) / t_sfu
    del s_in

    # Device memory streaming bandwidth (read + write = 2 * numel * element_size)
    big = torch.randn(n_vec, device=backend, dtype=torch.float16)
    bw = (big.numel() * big.element_size() * 2) / _best(lambda: big.clone(), backend=backend, it=it_bw, rep=3)
    del big, x

    # Host <-> Device PCIe bandwidth (unified on MPS/CPU)
    if backend == "cuda":
        xfer_elems = 32 * 1024 * 1024
        h_buf = torch.empty(xfer_elems, dtype=torch.float16, pin_memory=True)
        d_buf = torch.empty(xfer_elems, dtype=torch.float16, device=backend)
        t_h2d = _best(lambda: d_buf.copy_(h_buf, non_blocking=False), backend=backend, it=5, rep=3)
        host_bw_gbs = (h_buf.numel() * h_buf.element_size() / t_h2d) / 1e9
        del h_buf, d_buf
        torch.cuda.empty_cache()
    else:
        host_bw_gbs = bw / 1e9

    return {
        "device_name": device_name,
        "device": device_name,
        "backend": backend,
        "vram_gb": round(vram_gb, 2),
        "F": f_bf16,
        "F_bf16": f_bf16,
        "F_fp32": f_fp32,
        "F_sfu": f_sfu,
        "BW": bw,
        "host_bw_gbs": host_bw_gbs,
        "t": tau_eager,
        "tau_eager": tau_eager,
        "tau_graph": tau_graph,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Calibrate Proofline hardware constants.")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (auto-detected if omitted)")
    p.add_argument("--n", type=int, default=None, help="GEMM matrix dimension")
    p.add_argument("--json", action="store_true", help="print clean JSON output only")
    args = p.parse_args(argv)

    r = measure(dev=args.device, n=args.n)
    print(json.dumps(r, indent=2))
    if not args.json:
        print()
        print(
            '    GPU("%s", %.3e, %.3e, %.3e, F_fp32=%.3e, F_sfu=%.3e, tau_graph=%.3e, tau_eager=%.3e, vram_gb=%.2f, host_bw_gbs=%.2f),'
            % (
                r["device_name"].split()[-1],
                r["F_bf16"],
                r["BW"],
                r["tau_eager"],
                r["F_fp32"],
                r["F_sfu"],
                r["tau_graph"],
                r["tau_eager"],
                r["vram_gb"],
                r["host_bw_gbs"],
            )
        )


if __name__ == "__main__":
    main()
