"""Calibrate (F_bf16, F_fp32, BW, tau_eager, tau_graph) on the current GPU."""

from __future__ import annotations

import json
import time

import torch


def _best(fn, it: int = 20, rep: int = 3) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(rep):
        t0 = time.perf_counter()
        for _ in range(it):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / it)
    return best


def measure(dev: str = "cuda", n: int = 8192) -> dict:
    name = torch.cuda.get_device_name(0)
    x = torch.randn(1024, device=dev, dtype=torch.float32)

    # Eager dispatch latency on short chain
    tau_eager = _best(lambda: x + 1.0, it=5000, rep=3)

    # CUDA-Graph dependent chain floor per kernel (launch_chain=64 ops)
    g = torch.cuda.CUDAGraph()
    static_x = torch.ones(1024, device=dev, dtype=torch.float32)
    static_y = torch.empty_like(static_x)
    chain_len = 64
    with torch.cuda.graph(g):
        cur = static_x
        for _ in range(chain_len):
            cur = cur + 1.0
        static_y.copy_(cur)
    tau_graph = _best(lambda: g.replay(), it=500, rep=3) / float(chain_len)

    # Tensor Core BF16/FP16 peak GEMM
    a16 = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    b16 = torch.randn(n, n, device=dev, dtype=torch.bfloat16)
    f_bf16 = (2.0 * n ** 3) / _best(lambda: a16 @ b16, it=10, rep=3)
    del a16, b16

    # Pure FP32 GEMM peak (TF32 disabled)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        a32 = torch.randn(n // 2, n // 2, device=dev, dtype=torch.float32)
        b32 = torch.randn(n // 2, n // 2, device=dev, dtype=torch.float32)
        f_fp32 = (2.0 * (n // 2) ** 3) / _best(lambda: a32 @ b32, it=10, rep=3)
        del a32, b32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    # HBM streaming bandwidth (read + write = 2 * numel * element_size)
    big = torch.randn(128 * 1024 * 1024, device=dev, dtype=torch.float16)
    bw = (big.numel() * big.element_size() * 2) / _best(lambda: big.clone(), it=10, rep=3)
    del big, x, static_x, static_y, g
    torch.cuda.empty_cache()

    return {
        "device": name,
        "F": f_bf16,
        "F_fp32": f_fp32,
        "BW": bw,
        "t": tau_eager,
        "tau_eager": tau_eager,
        "tau_graph": tau_graph,
    }


if __name__ == "__main__":
    r = measure()
    print(json.dumps(r, indent=2))
    print()
    print(
        '    GPU("%s", %.3e, %.3e, %.3e, F_fp32=%.3e, tau_graph=%.3e, tau_eager=%.3e),'
        % (r["device"].split()[-1], r["F"], r["BW"], r["t"], r["F_fp32"], r["tau_graph"], r["tau_eager"])
    )
