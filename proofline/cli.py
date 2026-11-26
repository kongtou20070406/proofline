#!/usr/bin/env python3
"""CLI: compute hardware latency bounds of PyTorch models without GPU memory."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys


def load_target(spec: str):
    """'pkg.module:factory' or 'path/to/file.py:factory'"""
    target, _, fn = spec.rpartition(":")  # rpartition preserves Windows drive colon
    if not fn:
        raise SystemExit("model spec must be 'module:factory' or 'file.py:factory'")
    if target.endswith(".py"):
        name = os.path.splitext(os.path.basename(target))[0]
        s = importlib.util.spec_from_file_location(name, target)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    else:
        m = importlib.import_module(target)
    return getattr(m, fn)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", default=None, help="one or more 'module:factory' specs")
    p.add_argument("--input", default=None, help="comma-separated shape, e.g. 1,32,256,256")
    p.add_argument("--gpu", default="rtx5090_measured")
    p.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    p.add_argument("--phase", choices=("forward", "train", "both"), default="both")
    p.add_argument("--use", choices=("stage_fwd", "stage_train", "abs_floor", "eager"), default="stage_fwd")
    p.add_argument("--profile", default=None, help="calibrated profile json")
    p.add_argument("--list-gpus", action="store_true")
    p.add_argument("--formal-z3", action="store_true", help="run Z3 SMT lower-bound certificate check")
    p.add_argument("--emit-lean", default=None, help="emit and check Lean 4 speed-limit certificate file")
    p.add_argument("--symbolic", default=None, help="print SymPy symbolic bounds for op (gemm|sdpa|recurrent_wm)")
    p.add_argument("--precision-sweep", action="store_true", help="sweep FP32/TF32/BF16/FP8/INT8/INT4 and 2:4 sparsity bounds")
    p.add_argument("--rewrite-opt", action="store_true", help="run exact algebraic rewrite family (R_exact) Z3 optimization & prescriptions")
    p.add_argument("--compare-gpus", default=None, help="comma-separated list of GPU/TPU/Apple presets to compare")
    p.add_argument("--microarch", action="store_true", help="include 5-wall microarchitecture breakdown and advice")
    p.add_argument("--kv-cache-gb", type=float, default=0.0, help="KV cache size in GB for VRAM capacity / offload check")
    a = p.parse_args()

    from .calibrate import load_profile
    from .devices import GPU_PRESETS
    from .meter import print_compare

    if a.symbolic:
        from . import formal_prover
        import json as _json
        print(_json.dumps(formal_prover.derive_symbolic_bounds(a.symbolic), indent=2, ensure_ascii=False))
        return

    if a.list_gpus:
        for k, v in GPU_PRESETS.items():
            print("%-18s -> %s" % (k, v))
        return
    if not a.model or not a.input:
        raise SystemExit("--model and --input are required (or use --list-gpus / --symbolic)")

    gpu = GPU_PRESETS.get(a.gpu)
    if gpu is None:
        raise SystemExit("unknown gpu %r; see --list-gpus" % a.gpu)
    if a.profile:
        gpu, a0, _ = load_profile(a.profile)
        print("using calibrated profile: %s  a0=%.3fms" % (gpu, a0 * 1e3))

    shape = tuple(int(x) for x in a.input.split(","))
    models = {}
    for spec in a.model:
        factory = load_target(spec)
        name = spec.split(":")[-1]
        models[name] = factory()

    if a.compare_gpus:
        from .microarch import compare_hardware, print_hardware_compare

        gpu_keys = [k.strip() for k in a.compare_gpus.split(",") if k.strip()]
        target_phase = "fwd" if a.phase in ("forward", "both") else "train"
        for mname, mod in models.items():
            rows = compare_hardware(mod, shape, gpus=gpu_keys, dtype=a.dtype, phase=target_phase, kv_cache_gb=a.kv_cache_gb)
            print_hardware_compare(
                rows,
                title=f"\n[Microarch Hardware Compare] {mname} (input={shape}, dtype={a.dtype}, phase={target_phase})",
                show_advice=a.microarch,
            )
        return

    print_compare(models, shape, gpu, use=a.use, dtype=a.dtype, phase=a.phase)

    if a.microarch:
        from .microarch import microarch_report

        target_phase = "fwd" if a.phase in ("forward", "both") else "train"
        for mname, mod in models.items():
            rep = microarch_report(mod, shape, gpu=gpu, dtype=a.dtype, phase=target_phase, kv_cache_gb=a.kv_cache_gb)
            print(
                "\n[Microarch] %-16s T_uArch=%.2fus dominant=%s wave_eff=%.1f%% "
                "(TC=%.2fus ALU=%.2fus SFU=%.2fus Mem=%.2fus Sync=%.2fus grid_sync=%d l2=%s)"
                % (
                    mname,
                    rep["T_microarch_floor_us"],
                    rep["dominant_wall"],
                    rep["wave_efficiency"] * 100.0,
                    rep["T_tc_us"],
                    rep["T_alu_us"],
                    rep["T_sfu_us"],
                    rep["T_mem_us"],
                    rep["T_sync_and_serial_us"],
                    rep["grid_sync_stages"],
                    rep["l2_resident"],
                )
            )
            for adv in rep["advice"]:
                print("  -> %s" % adv)

    if a.formal_z3 or a.emit_lean:
        from . import autodiff_bounds
        from . import formal_prover
        target_phase = "forward" if a.phase == "forward" else "train_store"
        for mname, mod in models.items():
            dag = autodiff_bounds.dag_from_torch_module(mod, shape, dtype=a.dtype, gpu=gpu)
            if a.formal_z3:
                cert = formal_prover.prove_lower_bound_z3(dag, gpu, phase=target_phase)
                print(
                    "[Z3] %-16s T_abs=%.2fus binding=%s proven=%s core=%s"
                    % (mname, cert["T_abs_floor_us"], cert["binding"], cert["proven"], cert["unsat_core"])
                )
            if a.emit_lean:
                out_p = a.emit_lean if len(models) == 1 else f"{a.emit_lean}.{mname}.lean"
                lres = formal_prover.emit_lean4_certificate(dag, gpu, out_p, phase=target_phase, verify=True)
                print("[Lean4] %-14s certified=%dns exit=%s" % (mname, lres["certified_ns"], lres["exit_code"]))

    if a.precision_sweep:
        from .precision import precision_sweep, print_precision_sweep
        for mname, mod in models.items():
            res_sw = precision_sweep(mod, shape, gpu)
            print_precision_sweep(res_sw, title=f"\n[Precision Sweep] {mname} on {gpu.name} (shape={shape})")

    if a.rewrite_opt:
        from . import autodiff_bounds
        from . import formal_prover
        target_phase = "fwd" if a.phase == "forward" else "train"
        for mname, mod in models.items():
            dag = autodiff_bounds.dag_from_torch_module(mod, shape, dtype=a.dtype, gpu=gpu)
            rcert = formal_prover.prove_rewrite_family_z3(dag, gpu, phase=target_phase)
            print(
                "\n[RewriteOpt] %-14s T_base=%.2fus -> T_rewrite=%.2fus (%.2fx) proven=%s groups=%d"
                % (
                    mname,
                    rcert["T_baseline_us"],
                    rcert["T_rewrite_floor_us"],
                    rcert["speedup_vs_baseline"],
                    rcert["proven"],
                    rcert["n_rewrite_groups"],
                )
            )
            for pr in rcert["prescriptions"]:
                print("  - [%s] %s -> %s (save %.2fus): %s" % (pr["rule"], pr["op"], pr["variant"], pr["delta_us"], pr["reason"]))


if __name__ == "__main__":
    main()
