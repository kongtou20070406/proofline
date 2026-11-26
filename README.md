English · [简体中文](README.zh-CN.md)

# Proofline

**Zero-GPU, formally verified hardware latency lower bounds (Roofline + Z3 + Lean 4) for PyTorch models.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/kongtou20070406/proofline/actions/workflows/ci.yml/badge.svg)](https://github.com/kongtou20070406/proofline/actions/workflows/ci.yml)

Before spending days writing Triton/CUDA kernels or running GPU sweeps, **Proofline** traces any `torch.nn.Module` strictly on CPU (`FakeTensorMode` on `meta` tensors — zero VRAM, zero mutation of caller weights) and computes three speed-of-light latency floors backed by Z3 SMT UNSAT certificates and optional Lean 4 proofs.

## Quickstart

Compare candidate architectures and verify the DAG lower bound with Z3 without touching a GPU:

```bash
python -m proofline.cli \
  --model examples/quickstart_models.py:mlp_block examples/quickstart_models.py:transformer_block \
  --input 1,256,512 \
  --gpu rtx5090_measured \
  --dtype bf16 \
  --formal-z3
```

Actual output:

```text
RTX5090-measured: 195.7 TF(bf16) / 95.0 TF(fp32), 1508 GB/s, L2 96 MB, 0.93 us graph / 10.32 us eager
input (1, 256, 512) | dtype=bfloat16 | ranked by T_stage_fwd (lower = faster floor)

model             T_fwd(us)  T_trn(us)  T_eag(us)  vsBest  stages    W(GF)   Q_f(MB)  binding
------------------------------------------------------------------------------------------------
mlp_block               6.4       18.4       58.0   1.00x       3    1.077      7.60  compute
transformer_bloc       21.2       58.0      289.5   3.30x      13    1.752     23.86   memory
[Z3] mlp_block        T_abs=33.43us binding=memory proven=True core=['global_hbm_and_sram_io_ceiling', 'claim_beating_lower_bound']
[Z3] transformer_block T_abs=86.13us binding=memory proven=True core=['claim_beating_lower_bound', 'global_hbm_and_sram_io_ceiling']
```

Python API:

```python
from proofline import (
    GPU_PRESETS,
    analyze,
    dag_from_torch_module,
    emit_lean4_certificate,
    prove_lower_bound_z3,
)
from examples.quickstart_models import transformer_block

gpu = GPU_PRESETS["rtx5090_measured"]
model = transformer_block()

# 1. Stage-fused and persistent-kernel latency floors (zero VRAM)
report = analyze(model, (1, 256, 512), gpu, dtype="bf16")
print(report["T_stage_fwd_us"], report["T_stage_train_us"], report["binding"])

# 2. Z3 UNSAT+SAT speed-limit certificate & standalone Lean 4 proof
dag = dag_from_torch_module(model, (1, 256, 512), dtype="bf16", gpu=gpu)
z3_cert = prove_lower_bound_z3(dag, gpu, phase="train_store")
lean_cert = emit_lean4_certificate(dag, gpu, "TransformerCert.lean", phase="train_store", verify=False)
```

## What Proofline Computes

Proofline reports three hardware lower bounds assuming unchanged mathematics and numerical precision:

| Bound | Formula / Model | Optimization Target It Represents |
| --- | --- | --- |
| `T_abs_floor` | $\max(W_{\text{tot}}/F_{\text{eff}},\, Q_{\min}/B,\, S_{\text{crit}}\tau_{\text{graph}})$ | Cross-layer persistent kernel / SRAM-resident ideal limit (Hong–Kung I/O + critical path). No kernel can beat this without changing the math or precision. |
| `T_stage_fwd` / `T_stage_train` | $\sum_{\ell} \max(W_\ell/F_{\text{eff}},\, Q_\ell/B,\, \tau_{\text{graph}})$ | Well-tuned Triton / `torch.compile` + CUDA Graph execution fusing pointwise/residual chains while respecting tensor dependencies. |
| `T_eager` | Stage time $+ N_{\text{kernels}} \cdot \tau_{\text{eager}}$ | Unfused eager PyTorch dispatch reference line; the gap to `T_stage` shows how much kernel fusion headroom remains. |

Pure metadata view ops (`view`, `reshape`, `permute`, `transpose`, `slice`, `unfold`, `chunk`) contribute `0 FLOP`, `0 B`, and `0` kernel launches, while non-contiguous copy-triggering reshapes are detected automatically.

Each bottleneck tag maps directly to an actionable engineering knob:
- **`compute`**: Reduce FLOPs or raise arithmetic intensity; faster Tensor Cores / FP8 directly help.
- **`memory`**: Fuse operators and tile activations in SRAM; reducing MACs alone helps little.
- **`serial`**: Shorten sequential depth or fuse across recurrent steps; upgrading raw TFLOP/s barely moves runtime.

## Installation

Core PyTorch analyzer (CPU-only compatible, zero GPU needed):

```bash
pip install git+https://github.com/kongtou20070406/proofline.git
```

Include Z3 SMT + SymPy symbolic certification:

```bash
pip install "proofline[formal] @ git+https://github.com/kongtou20070406/proofline.git"
```

For local Lean 4 verification, install [Lean 4](https://leanprover.github.io/) (`>= v4.34.1`, `lean` in `PATH` or `PROOFLINE_LEAN_BIN`).

## Built-in Hardware Presets & Calibration

| Preset | Device | FP32 Peak | BF16 Peak | HBM Bandwidth | $\tau_{\text{graph}} / \tau_{\text{eager}}$ |
| --- | --- | ---: | ---: | ---: | ---: |
| `rtx5090_measured` | NVIDIA RTX 5090 (measured) | 95.0 TF | 195.7 TF | 1,508 GB/s | 0.93 / 10.32 $\mu$s |
| `rtx5090` | NVIDIA RTX 5090 (spec) | 104.8 TF | 209.5 TF | 1,792 GB/s | 0.93 / 10.32 $\mu$s |
| `h100` | NVIDIA H100 SXM | 67.0 TF | 989.4 TF | 3,350 GB/s | 1.00 / 10.00 $\mu$s |
| `a100` | NVIDIA A100 80GB | 19.5 TF | 312.0 TF | 2,039 GB/s | 1.20 / 10.00 $\mu$s |

Run `python -m proofline.cli --list-gpus` for all presets, instantiate `proofline.GPU(...)` for custom ASICs/GPUs, or measure your local GPU directly with `python -m proofline.measure_machine`.

## Caveats & v3 Extensions (Precision, L2 Residency & Exact Rewrites)

`T_abs_floor` is a hard lower bound **only under fixed math and precision**. Proofline v3 models three ways real code can move below a naive baseline floor:
1. **Precision & 2:4 structured sparsity (`--precision-sweep`)**: Quantizing to `FP8`/`INT8`/`INT4` or enabling 2:4 sparsity updates `F_eff` and byte width `b` (including group-quant scale metadata) while keeping sensitive ops (`LayerNorm`, `RMSNorm`, `Softmax`, residual adds) at high precision (`>= bf16/fp32`).
2. **L2 Cache resident steady state**: When weights + active activations fit inside chip L2 (e.g., 96 MB on RTX 5090), `phase_report` also emits `T_l2_resident_us` alongside cold-start HBM bounds.
3. **Exact equivalence rewrite family (`--rewrite-opt`)**: Uses Z3 `Optimize` + `UNSAT` over exact algebraic rewrites $\mathcal{R}_{\text{exact}}$ (MatMul chain associativity, shared-input GEMM fusion like QKV/SwiGLU, `Conv3x3` Direct vs. Winograd/FFT, and activation rematerialization) to prove the optimal rewrite-family floor and emit concrete prescriptions. Approximate algorithmic changes (e.g., sparse/linear attention) alter function outputs and stay outside $\mathcal{R}_{\text{exact}}$.

Actual `--precision-sweep --rewrite-opt` output on `transformer_block` (`RTX5090-measured`, `(1, 256, 512)`):

```text
[Precision Sweep] transformer_block on RTX5090-measured (shape=(1, 256, 512))
Format         2:4      T_fwd(us)  T_abs(us)     Q(MB)  vs_FP32 Binding  HighPrecOps
----------------------------------------------------------------------------------
FP32           no           41.03      19.53    49.033    1.00x memory   6 (norm=fp32)
BF16           no           25.98      19.53    24.517    1.58x serial   6 (norm=bf16)
FP8(g128)+2:4  yes          20.54      19.53    14.604    2.00x serial   6 (norm=bf16)

[RewriteOpt] transformer_block T_base=30.69us -> T_rewrite=30.69us (1.00x) proven=True groups=16
```

## Formal Proofs (Z3 + SymPy + Lean 4)

Proofline couples empirical tracing with machine-checked certificates:
- **Z3 SMT (`prove_lower_bound_z3`)**: Proves `UNSAT` (with minimal `unsat_core`) that no dependency-respecting schedule can execute strictly below `T_abs_floor`, alongside a `SAT` feasibility witness at `T_abs_floor`.
- **SymPy (`derive_symbolic_bounds`)**: Emits exact closed-form `W_fwd`, `W_train`, Hong–Kung tiled `Q_hk`, and compute-bound critical arithmetic intensity (`F / BW`).
- **Lean 4 ([`proofs/PerfRulerBounds.lean`](proofs/PerfRulerBounds.lean))**: Standalone `import Std` theorems (`EXIT=0` under `lean -DwarningAsError=true`, zero `sorry`) covering universal DAG makespan bounds, reverse-mode AD critical paths, recomputation-vs-storage crossovers, and Hong–Kung SRAM I/O lower bounds, plus auto-generated per-model certificates (`emit_lean4_certificate`, sample in [`proofs/examples/RecurrentWM_RTX5090_Cert.lean`](proofs/examples/RecurrentWM_RTX5090_Cert.lean)).

## Contributing & License

Contributions, custom device presets, and new operator handlers are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

Released under the [MIT License](LICENSE).
