[English](README.md) · 简体中文

# Proofline

**零 GPU、带 Z3 + Lean 4 形式化证明的 PyTorch 硬件理论时延下限分析工具。**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![CI](https://github.com/kongtou20070406/proofline/actions/workflows/ci.yml/badge.svg)](https://github.com/kongtou20070406/proofline/actions/workflows/ci.yml)

在花几天重写 Triton/CUDA kernel 或上卡反复压测之前，**Proofline** 只在纯 CPU 上通过 `FakeTensorMode` 与 `meta` 张量追踪任意 `torch.nn.Module`（**零显存占用、绝不修改原模型设备与权重**），算清三档硬件速度下限，并输出 **Z3 SMT 不可满足证明** 与 **Lean 4 机器证明证书**，先帮你摸清还剩多少优化空间、瓶颈在算力、显存带宽还是串行深度。

## 快速上手

不用占用任何 GPU，直接对比多个模型并输出 Z3 下界证明：

```bash
python -m proofline.cli \
  --model examples/quickstart_models.py:mlp_block examples/quickstart_models.py:transformer_block \
  --input 1,256,512 \
  --gpu rtx5090_measured \
  --dtype bf16 \
  --formal-z3
```

实测输出：

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

Python 调用：

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

# 1. 计算阶段融合与持久核理论下界（零显存占用）
report = analyze(model, (1, 256, 512), gpu, dtype="bf16")
print(report["T_stage_fwd_us"], report["T_stage_train_us"], report["binding"])

# 2. 生成 Z3 UNSAT+SAT 证明与独立 Lean 4 证书
dag = dag_from_torch_module(model, (1, 256, 512), dtype="bf16", gpu=gpu)
z3_cert = prove_lower_bound_z3(dag, gpu, phase="train_store")
lean_cert = emit_lean4_certificate(dag, gpu, "TransformerCert.lean", phase="train_store", verify=False)
```

## 三档下界与瓶颈诊断

在不改数学定义、不降数值精度的前提下，Proofline 同时给出三档时延地板：

| 指标 | 定义公式 | 对应的工程优化极限 |
| --- | --- | --- |
| `T_abs_floor` | $\max(W_{\text{tot}}/F_{\text{eff}},\, Q_{\min}/B,\, S_{\text{crit}}\tau_{\text{graph}})$ | **跨层持久核极限**：假设全网驻留片上 SRAM、最小化 HBM 读写（含 Hong–Kung 片上缓存下界）后的物理地板。不改算法/精度不可突破。 |
| `T_stage_fwd` / `T_stage_train` | $\sum_{\ell} \max(W_\ell/F_{\text{eff}},\, Q_\ell/B,\, \tau_{\text{graph}})$ | **阶段融合优化极限**：对应高质量 Triton / `torch.compile` + CUDA Graph 融合逐元素与残差链路后的真实工程上限。 |
| `T_eager` | 阶段时延 $+ N_{\text{kernels}} \cdot \tau_{\text{eager}}$ | **未融合 Eager 参考线**：和前两档对比即可直接看出继续卷算子融合最多还能快几倍。 |

零拷贝视图算子（`view`、`reshape`、`permute`、`transpose`、`slice`、`unfold`、`chunk`）一律计为 `0 FLOP`、`0 B`、`0` 次 kernel 发射；会触发物理重排的非连续形状变换则会被自动识别计入。

三种瓶颈标签直接对应优化方向：
- **`compute`（算力受限）**：需砍 FLOPs 或换用更高算力卡/张量核；
- **`memory`（显存带宽受限）**：优先做算子融合与 SRAM 分块复用，单纯减 MACs 收益极小；
- **`serial`（串行深度受限）**：需缩短串行步数或跨步融合，盲目升级算力几乎不提速。

## 安装方式

安装核心 PyTorch 分析器（支持纯 CPU 环境，无需 GPU）：

```bash
pip install git+https://github.com/kongtou20070406/proofline.git
```

安装 Z3 SMT 与 SymPy 形式化验证扩展：

```bash
pip install "proofline[formal] @ git+https://github.com/kongtou20070406/proofline.git"
```

如需在本地用 Lean 4 编译证书，请安装 [Lean 4](https://leanprover.github.io/)（`>= v4.34.1`，并确保 `lean` 在 `PATH` 中或设置 `PROOFLINE_LEAN_BIN`）。

## 内置硬件预设与实机标定

| 预设名 | 设备 | FP32 峰值 | BF16 峰值 | 显存带宽 | $\tau_{\text{graph}} / \tau_{\text{eager}}$ |
| --- | --- | ---: | ---: | ---: | ---: |
| `rtx5090_measured` | NVIDIA RTX 5090（实测） | 95.0 TF | 195.7 TF | 1,508 GB/s | 0.93 / 10.32 $\mu$s |
| `rtx5090` | NVIDIA RTX 5090（标称） | 104.8 TF | 209.5 TF | 1,792 GB/s | 0.93 / 10.32 $\mu$s |
| `h100` | NVIDIA H100 SXM | 67.0 TF | 989.4 TF | 3,350 GB/s | 1.00 / 10.00 $\mu$s |
| `a100` | NVIDIA A100 80GB | 19.5 TF | 312.0 TF | 2,039 GB/s | 1.20 / 10.00 $\mu$s |

可通过 `python -m proofline.cli --list-gpus` 查看全部预设，用 `proofline.GPU(...)` 自定义硬件参数，或在本机 GPU 上运行 `python -m proofline.measure_machine` 生成实测校准档。

## 适用边界（什么时候实测能突破下限）

`T_abs_floor` 仅在**计算图与精度不变**时构成硬下限；打破下列任一前提时实测可以更快：
1. **更改精度或稀疏格式**：例如量化为 FP8/INT4 或开启 2:4 硬件稀疏，实际改变了硬件算力峰值 $F_{\text{eff}}$ 与每字节流量 $b$（切换 `dtype` 或设备预设即可重算）。
2. **改写了算法本身**：例如将标准卷积换为 Winograd/FFT，或把稠密注意力换为稀疏/线性近似，属于更换计算图。
3. **整网工作集全驻留 L2 Cache**：若权重与激活总量低于芯片 L2 容量（例如 RTX 5090 的 96 MB），持久运行可免去 HBM 往返。

## 形式化证明（Z3 + SymPy + Lean 4）

Proofline 将图追踪与三层机器证明直接绑定：
- **Z3 SMT（`prove_lower_bound_z3`）**：用 `UNSAT` 与最小冲突核证明任何满足因果依赖的调度都无法低于 `T_abs_floor`，同时给出可达 `SAT` 构造见证。
- **SymPy（`derive_symbolic_bounds`）**：自动推导 `W_fwd`、`W_train`、Hong–Kung 分块 I/O 及算力/访存临界算术强度（`F / BW`）。
- **Lean 4（[`proofs/PerfRulerBounds.lean`](proofs/PerfRulerBounds.lean)）**：纯 `import Std` 零外部数学库依赖、零 `sorry`（通过 `lean -DwarningAsError=true` 零警告校验），严格证明通用 DAG 关键路径、反向传播依赖、重计算交叉阈值及 Hong–Kung I/O 下界；并支持为任意 PyTorch 模型一键生成可编译证书（参考样例 [`proofs/examples/RecurrentWM_RTX5090_Cert.lean`](proofs/examples/RecurrentWM_RTX5090_Cert.lean)）。

## 参与贡献与开源协议

欢迎提交新算子支持、自定义芯片档案或问题反馈，请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

本项目基于 [MIT License](LICENSE) 开源。
