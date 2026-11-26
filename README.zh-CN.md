[English](README.md) · 简体中文

# Proofline

**零 GPU PyTorch 性能预算器：在显式理想化 Roofline / 调度模型下输出 Z3 + Lean 4 机器核验时延下界证书。**

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

| 预设名 | 厂商 / 架构 | 核心单元数 | 显存 / 统一内存 | L2 / 缓存 | BF16 峰值 | 显存带宽 | 校准状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `rtx5090_measured` | NVIDIA `sm120` | 170 | 32 GB | 96 MB | 195.7 TF | 1,508 GB/s | **已实测** |
| `b200` | NVIDIA `sm100` | 148 | 192 GB | 128 MB | 2,250.0 TF | 8,000 GB/s | 官方参数·未实测待验证 |
| `h100` | NVIDIA `sm90` | 132 | 80 GB | 50 MB | 989.4 TF | 3,350 GB/s | 官方参数·未实测待验证 |
| `a100` | NVIDIA `sm80` | 108 | 80 GB | 40 MB | 312.0 TF | 2,039 GB/s | 官方参数·未实测待验证 |
| `l4` / `t4` | NVIDIA `sm89` / `sm75` | 58 / 40 | 24 / 16 GB | 48 / 4 MB | 121 / 65 TF | 300 / 320 GB/s | 官方参数·未实测待验证 |
| `mi300x` / `mi325x` | AMD `cdna3` | 304 | 192 / 256 GB | 256 MB | 1,307.4 TF | 5,300 / 6,000 GB/s | 官方参数·未实测待验证 |
| `tpu_v6e` / `tpu_v5p` | Google TPU | 2 / 8 MXUs | 31.25 / 95 GB | 64 / 128 MB | 918 / 459 TF | 1,638 / 2,765 GB/s | 官方参数·未实测待验证 |
| `m3_ultra` / `m4_max` | Apple Silicon | 80 / 40 | 512 / 128 GB | 96 / 48 MB | 64 / 34 TF | 819 / 546 GB/s | 官方参数·未实测待验证 |

> **实测状态说明：** 当前仅 `rtx5090_measured` 完成过真实硬件标定，其余芯片均基于官方公开白皮书参数建模（**未实测，待实机验证**）。如果你手头有 H100、B200、MI300X 或 Apple Silicon 设备，跑一条 15 秒命令即可生成实测 JSON 档：

```bash
python -m proofline.measure_machine --json
```

## v4 微架构五堵物理墙与跨芯片横评（`--compare-gpus --microarch`）

Proofline v4 在 Roofline + Z3 + Lean 4 中加入了**五堵硬件物理墙**：
1. **SM 波次量化 / TPU 脉动阵列填充**：当 CTA 数不足或不是 $N_{\text{SM}}$ 整数倍时计入空转惩罚（TPU 上按 $128\times 128$ / $256\times 256$ MXU 对齐率计算）。
2. **三单元执行上限**：把 FLOPs 拆进 **Tensor Core**、**FP32 ALU** 与 **SFU（`exp`/`rsqrt`/`softmax`/`gelu`）** 三条独立屋顶。
3. **三级存储与片上分块**：纳入 `SMEM + TMEM` 分块容量、L2 / Infinity Cache / SLC 驻留与 HBM 带宽。
4. **跨 SM 全局栅栏 vs. Cluster DSM**：工作集超出单 CTA / Cluster DSM 容量时强制计入跨 SM 同步延迟。
5. **大模型显存溢出与主机内存搬运（`--kv-cache-gb`）**：支持零内存 `meta` 张量追踪大模型，超出可用显存时自动算出 PCIe offload 搬运下界与最小不溢出卡数（Apple 统一内存免除 PCIe 搬运）。

`transformer_block`（`(1, 256, 512)`，`bf16`）跨芯片 `--compare-gpus` 实测输出：

```text
[Microarch Hardware Compare] transformer_block (input=(1, 256, 512), dtype=bf16, phase=fwd)
Device            Arch     SMs   VRAM Fit? Offload(ms) T_uArch(us) T_wave(us)  T_mem(us) WaveEff  L2? Chips DominantWall 
------------------------------------------------------------------------------------------------------------------------
B200              sm100    148   192G yes         0.00      15.50       8.14       0.85   13.7% yes      1 grid_sync    
H100              sm90     132    80G yes         0.00      26.50      16.51       2.04   15.4% yes      1 grid_sync    
RTX5090-measured  sm120    170    32G yes         0.00     107.48     107.48       4.52   11.9% yes      1 wave_compute 
MI300X            cdna3    304   192G yes         0.00      39.50      28.77       1.29    6.7% yes      1 grid_sync    
TPU-v6e           tpu_v6e    2    31G yes         0.00       4.16       2.12       4.16   97.1% yes      1 memory       
M3-Ultra          apple_m3   80   512G yes         0.00      61.00      51.12       8.33   58.5% yes      1 grid_sync    
```

## 适用边界与 v3 扩展（多精度、L2 驻留与等价改图）

`T_abs_floor` 仅在**计算图与精度不变**时构成硬下限。Proofline v3 对三种突破朴素基准下限的路径给出了显式建模：
1. **多精度与 2:4 结构化稀疏（`--precision-sweep`）**：切到 `FP8`/`INT8`/`INT4` 或开启 2:4 硬件稀疏会更新 `F_eff` 与字节流量 `b`（含分组量化 scale 元数据开销），同时强制 `LayerNorm`、`RMSNorm`、`Softmax` 与残差累加保留高精度（`>= bf16/fp32`）。
2. **L2 Cache 稳态驻留**：当权重与活跃激活总量小于芯片 L2 容量（例如 RTX 5090 的 96 MB）时，`phase_report` 会在冷启动 HBM 下界之外同步给出 `T_l2_resident_us`。
3. **严格等价改写族与改图处方（`--rewrite-opt`）**：通过 Z3 `Optimize` + `UNSAT` 在严格代数等价改写族 $\mathcal{R}_{\text{exact}}$（矩阵链结合律、同输入 QKV/SwiGLU 融合、`Conv3x3` Direct/Winograd/FFT 切换、激活重计算权衡）内求得最低时延下界并输出改图处方；稀疏/线性注意力等近似算法会改变输出函数，归为近似对照，不混入严格等价证明。

`transformer_block` 在 `RTX5090-measured`（`(1, 256, 512)`）下的 `--precision-sweep --rewrite-opt` 实测输出：

```text
[Precision Sweep] transformer_block on RTX5090-measured (shape=(1, 256, 512))
Format         2:4      T_fwd(us)  T_abs(us)     Q(MB)  vs_FP32 Binding  HighPrecOps
----------------------------------------------------------------------------------
FP32           no           41.03      19.53    49.033    1.00x memory   6 (norm=fp32)
BF16           no           25.98      19.53    24.517    1.58x serial   6 (norm=bf16)
FP8(g128)+2:4  yes          20.54      19.53    14.604    2.00x serial   6 (norm=bf16)

[RewriteOpt] transformer_block T_base=30.69us -> T_rewrite=30.69us (1.00x) proven=True groups=16
```

## 机器核验下界证书与三层显式假设（Z3 + SymPy + Lean 4）

Proofline 将纯 CPU 图追踪与**显式理想化 Roofline / 调度模型**下的机器核验绑定在一起。Z3 与 Lean 4 证明的是“在该组显式调度与硬件公理内部不存在更快的合法调度”，而不是对真实芯片微码、驱动或编译器实现做端到端语义验证。每份证书都显式依赖三层假设：
1. **计算图追踪保真度**：`FakeTensorMode` / FX 追踪是否完整捕获了真实运行时的全部算子、张量形状与依赖关系（未被追踪的动态控制流或外部副作用不在覆盖范围内）。
2. **必做计算与访存量（$W, Q$）的不可回避性**：所统计的 FLOPs 与访存流量在固定算法、精度或声明的严格等价改写族 $\mathcal{R}_{\text{exact}}$ 下是否确实无法绕开。
3. **理想化硬件与调度模型边界**：抽象硬件参数（$F_{\text{eff}}, B, \text{L2}, N_{\text{SM}}, \tau$）及缓存/融合/同步公理是否贴合目标芯片与编译器实际行为。

在此显式模型内：
- **Z3 SMT（`prove_lower_bound_z3`、`prove_microarch_bound_z3`、`prove_rewrite_family_z3`）**：显式编码逐算子/逐单元资源守恒（$w_i \ge W_i$、$\sum w_i \le F_{\text{eff}} T$）、DAG 因果依赖与鸽笼显存溢出约束，以 `UNSAT` 与最小冲突核检验不存在更低合法时延，并给出可达 `SAT` 见证。
- **SymPy（`derive_symbolic_bounds`）**：自动推导 `W_fwd`、`W_train`、Hong–Kung 分块 I/O 及算力/访存临界算术强度（`F / BW`）。
- **Lean 4（纯 `import Std`、零 `sorry`、`lean -DwarningAsError=true` 零警告）**：
  - [`proofs/OperationalBounds.lean`](proofs/OperationalBounds.lean)：从离散时间步的局部硬件容量轨迹出发，通过 Fubini 双重求和交换直接推导出算力、访存、三单元（`TC`/`ALU`/`SFU`）与关键路径下界（不再把全局不等式假设为结构字段）。
  - [`proofs/PebbleIOBounds.lean`](proofs/PebbleIOBounds.lean)：在显式缓存状态机（`CacheState` + `ValidRun`）上按结构归纳证明冷启动未命中、鸽笼显存溢出（`|req| - V <= loadCount`）与 Hong–Kung 时间片访问上界（`<= M + loadCount`）。
  - [`proofs/PerfRulerBounds.lean`](proofs/PerfRulerBounds.lean) 与 `emit_lean4_certificate`：通用调度下界定理与逐模型自动证书（参考样例 [`proofs/examples/RecurrentWM_RTX5090_Cert.lean`](proofs/examples/RecurrentWM_RTX5090_Cert.lean)）。

## 参与贡献与开源协议

欢迎提交新算子支持、自定义芯片档案或问题反馈，请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。

本项目基于 [MIT License](LICENSE) 开源。
