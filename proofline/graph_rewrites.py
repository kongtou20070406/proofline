"""
graph_rewrites -- Exact Algebraic Equivalence Rewrite Family (R_exact).

Enumerates mathematically exact, semantics-preserving algebraic rewrites on a ComputeDAG:
  (a) matmul_chain_assoc: optimal DP parenthesization of consecutive Linear/MatMul chains
      without intermediate nonlinearity (e.g. (AB)C vs A(BC)).
  (b) fuse_shared_input_linears: fuse sibling Linear ops sharing the same input activation
      (e.g. Q/K/V or SwiGLU gate+up) into a single wider GEMM.
  (c) conv3x3_winograd_fft: exact Conv2d stride-1 Direct vs Winograd F(2x2, 3x3) (2.25x MAC
      reduction + explicit transform & tile SRAM check) vs FFT (large kernels k >= 5).
  (d) remat_tradeoff: activation save (2*A/BW) vs rematerialization (W_remat/F_eff) in training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .autodiff_bounds import ComputeDAG, OpNode, _hong_kung_gemm_io
from .devices import GPU, dtype_bytes


@dataclass(frozen=True)
class RewriteVariant:
    variant: str
    W_flops: float
    Q_bytes: float
    S_steps: float
    reason: str
    valid: bool = True


@dataclass(frozen=True)
class RewriteGroup:
    group_id: str
    rule: str
    op_label: str
    node_ids: tuple[int, ...]
    baseline: RewriteVariant
    candidates: tuple[RewriteVariant, ...]


def _matrix_chain_dp(dims: list[int]) -> tuple[float, list[tuple[int, int]]]:
    """Standard Matrix Chain Multiplication DP returning (min_flops, split_order)."""
    n = len(dims) - 1
    if n <= 1:
        return 0.0, []
    dp = [[0.0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float("inf")
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + 2.0 * dims[i] * dims[k + 1] * dims[j + 1]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k
    return dp[0][n - 1], [(i, split[i][n - 1]) for i in range(n - 1)]


def _matmul_chain_groups(
    dag: ComputeDAG,
    device: GPU,
    used_nodes: set[int],
    b_elem: int,
) -> list[RewriteGroup]:
    """Detect consecutive Linear/MatMul chains without intermediate nonlinearity."""
    nodes = dag.nodes
    children: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for n in nodes:
        for p in n.parents:
            if 0 <= p < len(nodes):
                children[p].append(n.id)

    def _is_pure_chain_gemm(n: OpNode) -> bool:
        if n.kind != "gemm":
            return False
        if n.attrs.get("has_bias", False):
            return False
        return all(k in n.attrs for k in ("m", "k", "n"))

    groups: list[RewriteGroup] = []
    visited: set[int] = set()
    for i, node in enumerate(nodes):
        if i in visited or i in used_nodes or not _is_pure_chain_gemm(node):
            continue
        # Walk forward along single-child linear chain
        chain = [node]
        cur = node
        while len(children[cur.id]) == 1:
            nxt_id = children[cur.id][0]
            nxt = nodes[nxt_id]
            if (
                nxt_id in used_nodes
                or not _is_pure_chain_gemm(nxt)
                or len(nxt.parents) != 1
                or int(nxt.attrs["k"]) != int(cur.attrs["n"])
                or int(nxt.attrs.get("batch", 1)) != int(cur.attrs.get("batch", 1))
            ):
                break
            chain.append(nxt)
            cur = nxt

        if len(chain) < 2:
            continue
        for c in chain:
            visited.add(c.id)

        # Chain of k GEMMs corresponds to multiplying k+1 matrices:
        # X: (M, K_0), W_0: (K_0, N_0), W_1: (N_0, N_1), ...
        batch = int(chain[0].attrs.get("batch", 1))
        m = int(chain[0].attrs["m"])
        dims = [m, int(chain[0].attrs["k"])] + [int(c.attrs["n"]) for c in chain]

        # Left-to-right (direct) cost
        w_direct = sum(c.W_fwd for c in chain)
        q_direct = sum(max(c.Q_fwd, c.Q_hk_fwd) for c in chain)
        s_direct = sum(c.tau_fwd for c in chain)

        # Optimal parenthesization across (X, W_0, ..., W_{L-1})
        w_opt_single, _ = _matrix_chain_dp(dims)
        w_opt = float(batch) * w_opt_single
        # Estimate Q for optimal association: if W_0 @ W_1 collapses inner dim, HBM traffic changes
        # Compulsory input + all weights + final output
        w_bytes = sum(float(c.attrs.get("param_bytes", 0.0)) for c in chain)
        x_bytes = float(chain[0].attrs.get("act_in_bytes", m * dims[1] * b_elem))
        y_bytes = float(chain[-1].attrs.get("out_bytes", m * dims[-1] * b_elem))
        q_comp_opt = x_bytes + w_bytes + y_bytes
        # Hong-Kung bound for the resulting fused/reassociated GEMM (M x N_last x K_first)
        q_hk_opt = _hong_kung_gemm_io(
            m, dims[-1], dims[1], batch, b_elem, q_comp_opt, device.sm_sram_bytes
        )
        # If W_opt < W_direct, association collapsed weight-weight product first (e.g. A @ (W1 @ W2))
        if w_opt < w_direct * 0.999:
            # Two-step or reassociated chain: W_prod then X @ W_prod
            q_opt = max(q_comp_opt, q_hk_opt)
            s_opt = 2.0 if len(chain) >= 2 else float(len(chain))
            reason = f"Matrix-chain DP {dims} reduces FLOPs from {w_direct/1e6:.2f}M to {w_opt/1e6:.2f}M"
        else:
            w_opt = w_direct
            q_opt = q_direct
            s_opt = s_direct
            reason = f"Left-to-right association {dims} is already FLOP-optimal"

        base = RewriteVariant("direct", w_direct, q_direct, s_direct, "Sequential GEMM chain")
        opt = RewriteVariant("assoc_opt", w_opt, q_opt, s_opt, reason)
        groups.append(
            RewriteGroup(
                group_id=f"matmul_chain_{chain[0].id}",
                rule="matmul_chain_assoc",
                op_label=f"nodes[{chain[0].id}..{chain[-1].id}]",
                node_ids=tuple(c.id for c in chain),
                baseline=base,
                candidates=(base, opt),
            )
        )
        used_nodes.update(c.id for c in chain)
    return groups


def _shared_input_linear_groups(
    dag: ComputeDAG,
    device: GPU,
    used_nodes: set[int],
    b_elem: int,
) -> list[RewriteGroup]:
    """Detect sibling Linear ops sharing the exact same input activation tensor."""
    by_input: dict[int, list[OpNode]] = {}
    for n in dag.nodes:
        if n.id in used_nodes or n.kind != "gemm":
            continue
        if not n.attrs.get("is_linear", False):
            continue
        in_root = n.attrs.get("input_root_id")
        if in_root is None:
            continue
        by_input.setdefault(int(in_root), []).append(n)

    groups: list[RewriteGroup] = []
    for in_root, siblings in by_input.items():
        if len(siblings) < 2:
            continue
        # All siblings share same M, K and input activation
        m = int(siblings[0].attrs.get("m", 1))
        k = int(siblings[0].attrs.get("k", 1))
        n_sum = sum(int(s.attrs.get("n", 1)) for s in siblings)
        batch = int(siblings[0].attrs.get("batch", 1))

        w_unfused = sum(s.W_fwd for s in siblings)
        q_unfused = sum(max(s.Q_fwd, s.Q_hk_fwd) for s in siblings)
        s_unfused = sum(s.tau_fwd for s in siblings)

        # Fused GEMM reads shared input X once and writes concatenated output
        x_bytes = float(siblings[0].attrs.get("act_in_bytes", m * k * b_elem))
        w_bytes_sum = sum(float(s.attrs.get("param_bytes", k * int(s.attrs.get("n", 1)) * b_elem)) for s in siblings)
        y_bytes_sum = sum(float(s.attrs.get("out_bytes", m * int(s.attrs.get("n", 1)) * b_elem)) for s in siblings)
        q_comp_fused = x_bytes + w_bytes_sum + y_bytes_sum
        q_fused = max(q_comp_fused, q_unfused - float(len(siblings) - 1) * x_bytes)
        s_fused = 1.0

        saved_mb = max(0.0, q_unfused - q_fused) / 1e6
        base = RewriteVariant("unfused", w_unfused, q_unfused, s_unfused, f"{len(siblings)} separate GEMMs")
        fused = RewriteVariant(
            "fused_gemm",
            w_unfused,
            q_fused,
            s_fused,
            f"Fuse {len(siblings)} shared-input linears (N_tot={n_sum}): saves {saved_mb:.2f} MB HBM & {int(s_unfused - 1)} launches",
        )
        groups.append(
            RewriteGroup(
                group_id=f"shared_in_{in_root}",
                rule="fuse_shared_input_linears",
                op_label="+".join(f"{s.name}#{s.id}" for s in siblings),
                node_ids=tuple(s.id for s in siblings),
                baseline=base,
                candidates=(base, fused),
            )
        )
        used_nodes.update(s.id for s in siblings)
    return groups


def _conv_winograd_fft_groups(
    dag: ComputeDAG,
    device: GPU,
    used_nodes: set[int],
    b_elem: int,
) -> list[RewriteGroup]:
    """Exact Conv2d stride-1 Direct vs Winograd F(2x2, 3x3) vs FFT."""
    groups: list[RewriteGroup] = []
    for n in dag.nodes:
        if n.id in used_nodes or n.kind != "conv":
            continue
        a = n.attrs
        if not all(k in a for k in ("batch", "c_in", "c_out", "h_out", "w_out", "kh", "kw", "stride_h", "stride_w")):
            continue
        sh, sw = int(a["stride_h"]), int(a["stride_w"])
        kh, kw = int(a["kh"]), int(a["kw"])
        if sh != 1 or sw != 1:
            continue

        b = int(a["batch"])
        cin = int(a["c_in"])
        cout = int(a["c_out"])
        ho = int(a["h_out"])
        wo = int(a["w_out"])
        w_direct = n.W_fwd
        q_direct = max(n.Q_fwd, n.Q_hk_fwd)
        s_direct = n.tau_fwd

        base = RewriteVariant("direct", w_direct, q_direct, s_direct, f"Direct {kh}x{kw} Conv2d")
        cands = [base]

        if kh == 3 and kw == 3:
            # Winograd F(2x2, 3x3): 4x4 tile = 16 muls per 2x2=4 outputs -> 4/9 (2.25x reduction)
            n_tiles = float(b) * float(math.ceil(ho / 2.0)) * float(math.ceil(wo / 2.0))
            w_elem_muls = 2.0 * 16.0 * float(cin) * float(cout) * n_tiles
            # Explicit transform overheads: B^T d B (32*cin*n_tiles), G g G^T (16*cout*cin), A^T M A (24*cout*n_tiles)
            w_xform = 32.0 * float(cin) * n_tiles + 16.0 * float(cout) * float(cin) + 24.0 * float(cout) * n_tiles
            w_wino = w_elem_muls + w_xform
            # Tile SRAM check: 4x4 tile for 16-channel block must fit in sm_sram_bytes
            tile_sram_need = 16.0 * float(min(cin, 32) + min(cout, 32)) * float(b_elem)
            sram_ok = (device.sm_sram_bytes <= 0.0) or (tile_sram_need <= device.sm_sram_bytes)
            if sram_ok:
                cands.append(
                    RewriteVariant(
                        "winograd_f2x2_3x3",
                        w_wino,
                        q_direct,
                        s_direct,
                        f"Winograd F(2x2,3x3) reduces MACs by {w_direct / max(w_wino, 1.0):.2f}x (tile SRAM {tile_sram_need/1024:.1f}KB <= {device.sm_sram_bytes/1024:.0f}KB)",
                        valid=True,
                    )
                )

        if kh >= 5 and kw >= 5:
            # FFT-based convolution: ~5 * H * W * log2(H * W) * (C_in + C_out) + 8 * H * W * C_in * C_out (complex mul)
            hw = max(float(ho * wo), 4.0)
            log_hw = math.log2(hw)
            w_fft = float(b) * (5.0 * hw * log_hw * float(cin + cout) + 8.0 * hw * float(cin) * float(cout))
            cands.append(
                RewriteVariant(
                    "fft_conv",
                    w_fft,
                    q_direct,
                    s_direct,
                    f"FFT Conv ({kh}x{kw}): {w_fft/1e6:.2f}M FLOPs vs Direct {w_direct/1e6:.2f}M",
                    valid=True,
                )
            )

        if len(cands) > 1:
            groups.append(
                RewriteGroup(
                    group_id=f"conv_{n.id}",
                    rule="conv3x3_winograd_fft",
                    op_label=f"{n.name}#{n.id}({kh}x{kw})",
                    node_ids=(n.id,),
                    baseline=base,
                    candidates=tuple(cands),
                )
            )
            used_nodes.add(n.id)
    return groups


def _remat_tradeoff_groups(
    dag: ComputeDAG,
    phase: str,
) -> list[RewriteGroup]:
    """During training, choose activation save (2*A) vs rematerialization (W_remat) per node."""
    if phase in ("fwd", "forward"):
        return []
    groups: list[RewriteGroup] = []
    for n in dag.nodes:
        if n.act_save_bytes <= 0.0 or n.remat_flops <= 0.0:
            continue
        if not (n.requires_grad_in or n.requires_grad_param):
            continue
        # Baseline is store (2*act_save_bytes HBM traffic, 0 extra FLOPs)
        base = RewriteVariant(
            "store_act",
            0.0,
            2.0 * n.act_save_bytes,
            0.0,
            f"Store activation ({2.0 * n.act_save_bytes / 1e6:.2f} MB HBM R+W)",
        )
        remat = RewriteVariant(
            "remat_act",
            n.remat_flops,
            0.0,
            0.0,
            f"Rematerialize activation ({n.remat_flops / 1e6:.2f} MFLOP, 0 MB HBM save)",
        )
        groups.append(
            RewriteGroup(
                group_id=f"remat_{n.id}",
                rule="remat_tradeoff",
                op_label=f"{n.name}#{n.id}",
                node_ids=(n.id,),
                baseline=base,
                candidates=(base, remat),
            )
        )
    return groups


def enumerate_rewrite_candidates(
    dag: ComputeDAG,
    device: GPU,
    phase: str = "fwd",
) -> list[RewriteGroup]:
    """Enumerate all exact algebraic equivalence rewrite groups on a ComputeDAG."""
    b_elem = int(max(1, math.ceil(dtype_bytes(dag.dtype))))
    used_nodes: set[int] = set()
    groups: list[RewriteGroup] = []
    groups.extend(_matmul_chain_groups(dag, device, used_nodes, b_elem))
    groups.extend(_shared_input_linear_groups(dag, device, used_nodes, b_elem))
    groups.extend(_conv_winograd_fft_groups(dag, device, used_nodes, b_elem))
    groups.extend(_remat_tradeoff_groups(dag, phase))
    return groups
