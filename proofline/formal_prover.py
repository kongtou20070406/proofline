"""
formal_prover -- Universal Z3 SMT + SymPy + Lean 4 speed-limit prover.

Given ANY ComputeDAG (or PyTorch nn.Module / op) and ANY DeviceSpec (GPU, CPU,
TPU, custom ASIC):
  1. prove_lower_bound_z3(...) proves via Z3 UNSAT + minimal unsat_core that NO
     valid parallel overlapping schedule can beat T_abs_floor (or T_stage_floor),
     and confirms SAT tightness at T = T_star.
  2. derive_symbolic_bounds(...) uses SymPy to output analytical closed-form
     forward/backward formulas and compute/memory/SRAM crossover thresholds.
  3. emit_lean4_certificate(...) emits a standalone `import Std` Lean 4 proof
     certifying cleared-denominator integer bounds with Lean 4.34.1 verification.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from typing import Any

import sympy as sp
import z3

from .autodiff_bounds import ComputeDAG, evaluate_dag_bounds
from .devices import GPU
from .graph_rewrites import RewriteGroup, enumerate_rewrite_candidates


def _find_lean_bin() -> str | None:
    return os.environ.get("PROOFLINE_LEAN_BIN") or shutil.which("lean")


def _r(val: float, scale: float = 1.0) -> z3.RatNumRef:
    """Convert non-negative float precisely into scaled rational for Z3."""
    return z3.Q(int(round(max(val, 0.0) * scale)), 1) if scale != 1.0 else z3.RealVal(f"{float(val):.12g}")


def prove_lower_bound_z3(
    dag: ComputeDAG,
    device: GPU,
    phase: str = "train_store",
    eps_rel: float = 1e-7,
) -> dict[str, Any]:
    """Formally certify via Z3 SMT that no feasible DAG schedule can run faster than T*."""
    ev = evaluate_dag_bounds(dag, device, phase=phase)
    t_abs_us = float(ev["T_abs_floor_us"])
    f_eff = float(device.peak_flops(dag.dtype))
    bw = float(device.BW)
    tau_g = float(device.tau_graph)

    w_tot_us_scale = float(ev["W_tot_GFLOP"]) * 1e9 * 1e6  # FLOPs * 1e6 -> time in us when divided by F_eff
    q_min_us_scale = max(float(ev["Q_min_io_MB"]), float(ev["Q_hk_MB"])) * 1e6 * 1e6

    # Build linear real-arithmetic SMT model in microseconds (avoids nonlinear F * T explosion)
    solver = z3.Solver()
    solver.set(unsat_core=True)
    T_total = z3.Real("T_total_us")
    solver.assert_and_track(T_total >= 0, "non_negative_makespan")

    t_comp_bound_us = w_tot_us_scale / f_eff
    t_mem_bound_us = q_min_us_scale / bw
    solver.assert_and_track(T_total >= _r(t_comp_bound_us), "global_compute_ceiling")
    solver.assert_and_track(T_total >= _r(t_mem_bound_us), "global_hbm_and_sram_io_ceiling")

    # Explicit forward DAG schedule variables + dependency causality
    n_nodes = len(dag.nodes)
    fwd_starts = [z3.Real(f"s_fwd_{i}") for i in range(n_nodes)]
    fwd_ends = [z3.Real(f"e_fwd_{i}") for i in range(n_nodes)]
    for i, node in enumerate(dag.nodes):
        solver.assert_and_track(fwd_starts[i] >= 0, f"fwd_start_ge0_{i}")
        solver.assert_and_track(fwd_ends[i] <= T_total, f"fwd_end_le_T_{i}")
        step_us = node.tau_fwd * tau_g * 1e6
        solver.assert_and_track(fwd_ends[i] - fwd_starts[i] >= _r(step_us), f"fwd_step_{i}")
        for p in node.parents:
            if 0 <= p < i:
                solver.assert_and_track(fwd_starts[i] >= fwd_ends[p], f"fwd_dep_{p}_to_{i}")

    if phase != "forward" and n_nodes > 0:
        # Reverse-mode AD: backward dX starts after forward node finishes and flows reverse DAG
        bwd_starts = [z3.Real(f"s_bwd_{i}") for i in range(n_nodes)]
        bwd_ends = [z3.Real(f"e_bwd_{i}") for i in range(n_nodes)]
        children_map: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
        for n in dag.nodes:
            for p in n.parents:
                if 0 <= p < n_nodes:
                    children_map[p].append(n.id)
        has_any_dw = any(n.W_bwd_dw > 0 for n in dag.nodes)
        for i, node in enumerate(dag.nodes):
            solver.assert_and_track(bwd_starts[i] >= fwd_ends[i], f"bwd_after_fwd_{i}")
            solver.assert_and_track(bwd_ends[i] <= T_total, f"bwd_end_le_T_{i}")
            dx_us = (node.tau_bwd * tau_g * 1e6) if node.requires_grad_in else 0.0
            if dx_us > 0:
                solver.assert_and_track(bwd_ends[i] - bwd_starts[i] >= _r(dx_us), f"bwd_dx_step_{i}")
            for c in children_map[i]:
                solver.assert_and_track(bwd_starts[i] >= bwd_ends[c], f"bwd_rev_dep_{c}_to_{i}")
        if has_any_dw:
            # Final weight gradient emission/step contributes at least 1*tau_g after fwd
            dw_tail = z3.Real("e_dw_tail")
            solver.assert_and_track(dw_tail <= T_total, "dw_tail_le_T")
            solver.assert_and_track(dw_tail >= _r((ev["S_crit_fwd"] + ev["S_crit_bwd"]) * tau_g * 1e6), "critical_chain_with_dw")

    # 1) Refutation check: can ANY schedule beat T_abs_floor * (1 - eps_rel)?
    strict_target = max(t_abs_us * (1.0 - eps_rel), 0.0)
    solver.push()
    if t_abs_us > 0.0:
        solver.assert_and_track(T_total < _r(strict_target), "claim_beating_lower_bound")
        ref_res = solver.check()
        core = [str(x) for x in solver.unsat_core()] if ref_res == z3.unsat else []
    else:
        ref_res = z3.unsat
        core = ["zero_cost_graph"]
    solver.pop()

    # 2) Witness tightness check: is T_total <= T_abs_floor * (1 + 1e-6) satisfiable?
    solver.push()
    solver.assert_and_track(T_total <= _r(t_abs_us * (1.0 + 1e-6)), "tight_upper_witness")
    sat_res = solver.check()
    solver.pop()

    return {
        "model": dag.name,
        "device": device.name,
        "dtype": dag.dtype,
        "phase": phase,
        "T_abs_floor_us": t_abs_us,
        "T_stage_floor_us": float(ev["T_stage_floor_us"]),
        "max_throughput_hz": float(ev["max_throughput_hz"]),
        "binding": ev["binding"],
        "z3_refutation_unsat": (ref_res == z3.unsat),
        "z3_tight_witness_sat": (sat_res == z3.sat),
        "proven": (ref_res == z3.unsat) and (sat_res == z3.sat),
        "unsat_core": core,
        "S_crit_tot": float(ev["S_crit_tot"]),
    }


def derive_symbolic_bounds(op_kind: str = "gemm") -> dict[str, Any]:
    """Derive symbolic closed-form forward/backward bounds and crossover thresholds using SymPy."""
    key = op_kind.lower().strip()
    F, BW, tau, M_sram, b = sp.symbols("F BW tau M_sram b", positive=True, real=True)

    if key == "gemm":
        B, M, N, K = sp.symbols("B M N K", positive=True, real=True)
        w_fwd = 2 * B * M * N * K
        w_train = 3 * w_fwd
        q_comp = (B * M * K + K * N + B * M * N) * b
        q_hk = 2 * B * M * N * K * b / sp.sqrt(M_sram / (3 * b))
        q_fwd = sp.Max(q_comp, q_hk)
        q_train = sp.Max(q_comp + B * M * N * b, 3 * q_hk)
        t_fwd = sp.Max(w_fwd / F, q_fwd / BW, tau)
        t_train = sp.Max(w_train / F, q_train / BW, 3 * tau)
        crit_ai = F / BW
        crit_sram_tile = sp.solve(sp.Eq(2 * B * M * N * K * b / sp.sqrt(M_sram / (3 * b)), w_fwd * BW / F), M_sram)
        return {
            "op": "gemm",
            "W_fwd": str(w_fwd),
            "W_train": str(w_train),
            "Q_hk_fwd": str(q_hk),
            "T_fwd_latex": sp.latex(t_fwd),
            "T_train_latex": sp.latex(t_train),
            "critical_arithmetic_intensity": str(crit_ai),
            "critical_sram_for_compute_bound": str(crit_sram_tile),
        }

    if key == "sdpa":
        B, H, N, D = sp.symbols("B H N D", positive=True, real=True)
        w_fwd = 4 * B * H * N**2 * D
        w_bwd = sp.Rational(5, 2) * w_fwd
        w_train = sp.simplify(w_fwd + w_bwd)
        q_comp = 4 * B * H * N * D * b
        q_fa_io = 4 * B * H * N**2 * D**2 * b**2 / M_sram
        q_fwd = sp.Max(q_comp, q_fa_io)
        t_fwd = sp.Max(w_fwd / F, q_fwd / BW, tau)
        t_train = sp.Max(w_train / F, sp.Rational(7, 2) * q_fwd / BW, 2 * tau)
        n_crit_serial = sp.sqrt(F * tau / (4 * B * H * D))
        return {
            "op": "sdpa",
            "W_fwd": str(w_fwd),
            "W_bwd": str(w_bwd),
            "W_train": str(w_train),
            "Q_flash_io": str(q_fa_io),
            "T_fwd_latex": sp.latex(t_fwd),
            "T_train_latex": sp.latex(t_train),
            "critical_seq_len_compute_vs_serial": str(n_crit_serial),
        }

    if key in ("recurrent_wm", "rwm"):
        T_r, H_r, w_step, w_roll, tau_step, tau_roll = sp.symbols(
            "T_r H_r w_step w_roll tau_step tau_roll", positive=True, real=True
        )
        k_grad = sp.symbols("k_grad", positive=True, real=True)
        w_fwd = T_r * w_step + H_r * w_roll
        w_bwd = 2 * (k_grad * w_step + H_r * w_roll)
        s_fwd = T_r * tau_step + H_r * tau_roll
        s_bwd = k_grad * tau_step + H_r * tau_roll
        t_fwd = sp.Max(w_fwd / F, s_fwd * tau)
        t_train = sp.Max((w_fwd + w_bwd) / F, (s_fwd + s_bwd) * tau)
        return {
            "op": "recurrent_wm",
            "W_fwd": str(w_fwd),
            "W_bwd": str(w_bwd),
            "S_fwd": str(s_fwd),
            "S_bwd": str(s_bwd),
            "T_fwd_latex": sp.latex(t_fwd),
            "T_train_latex": sp.latex(t_train),
            "critical_arithmetic_intensity": str(F / BW),
        }

    raise ValueError(f"Unsupported symbolic op_kind={op_kind!r}")


def _safe_ident(s: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", s)
    if not clean or clean[0].isdigit():
        clean = "inst_" + clean
    return clean


def emit_lean4_certificate(
    dag: ComputeDAG,
    device: GPU,
    out_path: str,
    phase: str = "train_store",
    verify: bool = False,
) -> dict[str, Any]:
    """Emit a self-contained Lean 4 (`import Std`) certificate with exact cleared-denominator integer bounds."""
    ev = evaluate_dag_bounds(dag, device, phase=phase)
    # Use nanosecond integer arithmetic + MFLOP / MB precision so all bounds hold with zero float loss.
    f_gflops = max(1, int(round(float(device.peak_flops(dag.dtype)) / 1e9)))  # FLOPs / ns
    bw_gbps = max(1, int(round(float(device.BW) / 1e9)))                      # bytes / ns
    tau_ns = max(1, int(round(float(device.tau_graph) * 1e9)))                # ns / step

    w_mflops = int(math.floor(float(ev["W_tot_GFLOP"]) * 1000.0))             # integer MFLOPs (conservative floor)
    w_flops = w_mflops * 1_000_000
    q_kb = int(math.floor(max(float(ev["Q_min_io_MB"]), float(ev["Q_hk_MB"])) * 1000.0))
    q_bytes = q_kb * 1000
    s_crit = int(math.floor(float(ev["S_crit_tot"])))
    t_star_ns = max(
        (w_flops + f_gflops - 1) // f_gflops if w_flops > 0 else 0,
        (q_bytes + bw_gbps - 1) // bw_gbps if q_bytes > 0 else 0,
        s_crit * tau_ns,
    )
    # Wait: compute exact integer lower bound floor_ns:
    t_comp_floor = w_flops // f_gflops
    t_mem_floor = q_bytes // bw_gbps
    t_ser_floor = s_crit * tau_ns
    t_floor_ns = max(t_comp_floor, t_mem_floor, t_ser_floor)

    ns = _safe_ident(f"{dag.name}_{device.name}_{phase}")
    lean_code = f"""import Std

/-!
Auto-generated formal hardware latency lower-bound certificate by `Proofline`.
Model  : {dag.name}
Device : {device.name} (dtype={dag.dtype}, F={f_gflops} GFLOP/s, BW={bw_gbps} GB/s, tau={tau_ns} ns)
Phase  : {phase}
-/

namespace {ns}

def peakFlopsPerNs : Nat := {f_gflops}
def bwBytesPerNs   : Nat := {bw_gbps}
def tauNs          : Nat := {tau_ns}

def workFlops      : Nat := {w_flops}
def trafficBytes   : Nat := {q_bytes}
def critSteps      : Nat := {s_crit}
def certifiedNs    : Nat := {t_floor_ns}

structure ValidSchedule where
  makespanNs : Nat
  compute_ok : workFlops <= peakFlopsPerNs * makespanNs
  memory_ok  : trafficBytes <= bwBytesPerNs * makespanNs
  serial_ok  : critSteps * tauNs <= makespanNs

theorem schedule_meets_compute_floor (s : ValidSchedule) :
    {t_comp_floor} <= s.makespanNs := by
  have h1 : peakFlopsPerNs * {t_comp_floor} <= workFlops := by decide
  have h2 : peakFlopsPerNs * {t_comp_floor} <= peakFlopsPerNs * s.makespanNs :=
    Nat.le_trans h1 s.compute_ok
  exact Nat.le_of_mul_le_mul_left h2 (by decide : 0 < peakFlopsPerNs)

theorem schedule_meets_memory_floor (s : ValidSchedule) :
    {t_mem_floor} <= s.makespanNs := by
  have h1 : bwBytesPerNs * {t_mem_floor} <= trafficBytes := by decide
  have h2 : bwBytesPerNs * {t_mem_floor} <= bwBytesPerNs * s.makespanNs :=
    Nat.le_trans h1 s.memory_ok
  exact Nat.le_of_mul_le_mul_left h2 (by decide : 0 < bwBytesPerNs)

theorem schedule_meets_serial_floor (s : ValidSchedule) :
    {t_ser_floor} <= s.makespanNs := by
  have h1 : {t_ser_floor} = critSteps * tauNs := by rfl
  rw [h1]
  exact s.serial_ok

theorem certified_speed_limit (s : ValidSchedule) :
    certifiedNs <= s.makespanNs := by
  have hc := schedule_meets_compute_floor s
  have hm := schedule_meets_memory_floor s
  have hs := schedule_meets_serial_floor s
  have hmax : certifiedNs <= {t_comp_floor} \\/ certifiedNs <= {t_mem_floor} \\/ certifiedNs <= {t_ser_floor} := by
    decide
  rcases hmax with h | h | h
  · exact Nat.le_trans h hc
  · exact Nat.le_trans h hm
  · exact Nat.le_trans h hs

theorem witness_feasible : Exists (fun s : ValidSchedule => s.makespanNs = {t_star_ns}) := by
  refine ⟨⟨{t_star_ns}, by decide, by decide, by decide⟩, rfl⟩

#print axioms certified_speed_limit
#print axioms witness_feasible

end {ns}
"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(lean_code)

    out: dict[str, Any] = {
        "lean_file": out_path,
        "namespace": ns,
        "certified_ns": t_floor_ns,
        "witness_ns": t_star_ns,
        "binding": ev["binding"],
        "verified": False,
        "exit_code": None,
        "stdout": "",
    }
    if verify:
        lean_bin = _find_lean_bin()
        if not lean_bin:
            out["stdout"] = "lean binary not found in PATH (set PROOFLINE_LEAN_BIN to verify locally)"
            return out
        res = subprocess.run(
            [lean_bin, "-DwarningAsError=true", out_path],
            capture_output=True,
            text=True,
        )
        out["exit_code"] = res.returncode
        out["stdout"] = (res.stdout + "\n" + res.stderr).strip()
        out["verified"] = (res.returncode == 0)
    return out


def prove_rewrite_family_z3(
    dag: ComputeDAG,
    device: GPU,
    phase: str = "fwd",
    eps_rel: float = 1e-6,
) -> dict[str, Any]:
    """Optimize over R_exact via Z3 Optimize, then prove no valid rewrite+schedule beats T_rewrite_floor via Z3 UNSAT."""
    norm_phase = "forward" if phase in ("fwd", "forward") else "train_store"
    ev = evaluate_dag_bounds(dag, device, phase=norm_phase)
    f_eff = float(device.peak_flops(dag.dtype))
    bw = float(device.BW)
    tau_g = float(device.tau_graph)

    w_base_flops = float(ev["W_tot_GFLOP"]) * 1e9
    q_base_bytes = max(float(ev["Q_min_io_MB"]), float(ev["Q_hk_MB"])) * 1e6
    s_base_steps = float(ev["S_crit_tot"])
    t_base_us = float(ev["T_abs_floor_us"])
    q_compulsory_floor = max(dag.input_bytes + dag.output_bytes + min(dag.param_bytes, q_base_bytes), 0.0)

    groups: list[RewriteGroup] = enumerate_rewrite_candidates(dag, device, phase=norm_phase)

    def _build_constraints(ctx_add, is_optimize: bool):
        T_tot = z3.Real("T_rewrite_us")
        W_tot = z3.Real("W_tot_flops")
        Q_tot = z3.Real("Q_tot_bytes")
        S_tot = z3.Real("S_tot_steps")

        ctx_add(T_tot >= 0, "non_negative_makespan")
        ctx_add(W_tot >= 0, "non_negative_work")
        ctx_add(Q_tot >= _r(q_compulsory_floor), "compulsory_io_floor")
        ctx_add(S_tot >= _r(1.0 if dag.nodes else 0.0), "non_empty_serial_floor")

        choice_vars: list[tuple[RewriteGroup, list[z3.BoolRef]]] = []
        dw_terms = []
        dq_terms = []
        ds_terms = []

        for g in groups:
            bvars = [z3.Bool(f"rw_{g.group_id}_{idx}") for idx in range(len(g.candidates))]
            choice_vars.append((g, bvars))
            # Exactly one variant selected in this equivalence group
            ctx_add(z3.PbEq([(bv, 1) for bv in bvars], 1), f"choose_one_{g.group_id}")
            for bv, cand in zip(bvars, g.candidates):
                if not cand.valid:
                    ctx_add(z3.Not(bv), f"invalid_{g.group_id}_{cand.variant}")
                dw = cand.W_flops - g.baseline.W_flops
                dq = cand.Q_bytes - g.baseline.Q_bytes
                ds = cand.S_steps - g.baseline.S_steps
                if abs(dw) > 0.0:
                    dw_terms.append(z3.If(bv, _r(dw), z3.RealVal(0)))
                if abs(dq) > 0.0:
                    dq_terms.append(z3.If(bv, _r(dq), z3.RealVal(0)))
                if abs(ds) > 0.0:
                    ds_terms.append(z3.If(bv, _r(ds), z3.RealVal(0)))

        w_expr = _r(w_base_flops) + (z3.Sum(dw_terms) if dw_terms else z3.RealVal(0))
        q_expr = _r(q_base_bytes) + (z3.Sum(dq_terms) if dq_terms else z3.RealVal(0))
        s_expr = _r(s_base_steps) + (z3.Sum(ds_terms) if ds_terms else z3.RealVal(0))

        ctx_add(W_tot == w_expr, "rewrite_work_balance")
        ctx_add(Q_tot >= q_expr, "rewrite_traffic_balance")
        ctx_add(S_tot >= s_expr, "rewrite_serial_balance")

        # Hardware speed-limit inequalities in microseconds
        ctx_add(T_tot >= W_tot * _r(1e6 / f_eff), "rewrite_compute_ceiling")
        ctx_add(T_tot >= Q_tot * _r(1e6 / bw), "rewrite_memory_ceiling")
        ctx_add(T_tot >= S_tot * _r(tau_g * 1e6), "rewrite_serial_ceiling")
        return T_tot, W_tot, Q_tot, S_tot, choice_vars

    # 1) Z3 Optimize to find T_rewrite_floor and winning rewrite assignment
    opt = z3.Optimize()
    T_opt, W_opt, Q_opt, S_opt, opt_choices = _build_constraints(lambda c, _n: opt.add(c), is_optimize=True)
    opt.minimize(T_opt)
    opt_res = opt.check()
    if opt_res != z3.sat:
        raise RuntimeError(f"Z3 Optimize failed on rewrite family: {opt_res}")

    model = opt.model()
    t_rw_us = float(model[T_opt].as_fraction())
    w_rw_flops = float(model[W_opt].as_fraction())
    q_rw_bytes = float(model[Q_opt].as_fraction())
    s_rw_steps = float(model[S_opt].as_fraction())

    selected: dict[str, str] = {}
    prescriptions: list[dict[str, Any]] = []
    for g, bvars in opt_choices:
        chosen_idx = 0
        for idx, bv in enumerate(bvars):
            if z3.is_true(model.evaluate(bv, model_completion=True)):
                chosen_idx = idx
                break
        cand = g.candidates[chosen_idx]
        selected[g.group_id] = cand.variant
        if cand.variant != g.baseline.variant:
            t_b = max(g.baseline.W_flops / f_eff, g.baseline.Q_bytes / bw, g.baseline.S_steps * tau_g) * 1e6
            t_c = max(cand.W_flops / f_eff, cand.Q_bytes / bw, cand.S_steps * tau_g) * 1e6
            delta_us = max(0.0, t_b - t_c)
            prescriptions.append(
                {
                    "op": g.op_label,
                    "rule": g.rule,
                    "variant": cand.variant,
                    "delta_us": round(delta_us, 4),
                    "reason": cand.reason,
                }
            )

    # 2) Z3 UNSAT proof that NO rewrite choice in R_exact + schedule beats T_rewrite_floor * (1 - eps_rel)
    solver = z3.Solver()
    solver.set(unsat_core=True)
    T_ref, _, _, _, _ = _build_constraints(lambda c, name: solver.assert_and_track(c, name), is_optimize=False)
    strict_target = max(t_rw_us * (1.0 - eps_rel), 0.0)
    if t_rw_us > 0.0:
        solver.assert_and_track(T_ref < _r(strict_target), "claim_beating_rewrite_floor")
        ref_res = solver.check()
        core = [str(x) for x in solver.unsat_core()] if ref_res == z3.unsat else []
    else:
        ref_res = z3.unsat
        core = ["zero_cost_graph"]

    return {
        "model": dag.name,
        "device": device.name,
        "dtype": dag.dtype,
        "phase": phase,
        "T_baseline_us": t_base_us,
        "T_rewrite_floor_us": t_rw_us,
        "speedup_vs_baseline": (t_base_us / t_rw_us) if t_rw_us > 0 else 1.0,
        "W_rewrite_GFLOP": w_rw_flops / 1e9,
        "Q_rewrite_MB": q_rw_bytes / 1e6,
        "S_rewrite_steps": s_rw_steps,
        "n_rewrite_groups": len(groups),
        "selected_rewrites": selected,
        "prescriptions": prescriptions,
        "z3_refutation_unsat": (ref_res == z3.unsat),
        "proven": (ref_res == z3.unsat),
        "unsat_core": core,
    }


def emit_rewrite_lean4_certificate(
    dag: ComputeDAG,
    device: GPU,
    out_path: str,
    phase: str = "fwd",
    verify: bool = False,
) -> dict[str, Any]:
    """Emit a standalone Lean 4 (`import Std`) certificate for the R_exact rewrite floor."""
    cert = prove_rewrite_family_z3(dag, device, phase=phase)
    f_gflops = max(1, int(round(float(device.peak_flops(dag.dtype)) / 1e9)))
    bw_gbps = max(1, int(round(float(device.BW) / 1e9)))
    tau_ns = max(1, int(round(float(device.tau_graph) * 1e9)))

    w_mflops = int(math.floor(float(cert["W_rewrite_GFLOP"]) * 1000.0))
    w_flops = w_mflops * 1_000_000
    q_kb = int(math.floor(float(cert["Q_rewrite_MB"]) * 1000.0))
    q_bytes = q_kb * 1000
    s_crit = int(math.floor(float(cert["S_rewrite_steps"])))

    t_comp_floor = w_flops // f_gflops
    t_mem_floor = q_bytes // bw_gbps
    t_ser_floor = s_crit * tau_ns
    t_floor_ns = max(t_comp_floor, t_mem_floor, t_ser_floor)
    t_star_ns = max(
        (w_flops + f_gflops - 1) // f_gflops if w_flops > 0 else 0,
        (q_bytes + bw_gbps - 1) // bw_gbps if q_bytes > 0 else 0,
        s_crit * tau_ns,
    )

    ns = _safe_ident(f"Rewrite_{dag.name}_{device.name}_{phase}")
    lean_code = f"""import Std

/-!
Auto-generated formal R_exact rewrite speed-limit certificate by `Proofline v3`.
Model  : {dag.name}
Device : {device.name} (dtype={dag.dtype}, F={f_gflops} GFLOP/s, BW={bw_gbps} GB/s, tau={tau_ns} ns)
Phase  : {phase}
Groups : {cert["n_rewrite_groups"]}
-/

namespace {ns}

def peakFlopsPerNs : Nat := {f_gflops}
def bwBytesPerNs   : Nat := {bw_gbps}
def tauNs          : Nat := {tau_ns}

def rewriteWorkFlops    : Nat := {w_flops}
def rewriteTrafficBytes : Nat := {q_bytes}
def rewriteCritSteps    : Nat := {s_crit}
def certifiedRewriteNs  : Nat := {t_floor_ns}

structure ValidRewriteSchedule where
  makespanNs : Nat
  compute_ok : rewriteWorkFlops <= peakFlopsPerNs * makespanNs
  memory_ok  : rewriteTrafficBytes <= bwBytesPerNs * makespanNs
  serial_ok  : rewriteCritSteps * tauNs <= makespanNs

theorem rewrite_meets_compute_floor (s : ValidRewriteSchedule) :
    {t_comp_floor} <= s.makespanNs := by
  have h1 : peakFlopsPerNs * {t_comp_floor} <= rewriteWorkFlops := by decide
  have h2 : peakFlopsPerNs * {t_comp_floor} <= peakFlopsPerNs * s.makespanNs :=
    Nat.le_trans h1 s.compute_ok
  exact Nat.le_of_mul_le_mul_left h2 (by decide : 0 < peakFlopsPerNs)

theorem rewrite_meets_memory_floor (s : ValidRewriteSchedule) :
    {t_mem_floor} <= s.makespanNs := by
  have h1 : bwBytesPerNs * {t_mem_floor} <= rewriteTrafficBytes := by decide
  have h2 : bwBytesPerNs * {t_mem_floor} <= bwBytesPerNs * s.makespanNs :=
    Nat.le_trans h1 s.memory_ok
  exact Nat.le_of_mul_le_mul_left h2 (by decide : 0 < bwBytesPerNs)

theorem rewrite_meets_serial_floor (s : ValidRewriteSchedule) :
    {t_ser_floor} <= s.makespanNs := by
  have h1 : {t_ser_floor} = rewriteCritSteps * tauNs := by rfl
  rw [h1]
  exact s.serial_ok

theorem certified_rewrite_speed_limit (s : ValidRewriteSchedule) :
    certifiedRewriteNs <= s.makespanNs := by
  have hc := rewrite_meets_compute_floor s
  have hm := rewrite_meets_memory_floor s
  have hs := rewrite_meets_serial_floor s
  have hmax : certifiedRewriteNs <= {t_comp_floor} \\/ certifiedRewriteNs <= {t_mem_floor} \\/ certifiedRewriteNs <= {t_ser_floor} := by
    decide
  rcases hmax with h | h | h
  · exact Nat.le_trans h hc
  · exact Nat.le_trans h hm
  · exact Nat.le_trans h hs

theorem rewrite_witness_feasible : Exists (fun s : ValidRewriteSchedule => s.makespanNs = {t_star_ns}) := by
  refine ⟨⟨{t_star_ns}, by decide, by decide, by decide⟩, rfl⟩

#print axioms certified_rewrite_speed_limit
#print axioms rewrite_witness_feasible

end {ns}
"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(lean_code)

    out: dict[str, Any] = {
        "lean_file": out_path,
        "namespace": ns,
        "certified_ns": t_floor_ns,
        "witness_ns": t_star_ns,
        "T_rewrite_floor_us": cert["T_rewrite_floor_us"],
        "prescriptions": cert["prescriptions"],
        "verified": False,
        "exit_code": None,
        "stdout": "",
    }
    if verify:
        lean_bin = _find_lean_bin()
        if not lean_bin:
            out["stdout"] = "lean binary not found in PATH (set PROOFLINE_LEAN_BIN to verify locally)"
            return out
        res = subprocess.run(
            [lean_bin, "-DwarningAsError=true", out_path],
            capture_output=True,
            text=True,
        )
        out["exit_code"] = res.returncode
        out["stdout"] = (res.stdout + "\n" + res.stderr).strip()
        out["verified"] = (res.returncode == 0)
    return out
