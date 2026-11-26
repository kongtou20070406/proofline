"""Verify first-principles operational Lean 4 proofs and per-node Z3 conservation bounds."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from examples.quickstart_models import mlp_block
from proofline import GPU_PRESETS, dag_from_torch_module, prove_lower_bound_z3, prove_microarch_bound_z3

PROOFS_DIR = Path(__file__).resolve().parents[1] / "proofs"


@pytest.mark.parametrize("lean_file", ["PerfRulerBounds.lean", "OperationalBounds.lean", "PebbleIOBounds.lean"])
def test_lean4_operational_proofs_compile_cleanly(lean_file: str) -> None:
    lean_bin = shutil.which("lean")
    if not lean_bin:
        pytest.skip("lean binary not installed on PATH")
    target = PROOFS_DIR / lean_file
    assert target.is_file()
    assert "sorry" not in target.read_text(encoding="utf-8")
    res = subprocess.run([lean_bin, "-DwarningAsError=true", str(target)], capture_output=True, text=True)
    assert res.returncode == 0, f"{lean_file} failed:\n{res.stdout}\n{res.stderr}"


def test_z3_bounds_are_sensitive_to_node_work_not_tautological() -> None:
    pytest.importorskip("z3")
    gpu = GPU_PRESETS["rtx5090_measured"]
    dag = dag_from_torch_module(mlp_block(), (1, 256, 512), dtype="bf16", gpu=gpu)

    cert = prove_lower_bound_z3(dag, gpu, phase="forward")
    assert cert["proven"] is True
    assert cert["z3_refutation_unsat"] is True
    assert cert["z3_tight_witness_sat"] is True

    ucert = prove_microarch_bound_z3(dag, gpu, phase="fwd")
    assert ucert["proven"] is True
    assert ucert["z3_refutation_unsat"] is True
    assert ucert["z3_tight_witness_sat"] is True
