# Contributing to Proofline

Thanks for helping improve Proofline!

## Local Development Setup

```bash
git clone https://github.com/kongtou20070406/proofline.git
cd proofline
pip install -e ".[dev]"
```

All core tests run strictly on CPU (`FakeTensorMode` / `meta` tensors) with zero GPU required:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests
```

## Optional: Verifying Lean 4 Proofs Locally

Install [Lean 4](https://leanprover.github.io/) (`v4.34.1` or newer `Std`-compatible toolchain) and ensure `lean` is on your `PATH` (or set `PROOFLINE_LEAN_BIN=/path/to/lean`):

```bash
lean -DwarningAsError=true proofs/PerfRulerBounds.lean
lean -DwarningAsError=true proofs/examples/RecurrentWM_RTX5090_Cert.lean
```

## Calibration Profiles & New Hardware Presets

To benchmark and export a measured hardware profile on your own GPU (peak `bf16`/`fp32` FLOP/s, HBM bandwidth, eager dispatch latency, and CUDA-Graph kernel launch floor):

```bash
python -m proofline.measure_machine
python -m proofline.calibrate --help
```

Pass the resulting JSON to `proofline --profile your_gpu.json` or open a PR adding your device preset in [`proofline/devices.py`](proofline/devices.py).

## Documentation & Test Invariants

- Both [`README.md`](README.md) and [`README.zh-CN.md`](README.zh-CN.md) are covered by [`tests/test_readme_links.py`](tests/test_readme_links.py): every relative link must resolve and both READMEs must mirror top-level section structure.
- Core bounds changes must keep caller models unmutated (same device, same dtype, identical weights before and after `analyze()`).
