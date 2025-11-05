# Repository Guidelines

## Project Structure & Module Organization
Core design notes live in `project.plan.md`; review the relevant section before adding or modifying lowering logic. Prototype kernels and integration sketches belong in `examples/`, with `examples/helion_matmul.py` showing the expected Helion-to-MLIR workflow. Keep Python package code under a `src/helion_fx_mlir/` tree (create it if missing) so imports remain stable; place auxiliary utilities beside the features they support. Configuration lives at the repo root (`requirements.txt`, `.gitignore`, optional `build.sh` scripts), so update those files whenever dependencies or tooling change.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: establish an isolated environment aligned with the documented Python ≥3.10 baseline.
- `python -m pip install -r requirements.txt`: install PyTorch, Helion, Triton, and typed helpers; switch to the CUDA or ROCm extras noted in the file when targeting GPUs.
- `python examples/helion_matmul.py`: execute the reference kernel end-to-end to sanity-check FX capture and autotuning hooks.
- `pytest -q`: run the test suite; add `-k name` to scope runs while iterating.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and type hints for public helpers. Modules, functions, and variables use `snake_case`; classes use `CamelCase`; constants stay upper snake. Prefer explicit imports grouped as stdlib, third-party, local. Structure kernels and lowerings so that each helper handles one IR concern; add short docstrings explaining any MLIR construction strategy. Use f-strings for formatting, and keep lines ≤100 characters unless MLIR text needs more room.

## Testing Guidelines
Use `pytest` for all new behavior. Mirror source layout under a `tests/` package and name files `test_<feature>.py`. Exercise both positive and failure paths, and rely on parameterized cases to cover shape/dtype combinations. For graph-emission code, assert on emitted MLIR strings or structural helpers rather than string-matching entire modules, keeping tests resilient. Ensure new tests run quickly so they can be executed alongside the matmul example.

## Commit & Pull Request Guidelines
Write concise, present-tense commit messages (`lowering: add affine.emit helper`) roughly following the existing lower-case, action-first style. Each pull request should summarize the change, call out any dependencies (e.g., new requirements), and link to the relevant checklist item in `project.plan.md`. Include before/after evidence when altering generated MLIR, and highlight any follow-up tasks so reviewers can plan incremental landings. Request review once `pytest` and the matmul example finish cleanly.
