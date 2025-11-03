# Repository Guidelines

## Project Structure & Module Organization
The workspace covers the two-stage Helion FX pipeline described in `project.plan.md`. Keep Python capture code in `helion2json/core/` and its tests in `helion2json/tests/`. C++ lowering lives in `json2mlir/include/helion_mlir/` for headers, `json2mlir/lib/` for implementations, and `json2mlir/tests/FileCheck/` for golden MLIR fixtures. Place shared JSON samples under `samples/` and keep experimental notebooks or scratch files outside the repo to avoid noise.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates the expected local environment; reinstall whenever system Python changes.
- `pip install -r requirements.txt` pulls PyTorch, Triton, and Helion with the right CUDA/ROCm wheels; add `-e helion2json` when the package metadata lands.
- `cmake -S json2mlir -B build -G Ninja -DLLVM_DIR=/path/to/llvm` configures MLIR tooling; pass `-DTORCH_MLIR_HOME` when the torch dialect is required.
- `ninja -C build json2mlir` builds the CLI; `ninja -C build check-json2mlir` runs FileCheck tests; use `pytest helion2json/tests` for the Python suite.
- `cmake --build build --target format` (once enabled) should gate formatting before commits.

## Coding Style & Naming Conventions
Python follows PEP 8 with 4-space indents, type hints, and `snake_case` symbols; run `python -m black --target-version py310` and `python -m ruff check` prior to review. C++ mirrors the LLVM/MLIR style guide: 2-space indents, `UpperCamelCase` for types, `lower_case` for functions, and header include guards using `HELION_MLIR_`. Keep MLIR textual IR readable with consistent indenting and comments only for non-obvious semantics.

## Testing Guidelines
Add `pytest` cases alongside fixtures in `helion2json/tests/` and keep names `test_<feature>.py`. For lowering, extend the FileCheck corpus and drive them through `ninja -C build check-json2mlir`; complex flows can use `llvm-lit` when available. Include JSON samples that exercise edge tiles, dynamic shapes, and error paths. Aim for fail-fast checks and document new diagnostics in the expected MLIR files.

## Commit & Pull Request Guidelines
Use short, imperative commit subjects (e.g., `Add edge-tile clamp for matmul`) with optional bullet bodies explaining rationale and testing. Scope each PR to a logical slice, reference planning notes or GitHub issues, and show before/after MLIR snippets or CLI output when behavior changes. Confirm CI or local `pytest`/`ninja check-json2mlir` runs before requesting review.

## Security & Configuration Tips
Avoid committing real customer kernels or credentials; anonymize JSON before sharing. Pin the LLVM/MLIR build in PR descriptions so reviewers stay aligned. When testing GPU wheels, note the CUDA or ROCm index used so others can reproduce the environment.
