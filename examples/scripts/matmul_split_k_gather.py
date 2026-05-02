"""
Matmul Split-K with custom gather op
=====================================
Verification example: replaces ``hl.atomic_add`` in the split-K matmul
with the custom ``gather(tile_k, acc)`` op registered via Helion's decorator
API.  The resulting MLIR should contain a ``loom.gather`` placeholder op.

Usage::

    cd /root/loom-monorepo/third_party/helion-mlir
    python examples/scripts/matmul_split_k_gather.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

# ---------------------------------------------------------------------------
# Path setup: make sure helion_mlir and custom_op are importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
_CUSTOM_OP_ROOT = _REPO_ROOT / "custom_op"

for p in [str(_SRC_ROOT), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from helion_mlir import print_debug_info, generate_mlir, validate_with_mlir_opt
from custom_op import gather  # registers the op with Helion's decorator API


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@helion.kernel(static_shapes=False, dot_precision="ieee")
def split_k_matmul_gather(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Split-K matmul using the custom gather op instead of atomic_add.

    For each (tile_m, tile_n, tile_k) block:
      1. Compute a partial product: acc = a[tile_m, tile_k] @ b[tile_k, tile_n]
      2. Gather acc across all tile_k iterations → shape [K//tile_k, tile_m, tile_n]

    The gather result is stored back for demonstration purposes.
    In a real kernel the consumer would reduce (e.g. torch.sum) and write out.
    """
    m, k = a.shape
    _, n = b.shape
    out_ = torch.empty((m, n), device=a.device, dtype=a.dtype)

    for tile_m, tile_n, tile_k in hl.tile([m, n, k]):
        local_acc = torch.mm(a[tile_m, tile_k], b[tile_k, tile_n])
        # Gather partial results from all tile_k iterations.
        # Lowering emits a ``loom.gather`` placeholder MLIR op.
        gathered = gather(tile_k, local_acc)
        if tile_k.id == 0:
            acc_across_k = torch.sum(gathered, 0)
            out_[tile_m, tile_n] = acc_across_k

    return out_


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    m, k, n = 512, 4096, 512
    a = torch.randn([m, k], device="cpu", dtype=torch.float16)
    b = torch.randn([k, n], device="cpu", dtype=torch.float16)
    bound_kernel = split_k_matmul_gather.bind((a, b))

    print_debug_info(bound_kernel)

    try:
        mlir_text = generate_mlir(bound_kernel, cleanup=False, assume_divisible_tiles=True)
        print("=== MLIR Dump ===")
        print(mlir_text)

        # Validate with --allow-unregistered-dialect because loom.gather is
        # a placeholder op not registered in upstream MLIR.
        result = validate_with_mlir_opt(
            mlir_text,
            extra_args=["--allow-unregistered-dialect"],
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            print("mlir-opt validation FAILED.")
            sys.exit(1)
        else:
            print("mlir-opt validation succeeded.\n")

        # Quick sanity: confirm loom.gather appears in the output.
        if "loom.gather" in mlir_text:
            print("✓  loom.gather op found in MLIR output.")
        else:
            print("✗  loom.gather op NOT found — check visit_gather in ir_visitor.py")
    except Exception as e:
        import traceback
        print(f"\nMLIR generation failed: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
