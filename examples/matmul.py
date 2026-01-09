"""
Helion Matmul Kernel Example
============================
This example demonstrates a Helion kernel implementation of matrix multiplication
with support for a customizable epilogue function. It includes autotuning,
correctness checks against PyTorch baselines, and integration with tritonbench.
"""

# %%
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import sys
from pathlib import Path

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

# Make sure the repo-local src/ package layout is importable when the example is
# executed directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from helion_fx_mlir import generate_plan_stage0_mlir, validate_with_helion_opt


# %%
@helion.kernel(
    static_shapes=False,
    # Disable autotung over unrolling/range_num_stages
    # tl.dot is pipelined with num_stages
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def matmul(
    x: Tensor,
    y: Tensor,
    epilogue: Callable[[Tensor, tuple[Tensor, ...]], Tensor] = lambda acc, tile: acc,
) -> Tensor:
    """
    Performs matrix multiplication of x and y with an optional epilogue function.
    Args:
        x (Tensor): Left matrix of shape [m, k].
        y (Tensor): Right matrix of shape [k, n].
        epilogue (Callable, optional): Function applied to the accumulator and tile indices
            after the matmul. Defaults to identity (no change).
    Returns:
        Tensor: Resulting matrix of shape [m, n].
    """
    m, k = x.size()
    k2, n = y.size()
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = epilogue(acc, (tile_m, tile_n))
    return out


# %%
def autotune(m: int, k: int, n: int) -> None:
    """
    Runs autotuning on the matmul kernel with a ReLU epilogue and saves the best config.
    Args:
        m (int): Number of rows in matrix x.
        k (int): Number of columns in matrix x and rows in matrix y.
        n (int): Number of columns in matrix y.
    """
    x = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
    y = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
    bias = torch.randn([n], device=DEVICE, dtype=torch.float16)
    args = (x, y, lambda acc, tile: torch.relu(acc + bias[tile[1]]))
    best_config = matmul.autotune(args, force=True)
    print(f"Best config: {best_config}")
    best_config.save("best_config.json")


# %%
def check(m: int, k: int, n: int) -> None:
    """
    Checks the correctness of the matmul kernel against PyTorch baselines.
    Tests:
    - Plain matmul without bias.
    - Matmul with bias added in the epilogue.
    - Matmul with a more complex epilogue applying ReLU after bias addition.
    Args:
        m (int): Number of rows in matrix x.
        k (int): Number of columns in matrix x and rows in matrix y.
        n (int): Number of columns in matrix y.
    """
    x = torch.randn([m, k], device=DEVICE, dtype=torch.float16)
    y = torch.randn([k, n], device=DEVICE, dtype=torch.float16)
    bias = torch.randn([n], device=DEVICE, dtype=torch.float16)
    bias_scalar = torch.randn([1], device=DEVICE, dtype=torch.float16)
    # Test without bias
    run_example(matmul, torch.matmul, (x, y))

    # Test addmm forward + backward with different alpha/beta values
    print("\n\n=== AddMM Forward + Backward Test (Alpha=2.0, Beta=0.5) ===")
    run_example(
        addmm_autograd,
        lambda bias, mat1, mat2, alpha, beta: torch.addmm(
            bias, mat1, mat2, alpha=alpha, beta=beta
        ),
        (input_grad, mat1_grad, mat2_grad, 2.0, 0.5),
        kernel_name="helion_addmm_autograd_scaled",
        baseline_name="torch",
        rtol=1e-2,
        atol=1e-2,
        bwd=True,
    )


def main() -> None:
    """
    Main function to run autotuning (commented out) and correctness checks.
    """
    m, k, n = 128, 128, 256
    x = torch.randn([m, k], device="cpu", dtype=torch.float32)
    y = torch.randn([k, n], device="cpu", dtype=torch.float32)
    bound_kernel = matmul.bind((x, y))

    mlir_text = generate_plan_stage0_mlir(
        bound_kernel,
        kernel_name="matmul",
    )
    print("=== MLIR Dump ===")
    print(mlir_text)

    result = validate_with_helion_opt(
        mlir_text, opt_path="/mnt/fast/llvm-mlir/bin/mlir-opt"
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit("mlir-opt validation failed (see stderr above).")
    print("mlir-opt validation succeeded.\n")

    # autotune(1024, 1024, 1024)
    # check(1024, 1024, 1024)


# %%
if __name__ == "__main__":
    main()
