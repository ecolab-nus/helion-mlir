"""Debug utilities for Helion MLIR generation.

This module provides helper functions for debugging and validating MLIR output:
- print_device_ir: Print Device IR graphs (filtered for rolled reductions)
- print_nodes_with_symbols: Print nodes with symbolic values
- print_compile_env: Print CompileEnvironment block sizes and shape environment
- print_debug_info: Combined debug output
- validate_with_mlir_opt: Validate MLIR with mlir-opt
- run_dce_cleanup: Run dead code elimination on MLIR
"""

from typing import Any, Iterable
import subprocess
from pathlib import Path

MLIR_OPT_CANDIDATES = [
    Path("/mnt/fast/llvm-mlir/bin/mlir-opt"),
    Path("/usr/bin/mlir-opt"),
    Path("/usr/local/bin/mlir-opt"),
]


def print_device_ir(bound_kernel: Any) -> None:
    """Prints the Device IR graphs, filtering out rolled reductions."""
    print("=== Device IR ===")
    
    # Filter out graphs that are part of rolled reductions
    rolled_ids = set()
    # Check for rolled_reductions attribute safely, though it should exist on valid device_ir
    if hasattr(bound_kernel.host_function.device_ir, "rolled_reductions"):
        rolled_ids = {
            info.new_graph_id 
            for info in bound_kernel.host_function.device_ir.rolled_reductions 
            if info.new_graph_id is not None
        }

    for i, g in enumerate(bound_kernel.host_function.device_ir.graphs):
        if i in rolled_ids:
            continue
        print(f"Graph {i}: {type(g).__name__}")
        g.graph.print_tabular()
    print("\n")

def print_nodes_with_symbols(bound_kernel: Any) -> None:
    """Prints nodes that have symbolic values associated with them."""
    print("=== Nodes with symbols ===")
    for i, g in enumerate(bound_kernel.host_function.device_ir.graphs):
        for node in g.graph.nodes:
            if "val" in node.meta:
                print(f"Node {node.name} : {node.meta['val']}")
                    
    print("\n")

def print_compile_env(bound_kernel: Any) -> None:
    """Prints the CompileEnvironment information."""
    print("=== Compile Environment ===")
    env = bound_kernel.env
    print(f"Block Sizes ({len(env.block_sizes)}):")
    for bs in env.block_sizes:
        print(f"  Block {bs.block_id}: Size={bs.size}, Var={bs.var}, Reduction={bs.reduction}, Source={bs.block_size_source}")
    print(f"Shape Env ({len(env.shape_env.var_to_val)}):")
    for var, val in env.shape_env.var_to_val.items():
        print(f"  Var {var}: {val}")
    print("\n")

def print_debug_info(bound_kernel: Any) -> None:
    """Prints Device IR, nodes with symbols, and compile environment."""
    print_device_ir(bound_kernel)
    print_nodes_with_symbols(bound_kernel)
    print_compile_env(bound_kernel)


# -----------------------------------------------------------------------------
# MLIR Validation and Cleanup Utilities
# -----------------------------------------------------------------------------

def _find_mlir_opt(opt_path: str | Path | None = None) -> Path:
    """Find mlir-opt executable."""
    tool_candidates: Iterable[Path] = MLIR_OPT_CANDIDATES if opt_path is None else [Path(opt_path)]
    
    for candidate in tool_candidates:
        if candidate.exists():
            return candidate
    
    raise FileNotFoundError(
        "Unable to locate `mlir-opt`. "
        "Install LLVM/MLIR or pass `opt_path` explicitly."
    )


def validate_with_mlir_opt(
    mlir_text: str,
    *,
    opt_path: str | Path | None = None,
    extra_args: Iterable[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `mlir-opt` to confirm the emitted IR parses.
    
    Uses -allow-unregistered-dialect to allow loom.* and torch.* operations.
    """
    tool = _find_mlir_opt(opt_path)
    
    args = [str(tool), "-allow-unregistered-dialect"]
    if extra_args:
        args.extend(extra_args)
    
    return subprocess.run(
        args,
        input=mlir_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def run_dce_cleanup(
    mlir_text: str,
    *,
    opt_path: str | Path | None = None,
) -> str:
    """Run mlir-opt with dead code elimination pass.
    
    Returns the cleaned up MLIR text.
    Raises RuntimeError if mlir-opt fails.
    """
    tool = _find_mlir_opt(opt_path)
    
    args = [
        str(tool),
        "-allow-unregistered-dialect",
        "--canonicalize",
        "--cse",
    ]
    
    result = subprocess.run(
        args,
        input=mlir_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"mlir-opt DCE cleanup failed: {result.stderr}")
    
    return result.stdout
