"""Enhanced MLIR text builder with utility methods for structured emission.

This module provides a builder class that manages indentation, SSA value naming,
and provides convenient methods for emitting common MLIR constructs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    pass


class MLIROutputHelper:
    """Helper class to manage indentation and SSA name creation for MLIR text emission."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._indent = 0
        self._tmp_counter = 0

    # -------------------------------------------------------------------------
    # Core emission methods
    # -------------------------------------------------------------------------

    def emit(self, text: str) -> None:
        """Emit a line of MLIR text with current indentation."""
        self._lines.append("  " * self._indent + text)

    def push(self) -> None:
        """Increase indentation level."""
        self._indent += 1

    def pop(self) -> None:
        """Decrease indentation level."""
        assert self._indent > 0, "Cannot pop below zero indentation"
        self._indent -= 1

    def fresh(self, hint: str = "tmp") -> str:
        """Generate a fresh SSA value name with the given hint prefix."""
        name = f"%{hint}{self._tmp_counter}"
        self._tmp_counter += 1
        return name

    def build(self) -> str:
        """Build and return the complete MLIR text."""
        return "\n".join(self._lines) + "\n"


# -----------------------------------------------------------------------------
# Utility functions for MLIR text formatting
# -----------------------------------------------------------------------------

def torch_dtype_to_mlir_element_type(dtype: torch.dtype) -> str:
    """Convert a torch dtype to MLIR element type string."""
    mapping = {
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float64: "f64",
        torch.int32: "i32",
        torch.int64: "i64",
        torch.int8: "i8",
        torch.int16: "i16",
        torch.bool: "i1",
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype for MLIR emission: {dtype}")
    return mapping[dtype]


def format_tensor_type(shape: Sequence[int | None], element_type: str) -> str:
    """Format an MLIR tensor type string.
    
    Args:
        shape: Tensor shape, with None for dynamic dimensions
        element_type: MLIR element type (e.g., "f32")
        
    Returns:
        MLIR tensor type string (e.g., "tensor<?x?xf32>")
    """
    if not shape:
        return f"tensor<{element_type}>"
    dims = "x".join("?" if dim is None else str(dim) for dim in shape)
    return f"tensor<{dims}x{element_type}>"


def add_local_memory_kind_tensor_encoding(
    tensor_type: str, local_mem_kind: int
) -> str:
    """Add the structured local-memory-space encoding to a ranked tensor type."""
    if not tensor_type.startswith("tensor<") or not tensor_type.endswith(">"):
        raise ValueError(f"Expected ranked tensor type, got {tensor_type}")
    return f"{tensor_type[:-1]}, {{local_mem_kind = {local_mem_kind} : i64}}>"


def format_memref_type(shape: Sequence[int | None], element_type: str) -> str:
    """Format an MLIR memref type string.
    
    Args:
        shape: Memref shape, with None for dynamic dimensions
        element_type: MLIR element type (e.g., "f32")
        
    Returns:
        MLIR memref type string (e.g., "memref<?x?xf32>")
    """
    if not shape:
        return f"memref<{element_type}>"
    dims = "x".join("?" if dim is None else str(dim) for dim in shape)
    return f"memref<{dims}x{element_type}>"


def format_shape_attr(shape: Sequence[int | None]) -> str:
    """Format a shape as an MLIR array attribute.
    
    Uses -1 to represent dynamic dimensions.
    """
    if not shape:
        return "[]"
    body = ", ".join(str(dim) if dim is not None else "-1" for dim in shape)
    return f"[{body}]"


def format_string_attr(value: str | None) -> str:
    """Format a string as an MLIR string attribute."""
    if value is None:
        return '""'
    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def format_attr_dict(attrs: dict[str, str | None]) -> str:
    """Format a dictionary of attributes as an MLIR attribute dict.
    
    Filters out None values.
    """
    items = [f"{key} = {value}" for key, value in attrs.items() if value is not None]
    if not items:
        return ""
    return " {" + ", ".join(items) + "}"
