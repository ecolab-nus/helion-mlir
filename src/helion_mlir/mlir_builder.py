"""Enhanced MLIR text builder with utility methods for structured emission.

This module provides a builder class that manages indentation, SSA value naming,
and provides convenient methods for emitting common MLIR constructs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch

if TYPE_CHECKING:
    pass


class MLIRBuilder:
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

    # -------------------------------------------------------------------------
    # Module and function emission
    # -------------------------------------------------------------------------

    def emit_module_start(self, attrs: dict[str, tuple[object, str]] | None = None) -> None:
        """Emit the start of an MLIR module.
        
        Args:
            attrs: Optional dict of module attributes, mapping name to (value, type).
                   Example: {"loom.tile_m": (64, "index"), "loom.tensor_dims.x": ('"tile_m,tile_k"', "")}
                   If type is empty string, the value is emitted as-is (for string attrs).
        """
        if attrs:
            attr_strs = []
            for name, (value, typ) in attrs.items():
                if typ:
                    attr_strs.append(f"{name} = {value} : {typ}")
                else:
                    attr_strs.append(f"{name} = {value}")
            self.emit(f"module attributes {{{', '.join(attr_strs)}}} {{")
        else:
            self.emit("module {")
        self.push()

    def emit_module_end(self) -> None:
        """Emit the end of an MLIR module."""
        self.pop()
        self.emit("}")

    def emit_func_start(
        self,
        name: str,
        args: list[tuple[str, str]],
        result_type: str | None = None,
    ) -> None:
        """Emit the start of a function.
        
        Args:
            name: Function name (without @)
            args: List of (arg_name, type) tuples, where arg_name includes %
            result_type: Optional result type string
        """
        args_str = ", ".join(f"{arg_name}: {arg_type}" for arg_name, arg_type in args)
        if result_type:
            self.emit(f"func.func @{name}({args_str}) -> {result_type} {{")
        else:
            self.emit(f"func.func @{name}({args_str}) {{")
        self.push()

    def emit_func_end(self) -> None:
        """Emit the end of a function."""
        self.pop()
        self.emit("}")

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
