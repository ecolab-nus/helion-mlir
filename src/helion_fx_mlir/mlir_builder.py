"""Enhanced MLIR text builder with utility methods for structured emission.

This module provides a builder class that manages indentation, SSA value naming,
and provides convenient methods for emitting common MLIR constructs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch

if TYPE_CHECKING:
    pass

# Import BlockSizeInfo types for symbolic size detection
try:
    from helion._compiler.compile_environment import AutoSize
except ImportError:
    AutoSize = None  # type: ignore[misc,assignment]


class MLIRBuilder:
    """Helper class to manage indentation and SSA name creation for MLIR text emission."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._indent = 0
        self._tmp_counter = 0
        self._name_counters: dict[str, int] = {}

    # -------------------------------------------------------------------------
    # Core emission methods
    # -------------------------------------------------------------------------

    def emit(self, text: str) -> None:
        """Emit a line of MLIR text with current indentation."""
        self._lines.append("  " * self._indent + text)

    def emit_comment(self, comment: str) -> None:
        """Emit a comment line."""
        self.emit(f"// {comment}")

    def emit_blank(self) -> None:
        """Emit a blank line."""
        self._lines.append("")

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

    def fresh_named(self, name: str) -> str:
        """Generate a fresh SSA value name, incrementing if the name was used before."""
        if name not in self._name_counters:
            self._name_counters[name] = 0
            return f"%{name}"
        count = self._name_counters[name]
        self._name_counters[name] = count + 1
        return f"%{name}_{count}"

    def build(self) -> str:
        """Build and return the complete MLIR text."""
        return "\n".join(self._lines) + "\n"

    # -------------------------------------------------------------------------
    # Module and function emission
    # -------------------------------------------------------------------------

    def emit_module_start(self, attrs: dict[str, tuple[int, str]] | None = None) -> None:
        """Emit the start of an MLIR module.
        
        Args:
            attrs: Optional dict of module attributes, mapping name to (value, type).
                   Example: {"helion.tile_m": (64, "index")}
        """
        if attrs:
            attr_strs = [f"{name} = {value} : {typ}" for name, (value, typ) in attrs.items()]
            self.emit(f"module attributes {{{', '.join(attr_strs)}}} {{")
        else:
            self.emit("module {")
        self.push()

    def emit_module_end(self) -> None:
        """Emit the end of an MLIR module."""
        self.pop()
        self.emit("}")

    def emit_get_module_attribute(self, attr_name: str, result_hint: str = "attr") -> str:
        """Emit a helion.get_module_attribute operation.
        
        Args:
            attr_name: The attribute name to retrieve (e.g., "helion.tile_m")
            result_hint: Hint for the result SSA name
            
        Returns:
            The result SSA value name.
        """
        result = self.fresh(result_hint)
        # Use generic format for compatibility with mlir-opt -allow-unregistered-dialect
        self.emit(f'{result} = "loom.get_module_attribute"() {{attr_name = "{attr_name}"}} : () -> index')
        return result

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

    def emit_return(self, values: list[str], types: list[str]) -> None:
        """Emit a return statement."""
        if not values:
            self.emit("return")
        elif len(values) == 1:
            self.emit(f"return {values[0]} : {types[0]}")
        else:
            values_str = ", ".join(values)
            types_str = ", ".join(types)
            self.emit(f"return {values_str} : {types_str}")

    # -------------------------------------------------------------------------
    # Operation emission helpers
    # -------------------------------------------------------------------------

    def emit_op(
        self,
        op_name: str,
        operands: list[str],
        attrs: dict[str, str | None],
        operand_types: list[str],
        result_types: list[str],
        result_name: str | None = None,
    ) -> str | None:
        """Emit a generic MLIR operation.
        
        Args:
            op_name: Operation name (e.g., "helion.load_tile_dynamic")
            operands: List of SSA value names for operands
            attrs: Dictionary of attributes (None values are filtered out)
            operand_types: List of types for operands
            result_types: List of result types (empty for void ops)
            result_name: Optional hint for the result SSA name
            
        Returns:
            The result SSA value name, or None if no results.
        """
        operands_str = ", ".join(operands)
        attrs_str = format_attr_dict(attrs)
        types_str = f"({', '.join(operand_types)}) -> "
        
        if len(result_types) == 0:
            types_str += "()"
            self.emit(f'"{op_name}"({operands_str}){attrs_str} : {types_str}')
            return None
        elif len(result_types) == 1:
            types_str += result_types[0]
        else:
            types_str += f"({', '.join(result_types)})"
        
        result = self.fresh(result_name or "v")
        self.emit(f'{result} = "{op_name}"({operands_str}){attrs_str} : {types_str}')
        return result

    def emit_index_constant(self, value: int) -> str:
        """Emit an arith.constant of index type and return its SSA name."""
        name = self.fresh(f"c{value}_")
        self.emit(f"{name} = arith.constant {value} : index")
        return name

    def emit_tensor_dim(self, tensor: str, dim_index: str, tensor_type: str) -> str:
        """Emit a tensor.dim operation."""
        result = self.fresh("dim")
        self.emit(f"{result} = tensor.dim {tensor}, {dim_index} : {tensor_type}")
        return result

    # -------------------------------------------------------------------------
    # Affine dialect helpers
    # -------------------------------------------------------------------------

    def emit_affine_parallel_start(
        self,
        ivs: list[str],
        lower_bounds: list[str],
        upper_bounds: list[str],
        steps: list[str] | None = None,
    ) -> None:
        """Emit the start of an affine.parallel loop.
        
        Args:
            ivs: Induction variable names (including %)
            lower_bounds: Lower bound expressions
            upper_bounds: Upper bound expressions  
            steps: Step values (default all 1)
        """
        if steps is None:
            steps = ["1"] * len(ivs)
        
        ivs_str = ", ".join(ivs)
        lb_str = ", ".join(lower_bounds)
        ub_str = ", ".join(upper_bounds)
        steps_str = ", ".join(steps)
        
        self.emit(f"affine.parallel ({ivs_str}) = ({lb_str}) to ({ub_str}) step ({steps_str}) {{")
        self.push()

    def emit_affine_parallel_end(self) -> None:
        """Emit the end of an affine.parallel loop."""
        self.emit("affine.yield")
        self.pop()
        self.emit("}")

    def emit_affine_for_start(
        self,
        iv: str,
        lower_bound: str,
        upper_bound: str,
        iter_args: list[tuple[str, str, str]] | None = None,
        result_types: list[str] | None = None,
    ) -> str | None:
        """Emit the start of an affine.for loop.
        
        Args:
            iv: Induction variable name (including %)
            lower_bound: Lower bound (int or SSA value)
            upper_bound: Upper bound (int or SSA value)
            iter_args: List of (arg_name, init_value, type) for iter_args
            result_types: Types of loop results
            
        Returns:
            The result SSA value name if iter_args are present, None otherwise.
        """
        result = None
        if iter_args:
            iter_args_str = ", ".join(
                f"{arg_name} = {init_val}" for arg_name, init_val, _ in iter_args
            )
            types_str = ", ".join(t for _, _, t in iter_args)
            result = self.fresh("loop_result")
            self.emit(
                f"{result} = affine.for {iv} = {lower_bound} to {upper_bound} "
                f"iter_args({iter_args_str}) -> ({types_str}) {{"
            )
        else:
            self.emit(f"affine.for {iv} = {lower_bound} to {upper_bound} {{")
        self.push()
        return result

    def emit_affine_for_end(self, yield_values: list[str], yield_types: list[str]) -> None:
        """Emit the end of an affine.for loop."""
        if yield_values:
            values_str = ", ".join(yield_values)
            types_str = ", ".join(yield_types)
            self.emit(f"affine.yield {values_str} : {types_str}")
        else:
            self.emit("affine.yield")
        self.pop()
        self.emit("}")

    def emit_affine_apply(self, map_str: str, dims: list[str], symbols: list[str]) -> str:
        """Emit an affine.apply operation.
        
        Args:
            map_str: The affine map string (e.g., "(d0)[s0] -> (d0 + s0)")
            dims: Dimension operands
            symbols: Symbol operands
            
        Returns:
            The result SSA value name.
        """
        result = self.fresh("apply")
        dims_str = ", ".join(dims) if dims else ""
        symbols_str = ", ".join(symbols) if symbols else ""
        
        if dims and symbols:
            self.emit(f"{result} = affine.apply affine_map<{map_str}>({dims_str})[{symbols_str}]")
        elif dims:
            self.emit(f"{result} = affine.apply affine_map<{map_str}>({dims_str})")
        elif symbols:
            self.emit(f"{result} = affine.apply affine_map<{map_str}>()[{symbols_str}]")
        else:
            self.emit(f"{result} = affine.apply affine_map<{map_str}>()")
        return result

    def emit_affine_min(self, map_str: str, dims: list[str], symbols: list[str]) -> str:
        """Emit an affine.min operation.
        
        Args:
            map_str: The affine map string
            dims: Dimension operands
            symbols: Symbol operands
            
        Returns:
            The result SSA value name.
        """
        result = self.fresh("min")
        dims_str = ", ".join(dims) if dims else ""
        symbols_str = ", ".join(symbols) if symbols else ""
        
        if dims and symbols:
            self.emit(f"{result} = affine.min affine_map<{map_str}>({dims_str})[{symbols_str}]")
        elif dims:
            self.emit(f"{result} = affine.min affine_map<{map_str}>({dims_str})")
        elif symbols:
            self.emit(f"{result} = affine.min affine_map<{map_str}>()[{symbols_str}]")
        else:
            self.emit(f"{result} = affine.min affine_map<{map_str}>()")
        return result


# -----------------------------------------------------------------------------
# Utility functions for MLIR text formatting
# -----------------------------------------------------------------------------


def is_concrete_size(size: Any) -> bool:
    """Check if a block size is a concrete integer value.
    
    Returns True for int, False for SymInt, AutoSize, or None.
    """
    if size is None:
        return False
    if AutoSize is not None and isinstance(size, AutoSize):
        return False
    if hasattr(torch, 'SymInt') and isinstance(size, torch.SymInt):
        return False
    try:
        int(size)
        return True
    except (TypeError, ValueError):
        return False


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


def format_indices_attr(indices: Sequence[str | None]) -> str:
    """Format indices as an MLIR array of string attributes."""
    if not indices:
        return "[]"
    body = ", ".join(format_string_attr(index if index is not None else "?") for index in indices)
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


def format_dynamic_tensor_meta(dim0: str, dim1: str, element_type: str) -> str:
    """Format tensor metadata for dynamic dimensions."""
    return format_string_attr(f"[{dim0}, {dim1}]:{element_type}")


def as_optional_int(value: object) -> int | None:
    """Attempt to convert a value to int, return None on failure."""
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return None
