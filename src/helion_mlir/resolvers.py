from __future__ import annotations

import re
from typing import Any

import torch
import torch.fx as fx
from torch.ops import aten

import helion.language.tile_ops as hl_tile_ops

from .mlir_utils import format_memref_type, format_tensor_type, torch_dtype_to_mlir_element_type
from .session import LoweringSession


class SymbolResolver:
    def __init__(self, session: LoweringSession):
        self.session = session

    def resolve_dimension(
        self,
        dim_size: Any,
        dim_hint: int = 0,
        overrides: dict[int, str] | None = None,
    ) -> tuple[str | None, bool]:
        from helion._compiler.variable_origin import BlockSizeOrigin

        if overrides and dim_hint in overrides:
            return overrides[dim_hint], False
        if not hasattr(dim_size, "_sympy_"):
            return str(int(dim_size)), True

        sym = dim_size._sympy_()
        host_function = self.session.bound_kernel.host_function
        shape_env = self.session.bound_kernel.env.shape_env
        origin_info = host_function.expr_to_origin.get(sym)
        origin = origin_info.origin if origin_info else None

        if isinstance(origin, BlockSizeOrigin):
            raw_id = origin.block_id
            canonical_id = self.session.resolve_block_id(raw_id)
            block_info = self.session.env.block_sizes[raw_id]
            if isinstance(block_info.size, int):
                return str(block_info.size), True
            ssa = self.session.block_size_ssa.get(canonical_id)
            if ssa:
                return ssa, False

        if sym in shape_env.var_to_val:
            return str(int(shape_env.var_to_val[sym])), True
        return None, False

    def try_get_block_id_from_node(self, node: fx.Node, range_index_block_ids: dict[str, int]) -> int | None:
        from helion._compiler.variable_origin import BlockSizeOrigin

        if node.name in range_index_block_ids:
            return range_index_block_ids[node.name]
        if node.target is hl_tile_ops.tile_index and node.args and isinstance(node.args[0], fx.Node):
            return self.try_get_block_id_from_node(node.args[0], range_index_block_ids)

        raw_id: int | None = None
        sym_val = node.meta.get("val")
        if sym_val is not None and hasattr(sym_val, "_sympy_"):
            sym = sym_val._sympy_()
            origin_info = self.session.bound_kernel.host_function.expr_to_origin.get(sym)
            if origin_info and isinstance(origin_info.origin, BlockSizeOrigin):
                raw_id = origin_info.origin.block_id
        if raw_id is None:
            match = re.search(r"block_size_(\d+)", node.name)
            if match:
                raw_id = int(match.group(1))
        return self.session.resolve_block_id(raw_id) if raw_id is not None else None

    def get_block_size_value(self, canonical_block_id: int) -> tuple[str, bool]:
        info = self.session.env.block_sizes[canonical_block_id]
        if isinstance(info.size, int):
            return str(int(info.size)), True
        ssa = self.session.block_size_ssa.get(canonical_block_id)
        if ssa is not None:
            return ssa, False
        extent = self.session.get_loop_extent(canonical_block_id)
        return (str(extent), True) if extent is not None else ("1", True)

class TypeResolver:
    def __init__(self, session: LoweringSession, symbols: SymbolResolver):
        self.session = session
        self.symbols = symbols

    def compute_mlir_type_from_fake_tensor(self, fake_tensor: Any) -> str:
        from helion._compiler.variable_origin import BlockSizeOrigin

        if fake_tensor is None or not hasattr(fake_tensor, "shape"):
            raise RuntimeError("FakeTensor is None or does not have shape")
        dtype_str = torch_dtype_to_mlir_element_type(fake_tensor.dtype)
        host_function = self.session.bound_kernel.host_function
        shape_env = self.session.bound_kernel.env.shape_env
        block_sizes = self.session.bound_kernel.env.block_sizes
        shape: list[int | None] = []
        for dim_size in fake_tensor.shape:
            if hasattr(dim_size, "_sympy_"):
                sym = dim_size._sympy_()
                origin_info = host_function.expr_to_origin.get(sym)
                origin = origin_info.origin if origin_info else None
                if isinstance(origin, BlockSizeOrigin):
                    block_info = block_sizes[origin.block_id]
                    shape.append(block_info.size if isinstance(block_info.size, int) else None)
                elif sym.is_number:
                    shape.append(int(sym))
                elif sym in shape_env.var_to_val:
                    shape.append(int(shape_env.var_to_val[sym]))
                else:
                    shape.append(None)
            elif isinstance(dim_size, int):
                shape.append(int(dim_size))
            else:
                shape.append(None)
        return format_tensor_type(shape, dtype_str)

    def compute_mlir_memref_type_from_fake_tensor(self, fake_tensor: Any) -> str:
        tensor_type = self.compute_mlir_type_from_fake_tensor(fake_tensor)
        return "memref" + tensor_type[len("tensor") :]

    def get_tensor_type(self, tensor_node: fx.Node | str) -> str:
        if isinstance(tensor_node, fx.Node):
            name = tensor_node.name
            node = tensor_node
        else:
            name = str(tensor_node)
            node = None
        if name in self.session.node_types:
            return self.session.node_types[name]
        if name in self.session.host_tensor_types:
            return self.session.host_tensor_types[name]
        if name in self.session.arg_mlir_types:
            return self.session.arg_mlir_types[name]
        if node is not None and "val" in node.meta:
            return self.compute_mlir_type_from_fake_tensor(node.meta["val"])
        raise RuntimeError(f"Cannot compute MLIR type for node {name}")

    def get_element_type_from_node(self, node: fx.Node) -> str:
        fake = node.meta.get("val")
        if fake is not None and hasattr(fake, "dtype"):
            return torch_dtype_to_mlir_element_type(fake.dtype)
        return "f32"

    @staticmethod
    def split_top_level_commas(text: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        angle_depth = square_depth = paren_depth = 0
        for ch in text:
            if ch == "<":
                angle_depth += 1
            elif ch == ">":
                angle_depth = max(0, angle_depth - 1)
            elif ch == "[":
                square_depth += 1
            elif ch == "]":
                square_depth = max(0, square_depth - 1)
            elif ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            if ch == "," and angle_depth == square_depth == paren_depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            current.append(ch)
        if current:
            parts.append("".join(current).strip())
        return parts

    def parse_plain_memref_dimensions(self, memref_type: str, op_name: str) -> list[str]:
        if not (memref_type.startswith("memref<") and memref_type.endswith(">")):
            raise RuntimeError(f"{op_name} expects a memref type, got {memref_type}")
        content = memref_type[memref_type.find("<") + 1 : memref_type.rfind(">")]
        top_level_parts = self.split_top_level_commas(content)
        if len(top_level_parts) != 1:
            raise RuntimeError(
                f"{op_name} only supports implicit identity-layout memrefs without "
                f"extra layout annotations, got {memref_type}"
            )
        shape_and_dtype = top_level_parts[0]
        return [] if "x" not in shape_and_dtype else shape_and_dtype.split("x")[:-1]

    @staticmethod
    def extract_dtype(type_str: str) -> str:
        content = type_str[type_str.find("<") + 1 : type_str.rfind(">")]
        if "," in content:
            content = content.split(",")[0]
        return content.split("x")[-1] if "x" in content else content


class LoopResolver:
    def __init__(self, session: LoweringSession):
        self.session = session

    def get_block_loop_iv(self, canonical_block_id: int) -> str:
        return f"%iv_block_{canonical_block_id}"

    def get_active_loop_bounds(self, canonical_block_id: int) -> tuple[str, str] | None:
        return self.session.active_loop_bounds().get(canonical_block_id)
