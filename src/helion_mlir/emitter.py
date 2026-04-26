from __future__ import annotations

import math
import re

from .debug_utils import run_dce_cleanup
from .session import LoweringSession


class ModuleEmitter:
    def __init__(self, session: LoweringSession):
        self.session = session
        self.builder = session.mlir_output_helper

    def emit_module_prelude(self) -> None:
        module_attrs = self.session.analysis.module_attributes
        if module_attrs:
            attr_strs = []
            for name, (value, typ) in module_attrs.items():
                attr_strs.append(f"{name} = {value}" if not typ else f"{name} = {value} : {typ}")
            self.builder.emit(f"module attributes {{{', '.join(attr_strs)}}} {{")
        else:
            self.builder.emit("module {")
        self.builder.push()

        func_args = []
        for tensor_name, tensor_type in self.session.host_tensor_types.items():
            ssa_name = f"%{tensor_name}"
            func_args.append((ssa_name, tensor_type))
            self.session.host_tensors[tensor_name] = ssa_name
        args_str = ", ".join(f"{name}: {typ}" for name, typ in func_args)
        self.builder.emit(f"func.func @{self.session.kernel_name}({args_str}) {{")
        self.builder.push()

    def emit_block_size_symbols(self) -> None:
        emitted_canonical: set[int] = set()
        alias = self.session.analysis.block_info.canonical_aliases
        used_canonical = self.session.analysis.block_info.used_canonical_block_ids
        for info in self.session.env.block_sizes:
            canonical_id = alias.get(info.block_id, info.block_id)
            if canonical_id not in used_canonical:
                continue
            if canonical_id in emitted_canonical:
                self.session.block_size_ssa[info.block_id] = self.session.block_size_ssa[canonical_id]
                continue
            emitted_canonical.add(canonical_id)
            sym_name = next(iter(info.debug_names), f"block_{canonical_id}")
            ssa = f"%{sym_name}"
            if isinstance(info.size, int):
                self.builder.emit(f"{ssa} = arith.constant {info.size} : index")
            else:
                upper_bound = self.session.get_loop_extent(info.block_id)
                self.builder.emit(
                    f'{ssa} = "loom.sym"() {{symbol_ref = @{sym_name}, '
                    f"upper_bound = {upper_bound} : index, "
                    f"is_reduction = {str(info.reduction).lower()}}} : () -> index"
                )
            self.session.block_size_ssa[canonical_id] = ssa
            self.session.block_size_ssa[info.block_id] = ssa

    def emit_reduction_trip_counts(self) -> None:
        for block_id in self.session.analysis.reduction_block_ids:
            canonical_id = self.session.resolve_block_id(block_id)
            info = self.session.env.block_sizes[block_id]
            total_extent = self.session.get_loop_extent(block_id)
            if total_extent is None:
                continue
            if isinstance(info.size, int):
                trip_count_ssa = self.builder.fresh("trip_count")
                trip_count = max(1, math.ceil(total_extent / info.size))
                self.builder.emit(f"{trip_count_ssa} = arith.constant {trip_count} : index")
            else:
                tile_size_ssa = self.session.block_size_ssa[canonical_id]
                total_extent_ssa = self.builder.fresh("loop_extent")
                self.builder.emit(f"{total_extent_ssa} = arith.constant {total_extent} : index")
                trip_count_ssa = self.builder.fresh("trip_count")
                self.builder.emit(f"{trip_count_ssa} = arith.ceildivui {total_extent_ssa}, {tile_size_ssa} : index")
            self.session.reduction_trip_counts[block_id] = trip_count_ssa

    def close_module(self, *, cleanup: bool = True) -> str:
        self.builder.emit("return")
        self.builder.pop()
        self.builder.emit("}")
        self.builder.pop()
        self.builder.emit("}")
        mlir_text = self.builder.build()
        if cleanup:
            try:
                mlir_text = run_dce_cleanup(mlir_text)
                mlir_text = self._restore_host_arg_names(mlir_text)
            except (FileNotFoundError, RuntimeError):
                pass
        return mlir_text

    def _restore_host_arg_names(self, mlir_text: str) -> str:
        """Restore user-facing host tensor names after mlir-opt canonicalizes args."""
        host_names = list(self.session.host_tensor_types.keys())
        if not host_names:
            return mlir_text

        def repl(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            if idx < len(host_names):
                return f"%{host_names[idx]}"
            return match.group(0)

        return re.sub(r"%arg([0-9]+)\b", repl, mlir_text)
