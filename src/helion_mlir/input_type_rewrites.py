"""Caller-side operand type restoration for inlined torch-mlir operations."""

from __future__ import annotations

import re


def _split_mlir_types(types: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in types:
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def rewrite_linalg_input_types(rhs: str, operand_types: list[str]) -> str:
    """Rewrite linalg ins() types to caller types by %argN position."""
    ins_match = re.search(r"ins\(([^)]*)\)", rhs)
    if not ins_match or ":" not in ins_match.group(1):
        return rhs
    operands_part, types_part = ins_match.group(1).split(":", 1)
    operand_names = [operand.strip() for operand in operands_part.split(",")]
    type_strs = _split_mlir_types(types_part.strip())
    if len(operand_names) != len(type_strs):
        return rhs
    for index, operand_name in enumerate(operand_names):
        match = re.fullmatch(r"%arg(\d+)", operand_name)
        if match and int(match.group(1)) < len(operand_types):
            type_strs[index] = operand_types[int(match.group(1))]
    replacement = f"ins({', '.join(operand_names)} : {', '.join(type_strs)})"
    return rhs[: ins_match.start()] + replacement + rhs[ins_match.end() :]


def rewrite_tensor_dim_input_type(rhs: str, operand_types: list[str]) -> str:
    """Rewrite a tensor.dim argument type to its caller-side tensor type."""
    match = re.match(
        r"(tensor\.dim\s+%arg(\d+)\s*,\s*%[\w.+-]+\s*:\s*)(tensor<[^>]+>)(.*)",
        rhs,
    )
    if not match or int(match.group(2)) >= len(operand_types):
        return rhs
    return f"{match.group(1)}{operand_types[int(match.group(2))]}{match.group(4)}"
