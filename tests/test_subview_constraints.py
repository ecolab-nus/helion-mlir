from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helion_mlir.ir_visitor import IRVisitor


def test_parse_plain_memref_dimensions_accepts_plain_memref() -> None:
    dims = IRVisitor._parse_plain_memref_dimensions("memref<1x2048x1x?xf16>", "load")
    assert dims == ["1", "2048", "1", "?"]


def test_parse_plain_memref_dimensions_rejects_explicit_layout() -> None:
    with pytest.raises(RuntimeError, match="implicit identity-layout memrefs"):
        IRVisitor._parse_plain_memref_dimensions(
            "memref<4x8xf16, strided<[8, 1], offset: 0>>",
            "load",
        )


def test_validate_full_unit_slice_rejects_strided_slice() -> None:
    with pytest.raises(RuntimeError, match="full-dimension unit-step slices"):
        IRVisitor._validate_full_unit_slice(slice(None, None, 2), "load", 1)


def test_validate_full_unit_slice_rejects_bounded_slice() -> None:
    with pytest.raises(RuntimeError, match="full-dimension unit-step slices"):
        IRVisitor._validate_full_unit_slice(slice(0, 4, 1), "load", 1)
