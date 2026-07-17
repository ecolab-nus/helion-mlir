from __future__ import annotations

from helion_mlir.mlir_utils import add_integer_tensor_encoding
from helion_mlir.input_type_rewrites import rewrite_linalg_input_types


def test_add_integer_tensor_encoding() -> None:
    assert (
        add_integer_tensor_encoding("tensor<?x128x?xf16>", 1)
        == "tensor<?x128x?xf16, 1 : i64>"
    )


def test_rewrite_linalg_input_types_is_operand_positional() -> None:
    rhs = (
        "linalg.batch_matmul "
        "ins(%arg0, %arg1 : tensor<?x128x?xf16>, tensor<?x128x?xf16>) "
        "outs(%0 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>"
    )
    rewritten = rewrite_linalg_input_types(
        rhs,
        [
            "tensor<?x128x?xf16>",
            "tensor<?x128x?xf16, 1 : i64>",
        ],
    )

    assert (
        "ins(%arg0, %arg1 : tensor<?x128x?xf16>, "
        "tensor<?x128x?xf16, 1 : i64>)"
    ) in rewritten
