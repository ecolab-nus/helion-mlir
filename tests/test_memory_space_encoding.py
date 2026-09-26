from __future__ import annotations

from helion_mlir.mlir_utils import add_memory_tensor_encoding
from helion_mlir.input_type_rewrites import rewrite_linalg_input_types


def test_add_memory_tensor_encoding() -> None:
    assert (
        add_memory_tensor_encoding("tensor<?x128x?xf16>", "L1_S")
        == 'tensor<?x128x?xf16, {memory = "L1_S"}>'
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
            'tensor<?x128x?xf16, {memory = "L1_S"}>',
        ],
    )

    assert (
        "ins(%arg0, %arg1 : tensor<?x128x?xf16>, "
        'tensor<?x128x?xf16, {memory = "L1_S"}>)'
    ) in rewritten


def _attention():
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples.scripts.flash_attention import attention

    q = torch.randn([2, 128, 64], dtype=torch.float16)
    return attention.bind((q, q, q))


def _raises(fn, text: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc), exc
    else:
        raise AssertionError(f"expected ValueError containing {text!r}")


def test_set_memory_space_pins_one_load_by_name() -> None:
    from helion_mlir import generate_mlir

    mlir_text = generate_mlir(_attention(), cleanup=False)
    assert mlir_text.count('{memory = "L1_S"}') == 2  # load result + matmul operand
    assert "local_mem_kind" not in mlir_text


def test_residency_config_pins_every_load_of_a_host_tensor() -> None:
    from helion_mlir import generate_mlir

    mlir_text = generate_mlir(_attention(), cleanup=False, residency={"q_view": "L1_R"})
    assert '{memory = "L1_R"}' in mlir_text
    assert '{memory = "L1_S"}' in mlir_text


def test_residency_errors() -> None:
    from helion_mlir import generate_mlir

    bound = _attention()
    _raises(lambda: generate_mlir(bound, cleanup=False, residency={"k_view": "L1_R"}),
            "Conflicting residency")
    _raises(lambda: generate_mlir(bound, cleanup=False, residency={"k": "L1_R"}),
            "Unknown residency tensor(s): ['k']")
    _raises(lambda: generate_mlir(bound, cleanup=False, residency={"q_view": "L1-S"}),
            "platform memory name")
