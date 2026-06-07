from __future__ import annotations

from types import SimpleNamespace

from helion_mlir.models import BlockInfoSummary, GraphInventory, HostTensorInfo, KernelAnalysis
from helion_mlir.session import LoweringSession


def _make_analysis() -> KernelAnalysis:
    bound_kernel = SimpleNamespace(
        env=SimpleNamespace(block_sizes=[]),
        host_function=SimpleNamespace(device_ir=SimpleNamespace(grid_block_ids=[])),
    )
    return KernelAnalysis(
        bound_kernel=bound_kernel,
        kernel_name="dummy",
        graph_inventory=GraphInventory(
            root_ids=(),
            root_graphs={},
            inner_graphs={},
            reachable_inner_ids=frozenset(),
        ),
        block_info=BlockInfoSummary(
            canonical_aliases={},
            loop_extents={},
            natural_upper_bounds={},
            used_block_ids=frozenset(),
            used_canonical_block_ids=frozenset(),
        ),
        host_tensors=HostTensorInfo(
            tensor_types={},
            arg_types={},
        ),
        module_attributes={},
        reduction_block_ids=(),
    )


def test_graph_scope_restores_mutations() -> None:
    session = LoweringSession(_make_analysis())
    session.bind_node_value("outer", "%outer")

    with session.push_graph_scope():
        session.bind_node_value("inner", "%inner")
        session.record_loop_result("loop", "%loop")
        assert session.lookup_node_value("inner") == "%inner"

    assert session.lookup_node_value("outer") == "%outer"
    assert session.lookup_node_value("inner") is None
    assert "loop" not in session.loop_result_values


def test_loop_result_projection_uses_recorded_result() -> None:
    session = LoweringSession(_make_analysis())
    session.bind_node_value("loop_node", "%result")
    session.record_loop_result("loop_node", "%result", count=2)

    assert session.lookup_loop_result_projection("loop_node", 1) == "%result#1"
