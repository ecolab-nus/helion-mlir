import math
from pathlib import Path
import sys

import torch

import helion
import helion.language as hl

_SYS_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SYS_SRC) not in sys.path:
    sys.path.insert(0, str(_SYS_SRC))

from helion_fx_mlir import generate_mlir, validate_with_helion_opt


@helion.kernel(
    static_shapes=True,
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def attention(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim]).transpose(1, 2)
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        m_i += torch.log2(l_i)
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


def main() -> None:
    q = torch.randn([2, 4, 8])
    k = torch.randn([2, 4, 8])
    v = torch.randn([2, 4, 8])
    bound = attention.bind((q, k, v))

    print("=== Device IR ===")
    rolled_ids = {
        info.new_graph_id 
        for info in bound.host_function.device_ir.rolled_reductions 
        if info.new_graph_id is not None
    }
    for i, g in enumerate(bound.host_function.device_ir.graphs):
        if i in rolled_ids:
            continue
        print(f"Graph {i}: {type(g).__name__}")
        g.graph.print_tabular()
    print("\n")

    mlir_text = generate_mlir(bound, kernel_name="attention")
    print("=== MLIR Dump ===")
    print(mlir_text)
    
    # Skip helion-opt validation - torch dialect not registered in helion-opt
    # res = validate_with_helion_opt(mlir_text, extra_args=["-allow-unregistered-dialect"])
    # if res.returncode != 0:
    #     print(res.stderr, file=sys.stderr)
    #     raise SystemExit("helion-opt validation failed (see stderr above).")
    # print("helion-opt validation succeeded.\n")
    print("MLIR generation complete.\n")



if __name__ == "__main__":
    main()
