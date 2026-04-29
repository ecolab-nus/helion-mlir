import math
from pathlib import Path
import sys

import torch

import helion
import helion.language as hl

_SYS_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SYS_SRC) not in sys.path:
    sys.path.insert(0, str(_SYS_SRC))

from helion_mlir import generate_mlir, validate_with_mlir_opt, print_debug_info
from helion_mlir.custom_op import broadcast


@helion.kernel(
    static_shapes=False,
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
    out_ = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        qk_scale_dev = hl.full([], sm_scale, dtype=torch.float16)
        m_i = hl.full([tile_b, tile_m, 1], float("-inf"), dtype=torch.float16)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float16)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            k = k_view[tile_b, :, tile_n]
            qk = torch.bmm(q, k)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1, keepdim=True) * qk_scale_dev)
            m_ij_broad = broadcast(m_ij, 2, [m_ij.size(0), m_ij.size(1), tile_n])
            qk = qk * qk_scale_dev - m_ij_broad
            p = torch.exp(qk)
            l_ij = torch.sum(p, -1, keepdim=True)
            alpha = torch.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        m_i += torch.log(l_i)
        l_i_broadcast = broadcast(l_i, 2, [l_i.size(0), l_i.size(1), head_dim])
        acc = acc / l_i_broadcast
        out_[tile_b, tile_m, :] = acc.to(out_.dtype)
    return out_.view(q_in.size())


def main() -> None:
    q = torch.randn([32, 4096, 128], dtype=torch.float16)
    k = torch.randn([32, 4096, 128], dtype=torch.float16)
    v = torch.randn([32, 4096, 128], dtype=torch.float16)
    bound = attention.bind((q, k, v))

    print_debug_info(bound)

    mlir_text = generate_mlir(bound, cleanup=True, assume_divisible_tiles=True)
    print("=== MLIR Dump ===")
    print(mlir_text)
    
    # Validate with mlir-opt (use -allow-unregistered-dialect for loom.* and torch.* ops)
    res = validate_with_mlir_opt(mlir_text)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        raise SystemExit("mlir-opt validation failed (see stderr above).")
    print("mlir-opt validation succeeded.\n")



if __name__ == "__main__":
    main()
