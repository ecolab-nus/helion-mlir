"""
Helion Flash-Decoding Kernel Example (Split-KV)
===============================================
A robust Flash-Decoding kernel implementation that avoids Rank-3 tensors with 
size-1 dimensions, bypassing known compiler bugs in index-lowering.
"""

import math
from pathlib import Path
import sys

import torch

import helion
import helion.language as hl

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
_CUSTOME_OP_ROOT = _REPO_ROOT / "custome_op"

for path in [str(_SRC_ROOT), str(_REPO_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from helion_mlir import print_debug_info, generate_mlir, validate_with_mlir_opt
from custome_op import gather  # registers the op with Helion's decorator API


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------
@helion.kernel(
    static_shapes=False,
    autotune_config_overrides={
        "range_unroll_factors": [0, 0],
        "range_num_stages": [0, 0],
    },
)
def flash_decode(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    batch, _, num_q_head, head_dim = q_in.size()
    kvseq_len = k_in.size(-2)

    num_q_head = hl.specialize(num_q_head)
    head_dim = hl.specialize(head_dim)

    q_view = q_in.reshape([batch, num_q_head, head_dim]) 
    k_view = k_in.reshape([batch, kvseq_len, head_dim]).transpose(1, 2)
    v_view = v_in.reshape([batch, kvseq_len, head_dim])

    sm_scale = 1.0 / math.sqrt(head_dim)
    out = torch.zeros([batch, num_q_head, head_dim], dtype=torch.float16)
    

    # tile_h: head index, tile_s: split segment of sequence L
    for tile_b, tile_s in hl.tile([batch, kvseq_len]):
        qk_scale = hl.full([], sm_scale * 1.44269504, dtype=torch.float16)

        # States are Rank 1 and 2 for maximum stability
        m_i = hl.full([tile_b, num_q_head], float("-inf"), dtype=torch.float16)
        l_i = hl.full([tile_b, num_q_head], 1.0, dtype=torch.float16)
        acc = hl.zeros([tile_b, num_q_head, head_dim], dtype=torch.float16)
        q = q_view[tile_b, :, :]

        # (1) Intra-block sequential online softmax over the split range.
        for tile_n in hl.tile(tile_s.begin, tile_s.end):
            # k: [tile_b, head_dim, tile_n(from tile_s)]
            k = k_view[tile_b, :, tile_n]
            
            # Compute QK -> [tile_b, num_q_head, tile_n]
            qk = torch.bmm(q, k)
            
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None] 
            
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            
            v = v_view[tile_b, tile_n, :] # [tile_h, tile_n, head_dim]
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        
        split_lse = torch.log2(l_i) + m_i

        # (2) Inter-block softmax merge
        if tile_s.id == 0:
            gathered_lse = gather(tile_s, split_lse) # [N, tile_b, num_q_head]
            max_lse = torch.amax(gathered_lse, 0)
            weights = torch.exp2(gathered_lse - max_lse)
            lse_sum = torch.sum(weights, 0)
            norm_scale = weights / lse_sum
            
            gathered_acc = gather(tile_s, acc) # [N, tile_b, num_q_head, head_dim]
            weighted_acc = torch.sum(gathered_acc * norm_scale[:, :, :, None], 0)
            out[tile_b, :, :] = weighted_acc

    return out.view(q_in.size())


def main() -> None:
    B, H, L, D = 16, 32, 8192, 128
    # [batch, seq_len, num_q_heads, head_dim]
    q = torch.randn([B, 1, H, D], dtype=torch.float16)
    # [batch, num_kv_head, kvseq_len, head_dim]
    k = torch.randn([B, 1, L, D], dtype=torch.float16)
    v = torch.randn([B, 1, L, D], dtype=torch.float16)
    bound = flash_decode.bind((q, k, v))

    print_debug_info(bound)

    mlir_text = generate_mlir(bound, cleanup=True, assume_divisible_tiles=True)
    print("=== MLIR Dump ===")
    print(mlir_text)

    res = validate_with_mlir_opt(mlir_text)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        print("mlir-opt validation failed.")
    else:
        print("mlir-opt validation succeeded.\n")


if __name__ == "__main__":
    main()
