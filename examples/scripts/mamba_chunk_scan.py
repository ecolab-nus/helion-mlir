"""
Helion Mamba2 Chunk Scan Kernel Example
=======================================
This example demonstrates a Helion kernel implementation of Mamba2 chunk scan.
It includes MLIR generation and validation with mlir-opt.
"""

# %%
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"

for path in [str(_SRC_ROOT), str(_REPO_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from helion_mlir import generate_mlir, validate_with_mlir_opt, print_debug_info
from helion_mlir.custom_op import gather, broadcast  # registers the op with Helion's decorator API


# %%
@helion.kernel(
    static_shapes=False,
)
def helion_mamba2_chunk_scan_kernel(
    cb: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA_cumsum: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """
    Argument:
        cb: (batch, nchunks, ngroups, chunk_size, chunk_size)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        C: (batch, seqlen, ngroups, dstate)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads,)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """

    batch, nchunks, ngroups, chunk_size, _ = cb.shape
    _, seqlen, nheads, headdim = x.shape
    _, _, _, dstate = C.shape
    assert nchunks == (seqlen + chunk_size - 1) // chunk_size

    block_m = hl.register_block_size(chunk_size)
    block_n = hl.register_block_size(headdim)
    block_k = hl.register_block_size(64, 64)
    #dstate = hl.specialize(dstate)

    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
    assert D.shape == (nheads,)

    dtype = cb.dtype
    accum_dtype = torch.float16
    assert (
        x.dtype
        == dt.dtype
        == dA_cumsum.dtype
        == C.dtype
        == prev_states.dtype
        == D.dtype
        == dtype
    )
    prev_states_T = prev_states.transpose(3, 4)

    out_ = torch.empty_like(x)

    for tile_h, tile_m, tile_n, tile_b, tile_c in hl.tile(
        [nheads, chunk_size, headdim, batch, nchunks],
        block_size=[1, block_m, block_n, 1, 1],
    ):
        # tile_h: head tile (size 1)
        # tile_m: chunk-local sequence rows (M axis)
        # tile_n: head-dim columns (N axis)
        # tile_b: batch tile (size 1)
        # tile_c: chunk id tile (size 1)
        acc_o = hl.zeros([tile_m, tile_n], dtype=accum_dtype)
        # dA_cumsum_local_m: [tile_m]
        dA_cumsum_local_m = dA_cumsum[tile_b.begin, tile_h.begin, tile_c.begin, tile_m]
        dA_cumsum_local_m_bc_n = broadcast(
            dA_cumsum_local_m,
            0,
            [tile_n, dA_cumsum_local_m.size(0)],
        ).T

        # scale_m_local: [tile_m, tile_n]
        scale_m_local = torch.exp(dA_cumsum_local_m_bc_n)

        # C_local: [tile_m, dstate]
        # row index = tile_c * chunk_size + tile_m.index
        C_local = C[
            tile_b.begin,
            tile_m.index + tile_c.begin * chunk_size,
            tile_h.begin // (nheads // ngroups),
            :,
        ]
        # prev_states_local: [dstate, tile_b]
        prev_states_local = prev_states_T[
            tile_b.begin, tile_c.begin, tile_h.begin, :, tile_n
        ]
        # hl.dot([tile_m, dstate], [dstate, tile_n]) -> [tile_m, tile_n]
        acc_o = hl.dot(C_local, prev_states_local, acc=acc_o)
        acc_o *= scale_m_local

        for tile_k in hl.tile((tile_m.id + 1) * block_m, block_size=block_k):
            # cb_local: [tile_m, tile_k]
            cb_local = cb[
                tile_b.begin,
                tile_c.begin,
                tile_h.begin // (nheads // ngroups),
                tile_m,
                tile_k,
            ]
            # dA_cumsum_local_k: [tile_k]
            dA_cumsum_local_k = dA_cumsum[
                tile_b.begin, tile_h.begin, tile_c.begin, tile_k
            ]
            dA_cumsum_local_m_bc_k = broadcast(
                dA_cumsum_local_m,
                0,
                [tile_k, dA_cumsum_local_m.size(0)],
            ).T
            dA_cumsum_local_k = broadcast(dA_cumsum_local_k, 0, [tile_m, dA_cumsum_local_k.size(0)])
            # broadcast to [tile_m, tile_k]
            cb_local *= torch.exp(dA_cumsum_local_m_bc_k - dA_cumsum_local_k)
            # dt_local: [tile_k]
            dt_local = dt[tile_b.begin, tile_h.begin, tile_c.begin, tile_k]
            # dt_local[None, :]: [1, tile_k], broadcast over tile_m axis
            dt_local = broadcast(dt_local, 0, [tile_m, dt_local.size(0)])
            cb_local *= dt_local
            # Yet not support sparse matmul
            # pred = (tile_m.index + 0)[:, None] >= (tile_k.index + 0)[None, :]
            # cb_local = torch.where(pred, cb_local, torch.zeros_like(cb_local))
            # x_local: [tile_k, tile_n]
            x_local = x[
                tile_b.begin,
                tile_c.begin * chunk_size + tile_k.index,
                tile_h.begin,
                tile_n,
            ]
            # hl.dot([tile_m, tile_k], [tile_k, tile_n]) -> [tile_m, tile_n]
            acc_o = torch.addmm(acc_o, cb_local, x_local)

        # D_local: scalar
        D_local = D[tile_h.begin]
        # x_residual: [tile_m, tile_n]
        x_residual = x[
            tile_b.begin, tile_c.begin * chunk_size + tile_m.index, tile_h.begin, tile_n
        ]
        # D_local scalar broadcasts to [tile_m, tile_n]
        acc_o += x_residual * D_local
        # out[...] tile: [tile_m, tile_n]
        out_[
            tile_b.begin, tile_c.begin * chunk_size + tile_m.index, tile_h.begin, tile_n
        ] = acc_o.to(dtype=dtype)

    return out_


# %%
def main() -> None:
    """
    Main function to run MLIR generation and validation.
    """
    batch = 2
    seqlen = 2048
    nheads = 64
    headdim = 64
    chunk_size = 256
    ngroups = 1
    dstate = 64
    nchunks = (seqlen + chunk_size - 1) // chunk_size

    cb = torch.randn([batch, nchunks, ngroups, chunk_size, chunk_size], dtype=torch.float16)
    x = torch.randn([batch, seqlen, nheads, headdim], dtype=torch.float16)
    dt = torch.randn([batch, nheads, nchunks, chunk_size], dtype=torch.float16)
    dA_cumsum = torch.randn([batch, nheads, nchunks, chunk_size], dtype=torch.float16)
    C = torch.randn([batch, seqlen, ngroups, dstate], dtype=torch.float16)
    prev_states = torch.randn([batch, nchunks, nheads, headdim, dstate], dtype=torch.float16)
    D = torch.randn([nheads], dtype=torch.float16)

    args = (cb, x, dt, dA_cumsum, C, prev_states, D)
    bound_kernel = helion_mamba2_chunk_scan_kernel.bind(args)

    print_debug_info(bound_kernel)

    mlir_text = generate_mlir(bound_kernel, assume_divisible_tiles=True)
    print("=== MLIR Dump ===")
    print(mlir_text)

    # Validate with mlir-opt
    result = validate_with_mlir_opt(mlir_text)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit("mlir-opt validation failed (see stderr above).")
    print("mlir-opt validation succeeded.\n")


# %%
if __name__ == "__main__":
    main()
