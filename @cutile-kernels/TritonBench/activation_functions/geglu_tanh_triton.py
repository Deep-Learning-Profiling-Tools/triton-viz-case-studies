import math

import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


def _tile_size(n_cols: int) -> int:
    return 1 << math.ceil(math.log2(max(1, n_cols)))


@ct.kernel
def _geglu_tanh_forward(a, b, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO).astype(cp.float32)
    b_tile = ct.load(b, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_tile * a_tile * a_tile
    tanh_arg = sqrt_2_over_pi * (a_tile + 0.044715 * a_cubed)
    tanh_res = ct.tanh(tanh_arg)
    geglu_a = 0.5 * a_tile * (1 + tanh_res)
    ct.store(out, (pid,), tile=geglu_a * b_tile)


@ct.kernel
def _geglu_tanh_backward(dc, a, b, da, db, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    dc_tile = ct.load(dc, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    a_tile = ct.load(a, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO).astype(cp.float32)
    b_tile = ct.load(b, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)

    sqrt_2_over_pi = 0.7978845608028654
    a_cubed = a_tile * a_tile * a_tile
    tanh_arg = sqrt_2_over_pi * (a_tile + 0.044715 * a_cubed)
    tanh_res = ct.tanh(tanh_arg)
    geglu_a = 0.5 * a_tile * (1 + tanh_res)
    db_tile = dc_tile * geglu_a

    term1 = 0.5 * (1 + tanh_res)
    tanh_sq = tanh_res * tanh_res
    term2 = 0.5 * a_tile * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_tile * a_tile))
    da_tile = dc_tile * b_tile * (term1 + term2)

    ct.store(da, (pid,), tile=da_tile)
    ct.store(db, (pid,), tile=db_tile)


def geglu_forward(a, b):
    a_cp = to_cupy(a)
    b_cp = to_cupy(b)
    n_cols = a_cp.shape[-1]
    tile_size = _tile_size(n_cols)
    a_flat = a_cp.reshape(-1, n_cols)
    b_flat = b_cp.reshape(-1, n_cols)
    out = cp.empty_like(a_flat)
    grid = (a_flat.shape[0], 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _geglu_tanh_forward, (a_flat, b_flat, out, tile_size))
    return a_cp, b_cp, out.reshape(a_cp.shape)


def geglu_backward(a, b, dc):
    a_cp = to_cupy(a)
    b_cp = to_cupy(b)
    dc_cp = to_cupy(dc)
    n_cols = dc_cp.shape[-1]
    tile_size = _tile_size(n_cols)
    a_flat = a_cp.reshape(-1, n_cols)
    b_flat = b_cp.reshape(-1, n_cols)
    dc_flat = dc_cp.reshape(-1, n_cols)
    da = cp.empty_like(a_flat)
    db = cp.empty_like(b_flat)
    grid = (dc_flat.shape[0], 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _geglu_tanh_backward, (dc_flat, a_flat, b_flat, da, db, tile_size))
    return da.reshape(a_cp.shape), db.reshape(b_cp.shape)


def test_geglu():
    results = {}
    test_cases = [
        (2, 128),
        (3, 128),
        (2, 256),
        (1, 128),
    ]
    for idx, (bs, hidden) in enumerate(test_cases, start=1):
        a = cp.random.randn(bs, hidden, dtype=cp.float32)
        b = cp.random.randn(bs, hidden, dtype=cp.float32)
        dc = cp.random.randn(bs, hidden, dtype=cp.float32)
        a_out, b_out, c_out = geglu_forward(a, b)
        da_out, db_out = geglu_backward(a, b, dc)
        results[f"test_case_{idx}"] = (to_torch(a_out), to_torch(b_out), to_torch(c_out), to_torch(da_out), to_torch(db_out))
    return results


result_gold = test_geglu()
