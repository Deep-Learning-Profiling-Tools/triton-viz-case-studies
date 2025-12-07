import math

import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


def _tile_size_for_n(n):
    block = 1 << math.ceil(math.log2(max(1, n)))
    return min(block, 65536)


@ct.kernel
def _swiglu_forward(a, b, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO).astype(cp.float32)
    b_tile = ct.load(b, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    sig = 1.0 / (1.0 + ct.exp(-a_tile))
    ct.store(out, (pid,), tile=(a_tile * sig) * b_tile)


@ct.kernel
def _swiglu_backward(dc, a, b, da, db, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    dc_tile = ct.load(dc, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    a_tile = ct.load(a, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO).astype(cp.float32)
    b_tile = ct.load(b, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    sig = 1.0 / (1.0 + ct.exp(-a_tile))
    silu_a = a_tile * sig
    db_tile = dc_tile * silu_a
    da_tile = dc_tile * (silu_a * (1 - sig) + sig) * b_tile
    ct.store(da, (pid,), tile=da_tile)
    ct.store(db, (pid,), tile=db_tile)


def swiglu_forward(a, b):
    a_cp = to_cupy(a)
    b_cp = to_cupy(b)
    n = a_cp.size
    tile_size = _tile_size_for_n(a_cp.shape[-1])
    out = cp.empty_like(a_cp)
    grid = (ct.cdiv(n, tile_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _swiglu_forward, (a_cp, b_cp, out, tile_size))
    return a_cp, b_cp, out.reshape(a.shape)


def swiglu_backward(a, b, dc):
    a_cp = to_cupy(a)
    b_cp = to_cupy(b)
    dc_cp = to_cupy(dc)
    n = dc_cp.size
    tile_size = _tile_size_for_n(dc_cp.shape[-1])
    da = cp.empty_like(a_cp)
    db = cp.empty_like(b_cp)
    grid = (ct.cdiv(n, tile_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _swiglu_backward, (dc_cp, a_cp, b_cp, da, db, tile_size))
    return da.reshape(dc.shape), db.reshape(dc.shape)


def test_swiglu():
    a = cp.random.randn(4, 8, dtype=cp.float32)
    b = cp.random.randn(4, 8, dtype=cp.float32)
    dc = cp.random.randn(4, 8, dtype=cp.float32)

    a_out, b_out, c_out = swiglu_forward(a, b)
    da_out, db_out = swiglu_backward(a, b, dc)

    return {
        "test_case_1": (to_torch(a_out), to_torch(b_out), to_torch(c_out), to_torch(da_out), to_torch(db_out))
    }


result_gold = test_swiglu()
