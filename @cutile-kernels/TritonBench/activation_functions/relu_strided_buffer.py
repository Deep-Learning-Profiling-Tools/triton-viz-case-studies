import math
from typing import Union

import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


def heuristics_for_tile_size(max_tile_size, *sizes):
    ndim = len(sizes)
    tile_sizes = [0 for _ in range(ndim)]
    for i in range(ndim):
        size = sizes[ndim - 1 - i]
        tile_size = min(max_tile_size, 1 << math.ceil(math.log2(max(1, size))))
        tile_sizes[ndim - 1 - i] = tile_size
        max_tile_size = max(1, max_tile_size // tile_size)
    return tuple(tile_sizes)


class StridedBuffer:
    """Lightweight view wrapper to mirror the Triton version."""

    def __init__(self, base, shape=None, strides=None, dtype=None, offset=0):
        self._base = to_cupy(base)
        self.dtype = dtype or self._base.dtype
        self.shape = tuple(shape if shape is not None else self._base.shape)
        self._strides = tuple(strides if strides is not None else self._base.strides)
        self.ndim = len(self.shape)
        self.offset = offset

    def stride(self):
        return self._strides

    def size(self):
        return self.shape

    def numel(self):
        return math.prod(self.shape)

    def dim(self):
        return self.ndim

    def unwrap(self):
        return self._base


@ct.kernel
def _relu_kernel(inp, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    tile = ct.load(inp, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, (pid,), tile=ct.maximum(tile, 0))


def relu_forward_wrapper_rank_1(in0: Union[cp.ndarray, StridedBuffer], /, *, out0):
    # flatten to 1D contiguous buffers
    in_cp = in0.unwrap() if isinstance(in0, StridedBuffer) else to_cupy(in0)
    out_cp = out0.unwrap() if isinstance(out0, StridedBuffer) else to_cupy(out0)
    assert in_cp.shape == out_cp.shape
    tile_size = math.prod(heuristics_for_tile_size(512, *out_cp.shape))
    grid = (ct.cdiv(out_cp.size, tile_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _relu_kernel, (in_cp.reshape(-1), out_cp.reshape(-1), tile_size))
    return out_cp


def test_relu_forward():
    samples = [
        cp.asarray([-1.0, 0.0, 1.0], dtype=cp.float32),
        cp.asarray([3.0, -2.0, 5.0, -7.0], dtype=cp.float32),
    ]
    results = {}
    for idx, sample in enumerate(samples, start=1):
        out = cp.empty_like(sample)
        relu_forward_wrapper_rank_1(sample, out0=out)
        results[f"test_case_{idx}"] = to_torch(out)
    return results


result_gold = test_relu_forward()
