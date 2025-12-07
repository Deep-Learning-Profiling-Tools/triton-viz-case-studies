import cupy as cp

from cutile_kernels.TritonBench.utility_operations.vector_addition import add
from cutile_kernels.utils import to_torch


def add_wrapper(x, y, tile_size=4):
    # reuse the cuTile vector add with the original Triton block size
    return add(x, y, tile_size=tile_size)


def test_add_kernel():
    results = {}
    test_shapes = [16, 8, 32, 0]
    for idx, size in enumerate(test_shapes, start=1):
        lhs = cp.random.randn(size, dtype=cp.float32)
        rhs = cp.random.randn(size, dtype=cp.float32)
        out = add_wrapper(lhs, rhs, tile_size=4)
        results[f"test_case_{idx}"] = to_torch(out)
    return results


result_gold = test_add_kernel()
