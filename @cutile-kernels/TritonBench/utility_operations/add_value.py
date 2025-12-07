import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


@ct.kernel
def _add_value_kernel(x, out, value, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    tile = ct.load(x, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, (pid,), tile=tile + value)


def puzzle1(x, value=10, tile_size=1024):
    x_cp = to_cupy(x)
    out = cp.empty_like(x_cp)
    grid = (ct.cdiv(x_cp.size, tile_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _add_value_kernel, (x_cp, out, value, tile_size))
    return out


def test_puzzle():
    inputs = [
        cp.asarray([4, 5, 3, 2], dtype=cp.float32),
        cp.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=cp.float32),
        cp.asarray([10, 20, 30], dtype=cp.float32),
        cp.asarray([0, -1, -2, -3], dtype=cp.float32),
    ]
    results = {}
    for idx, arr in enumerate(inputs, start=1):
        results[f"test_case_{idx}"] = to_torch(puzzle1(arr, value=10, tile_size=1024))
    return results


result_gold = test_puzzle()
