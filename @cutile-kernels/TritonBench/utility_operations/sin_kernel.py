import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


@ct.kernel
def _sin_kernel(x, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    tile = ct.load(x, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, (pid,), tile=ct.sin(tile))


def call_kernel(x, tile_size=1024):
    x_cp = to_cupy(x)
    out = cp.empty_like(x_cp)
    grid = (ct.cdiv(x_cp.size, tile_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _sin_kernel, (x_cp, out, tile_size))
    return out


def test_call_kernel():
    results = {}
    test_inputs = [
        cp.asarray([0.0, 1.0, 2.0, 3.0], dtype=cp.float32),
        cp.linspace(0, 10, num=1024, dtype=cp.float32),
        cp.asarray([], dtype=cp.float32),
        cp.asarray([-1.0, -2.0, -3.0, -4.0], dtype=cp.float32),
    ]
    for idx, arr in enumerate(test_inputs, start=1):
        results[f"test_case_{idx}"] = to_torch(call_kernel(arr, tile_size=1024))
    return results


result_gold = test_call_kernel()
