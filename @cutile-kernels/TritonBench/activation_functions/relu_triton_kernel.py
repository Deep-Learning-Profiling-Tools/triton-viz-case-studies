import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


@ct.kernel
def relu_kernel(x, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    vals = ct.load(x, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, (pid,), tile=ct.maximum(vals, 0))


def relu(x, tile_size=1024):
    x_cp = to_cupy(x)
    out_cp = cp.empty_like(x_cp, dtype=cp.float32)
    grid = (ct.cdiv(x_cp.size, tile_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, relu_kernel, (x_cp, out_cp, tile_size))
    return out_cp


def test_relu():
    results = {}
    test_inputs = [
        cp.asarray([-3.0, -1.0, -0.5, -2.0, -5.0], dtype=cp.float32),
        cp.asarray([3.0, 1.0, 0.5, 2.0, 5.0], dtype=cp.float32),
        cp.asarray([-3.0, -1.0, 0.0, 2.0, 5.0], dtype=cp.float32),
        cp.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=cp.float32),
    ]
    for idx, tensor in enumerate(test_inputs, start=1):
        results[f"test_case_{idx}"] = to_torch(relu(tensor, tile_size=1024))
    return results


result_gold = test_relu()
