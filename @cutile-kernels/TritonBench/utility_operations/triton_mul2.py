import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


@ct.kernel
def _mul2_kernel(x, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    tile = ct.load(x, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, (pid,), tile=tile * 2)


@ct.kernel
def _mul2_inplace_kernel(x, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    tile = ct.load(x, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(x, (pid,), tile=tile * 2)


def triton_mul2(x, tile_size=16):
    x_cp = to_cupy(x)
    out = cp.empty_like(x_cp)
    grid = (ct.cdiv(x_cp.size, tile_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _mul2_kernel, (x_cp, out, tile_size))
    return out


def triton_mul2_inplace(x, tile_size=16):
    x_cp = to_cupy(x)
    grid = (ct.cdiv(x_cp.size, tile_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _mul2_inplace_kernel, (x_cp, tile_size))
    return x_cp


def test_mul():
    n_elements = 1024 * 1024
    x = cp.random.randn(n_elements, dtype=cp.float32)

    results = {
        "test_case_1": to_torch(triton_mul2(x, tile_size=1024)),
        "test_case_2": to_torch(triton_mul2_inplace(x.copy(), tile_size=1024)),
        "test_case_3": to_torch(triton_mul2(x, tile_size=512)),
        "test_case_4": to_torch(triton_mul2_inplace(x.copy(), tile_size=512)),
    }
    return results


result_gold = test_mul()
