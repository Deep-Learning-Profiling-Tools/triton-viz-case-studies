import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


@ct.kernel
def _pow_kernel(x, out, exponent, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    vals = ct.load(x, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, (pid,), tile=ct.pow(vals.astype(cp.float32), exponent))


def pow_func_scalar_tensor_wrapper_rank_1(val0, in0, out0=None, tile_size=512):
    x_cp = to_cupy(in0)
    out_cp = cp.empty_like(x_cp) if out0 is None else to_cupy(out0)
    grid = (ct.cdiv(x_cp.size, tile_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _pow_kernel, (x_cp.reshape(-1), out_cp.reshape(-1), val0, tile_size))
    return out_cp


def test_pow_func_scalar_tensor_wrapper_rank_1():
    in_tensor = cp.asarray([1.0, 2.0, 3.0, 4.0], dtype=cp.float32)
    out_tensor = cp.zeros_like(in_tensor)
    results = {}
    pow_func_scalar_tensor_wrapper_rank_1(2.0, in_tensor, out0=out_tensor)
    results["test_case_1"] = to_torch(out_tensor.copy())

    pow_func_scalar_tensor_wrapper_rank_1(0.5, in_tensor, out0=out_tensor)
    results["test_case_2"] = to_torch(out_tensor.copy())

    bigger = cp.linspace(1, 16, num=16, dtype=cp.float32)
    out_big = cp.zeros_like(bigger)
    pow_func_scalar_tensor_wrapper_rank_1(3.0, bigger, out0=out_big)
    results["test_case_3"] = to_torch(out_big.copy())

    return results


result_gold = test_pow_func_scalar_tensor_wrapper_rank_1()
