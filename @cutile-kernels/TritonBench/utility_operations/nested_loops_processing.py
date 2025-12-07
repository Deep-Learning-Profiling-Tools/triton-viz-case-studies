import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy


@ct.kernel
def nested3(in_ptr, out_ptr, stride_m: ct.Constant[int], stride_n: ct.Constant[int]):
    offs_am = ct.arange(0, 2)
    offs_an = ct.arange(0, 2)
    a_ptrs = in_ptr + offs_am[:, None] * stride_m + offs_an[None, :] * stride_n

    offs_cm = ct.arange(0, 2)
    offs_cn = ct.arange(0, 2)
    c_ptrs = out_ptr + offs_cm[:, None] * stride_m + offs_cn[None, :] * stride_n

    for _i in range(0, 2):
        a1 = ct.load(a_ptrs, shape=(2, 2), padding_mode=ct.PaddingMode.ZERO)
        a_ptrs = a_ptrs + 2 * stride_n
        a2 = ct.load(a_ptrs, shape=(2, 2), padding_mode=ct.PaddingMode.ZERO)
        a_ptrs = a_ptrs + 2 * stride_n
        a3 = ct.load(a_ptrs, shape=(2, 2), padding_mode=ct.PaddingMode.ZERO)

        ct.store(c_ptrs, a1)
        c_ptrs = c_ptrs + 2 * stride_n
        ct.store(c_ptrs, a2)
        c_ptrs = c_ptrs + 2 * stride_n
        ct.store(c_ptrs, a3)
        c_ptrs = c_ptrs + 2 * stride_n

        a_ptrs = a_ptrs + 2 * stride_n


def wrapper_nested3(n_rows, n_cols):
    x = cp.arange(0, n_rows * n_cols, dtype=cp.int32).reshape(n_rows, n_cols)
    output = cp.zeros((n_rows, n_cols), dtype=x.dtype)
    grid = (n_cols // 4, 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, nested3, (x, output, x.strides[0] // x.itemsize, x.strides[1] // x.itemsize))
    return output


def test_nested3():
    results = {}
    for idx, (r, c) in enumerate([(8, 8), (4, 4), (16, 16), (2, 2)], start=1):
        results[f"test_case_{idx}"] = wrapper_nested3(r, c)
    return results


result_gold = test_nested3()
