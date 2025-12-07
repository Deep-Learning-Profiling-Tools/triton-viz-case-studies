import math

import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


@ct.kernel
def _cos_kernel(a, b, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    vals = ct.load(a, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(b, (pid,), tile=ct.cos(vals))


def cos(A):
    a_cp = to_cupy(A)
    b_cp = cp.empty_like(a_cp)
    n_elements = a_cp.size
    block_size = 1 << math.ceil(math.log2(int(math.ceil(math.sqrt(max(1, n_elements))))))
    grid = (ct.cdiv(n_elements, block_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _cos_kernel, (a_cp, b_cp, block_size))
    return b_cp


def test_cos_function():
    test_cases = {
        "test_case_1": cp.random.rand(1024, dtype=cp.float32) * 2 * math.pi,
        "test_case_2": cp.random.rand(2048, dtype=cp.float32) * 2 * math.pi,
        "test_case_3": cp.random.rand(4096, dtype=cp.float32) * 2 * math.pi,
        "test_case_4": cp.random.rand(8192, dtype=cp.float32) * 2 * math.pi,
    }
    results = {name: to_torch(cos(tensor)) for name, tensor in test_cases.items()}
    return results


result_gold = test_cos_function()
