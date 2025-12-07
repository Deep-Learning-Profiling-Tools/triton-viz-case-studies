import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


@ct.kernel
def _masked_add_kernel(grad, p_data, p_mask, alpha, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    grad_tile = ct.load(grad, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    p_tile = ct.load(p_data, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    mask_tile = ct.load(p_mask, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    update = grad_tile + p_tile * alpha
    # keep original grad where mask is non-zero
    out_tile = ct.where(mask_tile != 0, grad_tile, update)
    ct.store(grad, (pid,), tile=out_tile)


def masked_add(grad, p_data, p_mask, alpha: float = 0.0, tile_size=1024):
    grad_cp = to_cupy(grad)
    p_data_cp = to_cupy(p_data)
    p_mask_cp = to_cupy(p_mask)
    grid = (ct.cdiv(grad_cp.size, tile_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _masked_add_kernel, (grad_cp, p_data_cp, p_mask_cp, alpha, tile_size))
    return grad_cp


def test_masked_add():
    cp.random.seed(0)
    n = 10000
    grad = cp.random.randn(n, dtype=cp.float32)
    p_data = cp.random.randn(n, dtype=cp.float32)
    p_mask = cp.random.randint(0, 2, size=n, dtype=cp.int32)

    results = {}
    grad_out = masked_add(grad.copy(), p_data, p_mask, alpha=0.5)
    results["test_case_1"] = to_torch(grad_out.copy())

    grad_out = masked_add(grad.copy(), p_data, p_mask, alpha=0.0)
    results["test_case_2"] = to_torch(grad_out.copy())

    grad_out = masked_add(grad.copy(), p_data, cp.zeros_like(p_mask), alpha=0.5)
    results["test_case_3"] = to_torch(grad_out.copy())

    grad_out = masked_add(grad.copy(), p_data, cp.ones_like(p_mask), alpha=0.5)
    results["test_case_4"] = to_torch(grad_out.copy())

    return results


result_gold = test_masked_add()
