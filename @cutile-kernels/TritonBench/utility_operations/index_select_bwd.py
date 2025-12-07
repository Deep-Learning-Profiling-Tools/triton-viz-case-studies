import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


@ct.kernel
def _index_select_cat_bwd(grad_output, index, grad_source, num_cols: ct.Constant[int], tile_size: ct.Constant[int]):
    pid = ct.bid(0)  # row in grad_output
    row_idx = ct.load(index, (pid,), shape=()).astype(cp.int32)
    go_tile = ct.load(grad_output, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(grad_source, (row_idx,), tile=go_tile)


def index_select_cat_bwd(grad_source, index, grad_output, tile_size=512):
    grad_source_cp = to_cupy(grad_source)
    grad_output_cp = to_cupy(grad_output)
    index_cp = to_cupy(index).astype(cp.int32)
    num_cols = grad_source_cp.shape[1]
    tile_size = min(tile_size, num_cols)

    go_flat = grad_output_cp.reshape(index_cp.size, num_cols)
    gs_flat = grad_source_cp.reshape(grad_source_cp.shape[0], num_cols)

    grid = (index_cp.size, 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _index_select_cat_bwd, (go_flat, index_cp, gs_flat, num_cols, tile_size))
    return grad_source_cp


def test_index_select_cat_bwd():
    results = {}
    num_rows = 10
    num_cols = 512
    cases = [
        cp.asarray([0, 2, 4, 6, 8], dtype=cp.int32),
        cp.asarray([1, 3, 5, 7, 9], dtype=cp.int32),
        cp.asarray([0, 0, 0, 0, 0], dtype=cp.int32),
        cp.asarray([9, 9, 9, 9, 9], dtype=cp.int32),
    ]
    for idx, inds in enumerate(cases, start=1):
        grad_source = cp.zeros((num_rows, num_cols), dtype=cp.float32)
        grad_output = cp.random.randn(inds.size, num_cols, dtype=cp.float32)
        index_select_cat_bwd(grad_source, inds, grad_output)
        results[f"test_case_{idx}"] = to_torch(grad_source.copy())
    return results


result_gold = test_index_select_cat_bwd()
