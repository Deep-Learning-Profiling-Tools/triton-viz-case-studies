import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


@ct.kernel
def _index_select_cat_fwd(source, index, out, num_cols: ct.Constant[int], tile_size: ct.Constant[int]):
    pid = ct.bid(0)  # row in output
    row_idx = ct.load(index, (pid,), shape=()).astype(cp.int32)
    src_tile = ct.load(source, (row_idx,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, (pid,), tile=src_tile)


def index_select_cat_fwd(output, source, index, tile_size=512):
    src_cp = to_cupy(source)
    idx_cp = to_cupy(index).astype(cp.int32)
    out_cp = to_cupy(output)

    num_rows, num_cols = src_cp.shape
    tile_size = min(tile_size, num_cols)

    src_flat = src_cp.reshape(num_rows, num_cols)
    out_flat = out_cp.reshape(idx_cp.size, num_cols)

    grid = (idx_cp.size, 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _index_select_cat_fwd, (src_flat, idx_cp, out_flat, num_cols, tile_size))
    return out_cp


def test_index_select_cat_fwd():
    results = {}
    source = cp.random.randn(10, 512, dtype=cp.float32)

    cases = [
        cp.asarray([0, 2, 4, 6, 8], dtype=cp.int32),
        cp.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=cp.int32),
        cp.asarray([0], dtype=cp.int32),
        cp.asarray([9, 7, 5, 3, 1], dtype=cp.int32),
    ]
    for idx, inds in enumerate(cases, start=1):
        out = cp.empty((inds.size, source.shape[1]), dtype=source.dtype)
        index_select_cat_fwd(out, source, inds)
        results[f"test_case_{idx}"] = to_torch(out.copy())
    return results


result_gold = test_index_select_cat_fwd()
