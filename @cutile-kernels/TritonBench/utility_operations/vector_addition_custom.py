import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


@ct.kernel
def _add_kernel(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(a, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(b, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(c, (pid,), tile=a_tile + b_tile)


def custom_add(a, b, tile_size=16):
    a_cp = to_cupy(a)
    b_cp = to_cupy(b)
    out = cp.empty_like(a_cp)
    if a_cp.size != b_cp.size:
        raise ValueError(f"Input sizes must match (got {a_cp.size} and {b_cp.size}).")

    grid = (ct.cdiv(a_cp.size, tile_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _add_kernel, (a_cp, b_cp, out, tile_size))
    return out


def test_add():
    test_vectors = [
        (cp.arange(1, 17, dtype=cp.float32), cp.arange(16, 0, -1, dtype=cp.float32)),
        (cp.arange(1, 9, dtype=cp.float32), cp.arange(8, 0, -1, dtype=cp.float32)),
        (cp.arange(32, dtype=cp.float32), cp.arange(32, 0, -1, dtype=cp.float32)),
        (cp.asarray([], dtype=cp.float32), cp.asarray([], dtype=cp.float32)),
    ]
    results = {}
    for idx, (lhs, rhs) in enumerate(test_vectors, start=1):
        out = custom_add(lhs, rhs, tile_size=16)
        results[f"test_case_{idx}"] = to_torch(out)
    return results


result_gold = test_add()
