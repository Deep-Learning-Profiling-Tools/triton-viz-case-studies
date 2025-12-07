import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


@ct.kernel
def _vector_add_kernel(x, y, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    a_tile = ct.load(x, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(y, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, (pid,), tile=a_tile + b_tile)


def add(x, y, tile_size=1024):
    """Elementwise add using cuTile; defaults mirror the Triton BLOCK_SIZE."""
    x_cp = to_cupy(x)
    y_cp = to_cupy(y)
    if x_cp.size != y_cp.size:
        raise ValueError(f"Input sizes must match (got {x_cp.size} and {y_cp.size}).")

    out = cp.empty_like(x_cp)
    grid = (ct.cdiv(x_cp.size, tile_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _vector_add_kernel, (x_cp, y_cp, out, tile_size))
    return out


def test_add():
    sizes = [98432, 1024, 2048, 4096]
    results = {}

    for idx, size in enumerate(sizes, start=1):
        lhs = cp.random.rand(size, dtype=cp.float32)
        rhs = cp.random.rand(size, dtype=cp.float32)
        out = add(lhs, rhs, tile_size=1024)
        # convert to torch for easier downstream comparison if needed
        results[f"test_case_{idx}"] = to_torch(out)

    return results


result_gold = test_add()
