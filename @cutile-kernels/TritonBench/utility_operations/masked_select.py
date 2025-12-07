import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


def broadcastable(s1, s2) -> bool:
    if not s1 or not s2:
        return True
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    d = len(s1) - len(s2)
    for i in range(len(s2)):
        if s1[d + i] == 1 or s2[i] == 1 or s1[d + i] == s2[i]:
            continue
        return False
    return True


@ct.kernel
def _masked_select(inp, mask, prefix_sum, out, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    offsets = pid * tile_size + ct.arange(0, tile_size)
    valid = offsets < inp.size
    vals = ct.load(inp, (offsets,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    mask_vals = ct.load(mask, (offsets,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    out_offsets = ct.load(prefix_sum, (offsets,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO) - 1
    ct.store(out, (out_offsets,), tile=vals, mask=(mask_vals != 0) & valid)


def masked_select(inp, mask, tile_size=256):
    inp_cp = to_cupy(inp)
    mask_cp = to_cupy(mask)
    assert broadcastable(tuple(inp_cp.shape), tuple(mask_cp.shape)), "Shapes must be broadcastable"
    inp_b, mask_b = cp.broadcast_arrays(inp_cp, mask_cp)
    inp_flat = inp_b.ravel()
    mask_flat = mask_b.ravel().astype(cp.int32)
    prefix_sum = cp.cumsum(mask_flat)
    out = cp.empty(int(prefix_sum[-1]), dtype=inp_flat.dtype)
    grid = (ct.cdiv(inp_flat.size, tile_size), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, _masked_select, (inp_flat, mask_flat, prefix_sum, out, tile_size))
    return out


def test_masked_select():
    results = {}
    cases = [
        (cp.random.rand(4, 4).astype(cp.float32), cp.random.randint(0, 2, (4, 4), dtype=cp.bool_)),
        (cp.random.rand(2, 3, 4).astype(cp.float64), cp.ones((2, 3, 4), dtype=cp.bool_)),
        (cp.random.randint(0, 100, (2, 2, 2, 2), dtype=cp.int64), cp.zeros((2, 2, 2, 2), dtype=cp.bool_)),
        (cp.random.rand(512, 1024).astype(cp.float32), cp.random.randint(0, 2, (512, 1024), dtype=cp.bool_)),
    ]
    for idx, (x, m) in enumerate(cases):
        res = masked_select(x, m)
        results[f"test_case_{idx}"] = to_torch(res)
    return results


result_gold = test_masked_select()
