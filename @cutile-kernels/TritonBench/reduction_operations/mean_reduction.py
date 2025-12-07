import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


def _dim_compress(inp: cp.ndarray, dims):
    if isinstance(dims, int):
        dims = [dims]
    dim = inp.ndim
    batch_dim = [i for i in range(dim) if i not in dims]
    sorted_reduction_dim = sorted(dims, key=lambda x: inp.strides[x], reverse=True)
    order = batch_dim + sorted_reduction_dim
    return cp.transpose(inp, axes=order).copy(), order, batch_dim, sorted_reduction_dim


@ct.kernel
def mean_dim_kernel(x, out, M: ct.Constant[int], N: ct.Constant[int], block_m: ct.Constant[int], block_n: ct.Constant[int]):
    pid = ct.bid(0)
    row_start = pid * block_m
    row_offsets = row_start + ct.arange(0, block_m)
    row_mask = row_offsets < M
    acc = ct.zeros((block_m, block_n), dtype=ct.float32)
    for off in range(0, N, block_n):
        cols = off + ct.arange(0, block_n)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        vals = ct.load(x, (row_offsets[:, None], cols[None, :]), shape=(block_m, block_n), padding_mode=ct.PaddingMode.ZERO)
        acc = acc + vals.astype(cp.float32) * mask
    mean = ct.sum(acc, axis=1) / N
    ct.store(out, (row_offsets, ct.zeros((block_m,), dtype=cp.int32)), tile=mean[:, None], mask=row_mask[:, None])


def mean_dim(x, dim, keepdim=False, dtype=None, block_m=8, block_n=8):
    x_cp = to_cupy(x)
    if dtype is None:
        dtype = x_cp.dtype

    shape = list(x_cp.shape)
    if isinstance(dim, int):
        dim = [dim]
    dim = [d % x_cp.ndim for d in dim]
    x_perm, order, batch_dim, red_dims = _dim_compress(x_cp, dim)
    N = 1
    for d in dim:
        N *= shape[d]
        shape[d] = 1
    M = x_perm.size // N
    x_2d = x_perm.reshape(M, N)
    out_cp = cp.empty((M, 1), dtype=dtype)

    grid = (ct.cdiv(M, block_m), 1, 1)
    ct.launch(cp.cuda.get_current_stream(), grid, mean_dim_kernel, (x_2d, out_cp, M, N, block_m, block_n))
    out = out_cp.reshape([shape[i] for i in batch_dim] + [shape[d] for d in dim])
    if not keepdim:
        out = out.reshape([shape[i] for i in range(len(shape)) if i not in dim])
    return out


def test_mean_dim():
    results = {}
    b1 = cp.random.randn(2, 3, 4, 5).astype(cp.float32)
    results["test_case_1"] = to_torch(mean_dim(b1, 1))
    b2 = cp.random.randn(2, 3, 4, 5).astype(cp.float32)
    results["test_case_2"] = to_torch(mean_dim(b2, [1, 2]))
    b3 = cp.random.randn(2, 3, 4, 5).astype(cp.float32)
    results["test_case_3"] = to_torch(mean_dim(b3, [1, 2], keepdim=True))
    b4 = cp.random.randn(2, 3, 4, 5).astype(cp.float64)
    results["test_case_4"] = to_torch(mean_dim(b4, [1, 2], dtype=cp.float32))
    return results


result_gold = test_mean_dim()
