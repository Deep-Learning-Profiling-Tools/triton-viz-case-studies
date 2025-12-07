import torch

from cutile_kernels.TritonBench.utility_operations.sin_kernel import call_kernel
from cutile_kernels.utils import to_cupy, to_torch


def sin_triton(x, out=None):
    x_cp = to_cupy(x)
    res_cp = call_kernel(x_cp, tile_size=4)
    if out is None:
        return res_cp
    out.copy_(to_torch(res_cp))
    return out


def test_sin_triton():
    results = {}
    samples = [
        torch.tensor([0.0, 1.0, 2.0, 3.0], device="cuda"),
        torch.tensor([4.0, 5.0, 6.0, 7.0], device="cuda"),
        torch.tensor([8.0, 9.0, 10.0, 11.0], device="cuda"),
        torch.tensor([12.0, 13.0, 14.0, 15.0], device="cuda"),
    ]
    for idx, sample in enumerate(samples, start=1):
        out = torch.empty_like(sample)
        sin_triton(sample, out)
        results[f"test_case_{idx}"] = out
    return results


result_gold = test_sin_triton()
