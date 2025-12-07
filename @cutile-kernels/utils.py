import cupy as cp


def to_cupy(x):
    """Convert torch/NumPy inputs to CuPy without copies when possible."""
    if isinstance(x, cp.ndarray):
        return x
    try:
        import torch

        if torch.is_tensor(x):
            return cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
    except Exception:
        pass
    return cp.asarray(x)


def to_torch(x):
    """Convert a CuPy array back to torch via DLPack."""
    import torch

    return torch.utils.dlpack.from_dlpack(x.toDlpack())
