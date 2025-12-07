# cuTile ports

This directory mirrors `kernels/TritonBench` with cuTile equivalents. A symlink `cutile_kernels/` points here so the modules can be imported with a valid Python identifier.

## Dependencies
- `cuda-tile` (cuTile Python)
- A matching `cupy` wheel for your CUDA version (e.g., `cupy-cuda13x`)
- PyTorch (only used in a few helper conversions/tests)

## Converted so far
Utility operations:
- `add_example.py`
- `add_value.py`
- `masked_add_cuda.py`
- `masked_select.py`
- `sin_computation.py`
- `sin_kernel.py`
- `cosine_compute.py`
- `triton_mul2.py`
- `vector_addition.py`
- `vector_addition_custom.py`
- `index_select_cat.py`
- `index_select_bwd.py`
- `var_len_copy.py`
- `nested_loops_processing.py`
- `pow_scalar_tensor.py`
- `mean_reduction.py`

Activation functions:
- `relu_triton_kernel.py`
- `fused_activation.py`
- `geglu_tanh_triton.py`
- `relu_strided_buffer.py`
- `swiglu_triton.py`
- `swiglu_fwd.py`
- `swiglu_backward.py`

The remaining TritonBench kernels still need cuTile rewrites.
