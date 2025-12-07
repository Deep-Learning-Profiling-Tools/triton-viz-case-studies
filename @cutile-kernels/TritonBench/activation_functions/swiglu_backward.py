from cutile_kernels.TritonBench.activation_functions.swiglu_triton import swiglu_backward, test_swiglu

__all__ = ["swiglu_backward", "test_swiglu", "result_gold"]

result_gold = test_swiglu()
