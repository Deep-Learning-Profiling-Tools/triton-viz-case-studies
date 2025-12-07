import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


@ct.kernel
def fused_add_mul_activation_kernel(x, bias, other, out, multiplier, tile_size: ct.Constant[int], activation: ct.Constant[int]):
    pid = ct.bid(0)
    tile_x = ct.load(x, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    tile_bias = ct.load(bias, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    tile_in = ct.load(other, (pid,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
    activ_input = multiplier * tile_in + tile_x + tile_bias
    if activation == 0:
        activ_out = 1.0 / (1.0 + ct.exp(-activ_input))
    else:
        activ_out = ct.maximum(activ_input, 0)
    ct.store(out, (pid,), tile=activ_out)


def fused_add_mul_activation(in_out_tensor, bias, in_tensor, multiplier=0.5, activation="sigmoid", tile_size=2048):
    x_cp = to_cupy(in_out_tensor)
    bias_cp = to_cupy(bias)
    in_cp = to_cupy(in_tensor)
    out = cp.empty_like(x_cp)
    activation_flag = 0 if activation == "sigmoid" else 1
    grid = (ct.cdiv(x_cp.size, tile_size), 1, 1)
    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        fused_add_mul_activation_kernel,
        (x_cp, bias_cp, in_cp, out, multiplier, tile_size, activation_flag),
    )
    return out


def test_fused_add_mul_activation():
    num_elements = 8192
    num_weights = 64
    in_out = cp.random.randn(num_elements, dtype=cp.float32)
    bias = cp.random.randn(num_weights, dtype=cp.float32)
    inp = cp.random.randn(num_elements, dtype=cp.float32)

    results = {}
    results["test_case_1"] = to_torch(fused_add_mul_activation(in_out.copy(), bias, inp, activation="sigmoid"))
    results["test_case_2"] = to_torch(fused_add_mul_activation(in_out.copy(), bias, inp, activation="relu"))
    return results


result_gold = test_fused_add_mul_activation()
