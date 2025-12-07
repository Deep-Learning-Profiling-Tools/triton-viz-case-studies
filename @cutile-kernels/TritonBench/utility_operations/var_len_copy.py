import cupy as cp
import cuda.tile as ct

from cutile_kernels.utils import to_cupy, to_torch


@ct.kernel
def _var_len_copy(old_start, old_len, old_loc, new_start, new_loc, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    length = ct.load(old_len, (pid,), shape=()).astype(cp.int32)
    src_start = ct.load(old_start, (pid,), shape=()).astype(cp.int32)
    dst_start = ct.load(new_start, (pid,), shape=()).astype(cp.int32)

    # loop over chunks of tile_size
    for offset in range(0, length, tile_size):
        tile = ct.load(old_loc, (src_start + offset,), shape=(tile_size,), padding_mode=ct.PaddingMode.ZERO)
        ct.store(new_loc, (dst_start + offset,), tile=tile)


def launch_var_len_copy_triton(old_a_start, old_a_len, old_location, new_a_start, new_a_location, tile_size=256):
    old_start_cp = to_cupy(old_a_start).astype(cp.int32)
    old_len_cp = to_cupy(old_a_len).astype(cp.int32)
    new_start_cp = to_cupy(new_a_start).astype(cp.int32)
    old_loc_cp = to_cupy(old_location)
    new_loc_cp = to_cupy(new_a_location)

    grid = (old_start_cp.size, 1, 1)
    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        _var_len_copy,
        (old_start_cp, old_len_cp, old_loc_cp, new_start_cp, new_loc_cp, tile_size),
    )
    return new_loc_cp


def test_launch_var_len_copy_kernel_triton():
    num_arrays = 3
    old_a_start = cp.asarray([0, 100, 300], dtype=cp.int32)
    old_a_len = cp.asarray([50, 150, 200], dtype=cp.int32)
    old_a_location = cp.arange(500, dtype=cp.float32)
    new_a_start = cp.asarray([0, 60, 260], dtype=cp.int32)
    new_a_location = cp.zeros(500, dtype=cp.float32)

    launch_var_len_copy_triton(old_a_start, old_a_len, old_a_location, new_a_start, new_a_location)

    results = {}
    for i in range(num_arrays):
        old_start = int(old_a_start[i])
        new_start = int(new_a_start[i])
        length = int(old_a_len[i])
        results[f"test_case_{i+1}"] = bool(
            cp.all(old_a_location[old_start : old_start + length] == new_a_location[new_start : new_start + length])
        )
    return results


result_gold = test_launch_var_len_copy_kernel_triton()
