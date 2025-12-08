"""
Triton Kernel Performance Profiler
Copied from liger_kernel.triton.monkey_patch for standalone use in TritonBench.
"""
import importlib
import time

try:
    import torch
except ImportError:  # pragma: no cover - torch is required at runtime but keep import safe
    torch = None

_LAUNCH_TIMING_STATE_KEY = "_triton_profiler_timing_state"
_launch_hooks_enabled = False


def _torch_cuda_ready():
    """Check if PyTorch and CUDA are available."""
    global torch
    if torch is None:
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            return False
    return torch.cuda.is_available()


def _resolve_stream(stream_handle):
    """Convert stream handle to PyTorch CUDA stream object."""
    if not _torch_cuda_ready():
        return None
    if stream_handle is None:
        return torch.cuda.current_stream()
    external_stream_ctor = getattr(torch.cuda, "ExternalStream", None)
    if external_stream_ctor is None:
        return torch.cuda.current_stream()
    try:
        return external_stream_ctor(stream_handle)
    except (TypeError, RuntimeError):
        return torch.cuda.current_stream()


def _liger_launch_enter_hook(metadata):
    """Hook function called before kernel launch."""
    if metadata is None or not _torch_cuda_ready():
        return
    try:
        data = metadata.get()
    except AttributeError:
        return
    stream = _resolve_stream(data.get("stream"))
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    cpu_start = time.perf_counter_ns()
    start_event.record(stream)
    data[_LAUNCH_TIMING_STATE_KEY] = {
        "stream": stream,
        "start_event": start_event,
        "end_event": end_event,
        "cpu_start": cpu_start,
    }


def _liger_launch_exit_hook(metadata):
    """Hook function called after kernel launch."""
    if metadata is None or not _torch_cuda_ready():
        return
    try:
        data = metadata.get()
    except AttributeError:
        return
    state = data.pop(_LAUNCH_TIMING_STATE_KEY, None)
    if not state:
        return
    stream = state["stream"]
    cpu_duration_ns = time.perf_counter_ns() - state["cpu_start"]
    state["end_event"].record(stream)
    state["end_event"].synchronize()
    gpu_time_ms = state["start_event"].elapsed_time(state["end_event"])
    kernel_name = data.get("name", "unknown_kernel")
    print(
        f"[triton-profiler] kernel={kernel_name} cpu_launch_ms={cpu_duration_ns / 1e6:.3f} gpu_time_ms={gpu_time_ms:.3f}",
        flush=True,
    )


def enable_triton_kernel_timing():
    """Enable Triton kernel timing instrumentation."""
    global _launch_hooks_enabled
    if _launch_hooks_enabled:
        return
    try:
        from triton import knobs
    except ImportError:
        return
    knobs.runtime.launch_enter_hook = _liger_launch_enter_hook
    knobs.runtime.launch_exit_hook = _liger_launch_exit_hook
    _launch_hooks_enabled = True
    print("[triton-profiler] Triton kernel timing enabled")


def disable_triton_kernel_timing():
    """Disable Triton kernel timing instrumentation."""
    global _launch_hooks_enabled
    if not _launch_hooks_enabled:
        return
    try:
        from triton import knobs
    except ImportError:
        return
    knobs.runtime.launch_enter_hook = None
    knobs.runtime.launch_exit_hook = None
    _launch_hooks_enabled = False
    print("[triton-profiler] Triton kernel timing disabled")


if __name__ == "__main__":
    # When run as a script, just enable the timing
    enable_triton_kernel_timing()
