#!/usr/bin/env python3
"""
Wrapper script to run TritonBench tests with optional profiling enabled.

Usage:
    python tritonbench_profiler_wrapper.py <test_script.py>

Environment Variables:
    ENABLE_TRITON_PROFILER=1  - Enable Triton kernel timing profiler
"""
import sys
import os
import runpy

# Check if profiling should be enabled
if os.getenv("ENABLE_TRITON_PROFILER", "0") == "1":
    # Add the base directory to Python path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    # Import and enable the profiler
    from triton_profiler import enable_triton_kernel_timing
    enable_triton_kernel_timing()

# Get the test script path from command line arguments
if len(sys.argv) < 2:
    print("Error: No test script specified", file=sys.stderr)
    print("Usage: python tritonbench_profiler_wrapper.py <test_script.py>", file=sys.stderr)
    sys.exit(1)

test_script = sys.argv[1]

# Remove the wrapper script from argv so the test script sees correct arguments
sys.argv = sys.argv[1:]

# Run the test script as __main__
try:
    runpy.run_path(test_script, run_name="__main__")
except SystemExit as e:
    # Preserve exit code from the test script
    sys.exit(e.code)
except Exception as e:
    print(f"Error running test script: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
