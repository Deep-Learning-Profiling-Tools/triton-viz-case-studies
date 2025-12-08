#!/usr/bin/env python3
"""
Common test registry utilities for discovering and loading test cases.

This module provides shared functionality for loading test registries
and discovering tests across different repositories (TritonBench, Liger-Kernel).
"""

from pathlib import Path
from typing import Dict, List, Any, Optional

# Base directory (tile-lens-experiments root)
BASE_DIR = Path(__file__).parent.parent.resolve()

# Repository directories
TRITONBENCH_DIR = BASE_DIR / "submodules" / "TritonBench" / "data" / "TritonBench_G_v1"
LIGER_KERNEL_DIR = BASE_DIR / "submodules" / "Liger-Kernel" / "test" / "transformers"

# Default test registry file location
DEFAULT_REGISTRY = Path(__file__).parent / "test_registry.txt"

# Repository configurations
REPO_CONFIGS = {
    "tritonbench": {
        "test_dir": TRITONBENCH_DIR,
        "test_pattern": "*.py",
        "is_pytest": False,
    },
    "liger_kernel": {
        "test_dir": LIGER_KERNEL_DIR,
        "test_pattern": "test_*.py",
        "is_pytest": True,
    }
}


def load_registry(registry_file: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Load test registry from file.

    Format: repo:test_name (one per line)
    Lines starting with # are comments.
    Line number (excluding comments/blank lines) determines global test ID.

    Args:
        registry_file: Path to registry file. If None, uses DEFAULT_REGISTRY.

    Returns:
        List of dicts with keys: global_id, repo, test_spec
    """
    if registry_file is None:
        registry_file = DEFAULT_REGISTRY

    registry = []
    if not registry_file.exists():
        return registry

    global_id = 0
    with open(registry_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            global_id += 1

            # Parse repo:test_spec format
            if ':' in line:
                # Find the first colon (repo separator)
                first_colon = line.index(':')
                repo = line[:first_colon]
                test_spec = line[first_colon + 1:]
            else:
                # Legacy format: assume tritonbench
                repo = "tritonbench"
                test_spec = line

            registry.append({
                "global_id": global_id,
                "repo": repo,
                "test_spec": test_spec
            })

    return registry


def discover_tests(
    repo: str,
    registry: Optional[List[Dict[str, Any]]] = None,
    case: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Discover tests to run for a given repository using the test registry.

    Args:
        repo: Repository name (tritonbench or liger_kernel)
        registry: List of registry entries from load_registry(). If None, loads default.
        case: Optional single test case to run

    Returns:
        List of test dicts with keys:
          - global_id: Global unique test ID from registry
          - name: Display name (e.g., "matmul_triton1" or "test_rms_norm::test_correctness")
          - file_path: Path to the test file
          - test_function: For pytest repos, the specific test function (None for direct python)
          - is_pytest: Whether to run with pytest
    """
    if repo not in REPO_CONFIGS:
        print(f"Error: Unknown repository: {repo}")
        return []

    if registry is None:
        registry = load_registry()

    config = REPO_CONFIGS[repo]
    test_dir = config["test_dir"]
    is_pytest = config["is_pytest"]

    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return []

    tests = []

    # Filter registry by repo
    repo_entries = [e for e in registry if e["repo"] == repo]

    if not repo_entries:
        print(f"Warning: No tests found in registry for repo: {repo}")
        return []

    # If case is specified, find it in registry
    if case:
        case_spec = case if case.endswith('.py') or '::' in case else f"{case}.py"
        matching = [e for e in repo_entries if e["test_spec"] == case_spec]
        if matching:
            repo_entries = matching
            print(f"Running single case: {case_spec} (ID: {matching[0]['global_id']})")
        else:
            print(f"Error: Case not found in registry: {case_spec}")
            return []

    # Build test list from registry entries
    for entry in repo_entries:
        test_spec = entry["test_spec"]
        global_id = entry["global_id"]

        if is_pytest:
            # Liger-Kernel style: test_file.py::test_function
            if "::" in test_spec:
                file_part, func_part = test_spec.split("::", 1)
                file_path = test_dir / file_part
                if file_path.exists():
                    tests.append({
                        "global_id": global_id,
                        "name": f"{file_path.stem}::{func_part}",
                        "file_path": file_path,
                        "test_function": func_part,
                        "is_pytest": True
                    })
                else:
                    print(f"Warning: Test file not found: {file_path}")
            else:
                print(f"Warning: Invalid pytest spec (missing ::): {test_spec}")
        else:
            # TritonBench style: direct Python files
            file_path = test_dir / test_spec
            if file_path.exists():
                tests.append({
                    "global_id": global_id,
                    "name": file_path.stem,
                    "file_path": file_path,
                    "test_function": None,
                    "is_pytest": False
                })
            else:
                print(f"Warning: Test file not found: {file_path}")

    print(f"Loaded {len(tests)} tests from registry for {repo}")
    return tests


def get_test_dir(repo: str) -> Path:
    """Get the test directory for a given repository."""
    if repo not in REPO_CONFIGS:
        raise ValueError(f"Unknown repository: {repo}")
    return REPO_CONFIGS[repo]["test_dir"]


def list_available_repos() -> List[str]:
    """List all available repository names."""
    return list(REPO_CONFIGS.keys())
