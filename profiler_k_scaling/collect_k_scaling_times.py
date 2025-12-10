#!/usr/bin/env python3
"""
Collect times from k_scaling_runner.py output logs and generate CSV files.
Collects both kernel_time and e2e_time for each K value from a single run.

Reads from results/kN/runM/ folders (both metrics are in each log file) and outputs:
  - kernel_time.csv
  - e2e_time.csv
"""

import re
import csv
from pathlib import Path
import argparse
import statistics


# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()

# Default K values
K_VALUES = [0, 1, 5, 10]


def check_exit_code(content):
    """Check if exit code is 0 in log content."""
    match = re.search(r'Exit code:\s*(\d+)', content)
    if match:
        return int(match.group(1)) == 0
    return False


def extract_kernel_time_from_log(log_file):
    """Extract kernel time from a log file (only if exit code is 0)."""
    if not log_file.exists():
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Check exit code first
    if not check_exit_code(content):
        return None

    # Look for "Kernel time: X.XXX ms"
    match = re.search(r'Kernel time:\s*([\d.]+)\s*ms', content)
    if match:
        return float(match.group(1))

    return None


def extract_e2e_time_from_log(log_file):
    """Extract E2E time from a log file (only if exit code is 0)."""
    if not log_file.exists():
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Check exit code first
    if not check_exit_code(content):
        return None

    # Look for "E2E time: X.XXX ms"
    match = re.search(r'E2E time:\s*([\d.]+)\s*ms', content)
    if match:
        return float(match.group(1))

    return None


def extract_total_blocks_from_log(log_file):
    """Extract total_blocks from a log file.

    Looks for line: Triton-Viz: total_blocks = N (grid: X x Y x Z)
    """
    if not log_file.exists():
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # Look for "Triton-Viz: total_blocks = N"
    match = re.search(r'Triton-Viz:\s+total_blocks\s*=\s*(\d+)', content)
    if match:
        return int(match.group(1))

    return None


def extract_case_name_from_log(log_filename):
    """Extract case name from log filename (e.g., '01_matmul_triton1.log' -> 'matmul_triton1')."""
    name = log_filename.stem
    match = re.match(r'^\d+_(.+)$', name)
    if match:
        return match.group(1)
    return name


def discover_runs(k_dir):
    """Discover run directories under a K value directory."""
    runs = []
    if k_dir.exists():
        for run_dir in sorted(k_dir.glob("run*")):
            if run_dir.is_dir():
                runs.append(run_dir)
    return runs


def discover_k_values(base_dir):
    """Discover K value directories under the base results directory."""
    k_values = []
    if base_dir.exists():
        for k_dir in sorted(base_dir.glob("k*")):
            if k_dir.is_dir():
                match = re.match(r'k(\d+)', k_dir.name)
                if match:
                    k_values.append(int(match.group(1)))
    return sorted(k_values)


def collect_times_for_metric(base_dir, metric, k_values=None):
    """Collect times for a specific metric from all K values across multiple runs.

    Directory structure: base_dir/k{N}/run{M}/*.log
    Each log file contains both kernel_time and e2e_time.
    """
    # Select the appropriate extraction function
    if metric == "kernel_time":
        extract_func = extract_kernel_time_from_log
    else:  # e2e_time
        extract_func = extract_e2e_time_from_log

    # Discover K values if not specified (look directly in base_dir)
    if k_values is None:
        k_values = discover_k_values(base_dir)

    if not k_values:
        print(f"No K value directories found in {base_dir}")
        return [], []

    # Discover all case names from all runs
    all_cases = set()

    for k in k_values:
        k_dir = base_dir / f"k{k}"
        for run_dir in discover_runs(k_dir):
            for log_file in run_dir.glob("*.log"):
                case_name = extract_case_name_from_log(log_file)
                all_cases.add(case_name)

    if not all_cases:
        print(f"No log files found in {base_dir}")
        return [], []

    # Sort case names
    all_cases = sorted(all_cases)

    # Collect times for each case
    results = []
    for case_name in all_cases:
        row = {"case": case_name, "total_blocks": None}

        # For each K value, collect times from all runs
        for k in k_values:
            k_dir = base_dir / f"k{k}"
            times = []

            for run_dir in discover_runs(k_dir):
                # Find log file matching this case
                for log_file in run_dir.glob("*.log"):
                    if extract_case_name_from_log(log_file) == case_name:
                        time_value = extract_func(log_file)
                        if time_value is not None:
                            times.append(time_value)
                        # Extract total_blocks (same for all K values, so only get once)
                        if row["total_blocks"] is None:
                            row["total_blocks"] = extract_total_blocks_from_log(log_file)
                        break

            # Calculate statistics
            if times:
                row[f"k{k}_mean"] = statistics.mean(times)
                row[f"k{k}_min"] = min(times)
                row[f"k{k}_max"] = max(times)
                row[f"k{k}_runs"] = len(times)
            else:
                row[f"k{k}_mean"] = None
                row[f"k{k}_min"] = None
                row[f"k{k}_max"] = None
                row[f"k{k}_runs"] = 0

        results.append(row)

    return results, k_values


def write_csv(results, output_file, metric, k_values):
    """Write results to CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        unit = "ms"
        header = ["case", "total_blocks"]
        for k in k_values:
            header.append(f"k{k}_mean ({unit})")

        # Add speedup columns (k0 as baseline, speedup = k0_time / kN_time)
        if len(k_values) > 1:
            for k in k_values[1:]:
                header.append(f"k{k}_speedup")

        writer.writerow(header)

        # Write data
        for row in results:
            total_blocks = row.get("total_blocks")
            data = [row["case"], str(total_blocks) if total_blocks is not None else ""]

            # Add mean times
            for k in k_values:
                mean = row[f"k{k}_mean"]
                data.append(f"{mean:.3f}" if mean is not None else "")

            # Add speedup columns (speedup = k0_time / kN_time)
            if len(k_values) > 1:
                first_k = k_values[0]
                first_k_mean = row.get(f"k{first_k}_mean")
                for k in k_values[1:]:
                    k_mean = row[f"k{k}_mean"]
                    if first_k_mean is not None and k_mean is not None and k_mean > 0:
                        data.append(f"{first_k_mean/k_mean:.2f}x")
                    else:
                        data.append("")

            writer.writerow(data)

    print(f"CSV written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect times from K-scaling experiment logs and generate CSV files"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help=f"Base directory containing results (default: {SCRIPT_DIR / 'results'})"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=str(SCRIPT_DIR / "results"),
        help=f"Output directory for CSV files (default: {SCRIPT_DIR / 'results'})"
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        choices=["kernel_time", "e2e_time", "all"],
        default=["all"],
        help="Metric(s) to collect: kernel_time, e2e_time, or all (default: all)"
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=None,
        help=f"K values to look for (default: auto-discover)"
    )
    args = parser.parse_args()

    base_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine metrics to collect
    if "all" in args.metric:
        metrics_to_collect = ["kernel_time", "e2e_time"]
    else:
        metrics_to_collect = args.metric

    print(f"Collecting times from: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Metrics: {', '.join(metrics_to_collect)}")
    print()

    # First pass: discover K values (same for all metrics since they share the same log files)
    k_values_found = args.k if args.k else discover_k_values(base_dir)

    if not k_values_found:
        print(f"No K value directories found in {base_dir}")
        return 1

    print(f"K values found: {k_values_found}")
    for k in k_values_found:
        print(f"  - k{k}/run*/")
    print()

    # Collect results for all metrics (from the same log files)
    all_metric_results = {}
    valid_cases_per_metric = {}

    for metric in metrics_to_collect:
        print(f"{'=' * 60}")
        print(f"Processing metric: {metric}")
        print(f"{'=' * 60}")

        results, k_values = collect_times_for_metric(base_dir, metric, k_values_found)

        if not results:
            print(f"  No results found for metric '{metric}'")
            continue

        # Print summary before filtering
        print(f"  Found {len(results)} test cases (before filtering)")
        for k in k_values:
            count = sum(1 for r in results if r[f"k{k}_mean"] is not None)
            print(f"    - k{k}: {count} results")

        # Filter to only keep cases where all K values have valid data (time > 0)
        filtered_results = []
        for r in results:
            all_valid = True
            for k in k_values:
                if r[f"k{k}_mean"] is None or r[f"k{k}_mean"] <= 0:
                    all_valid = False
                    break
            if all_valid:
                filtered_results.append(r)

        print(f"  After filtering (all K values passed): {len(filtered_results)} test cases")
        print()

        all_metric_results[metric] = {r["case"]: r for r in filtered_results}
        valid_cases_per_metric[metric] = set(r["case"] for r in filtered_results)

    # Find intersection of valid cases across all metrics
    if len(valid_cases_per_metric) > 1:
        common_cases = set.intersection(*valid_cases_per_metric.values())
        print(f"{'=' * 60}")
        print(f"Cross-metric filtering")
        print(f"{'=' * 60}")
        for metric, cases in valid_cases_per_metric.items():
            print(f"  {metric}: {len(cases)} valid cases")
        print(f"  Common cases (intersection): {len(common_cases)}")
        print()
    else:
        common_cases = list(valid_cases_per_metric.values())[0] if valid_cases_per_metric else set()

    # Write CSVs with only common cases
    for metric in metrics_to_collect:
        if metric not in all_metric_results:
            continue

        # Filter to only common cases
        final_results = [
            all_metric_results[metric][case]
            for case in sorted(common_cases)
            if case in all_metric_results[metric]
        ]

        csv_file = output_dir / f"{metric}.csv"
        write_csv(final_results, csv_file, metric, k_values_found)
        print()

    print(f"{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    exit(main())
