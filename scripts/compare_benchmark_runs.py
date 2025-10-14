#!/usr/bin/env python3
"""Benchmark Comparison Tool for Jetson Orin Nano Matrix Multiplication Analysis.

Copyright 2025 ByteStack Labs
SPDX-License-Identifier: MIT

This module provides benchmark comparison and analysis capabilities for comparing
performance metrics across different benchmark runs on NVIDIA Jetson Orin Nano.

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 1.0.0
License: MIT

Usage:
    python scripts/compare_benchmark_runs.py <baseline_dir> <new_dir>

Example:
    python scripts/compare_benchmark_runs.py \
        data/archive/run_20251005_122213/raw/power_modes \
        data/raw/power_modes
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkMetrics:
    """Store metrics from a single benchmark run."""

    gflops: float
    power_w: float
    efficiency: float
    gflops_per_watt: float

    @property
    def summary(self) -> str:
        return f"{self.gflops:.1f} GFLOPS @ {self.power_w:.1f}W = {self.gflops_per_watt:.1f} GFLOPS/W"


class BenchmarkComparator:
    """Compare two benchmark runs and generate reports."""

    def __init__(self, baseline_dir: Path, new_dir: Path):
        self.baseline_dir = baseline_dir
        self.new_dir = new_dir

    def load_benchmark_data(self, json_file: Path) -> dict[str, Any]:
        """Load benchmark data from JSON file."""
        with open(json_file) as f:
            data: dict[str, Any] = json.load(f)
            return data

    def extract_metrics(
        self, data: dict, power_mode: str, matrix_size: str = "matrix_1024"
    ) -> BenchmarkMetrics | None:
        """Extract metrics for a specific power mode and matrix size."""
        try:
            result = data["results"][power_mode][matrix_size]
            return BenchmarkMetrics(
                gflops=result["gflops"],
                power_w=result.get("power_watts", 0.0),
                efficiency=result["efficiency_percent"],
                gflops_per_watt=result.get("gflops_per_watt", 0.0),
            )
        except KeyError:
            return None

    def find_implementation_file(
        self, directory: Path, implementation: str
    ) -> Path | None:
        """Find the latest benchmark file for a given implementation."""
        pattern = f"{implementation}_*_analysis_*.json"
        files = sorted(
            directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
        )
        return files[0] if files else None

    def calculate_speedup(self, baseline: float, new: float) -> tuple[float, str]:
        """Calculate speedup and return with improvement indicator."""
        speedup = new / baseline if baseline > 0 else 0
        change_pct = ((new - baseline) / baseline * 100) if baseline > 0 else 0

        if abs(change_pct) < 1:
            indicator = "≈"  # Approximately equal
        elif change_pct > 0:
            indicator = "↑"  # Improvement
        else:
            indicator = "↓"  # Regression

        return speedup, indicator

    def compare_implementation(self, implementation: str) -> dict[str, Any]:
        """Compare a single implementation between baseline and new run."""
        baseline_file = self.find_implementation_file(self.baseline_dir, implementation)
        new_file = self.find_implementation_file(self.new_dir, implementation)

        if not baseline_file or not new_file:
            return {
                "error": f"Missing data files for {implementation}",
                "baseline_file": str(baseline_file) if baseline_file else "NOT FOUND",
                "new_file": str(new_file) if new_file else "NOT FOUND",
            }

        baseline_data = self.load_benchmark_data(baseline_file)
        new_data = self.load_benchmark_data(new_file)

        comparison: dict[str, Any] = {
            "implementation": implementation,
            "baseline_file": baseline_file.name,
            "new_file": new_file.name,
            "power_modes": {},
        }

        for power_mode in ["15W", "25W", "MAXN_SUPER"]:
            baseline_metrics = self.extract_metrics(baseline_data, power_mode)
            new_metrics = self.extract_metrics(new_data, power_mode)

            if baseline_metrics and new_metrics:
                speedup, indicator = self.calculate_speedup(
                    baseline_metrics.gflops, new_metrics.gflops
                )

                comparison["power_modes"][power_mode] = {
                    "baseline": baseline_metrics,
                    "new": new_metrics,
                    "speedup": speedup,
                    "indicator": indicator,
                    "gflops_delta": new_metrics.gflops - baseline_metrics.gflops,
                    "efficiency_delta": new_metrics.efficiency
                    - baseline_metrics.efficiency,
                }

        return comparison

    def generate_report(self) -> str:
        """Generate a comprehensive comparison report."""
        implementations = ["naive", "blocked", "cublas", "tensor_core"]

        report = []
        report.append("# Benchmark Comparison Report\n")
        report.append(f"**Baseline:** {self.baseline_dir}")
        report.append(f"**New Run:**  {self.new_dir}\n")

        all_comparisons = {}

        for impl in implementations:
            comparison = self.compare_implementation(impl)
            all_comparisons[impl] = comparison

            if "error" in comparison:
                report.append(f"\n## [!] {impl.upper()}: {comparison['error']}")
                continue

            report.append(f"\n## {impl.upper()} Implementation")
            report.append(
                f"**Files:** `{comparison['baseline_file']}` -> `{comparison['new_file']}`\n"
            )

            report.append("| Power Mode | Baseline | New Run | Change | Status |")
            report.append("|------------|----------|---------|--------|--------|")

            for power_mode, data in comparison["power_modes"].items():
                baseline = data["baseline"]
                new = data["new"]
                indicator = data["indicator"]
                delta = data["gflops_delta"]

                status = (
                    "[OK] Similar"
                    if indicator == "≈"
                    else ("[+] Improved" if indicator == "↑" else "[!] Regressed")
                )

                report.append(
                    f"| {power_mode:10s} | {baseline.summary} | "
                    f"{new.summary} | {delta:+.1f} GFLOPS ({delta / baseline.gflops * 100:+.1f}%) | {status} |"
                )

        # Add speedup comparison section
        report.append("\n## Speedup Analysis (vs Naive Baseline)")

        if "naive" not in all_comparisons or "error" in all_comparisons["naive"]:
            report.append(
                "[!] Cannot calculate speedups - naive implementation data missing"
            )
        else:
            naive_baseline = all_comparisons["naive"]["power_modes"]["MAXN_SUPER"][
                "baseline"
            ].gflops
            naive_new = all_comparisons["naive"]["power_modes"]["MAXN_SUPER"][
                "new"
            ].gflops

            report.append(
                f"\n**Naive Baseline (MAXN):** {naive_baseline:.2f} GFLOPS (baseline) -> {naive_new:.2f} GFLOPS (new)"
            )
            report.append(
                "\n| Implementation | Baseline Speedup | New Speedup | Change |"
            )
            report.append(
                "|----------------|-----------------|-------------|---------|"
            )

            for impl in ["blocked", "cublas", "tensor_core"]:
                if impl in all_comparisons and "power_modes" in all_comparisons[impl]:
                    maxn_data = all_comparisons[impl]["power_modes"].get("MAXN_SUPER")
                    if maxn_data:
                        baseline_speedup = maxn_data["baseline"].gflops / naive_baseline
                        new_speedup = maxn_data["new"].gflops / naive_new
                        speedup_delta = new_speedup - baseline_speedup

                        indicator = (
                            "↑"
                            if speedup_delta > 0.1
                            else ("↓" if speedup_delta < -0.1 else "≈")
                        )

                        report.append(
                            f"| {impl.upper():14s} | {baseline_speedup:.1f}× | "
                            f"{new_speedup:.1f}× | {speedup_delta:+.1f}× {indicator} |"
                        )

        # Add summary
        report.append("\n## Summary\n")

        total_regressions = sum(
            1
            for impl in all_comparisons.values()
            if "power_modes" in impl
            for pm in impl["power_modes"].values()
            if pm["indicator"] == "↓"
        )

        total_improvements = sum(
            1
            for impl in all_comparisons.values()
            if "power_modes" in impl
            for pm in impl["power_modes"].values()
            if pm["indicator"] == "↑"
        )

        if total_regressions > 0:
            report.append(
                f"[!] **{total_regressions} performance regressions detected**"
            )

        if total_improvements > 0:
            report.append(
                f"[+] **{total_improvements} performance improvements detected**"
            )

        if total_regressions == 0 and total_improvements == 0:
            report.append(
                "[OK] **Results are consistent with baseline** (within 1% tolerance)"
            )

        report.append("\n---\n")
        report.append(
            "*Legend: [OK] Similar (<1% change) | [+] Improved | [!] Regressed*"
        )

        return "\n".join(report)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    baseline_dir = Path(sys.argv[1])
    new_dir = Path(sys.argv[2])

    if not baseline_dir.exists():
        print(f"Error: Baseline directory not found: {baseline_dir}")
        sys.exit(1)

    if not new_dir.exists():
        print(f"Error: New run directory not found: {new_dir}")
        sys.exit(1)

    comparator = BenchmarkComparator(baseline_dir, new_dir)
    report = comparator.generate_report()

    print(report)

    # Also save to file
    output_file = Path("data/reports/benchmark_comparison.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(report)
    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()
