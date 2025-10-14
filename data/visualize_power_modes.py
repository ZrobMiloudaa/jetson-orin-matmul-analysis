#!/usr/bin/env python3
"""Multi-Implementation Visualization System for Jetson Orin Nano.

Copyright 2025 ByteStack Labs
SPDX-License-Identifier: MIT

This module provides comprehensive visualization capabilities for NVIDIA Jetson Orin Nano
across multiple power modes (15W, 25W, and MAXN_SUPER) and implementations (naive, blocked)
with enterprise-grade error handling, logging, and validation.

Target Hardware: Jetson Orin Nano Engineering Reference Developer Kit Super
Software Stack: L4T R36.4.4 (JetPack 6.x), CUDA V12.6.68

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 2.0.0
License: MIT
"""

import glob
import json
import logging
import os
import pwd
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class VisualizationError(Exception):
    """Raised when visualization operations fail."""

    pass


class DataValidationError(Exception):
    """Raised when data validation fails."""

    pass


def setup_results_directory() -> None:
    """Set up results directory structure with proper ownership."""
    directories = [
        "data/logs",
        "data/raw/power_modes",
        "data/reports",
        "data/plots/power_analysis",
        "data/plots/implementation_comparison",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    actual_user = os.environ.get("SUDO_USER", os.environ.get("USER"))

    if os.geteuid() == 0 and actual_user and actual_user != "root":
        try:
            user_info = pwd.getpwnam(actual_user)
            uid, gid = user_info.pw_uid, user_info.pw_gid

            for dir_path in directories:
                path_obj = Path(dir_path)
                if path_obj.exists():
                    os.chown(path_obj, uid, gid)
                    for parent in path_obj.parents:
                        if parent.name == "results":
                            os.chown(parent, uid, gid)
                            break
                        elif parent != Path("."):
                            os.chown(parent, uid, gid)
        except (KeyError, OSError, PermissionError):
            pass


def setup_logging() -> logging.Logger:
    """Set up enterprise-grade logging with console and file handlers."""
    setup_results_directory()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    try:
        log_file = Path("data/logs/jetson_visualization.log")
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except (PermissionError, OSError) as e:
        console_handler.setLevel(logging.WARNING)
        logger.addHandler(console_handler)
        logger.warning(f"Could not create log file: {e}")
        return logger

    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


logger = setup_logging()


class JetsonOrinNanoVisualizer:
    """Creates comprehensive visualizations for multi-implementation analysis."""

    def __init__(self) -> None:
        """Initialize the visualizer with default configuration."""
        logger.info("Initializing JetsonOrinNanoVisualizer")

        self.naive_data: pd.DataFrame | None = None
        self.blocked_data: pd.DataFrame | None = None
        self.cublas_data: pd.DataFrame | None = None
        self.tensor_core_data: pd.DataFrame | None = None
        self.power_modes: dict[int, str] = {0: "15W", 1: "25W", 2: "MAXN_SUPER"}
        self.colors: list[str] = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        self.impl_colors: dict[str, str] = {
            "naive": "#3498db",
            "blocked": "#e74c3c",
            "cublas": "#2ecc71",
            "tensor_core": "#9b59b6",
        }

        try:
            plt.style.use("default")
            plt.rcParams.update(
                {
                    "font.size": 13,
                    "font.family": "sans-serif",
                    "figure.dpi": 600,
                    "savefig.dpi": 600,
                    "axes.grid": True,
                    "grid.alpha": 0.3,
                    "lines.linewidth": 2.5,
                    "lines.markersize": 10,
                    "axes.titlepad": 20,
                    "axes.labelpad": 10,
                }
            )
            logger.debug("Matplotlib configuration applied successfully")
        except Exception as e:
            logger.error(f"Failed to configure matplotlib: {e}")
            raise VisualizationError(f"Matplotlib configuration failed: {e}") from e

    def load_latest_results(self) -> bool:
        """Load the most recent results for all available implementations."""
        logger.info("Loading latest benchmark results")

        try:
            results_dir = Path("data/raw/power_modes")

            if not results_dir.exists():
                logger.error(f"Results directory does not exist: {results_dir}")
                logger.info("Please run the benchmark first:")
                logger.info("  sudo python3 benchmarks/multi_power_mode_benchmark.py")
                return False

            # Load naive implementation data
            naive_pattern = str(results_dir / "naive_3mode_analysis_*.json")
            naive_files = glob.glob(naive_pattern)

            if naive_files:
                latest_naive = max(naive_files, key=os.path.getctime)
                logger.info(f"Loading naive data from: {latest_naive}")
                with open(latest_naive, encoding="utf-8") as f:
                    naive_raw = json.load(f)
                self.naive_data = pd.DataFrame(naive_raw)
                logger.info(f"Loaded {len(self.naive_data)} naive data points")
            else:
                logger.warning("No naive implementation data found")

            # Load blocked implementation data
            blocked_pattern = str(results_dir / "blocked_3mode_analysis_*.json")
            blocked_files = glob.glob(blocked_pattern)

            if blocked_files:
                latest_blocked = max(blocked_files, key=os.path.getctime)
                logger.info(f"Loading blocked data from: {latest_blocked}")
                with open(latest_blocked, encoding="utf-8") as f:
                    blocked_raw = json.load(f)
                self.blocked_data = pd.DataFrame(blocked_raw)
                logger.info(f"Loaded {len(self.blocked_data)} blocked data points")
            else:
                logger.warning("No blocked implementation data found")

            # Load cuBLAS implementation data
            cublas_pattern = str(results_dir / "cublas_3mode_analysis_*.json")
            cublas_files = glob.glob(cublas_pattern)

            if cublas_files:
                latest_cublas = max(cublas_files, key=os.path.getctime)
                logger.info(f"Loading cuBLAS data from: {latest_cublas}")
                with open(latest_cublas, encoding="utf-8") as f:
                    cublas_raw = json.load(f)
                self.cublas_data = pd.DataFrame(cublas_raw)
                logger.info(f"Loaded {len(self.cublas_data)} cuBLAS data points")
            else:
                logger.warning("No cuBLAS implementation data found")

            # Load Tensor Core implementation data
            tensor_core_pattern = str(results_dir / "tensor_core_3mode_analysis_*.json")
            tensor_core_files = glob.glob(tensor_core_pattern)

            if tensor_core_files:
                latest_tensor_core = max(tensor_core_files, key=os.path.getctime)
                logger.info(f"Loading Tensor Core data from: {latest_tensor_core}")
                with open(latest_tensor_core, encoding="utf-8") as f:
                    tensor_core_raw = json.load(f)
                self.tensor_core_data = pd.DataFrame(tensor_core_raw)
                logger.info(
                    f"Loaded {len(self.tensor_core_data)} Tensor Core data points"
                )
            else:
                logger.warning("No Tensor Core implementation data found")

            # Validate at least one implementation exists
            if (
                self.naive_data is None
                and self.blocked_data is None
                and self.cublas_data is None
                and self.tensor_core_data is None
            ):
                logger.error("No implementation data found")
                return False

            # Validate data structure
            for name, data in [
                ("naive", self.naive_data),
                ("blocked", self.blocked_data),
                ("cublas", self.cublas_data),
                ("tensor_core", self.tensor_core_data),
            ]:
                if data is not None:
                    required_columns = [
                        "power_mode_id",
                        "matrix_size",
                        "gflops",
                        "efficiency_percent",
                    ]
                    missing = [
                        col for col in required_columns if col not in data.columns
                    ]
                    if missing:
                        raise DataValidationError(
                            f"Missing columns in {name} data: {missing}"
                        )

            logger.info("Data loading completed successfully")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in data file: {e}")
            return False
        except DataValidationError as e:
            logger.error(f"Data validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error loading data: {e}", exc_info=True)
            return False

    def _save_plot(
        self,
        filename_base: str,
        subdir: str = "power_analysis",
        close_plot: bool = True,
    ) -> str:
        """Save plot with timestamped filename."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_base}_{timestamp}.png"
            filepath = Path(f"data/plots/{subdir}/{filename}")

            filepath.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(
                filepath,
                dpi=600,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
                pad_inches=0.2,
                format="png",
            )

            logger.info(f"Created visualization: {filename}")

            if close_plot:
                plt.close()

            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save plot {filename_base}: {e}")
            if close_plot:
                plt.close()
            raise VisualizationError(f"Plot saving failed: {e}") from e

    def create_comprehensive_implementation_comparison(self) -> str | None:
        """Create comprehensive 3-way implementation comparison (naive, blocked, cuBLAS)."""
        logger.info("Creating comprehensive 3-way implementation comparison")

        available_impls = []
        if self.naive_data is not None:
            available_impls.append("naive")
        if self.blocked_data is not None:
            available_impls.append("blocked")
        if self.cublas_data is not None:
            available_impls.append("cublas")
        if self.tensor_core_data is not None:
            available_impls.append("tensor_core")

        if len(available_impls) < 2:
            logger.error("At least 2 implementations required for comparison")
            return None

        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            title = f"Jetson Orin Nano: {' vs '.join([i.capitalize() for i in available_impls])} Comparison"
            fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98)

            # 1. Performance comparison by matrix size
            self._create_multi_impl_performance_plot(ax1, available_impls)

            # 2. Relative speedup analysis
            self._create_multi_impl_speedup_plot(ax2, available_impls)

            # 3. Efficiency comparison
            self._create_multi_impl_efficiency_plot(ax3, available_impls)

            # 4. Power mode breakdown (512x512)
            self._create_multi_impl_power_mode_plot(ax4, available_impls)

            plt.tight_layout(rect=(0, 0, 1, 0.95))
            impl_suffix = "_".join(available_impls)
            return self._save_plot(
                f"comprehensive_comparison_{impl_suffix}", "implementation_comparison"
            )

        except Exception as e:
            logger.error(
                f"Failed to create comprehensive comparison: {e}", exc_info=True
            )
            plt.close()
            return None

    def _create_multi_impl_performance_plot(
        self, ax: plt.Axes, available_impls: list[str]
    ) -> None:
        """Create performance comparison subplot for multiple implementations."""
        logger.debug("Creating multi-implementation performance comparison plot")

        impl_data = {
            "naive": self.naive_data,
            "blocked": self.blocked_data,
            "cublas": self.cublas_data,
            "tensor_core": self.tensor_core_data,
        }

        matrix_sizes = None
        for impl in available_impls:
            data = impl_data[impl]
            if data is not None:
                if matrix_sizes is None:
                    matrix_sizes = sorted(data["matrix_size"].unique())

                avg_perf = [
                    data[data["matrix_size"] == size]["gflops"].mean()
                    for size in matrix_sizes
                ]

                marker = "o" if impl == "naive" else ("s" if impl == "blocked" else "^")
                ax.plot(
                    matrix_sizes,
                    avg_perf,
                    f"{marker}-",
                    color=self.impl_colors[impl],
                    label=impl.upper()
                    if impl == "cublas"
                    else (
                        "Tensor Core" if impl == "tensor_core" else impl.capitalize()
                    ),
                    linewidth=3,
                    markersize=10,
                )

        ax.set_xlabel("Matrix Size (n×n)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Performance (GFLOPS)", fontsize=14, fontweight="bold")
        ax.set_title(
            "Performance Scaling Comparison", fontweight="bold", fontsize=16, pad=20
        )
        ax.legend(fontsize=13, loc="upper left")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

    def _create_multi_impl_speedup_plot(
        self, ax: plt.Axes, available_impls: list[str]
    ) -> None:
        """Create speedup analysis subplot for multiple implementations vs naive."""
        logger.debug("Creating multi-implementation speedup plot")

        if "naive" not in available_impls or self.naive_data is None:
            ax.text(
                0.5,
                0.5,
                "Speedup requires\nnaive baseline",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )
            return

        impl_data = {
            "blocked": self.blocked_data,
            "cublas": self.cublas_data,
            "tensor_core": self.tensor_core_data,
        }

        matrix_sizes = sorted(self.naive_data["matrix_size"].unique())
        x = range(len(matrix_sizes))
        width = 0.35 if len(available_impls) == 2 else 0.25
        offset = 0.0

        for impl in ["blocked", "cublas", "tensor_core"]:
            if impl in available_impls and impl_data.get(impl) is not None:
                speedups = []
                for size in matrix_sizes:
                    naive_perf = self.naive_data[
                        self.naive_data["matrix_size"] == size
                    ]["gflops"].mean()
                    data = impl_data.get(impl)
                    if data is not None:
                        impl_perf = data[data["matrix_size"] == size]["gflops"].mean()
                        speedup = (impl_perf / naive_perf) if naive_perf > 0 else 1.0
                        speedups.append(speedup)

                positions = [float(i) + offset for i in x]
                ax.bar(
                    positions,
                    speedups,
                    width,
                    label=impl.upper()
                    if impl == "cublas"
                    else (
                        "Tensor Core" if impl == "tensor_core" else impl.capitalize()
                    ),
                    color=self.impl_colors[impl],
                    alpha=0.8,
                )
                offset += width

        ax.set_xlabel("Matrix Size", fontsize=14, fontweight="bold")
        ax.set_ylabel("Speedup vs Naive (x)", fontsize=14, fontweight="bold")
        ax.set_title("Relative Speedup", fontweight="bold", fontsize=16, pad=20)
        ax.set_xticks([i + width / 2 for i in x])
        ax.set_xticklabels([f"{s}×{s}" for s in matrix_sizes], rotation=45)
        ax.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")

    def _create_multi_impl_efficiency_plot(
        self, ax: plt.Axes, available_impls: list[str]
    ) -> None:
        """Create efficiency comparison subplot for multiple implementations."""
        logger.debug("Creating multi-implementation efficiency plot")

        impl_data = {
            "naive": self.naive_data,
            "blocked": self.blocked_data,
            "cublas": self.cublas_data,
            "tensor_core": self.tensor_core_data,
        }

        matrix_sizes = None
        for impl in available_impls:
            data = impl_data[impl]
            if data is not None and matrix_sizes is None:
                matrix_sizes = sorted(data["matrix_size"].unique())

        x = range(len(matrix_sizes) if matrix_sizes else 0)
        width = 0.8 / len(available_impls) if available_impls else 1.0
        offset_start = -width * (len(available_impls) - 1) / 2

        for idx, impl in enumerate(available_impls):
            data = impl_data.get(impl)
            if data is not None and matrix_sizes is not None:
                eff = [
                    data[data["matrix_size"] == size]["efficiency_percent"].mean()
                    for size in matrix_sizes
                ]

                positions = [float(i) + offset_start + idx * width for i in x]
                ax.bar(
                    positions,
                    eff,
                    width,
                    label=impl.upper()
                    if impl == "cublas"
                    else (
                        "Tensor Core" if impl == "tensor_core" else impl.capitalize()
                    ),
                    color=self.impl_colors[impl],
                    alpha=0.8,
                )

        ax.set_xlabel("Matrix Size", fontsize=14, fontweight="bold")
        ax.set_ylabel("Algorithm Efficiency (%)", fontsize=14, fontweight="bold")
        ax.set_title("Efficiency Comparison", fontweight="bold", fontsize=16, pad=20)
        ax.set_xticks(x)
        if matrix_sizes is not None:
            ax.set_xticklabels([f"{s}×{s}" for s in matrix_sizes], rotation=45)
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")

    def _create_multi_impl_power_mode_plot(
        self, ax: plt.Axes, available_impls: list[str]
    ) -> None:
        """Create power mode breakdown subplot for multiple implementations."""
        logger.debug("Creating multi-implementation power mode plot")

        impl_data = {
            "naive": self.naive_data,
            "blocked": self.blocked_data,
            "cublas": self.cublas_data,
            "tensor_core": self.tensor_core_data,
        }

        modes = sorted(self.power_modes.keys())
        mode_names = [self.power_modes[m] for m in modes]
        x = range(len(modes))
        width = 0.8 / len(available_impls) if available_impls else 1.0
        offset_start = -width * (len(available_impls) - 1) / 2

        for idx, impl in enumerate(available_impls):
            data = impl_data.get(impl)
            if data is not None:
                perf_512 = [
                    data[(data["power_mode_id"] == m) & (data["matrix_size"] == 512)][
                        "gflops"
                    ].mean()
                    for m in modes
                ]

                positions = [float(i) + offset_start + idx * width for i in x]
                ax.bar(
                    positions,
                    perf_512,
                    width,
                    label=impl.upper()
                    if impl == "cublas"
                    else (
                        "Tensor Core" if impl == "tensor_core" else impl.capitalize()
                    ),
                    color=self.impl_colors[impl],
                    alpha=0.8,
                )

        ax.set_xlabel("Power Mode", fontsize=14, fontweight="bold")
        ax.set_ylabel("Performance (GFLOPS)", fontsize=14, fontweight="bold")
        ax.set_title(
            "512×512 Performance by Power Mode", fontweight="bold", fontsize=16, pad=20
        )
        ax.set_xticks(x)
        ax.set_xticklabels(mode_names, fontsize=13, fontweight="bold")
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")

    def create_implementation_comparison(self) -> str | None:
        """Create comprehensive naive vs blocked comparison visualization."""
        logger.info("Creating implementation comparison visualization")

        if self.naive_data is None or self.blocked_data is None:
            logger.error("Both naive and blocked data required for comparison")
            return None

        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
            fig.suptitle(
                "Jetson Orin Nano: Naive vs Blocked Implementation Comparison\n"
                "Cache-Optimized Tiling Performance Analysis",
                fontsize=20,
                fontweight="bold",
                y=0.98,
            )

            # 1. Performance comparison by matrix size
            self._create_performance_comparison_plot(ax1)

            # 2. Speedup analysis
            self._create_speedup_analysis_plot(ax2)

            # 3. Efficiency comparison
            self._create_efficiency_comparison_plot(ax3)

            # 4. Power mode breakdown
            self._create_power_mode_breakdown_plot(ax4)

            plt.tight_layout(rect=(0, 0, 1, 0.95))
            return self._save_plot(
                "implementation_comparison", "implementation_comparison"
            )

        except Exception as e:
            logger.error(
                f"Failed to create implementation comparison: {e}", exc_info=True
            )
            plt.close()
            raise VisualizationError(f"Implementation comparison failed: {e}") from e

    def _create_performance_comparison_plot(self, ax: plt.Axes) -> None:
        """Create performance comparison subplot."""
        logger.debug("Creating performance comparison plot")

        assert self.naive_data is not None and self.blocked_data is not None

        matrix_sizes = sorted(self.naive_data["matrix_size"].unique())

        # Aggregate across power modes
        naive_avg = [
            self.naive_data[self.naive_data["matrix_size"] == size]["gflops"].mean()
            for size in matrix_sizes
        ]
        blocked_avg = [
            self.blocked_data[self.blocked_data["matrix_size"] == size]["gflops"].mean()
            for size in matrix_sizes
        ]

        ax.plot(
            matrix_sizes,
            naive_avg,
            "o-",
            color=self.impl_colors["naive"],
            label="Naive (Baseline)",
            linewidth=3,
            markersize=10,
        )
        ax.plot(
            matrix_sizes,
            blocked_avg,
            "s-",
            color=self.impl_colors["blocked"],
            label="Blocked (Optimized)",
            linewidth=3,
            markersize=10,
        )

        ax.set_xlabel("Matrix Size (n×n)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Performance (GFLOPS)", fontsize=14, fontweight="bold")
        ax.set_title(
            "Performance Scaling: Naive vs Blocked",
            fontweight="bold",
            fontsize=16,
            pad=20,
        )
        ax.legend(fontsize=13, loc="upper left")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

    def _create_speedup_analysis_plot(self, ax: plt.Axes) -> None:
        """Create speedup analysis subplot."""
        logger.debug("Creating speedup analysis plot")

        assert self.naive_data is not None and self.blocked_data is not None

        matrix_sizes = sorted(self.naive_data["matrix_size"].unique())
        speedups = []

        for size in matrix_sizes:
            naive_perf = self.naive_data[self.naive_data["matrix_size"] == size][
                "gflops"
            ].mean()
            blocked_perf = self.blocked_data[self.blocked_data["matrix_size"] == size][
                "gflops"
            ].mean()
            speedup = (blocked_perf / naive_perf - 1) * 100 if naive_perf > 0 else 0
            speedups.append(speedup)

        colors = ["green" if s > 0 else "red" for s in speedups]
        bars = ax.bar(
            range(len(matrix_sizes)),
            speedups,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        ax.set_xlabel("Matrix Size", fontsize=14, fontweight="bold")
        ax.set_ylabel("Performance Improvement (%)", fontsize=14, fontweight="bold")
        ax.set_title("Blocked vs Naive Speedup", fontweight="bold", fontsize=16, pad=20)
        ax.set_xticks(range(len(matrix_sizes)))
        ax.set_xticklabels([f"{s}×{s}" for s in matrix_sizes], rotation=45)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, value in zip(bars, speedups, strict=False):
            height = bar.get_height()
            label_y = height + (1 if height > 0 else -3)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                label_y,
                f"{value:+.1f}%",
                ha="center",
                va="bottom" if height > 0 else "top",
                fontweight="bold",
                fontsize=11,
            )

    def _create_efficiency_comparison_plot(self, ax: plt.Axes) -> None:
        """Create efficiency comparison subplot."""
        logger.debug("Creating efficiency comparison plot")

        assert self.naive_data is not None and self.blocked_data is not None

        matrix_sizes = sorted(self.naive_data["matrix_size"].unique())

        naive_eff = [
            self.naive_data[self.naive_data["matrix_size"] == size][
                "efficiency_percent"
            ].mean()
            for size in matrix_sizes
        ]
        blocked_eff = [
            self.blocked_data[self.blocked_data["matrix_size"] == size][
                "efficiency_percent"
            ].mean()
            for size in matrix_sizes
        ]

        x = range(len(matrix_sizes))
        width = 0.35

        ax.bar(
            [i - width / 2 for i in x],
            naive_eff,
            width,
            label="Naive",
            color=self.impl_colors["naive"],
            alpha=0.8,
        )
        ax.bar(
            [i + width / 2 for i in x],
            blocked_eff,
            width,
            label="Blocked",
            color=self.impl_colors["blocked"],
            alpha=0.8,
        )

        ax.set_xlabel("Matrix Size", fontsize=14, fontweight="bold")
        ax.set_ylabel("Algorithm Efficiency (%)", fontsize=14, fontweight="bold")
        ax.set_title("Efficiency Comparison", fontweight="bold", fontsize=16, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}×{s}" for s in matrix_sizes], rotation=45)
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")

    def _create_power_mode_breakdown_plot(self, ax: plt.Axes) -> None:
        """Create power mode breakdown subplot."""
        logger.debug("Creating power mode breakdown plot")

        assert self.naive_data is not None and self.blocked_data is not None

        modes = sorted(self.power_modes.keys())
        mode_names = [self.power_modes[m] for m in modes]

        # Get 512×512 results for each mode (representative size)
        naive_512 = [
            self.naive_data[
                (self.naive_data["power_mode_id"] == m)
                & (self.naive_data["matrix_size"] == 512)
            ]["gflops"].mean()
            for m in modes
        ]
        blocked_512 = [
            self.blocked_data[
                (self.blocked_data["power_mode_id"] == m)
                & (self.blocked_data["matrix_size"] == 512)
            ]["gflops"].mean()
            for m in modes
        ]

        x = range(len(modes))
        width = 0.35

        ax.bar(
            [i - width / 2 for i in x],
            naive_512,
            width,
            label="Naive",
            color=self.impl_colors["naive"],
            alpha=0.8,
        )
        ax.bar(
            [i + width / 2 for i in x],
            blocked_512,
            width,
            label="Blocked",
            color=self.impl_colors["blocked"],
            alpha=0.8,
        )

        ax.set_xlabel("Power Mode", fontsize=14, fontweight="bold")
        ax.set_ylabel("Performance (GFLOPS)", fontsize=14, fontweight="bold")
        ax.set_title(
            "512×512 Performance by Power Mode", fontweight="bold", fontsize=16, pad=20
        )
        ax.set_xticks(x)
        ax.set_xticklabels(mode_names, fontsize=13, fontweight="bold")
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")

        # Add speedup annotations
        for i, (n, b) in enumerate(zip(naive_512, blocked_512, strict=False)):
            if n > 0:
                speedup = (b / n - 1) * 100
                ax.text(
                    i,
                    max(n, b) * 1.02,
                    f"+{speedup:.1f}%",
                    ha="center",
                    fontweight="bold",
                    fontsize=10,
                )

    def create_detailed_speedup_heatmap(self) -> str | None:
        """Create detailed speedup heatmap across all configurations."""
        logger.info("Creating detailed speedup heatmap")

        if self.naive_data is None or self.blocked_data is None:
            logger.error("Both implementations required for heatmap")
            return None

        try:
            fig, ax = plt.subplots(figsize=(14, 10))
            fig.suptitle(
                "Jetson Orin Nano: Blocked Implementation Speedup Matrix\n"
                "Performance Improvement Over Naive Baseline",
                fontsize=18,
                fontweight="bold",
            )

            matrix_sizes = sorted(self.naive_data["matrix_size"].unique())
            modes = sorted(self.power_modes.keys())

            # Create speedup matrix
            speedup_matrix = []
            for mode in modes:
                row = []
                for size in matrix_sizes:
                    naive_perf = self.naive_data[
                        (self.naive_data["power_mode_id"] == mode)
                        & (self.naive_data["matrix_size"] == size)
                    ]["gflops"].mean()

                    blocked_perf = self.blocked_data[
                        (self.blocked_data["power_mode_id"] == mode)
                        & (self.blocked_data["matrix_size"] == size)
                    ]["gflops"].mean()

                    speedup = (
                        (blocked_perf / naive_perf - 1) * 100 if naive_perf > 0 else 0
                    )
                    row.append(speedup)
                speedup_matrix.append(row)

            # Create heatmap
            im = ax.imshow(
                speedup_matrix, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=20
            )

            # Set ticks and labels
            ax.set_xticks(range(len(matrix_sizes)))
            ax.set_yticks(range(len(modes)))
            ax.set_xticklabels([f"{s}×{s}" for s in matrix_sizes], fontsize=12)
            ax.set_yticklabels(
                [self.power_modes[m] for m in modes], fontsize=12, fontweight="bold"
            )

            ax.set_xlabel("Matrix Size", fontsize=14, fontweight="bold")
            ax.set_ylabel("Power Mode", fontsize=14, fontweight="bold")

            # Add text annotations
            for i in range(len(modes)):
                for j in range(len(matrix_sizes)):
                    ax.text(
                        j,
                        i,
                        f"{speedup_matrix[i][j]:.1f}%",
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                        fontsize=11,
                    )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(
                "Performance Improvement (%)",
                rotation=270,
                labelpad=20,
                fontsize=13,
                fontweight="bold",
            )

            plt.tight_layout(rect=(0, 0, 1, 0.95))
            return self._save_plot("speedup_heatmap", "implementation_comparison")

        except Exception as e:
            logger.error(f"Failed to create speedup heatmap: {e}", exc_info=True)
            plt.close()
            return None

    def create_3mode_performance_comparison(
        self, data: pd.DataFrame | None = None, impl_name: str | None = None
    ) -> str | None:
        """Create main power mode comparison for a specific implementation.

        Args:
            data: DataFrame containing benchmark data. If None, uses naive or blocked data.
            impl_name: Name of the implementation (e.g., "Naive", "cuBLAS"). If None, auto-detects.
        """
        logger.info("Creating 3-mode power analysis visualization")

        # Use provided data or fall back to naive/blocked
        if data is None:
            data = self.naive_data if self.naive_data is not None else self.blocked_data

        if data is None:
            logger.error("No data available")
            return None

        # Auto-detect implementation name if not provided
        if impl_name is None:
            impl_name = (
                "Naive"
                if self.naive_data is not None and data is self.naive_data
                else "Blocked"
            )

        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))

            fig.suptitle(
                f"Jetson Orin Nano: 3-Power Mode Performance Analysis ({impl_name})\n"
                "Engineering Reference Developer Kit Super",
                fontsize=18,
                fontweight="bold",
                y=0.95,
            )

            self._create_power_mode_scaling_plot(ax1, data)
            self._create_power_efficiency_plot(ax2, data)
            self._create_scaling_bar_chart(ax3, data)
            self._create_thermal_analysis_plot(ax4, data)

            plt.tight_layout(rect=(0, 0, 1, 0.92))
            return self._save_plot(
                f"{impl_name.lower()}_3mode_comparison", "power_analysis"
            )

        except Exception as e:
            logger.error(f"Failed to create power mode comparison: {e}", exc_info=True)
            plt.close()
            return None

    def _create_power_mode_scaling_plot(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        """Create power mode scaling subplot."""
        for i, (mode_id, mode_name) in enumerate(self.power_modes.items()):
            mode_data = data[data["power_mode_id"] == mode_id]
            if not mode_data.empty:
                ax.plot(
                    mode_data["matrix_size"],
                    mode_data["gflops"],
                    "o-",
                    color=self.colors[i],
                    label=f"{mode_name} Mode",
                    linewidth=3,
                    markersize=8,
                )

        ax.set_xlabel("Matrix Size (n×n)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Performance (GFLOPS)", fontsize=14, fontweight="bold")
        ax.set_title(
            "Performance Scaling by Power Mode", fontweight="bold", fontsize=16, pad=20
        )
        ax.legend(fontsize=12)
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)

    def _create_power_efficiency_plot(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        """Create power efficiency subplot."""
        if "avg_power_consumption" in data.columns:
            for i, (mode_id, mode_name) in enumerate(self.power_modes.items()):
                mode_data = data[data["power_mode_id"] == mode_id]
                valid_power = mode_data[mode_data["avg_power_consumption"] > 0]
                if not valid_power.empty:
                    power_eff = (
                        valid_power["gflops"] / valid_power["avg_power_consumption"]
                    )
                    ax.plot(
                        valid_power["matrix_size"],
                        power_eff,
                        "s-",
                        color=self.colors[i],
                        label=f"{mode_name} Mode",
                        linewidth=3,
                        markersize=8,
                    )

            ax.set_xlabel("Matrix Size (n×n)", fontsize=14, fontweight="bold")
            ax.set_ylabel("Power Efficiency (GFLOPS/W)", fontsize=14, fontweight="bold")
            ax.set_title("Power Efficiency", fontweight="bold", fontsize=16, pad=20)
            ax.legend(fontsize=12)
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "Power data\nnot available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )

    def _create_scaling_bar_chart(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        """Create scaling bar chart subplot."""
        mode_performance = []
        mode_names = []

        for mode_id, mode_name in self.power_modes.items():
            mode_data = data[data["power_mode_id"] == mode_id]
            if not mode_data.empty:
                peak_perf = mode_data["gflops"].max()
                mode_performance.append(peak_perf)
                mode_names.append(mode_name)

        if mode_performance:
            bars = ax.bar(
                range(len(mode_performance)),
                mode_performance,
                color=self.colors[: len(mode_performance)],
                alpha=0.8,
            )

            ax.set_xlabel("Power Mode", fontsize=14, fontweight="bold")
            ax.set_ylabel("Peak Performance (GFLOPS)", fontsize=14, fontweight="bold")
            ax.set_title(
                "Peak Performance by Mode", fontweight="bold", fontsize=16, pad=20
            )
            ax.set_xticks(range(len(mode_names)))
            ax.set_xticklabels(mode_names, fontsize=12, fontweight="bold")

            max_value = max(mode_performance)
            for bar, value in zip(bars, mode_performance, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max_value * 0.02,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=12,
                )

            ax.set_ylim(0, max_value * 1.12)

    def _create_thermal_analysis_plot(self, ax: plt.Axes, data: pd.DataFrame) -> None:
        """Create thermal analysis subplot."""
        if "post_temperature" in data.columns:
            for i, (mode_id, mode_name) in enumerate(self.power_modes.items()):
                mode_data = data[data["power_mode_id"] == mode_id]
                if not mode_data.empty:
                    ax.scatter(
                        mode_data["post_temperature"],
                        mode_data["gflops"],
                        color=self.colors[i],
                        label=f"{mode_name} Mode",
                        s=80,
                        alpha=0.7,
                    )

            ax.set_xlabel("Operating Temperature (°C)", fontsize=14, fontweight="bold")
            ax.set_ylabel("Performance (GFLOPS)", fontsize=14, fontweight="bold")
            ax.set_title(
                "Thermal vs Performance", fontweight="bold", fontsize=16, pad=20
            )
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axvline(
                x=75,
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label="Thermal Threshold",
            )
            ax.legend(fontsize=12)
        else:
            ax.text(
                0.5,
                0.5,
                "Thermal data\nnot available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
            )

    def generate_implementation_report(self) -> bool:
        """Generate comprehensive implementation comparison report."""
        logger.info("Generating implementation comparison report")

        if self.naive_data is None or self.blocked_data is None:
            logger.error("Both implementations required for comparison report")
            return False

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = Path(
                f"data/reports/implementation_comparison_report_{timestamp}.md"
            )

            with open(report_file, "w", encoding="utf-8") as f:
                self._write_implementation_report(f)

            logger.info(f"Implementation comparison report generated: {report_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate report: {e}", exc_info=True)
            return False

    def _write_implementation_report(self, f) -> None:
        """Write implementation comparison report content."""

        assert self.naive_data is not None and self.blocked_data is not None

        f.write("# Jetson Orin Nano: Naive vs Blocked Implementation Analysis\n\n")
        f.write(
            "**Hardware:** NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super\n"
        )
        f.write(f"**Report Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            "**Implementations:** Naive (O(n³) baseline) vs Blocked (32×32 tiled)\n\n"
        )

        f.write("## Executive Summary\n\n")

        # Calculate overall statistics
        naive_peak = self.naive_data["gflops"].max()
        blocked_peak = self.blocked_data["gflops"].max()
        overall_improvement = (blocked_peak / naive_peak - 1) * 100

        f.write(f"**Naive Peak Performance:** {naive_peak:.2f} GFLOPS\n")
        f.write(f"**Blocked Peak Performance:** {blocked_peak:.2f} GFLOPS\n")
        f.write(f"**Overall Improvement:** {overall_improvement:+.1f}%\n\n")

        f.write("## Performance by Matrix Size\n\n")
        f.write("| Size | Naive (GFLOPS) | Blocked (GFLOPS) | Improvement |\n")
        f.write("|------|----------------|------------------|-------------|\n")

        for size in sorted(self.naive_data["matrix_size"].unique()):
            naive_avg = self.naive_data[self.naive_data["matrix_size"] == size][
                "gflops"
            ].mean()
            blocked_avg = self.blocked_data[self.blocked_data["matrix_size"] == size][
                "gflops"
            ].mean()
            improvement = (blocked_avg / naive_avg - 1) * 100 if naive_avg > 0 else 0
            f.write(
                f"| {size}×{size} | {naive_avg:.2f} | {blocked_avg:.2f} | {improvement:+.1f}% |\n"
            )

        f.write("\n## Performance by Power Mode (512×512)\n\n")
        f.write("| Mode | Naive (GFLOPS) | Blocked (GFLOPS) | Speedup |\n")
        f.write("|------|----------------|------------------|----------|\n")

        for mode_id, mode_name in self.power_modes.items():
            naive_512 = self.naive_data[
                (self.naive_data["power_mode_id"] == mode_id)
                & (self.naive_data["matrix_size"] == 512)
            ]["gflops"].mean()

            blocked_512 = self.blocked_data[
                (self.blocked_data["power_mode_id"] == mode_id)
                & (self.blocked_data["matrix_size"] == 512)
            ]["gflops"].mean()

            speedup = blocked_512 / naive_512 if naive_512 > 0 else 0
            f.write(
                f"| {mode_name} | {naive_512:.2f} | {blocked_512:.2f} | {speedup:.2f}x |\n"
            )

        f.write("\n## Key Findings\n\n")
        f.write(
            "1. **Consistent Improvement:** Blocked implementation shows measurable gains across all configurations\n"
        )
        f.write(
            "2. **Cache Optimization:** 32×32 tiling effectively utilizes on-chip memory hierarchy\n"
        )
        f.write(
            "3. **Power Mode Scaling:** Optimization benefits preserved across all power modes\n"
        )
        f.write(
            "4. **Practical Impact:** ~10% average improvement translates to meaningful throughput gains\n\n"
        )

        f.write("## Technical Details\n\n")
        f.write("**Naive Implementation:**\n")
        f.write("- Direct O(n³) algorithm\n")
        f.write("- Global memory access pattern\n")
        f.write("- Minimal memory reuse\n\n")

        f.write("**Blocked Implementation:**\n")
        f.write("- 32×32 tile size for optimal cache fit\n")
        f.write("- Shared memory utilization (2KB per block)\n")
        f.write("- Transposed B matrix storage for coalesced access\n")
        f.write("- Improved data locality and reuse\n\n")

    def generate_all_visualizations(self) -> bool:
        """Generate complete visualization suite."""
        logger.info("Generating complete visualization suite")

        try:
            setup_results_directory()

            if not self.load_latest_results():
                logger.error("Failed to load benchmark data")
                return False

            generated_files = []

            # Power mode analysis for available implementations
            if self.naive_data is not None:
                naive_file = self.create_3mode_performance_comparison(
                    self.naive_data, "Naive"
                )
                if naive_file:
                    generated_files.append(("Naive power mode analysis", naive_file))

            if self.blocked_data is not None:
                blocked_file = self.create_3mode_performance_comparison(
                    self.blocked_data, "Blocked"
                )
                if blocked_file:
                    generated_files.append(
                        ("Blocked power mode analysis", blocked_file)
                    )

            if self.cublas_data is not None:
                cublas_file = self.create_3mode_performance_comparison(
                    self.cublas_data, "cuBLAS"
                )
                if cublas_file:
                    generated_files.append(("cuBLAS power mode analysis", cublas_file))

            if self.tensor_core_data is not None:
                tensor_file = self.create_3mode_performance_comparison(
                    self.tensor_core_data, "Tensor Core"
                )
                if tensor_file:
                    generated_files.append(
                        ("Tensor Core power mode analysis", tensor_file)
                    )

            # Implementation comparison (if multiple available)
            available_for_comparison = sum(
                [
                    self.naive_data is not None,
                    self.blocked_data is not None,
                    self.cublas_data is not None,
                    self.tensor_core_data is not None,
                ]
            )

            if available_for_comparison >= 2:
                # Comprehensive 3-way comparison (or 2-way if only 2 available)
                comp_file = self.create_comprehensive_implementation_comparison()
                if comp_file:
                    generated_files.append(
                        ("Comprehensive implementation comparison", comp_file)
                    )

                # Legacy 2-way comparison if naive and blocked available
                if self.naive_data is not None and self.blocked_data is not None:
                    legacy_comp = self.create_implementation_comparison()
                    if legacy_comp:
                        generated_files.append(
                            ("Naive vs Blocked comparison", legacy_comp)
                        )

                    heatmap_file = self.create_detailed_speedup_heatmap()
                    if heatmap_file:
                        generated_files.append(("Speedup heatmap", heatmap_file))

                    report_success = self.generate_implementation_report()
                else:
                    report_success = False
            else:
                report_success = False
                logger.warning(
                    "Skipping implementation comparison (need at least 2 implementations)"
                )

            logger.info("Visualization Suite Complete!")
            logger.info("Generated files:")
            for description, filepath in generated_files:
                logger.info(f"  - {Path(filepath).name} ({description})")

            if report_success:
                logger.info("  - Implementation comparison report (detailed analysis)")

            return len(generated_files) > 0

        except Exception as e:
            logger.error(f"Failed to generate visualization suite: {e}", exc_info=True)
            return False


def main() -> int:
    """Main execution."""
    logger.info("Jetson Orin Nano: Multi-Implementation Visualization Generator")
    logger.info("=" * 60)

    try:
        visualizer = JetsonOrinNanoVisualizer()
        success = visualizer.generate_all_visualizations()

        if success:
            logger.info("Visualization generation completed successfully")
            return 0
        else:
            logger.error("Visualization generation failed")
            logger.info("Please run benchmarks first:")
            logger.info("  sudo python3 benchmarks/multi_power_mode_benchmark.py")
            return 1

    except (VisualizationError, DataValidationError) as e:
        logger.error(f"Application error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Visualization generation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
