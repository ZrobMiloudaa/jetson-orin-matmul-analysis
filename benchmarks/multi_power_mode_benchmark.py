#!/usr/bin/env python3
"""Multi-Power Mode Benchmarking System for Jetson Orin Nano.

Copyright 2025 ByteStack Labs
SPDX-License-Identifier: MIT

This module provides comprehensive benchmarking capabilities for NVIDIA Jetson Orin Nano
across multiple power modes (15W, 25W, MAXN_SUPER) with support for naive and blocked
implementations, with enterprise-grade error handling, logging, and validation.

Target Hardware: Jetson Orin Nano Engineering Reference Developer Kit Super
Software Stack: L4T R36.4.4 (JetPack 6.x), CUDA V12.6.68

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 2.0.0
License: MIT
"""

import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def setup_results_directory():
    """Set up results directory structure with proper ownership."""
    import pwd

    directories = [
        "data/logs",
        "data/raw/power_modes",
        "data/reports",
        "data/plots",
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


def ensure_log_file_writable(log_file: Path) -> bool:
    """
    Ensure log file is writable by removing root-owned files.

    Returns True if file is writable, False if unrecoverable permission issue.
    """
    if not log_file.exists():
        return True  # File doesn't exist, will be created fresh

    try:
        # Check if we can write to the file
        with open(log_file, "a"):
            pass
        return True
    except PermissionError:
        # File exists but is not writable (likely root-owned)
        try:
            # Try to remove the root-owned file
            log_file.unlink()
            print(f"Removed root-owned log file: {log_file}")
            return True
        except PermissionError:
            # Can't remove it - it's root-owned and we lack permissions
            print(f"ERROR: Cannot write to or remove log file: {log_file}")
            print("This file was created by a previous run with sudo.")
            print(
                f"Fix: Run 'sudo chown $USER:$USER {log_file}' or 'sudo rm {log_file}'"
            )
            return False


def create_log_file_handler(log_file: Path) -> logging.FileHandler | None:
    """
    Create log file handler with proper permission handling.

    Returns None if log file cannot be created (will fall back to console only).
    """
    if not ensure_log_file_writable(log_file):
        return None

    try:
        return logging.FileHandler(log_file, mode="w")
    except Exception as e:
        print(f"ERROR: Failed to create log file handler: {e}")
        return None


setup_results_directory()

log_dir = Path("data/logs")
log_dir.mkdir(parents=True, exist_ok=True)

benchmark_log = log_dir / "jetson_benchmark.log"

# Create log file handler with graceful permission error handling
file_handler = create_log_file_handler(benchmark_log)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if file_handler:
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
else:
    # Console-only logging if file handler couldn't be created
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)

logger.propagate = False


def print_info(message: str) -> None:
    """Print clean information message to console."""
    print(message)


def print_success(message: str) -> None:
    """Print success message to console."""
    print(f"[*] {message}")


def print_warning(message: str) -> None:
    """Print warning message to console."""
    print(f"Warning: {message}")


def print_error(message: str) -> None:
    """Print error message to console."""
    print(f"Error: {message}", file=sys.stderr)


def print_status(message: str) -> None:
    """Print status message to console."""
    print(f"-> {message}")


def print_progress(current: int, total: int, message: str = "") -> None:
    """Print progress message to console."""
    progress_msg = f"Progress: {current}/{total}"
    if message:
        progress_msg += f" - {message}"
    print(progress_msg)


class JetsonPowerModeError(Exception):
    """Custom exception for power mode management errors."""

    pass


class BenchmarkValidationError(Exception):
    """Custom exception for benchmark validation failures."""

    pass


class JetsonPowerModeManager:
    """Manages Jetson power modes with enterprise-grade error handling."""

    nvpmodel_path: str
    jetson_clocks_path: str
    sudo_path: str
    timeout_path: str
    tegrastats_path: str | None

    def __init__(self) -> None:
        """Initialize the power mode manager with proper error handling."""
        logger.info("Initializing Jetson Power Mode Manager")
        self._initialize_system_paths()

        self.power_modes = {
            0: "15W",
            1: "25W",
            2: "MAXN_SUPER",
        }

        self.expected_performance_ranges = {
            0: {"min_gflops": 20, "max_gflops": 120, "typical_gflops": 50},
            1: {"min_gflops": 40, "max_gflops": 180, "typical_gflops": 80},
            2: {"min_gflops": 60, "max_gflops": 220, "typical_gflops": 110},
        }

        try:
            self.current_mode = self.get_current_mode()
            self.baseline_temp = self.get_temperature()
            logger.info(
                "System initialized: Mode %d (%s), Temperature: %.1f°C",
                self.current_mode,
                self.power_modes[self.current_mode],
                self.baseline_temp,
            )
        except Exception as e:
            logger.error("Failed to initialize system state: %s", e)
            raise JetsonPowerModeError(f"System initialization failed: {e}") from e

        print_success("Jetson Orin Nano Engineering Reference Developer Kit Super")
        print_info("Target power modes: 15W, 25W, MAXN SUPER")

    def _initialize_system_paths(self) -> None:
        """Initialize and validate system executable paths."""
        required_paths = {
            "nvpmodel_path": ("nvpmodel", "/usr/bin/nvpmodel"),
            "jetson_clocks_path": ("jetson_clocks", "/usr/sbin/jetson_clocks"),
            "sudo_path": ("sudo", "/usr/bin/sudo"),
            "timeout_path": ("timeout", "/usr/bin/timeout"),
        }

        optional_paths = {
            "tegrastats_path": ("tegrastats", "/usr/bin/tegrastats"),
        }

        for attr_name, (cmd, fallback) in required_paths.items():
            path = shutil.which(cmd) or fallback
            if not os.path.exists(path):
                raise JetsonPowerModeError(
                    f"Required utility {cmd} not found at {path}"
                )
            setattr(self, attr_name, path)
            logger.debug("Found %s at %s", cmd, path)

        for attr_name, (cmd, fallback) in optional_paths.items():
            path = shutil.which(cmd) or fallback
            if os.path.exists(path):
                setattr(self, attr_name, path)
                logger.debug("Found optional %s at %s", cmd, path)
            else:
                setattr(self, attr_name, None)
                logger.warning("Optional utility %s not found", cmd)

    def get_current_mode(self) -> int:
        """Get the currently active power mode."""
        try:
            result = subprocess.run(
                [self.nvpmodel_path, "-q"],
                capture_output=True,
                text=True,
                shell=False,
                timeout=10,
            )

            if result.returncode != 0:
                logger.warning(
                    "nvpmodel query returned %d: %s", result.returncode, result.stderr
                )
                return 0

            for line in result.stdout.split("\n"):
                if "NV Power Mode:" in line:
                    line_lower = line.lower()
                    if "15w" in line_lower:
                        return 0
                    elif "25w" in line_lower:
                        return 1
                    elif "maxn" in line_lower or "max" in line_lower:
                        return 2

            logger.warning("Could not parse power mode from nvpmodel output")
            return 0

        except subprocess.TimeoutExpired:
            logger.error("Timeout while querying power mode")
            raise JetsonPowerModeError("Power mode query timeout") from None
        except Exception as e:
            logger.error("Power mode detection failed: %s", e)
            raise JetsonPowerModeError(f"Failed to detect power mode: {e}") from e

    def set_power_mode(self, mode_id: int) -> bool:
        """Set the system power mode with comprehensive validation."""
        if mode_id not in self.power_modes:
            raise JetsonPowerModeError(
                f"Invalid power mode: {mode_id}. Valid modes: {list(self.power_modes.keys())}"
            )

        mode_name = self.power_modes[mode_id]
        logger.info("Setting power mode to %d: %s", mode_id, mode_name)
        print_status(f"Setting power mode to {mode_id}: {mode_name}...")

        try:
            result = subprocess.run(
                [self.sudo_path, self.nvpmodel_path, "-m", str(mode_id)],
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
            )

            if result.returncode != 0:
                logger.warning(
                    "nvpmodel set mode returned %d: %s",
                    result.returncode,
                    result.stderr,
                )

            clocks_result = subprocess.run(
                [self.sudo_path, self.jetson_clocks_path],
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
            )

            if clocks_result.returncode != 0:
                logger.warning("jetson_clocks failed: %s", clocks_result.stderr)

            print_info("Waiting for power mode stabilization...")
            time.sleep(8)

            current = self.get_current_mode()
            if current == mode_id:
                print_success(f"Power mode confirmed: {mode_name}")
                logger.info("Power mode successfully set to %s", mode_name)
                return True
            else:
                logger.warning(
                    "Power mode verification mismatch: expected %d, got %d",
                    mode_id,
                    current,
                )
                print_warning(
                    f"Power mode verification: expected {mode_id}, detected {current}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error("Timeout setting power mode %d", mode_id)
            return False
        except Exception as e:
            logger.error("Error setting power mode %d: %s", mode_id, e)
            return False

    def get_temperature(self) -> float:
        """Get current system temperature with multiple fallback methods."""
        thermal_paths = [
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/thermal/thermal_zone1/temp",
            "/sys/class/thermal/thermal_zone2/temp",
            "/sys/devices/virtual/thermal/thermal_zone0/temp",
            "/sys/devices/virtual/thermal/thermal_zone1/temp",
        ]

        for path in thermal_paths:
            try:
                with open(path) as f:
                    temp_millic = int(f.read().strip())
                    temp_c = temp_millic / 1000.0
                    if 20.0 <= temp_c <= 100.0:
                        logger.debug("Temperature: %.1f°C from %s", temp_c, path)
                        return temp_c
            except (FileNotFoundError, ValueError, PermissionError) as e:
                logger.debug("Failed to read %s: %s", path, e)
                continue

        if self.tegrastats_path:
            try:
                result = subprocess.run(
                    [self.tegrastats_path, "--interval", "100"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.stdout:
                    temp_match = re.search(r"thermal@(\d+(?:\.\d+)?)C", result.stdout)
                    if temp_match:
                        temp = float(temp_match.group(1))
                        logger.debug("Temperature: %.1f°C from tegrastats", temp)
                        return temp
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.debug("Tegrastats temperature reading failed: %s", e)

        logger.warning("No temperature readings available")
        return 0.0

    def get_power_consumption(self) -> float:
        """Get current power consumption using multiple detection methods."""
        power_w = self._get_power_from_hwmon()
        if power_w > 0:
            return power_w

        power_w = self._get_power_from_tegrastats()
        if power_w > 0:
            return power_w

        power_w = self._get_power_from_ina3221()
        if power_w > 0:
            return power_w

        return self._estimate_power_consumption()

    def _get_power_from_hwmon(self) -> float:
        """Get power consumption from HWMON sensors."""
        try:
            hwmon_path = "/sys/class/hwmon/hwmon1"
            total_power_w = 0.0
            valid_readings = 0

            for channel in range(1, 5):
                try:
                    volt_file = f"{hwmon_path}/in{channel}_input"
                    curr_file = f"{hwmon_path}/curr{channel}_input"

                    if not (os.path.exists(volt_file) and os.path.exists(curr_file)):
                        continue

                    with open(volt_file) as vf, open(curr_file) as cf:
                        voltage_mv = float(vf.read().strip())
                        current_ma = float(cf.read().strip())

                    power_w = (voltage_mv * current_ma) / 1000000.0

                    if 0.1 <= power_w <= 30.0:
                        total_power_w += power_w
                        valid_readings += 1

                except (ValueError, FileNotFoundError) as e:
                    logger.debug("Failed to read channel %d: %s", channel, e)
                    continue

            if valid_readings > 0:
                logger.debug(
                    "HWMON power: %.2fW from %d channels", total_power_w, valid_readings
                )
                return total_power_w

        except Exception as e:
            logger.debug("HWMON power reading failed: %s", e)

        return 0.0

    def _get_power_from_tegrastats(self) -> float:
        """Get power consumption from tegrastats."""
        if not self.tegrastats_path:
            return 0.0

        try:
            result = subprocess.run(
                [
                    self.sudo_path,
                    "timeout",
                    "5s",
                    self.tegrastats_path,
                    "--interval",
                    "1000",
                ],
                capture_output=True,
                text=True,
                shell=False,
            )

            if not result.stdout:
                return 0.0

            lines = result.stdout.strip().split("\n")
            for line in reversed(lines):
                power_patterns = [
                    r"VDD_IN\s+(\d+)mW/(\d+)mW",
                    r"VDD_CPU_GPU_CV\s+(\d+)mW/(\d+)mW",
                    r"VDD_SOC\s+(\d+)mW/(\d+)mW",
                ]

                total_power_mw = 0.0
                components_found = 0

                for pattern in power_patterns:
                    match = re.search(pattern, line)
                    if match:
                        try:
                            current_mw = float(match.group(1))
                            if current_mw > 0:
                                total_power_mw += current_mw
                                components_found += 1
                        except (ValueError, IndexError):
                            continue

                if components_found > 0:
                    power_w = total_power_mw / 1000.0
                    if 3.0 <= power_w <= 50.0:
                        logger.debug("Tegrastats power: %.2fW", power_w)
                        return power_w

        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug("Tegrastats power reading failed: %s", e)

        return 0.0

    def _get_power_from_ina3221(self) -> float:
        """Get power consumption from INA3221 direct calculation."""
        try:
            base_path = "/sys/class/hwmon/hwmon1"

            if not os.path.exists(f"{base_path}/name"):
                return 0.0

            with open(f"{base_path}/name") as f:
                device_name = f.read().strip().lower()
                if "ina3221" not in device_name:
                    return 0.0

            total_power = 0.0

            for channel in range(1, 8):
                volt_path = f"{base_path}/in{channel}_input"
                curr_path = f"{base_path}/curr{channel}_input"

                if not (os.path.exists(volt_path) and os.path.exists(curr_path)):
                    continue

                try:
                    with open(volt_path) as vf, open(curr_path) as cf:
                        voltage_mv = float(vf.read().strip())
                        current_ma = float(cf.read().strip())

                    power_w = (voltage_mv * current_ma) / 1000000.0

                    if 0.1 <= power_w <= 25.0:
                        total_power += power_w

                except (ValueError, FileNotFoundError):
                    continue

            if total_power > 3.0:
                logger.debug("INA3221 direct power: %.2fW", total_power)
                return total_power

        except Exception as e:
            logger.debug("INA3221 direct reading failed: %s", e)

        return 0.0

    def _estimate_power_consumption(self) -> float:
        """Estimate power consumption based on current power mode."""
        power_estimates = {
            0: 6.0,
            1: 8.0,
            2: 12.0,
        }

        estimated_power = power_estimates.get(self.current_mode, 7.0)
        logger.warning("Using power estimate: %.1fW", estimated_power)
        return estimated_power

    def wait_for_thermal_stability(
        self, max_temp: float = 75.0, timeout: float = 300.0
    ) -> bool:
        """Wait for system thermal stability with comprehensive monitoring."""
        logger.info(
            "Waiting for thermal stability (max %.1f°C, timeout %.1fs)",
            max_temp,
            timeout,
        )
        print_info(f"Waiting for thermal stability (max {max_temp:.1f}°C)...")

        start_time = time.time()
        stable_readings = 0
        temp_history: list[float] = []

        while time.time() - start_time < timeout:
            current_temp = self.get_temperature()
            temp_history.append(current_temp)

            if len(temp_history) > 5:
                temp_history.pop(0)

            if current_temp > max_temp:
                logger.warning(
                    "Temperature too high: %.1f°C. Cooling down...", current_temp
                )
                print_warning(
                    f"Temperature too high: {current_temp:.1f}°C. Cooling down..."
                )
                time.sleep(20)
                stable_readings = 0
                continue

            if len(temp_history) >= 5:
                temp_range = max(temp_history) - min(temp_history)
                if temp_range < 1.0:
                    stable_readings += 1
                    if stable_readings >= 3:
                        logger.info(
                            "Thermal stability achieved at %.1f°C", current_temp
                        )
                        print_success(
                            f"Thermal stability achieved at {current_temp:.1f}°C"
                        )
                        return True
                else:
                    stable_readings = 0

            time.sleep(5)

        current_temp = self.get_temperature()
        logger.warning("Thermal stability timeout. Current temp: %.1f°C", current_temp)
        print_warning(f"Thermal stability timeout. Current temp: {current_temp:.1f}°C")
        return current_temp <= max_temp


class PowerModeComparativeBenchmark:
    """Comprehensive benchmarking system supporting naive and blocked implementations."""

    def __init__(self, implementations: list[str] | None = None) -> None:
        """Initialize the benchmark system.

        Args:
            implementations: List of implementations to test ("naive", "blocked", "cublas", or any combination).
                           If None, auto-detects available implementations.
        """
        logger.info("Initializing PowerModeComparativeBenchmark")

        try:
            self.power_manager = JetsonPowerModeManager()
        except JetsonPowerModeError as e:
            logger.error("Failed to initialize power manager: %s", e)
            raise

        # Detect available implementations
        available_impls = []
        if os.path.exists("cuda/naive_benchmark"):
            available_impls.append("naive")
        if os.path.exists("cuda/blocked_benchmark"):
            available_impls.append("blocked")
        if os.path.exists("cuda/cublas_benchmark"):
            available_impls.append("cublas")
        if os.path.exists("cuda/tensor_core_benchmark"):
            available_impls.append("tensor_core")

        if not available_impls:
            raise FileNotFoundError(
                "No benchmark binaries found. Run 'make compile' first."
            )

        if implementations is None:
            self.implementations = available_impls
        else:
            self.implementations = [
                impl for impl in implementations if impl in available_impls
            ]
            if not self.implementations:
                raise ValueError(
                    f"None of the requested implementations {implementations} are available"
                )

        logger.info("Available implementations: %s", self.implementations)
        print_info(f"Testing implementations: {', '.join(self.implementations)}")

        self.results: dict[str, list[dict]] = {
            impl: [] for impl in self.implementations
        }
        self.test_sizes = [64, 128, 256, 512, 1024]

        self.performance_validation = {
            "min_reasonable_gflops": 5.0,
            "max_reasonable_gflops": 2000.0,  # Increased for cuBLAS vendor-optimized performance
            "efficiency_max": 100.0,  # Based on corrected theoretical peak calculation
        }

        logger.info("Benchmark system initialized with test sizes: %s", self.test_sizes)

    def validate_benchmark_result(
        self, result: dict | None, matrix_size: int, power_mode: int
    ) -> list[str]:
        """Validate benchmark results for technical accuracy."""
        if result is None:
            return ["Benchmark failed to complete"]

        issues = []
        gflops = result.get("gflops", 0)

        if gflops < self.performance_validation["min_reasonable_gflops"]:
            issues.append(f"Performance too low: {gflops:.2f} GFLOPS")

        if gflops > self.performance_validation["max_reasonable_gflops"]:
            issues.append(f"Performance unrealistically high: {gflops:.2f} GFLOPS")

        # Note: Efficiency >100% is now possible with corrected theoretical peak calculation
        # Vendor-optimized implementations like cuBLAS should achieve 40-80% efficiency

        expected = self.power_manager.expected_performance_ranges.get(power_mode, {})
        min_expected = expected.get("min_gflops", 0)
        max_expected = expected.get("max_gflops", 999)

        if gflops < min_expected * 0.7:
            issues.append(
                f"Performance below expected range for power mode {power_mode}"
            )

        if gflops > max_expected * 1.5:
            issues.append(
                f"Performance above expected range for power mode {power_mode}"
            )

        if issues:
            logger.warning(
                "Validation issues for %dx%d in mode %d: %s",
                matrix_size,
                matrix_size,
                power_mode,
                issues,
            )

        return issues

    def run_single_benchmark(
        self, power_mode: int, matrix_size: int, implementation: str
    ) -> dict | None:
        """Execute a single benchmark for specified implementation."""
        if implementation not in self.implementations:
            logger.error("Implementation not available: %s", implementation)
            return None

        logger.info(
            "Running %s benchmark: %dx%d in mode %d",
            implementation,
            matrix_size,
            matrix_size,
            power_mode,
        )
        print_info(f"  Testing {matrix_size}x{matrix_size} ({implementation})...")

        pre_temp = self.power_manager.get_temperature()
        pre_power = self.power_manager.get_power_consumption()

        try:
            original_dir = os.getcwd()
            benchmark_path = f"cuda/{implementation}_benchmark"

            if not os.path.exists(benchmark_path):
                logger.error("Benchmark binary not found: %s", benchmark_path)
                print_error(f"{implementation}_benchmark not found")
                return None

            os.chdir("cuda")

            start_time = time.time()
            try:
                result = subprocess.run(
                    [f"./{implementation}_benchmark", str(matrix_size)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except subprocess.TimeoutExpired:
                logger.error("Benchmark timeout for %dx%d", matrix_size, matrix_size)
                print_error("Timeout (>10 minutes)")
                return None

            end_time = time.time()
            os.chdir(original_dir)

            if result.returncode != 0:
                logger.error(
                    "Benchmark failed with return code %d: %s",
                    result.returncode,
                    result.stderr,
                )
                print_error(f"Failed: {result.stderr}")
                return None

            post_temp = self.power_manager.get_temperature()
            post_power = self.power_manager.get_power_consumption()

            metrics = self._parse_benchmark_output(result.stdout)
            if not metrics:
                logger.error("Failed to parse benchmark output")
                print_error("Failed to parse benchmark output")
                return None

            metrics.update(
                {
                    "implementation": implementation,
                    "power_mode_id": power_mode,
                    "power_mode_name": self.power_manager.power_modes[power_mode],
                    "matrix_size": matrix_size,
                    "timestamp": datetime.now().isoformat(),
                    "wall_clock_time": end_time - start_time,
                    "pre_temperature": pre_temp,
                    "post_temperature": post_temp,
                    "avg_power_consumption": (pre_power + post_power) / 2.0
                    if pre_power > 0 and post_power > 0
                    else 0.0,
                    "thermal_rise": post_temp - pre_temp
                    if pre_temp > 0 and post_temp > 0
                    else 0.0,
                    "hardware_model": "Jetson Orin Nano Engineering Reference Developer Kit Super",
                    "l4t_version": "R36.4.4",
                    "cuda_version": "V12.6.68",
                }
            )

            validation_issues = self.validate_benchmark_result(
                metrics, matrix_size, power_mode
            )
            if validation_issues:
                print_warning("Validation warnings:")
                for issue in validation_issues:
                    print_warning(f"  * {issue}")
                metrics["validation_warnings"] = validation_issues

            gflops = metrics.get("gflops", 0)
            efficiency = metrics.get("efficiency_percent", 0)

            print_success(
                f"{implementation}: {gflops:.2f} GFLOPS, {efficiency:.1f}% efficiency"
            )
            logger.info("%s benchmark completed: %.2f GFLOPS", implementation, gflops)

            return metrics

        except Exception as e:
            logger.error("Benchmark execution failed: %s", e)
            print_error(str(e))
            return None
        finally:
            if os.getcwd() != original_dir:
                os.chdir(original_dir)

    def _parse_benchmark_output(self, output: str) -> dict:
        """Parse benchmark output to extract performance metrics."""
        metrics = {}

        for line in output.split("\n"):
            try:
                if any(
                    phrase in line
                    for phrase in ["Measured Performance:", "Performance:"]
                ):
                    gflops_match = re.search(r"(\d+(?:\.\d+)?)\s*GFLOPS", line)
                    if gflops_match:
                        metrics["gflops"] = float(gflops_match.group(1))

                elif "Elapsed Time:" in line:
                    time_match = re.search(r"(\d+(?:\.\d+)?)\s*ms", line)
                    if time_match:
                        metrics["elapsed_ms"] = float(time_match.group(1))

                elif "Memory Bandwidth:" in line:
                    bw_match = re.search(r"(\d+(?:\.\d+)?)\s*GB/s", line)
                    if bw_match:
                        metrics["memory_bandwidth"] = float(bw_match.group(1))

                elif any(
                    phrase in line
                    for phrase in ["Efficiency:", "Algorithm Efficiency:"]
                ):
                    eff_match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
                    if eff_match:
                        metrics["efficiency_percent"] = float(eff_match.group(1))

                elif "Arithmetic Intensity:" in line:
                    ai_match = re.search(r"(\d+(?:\.\d+)?)\s*FLOPS/byte", line)
                    if ai_match:
                        metrics["arithmetic_intensity"] = float(ai_match.group(1))

                elif "Numerical Accuracy" in line and "Error" in line:
                    error_match = re.search(
                        r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", line.split(":")[-1]
                    )
                    if error_match:
                        metrics["numerical_error"] = float(error_match.group(1))

            except (ValueError, IndexError, AttributeError) as e:
                logger.debug("Failed to parse line '%s': %s", line, e)
                continue

        logger.debug("Parsed metrics: %s", metrics)
        return metrics

    def run_comprehensive_analysis(self) -> None:
        """Execute the complete 3-power mode analysis for all implementations."""
        logger.info("Starting comprehensive multi-implementation analysis")
        print_info("Jetson Orin Nano: Multi-Implementation Comparative Analysis")
        print_info("Target Hardware: Engineering Reference Developer Kit Super")
        print_info("=" * 70)
        print_info(f"Implementations: {', '.join(self.implementations)}")
        print_info(f"Power modes: {list(self.power_manager.power_modes.values())}")
        print_info(f"Test matrix sizes: {self.test_sizes}")
        print_info("")

        initial_mode = self.power_manager.current_mode
        total_tests = (
            len(self.power_manager.power_modes)
            * len(self.test_sizes)
            * len(self.implementations)
        )
        current_test = 0
        failed_tests = 0

        try:
            for power_mode in sorted(self.power_manager.power_modes.keys()):
                mode_name = self.power_manager.power_modes[power_mode]
                logger.info("Testing power mode %d: %s", power_mode, mode_name)
                print_info(f"Testing Power Mode {power_mode}: {mode_name}")
                print_info("-" * 50)

                if not self.power_manager.set_power_mode(power_mode):
                    logger.error("Failed to set power mode %d", power_mode)
                    print_error(f"Failed to set power mode {power_mode}, skipping...")
                    failed_tests += len(self.test_sizes) * len(self.implementations)
                    continue

                if not self.power_manager.wait_for_thermal_stability():
                    logger.warning(
                        "Thermal stability not achieved, proceeding with caution"
                    )
                    print_warning(
                        "Thermal stability not achieved, proceeding with caution"
                    )

                mode_results: dict[str, list[dict]] = {
                    impl: [] for impl in self.implementations
                }

                for matrix_size in self.test_sizes:
                    for impl in self.implementations:
                        current_test += 1
                        print_progress(current_test, total_tests)

                        result = self.run_single_benchmark(
                            power_mode, matrix_size, impl
                        )
                        if result:
                            mode_results[impl].append(result)
                            self.results[impl].append(result)
                        else:
                            failed_tests += 1

                        current_temp = self.power_manager.get_temperature()
                        if current_temp > 70:
                            logger.info("Cooling down (current: %.1f°C)", current_temp)
                            print_info(
                                f"    Cooling down (current: {current_temp:.1f}°C)..."
                            )
                            time.sleep(20)

                # Save results for each implementation
                for impl in self.implementations:
                    self._save_implementation_results(
                        power_mode, mode_results[impl], impl
                    )

                results_summary = ", ".join(
                    [
                        f"{len(mode_results[impl])} {impl}"
                        for impl in self.implementations
                    ]
                )
                print_success(f"Completed {mode_name}: {results_summary}")
                print_info("")

        finally:
            logger.info("Restoring initial power mode: %d", initial_mode)
            print_info("Restoring initial power mode...")
            self.power_manager.set_power_mode(initial_mode)

        self._save_comprehensive_results()

        print_info("=" * 70)
        print_success("Multi-Implementation Analysis Completed!")
        for impl in self.implementations:
            print_info(f"{impl.capitalize()} tests: {len(self.results[impl])}")
        print_info(f"Failed tests: {failed_tests}")
        print_info("Results saved in data/raw/power_modes/")

        logger.info("Comprehensive analysis completed successfully")

    def _save_implementation_results(
        self, power_mode: int, results: list[dict], implementation: str
    ) -> None:
        """Save results for specific implementation and power mode."""
        if not results:
            return

        mode_name = self.power_manager.power_modes[power_mode]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_dir = Path("data/raw/power_modes")
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON format
        json_file = (
            output_dir
            / f"{implementation}_mode_{power_mode}_{mode_name}_{timestamp}.json"
        )
        try:
            with open(json_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info("Saved %s results: %s", implementation, json_file)
        except Exception as e:
            logger.error("Failed to save JSON results: %s", e)

        # CSV format
        csv_file = (
            output_dir
            / f"{implementation}_mode_{power_mode}_{mode_name}_{timestamp}.csv"
        )
        try:
            with open(csv_file, "w", newline="") as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
            logger.info("Saved %s CSV: %s", implementation, csv_file)
        except Exception as e:
            logger.error("Failed to save CSV results: %s", e)

    def _save_comprehensive_results(self) -> None:
        """Save combined results from all implementations and power modes."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/raw/power_modes")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save comprehensive results for each implementation
        for impl in self.implementations:
            json_file = output_dir / f"{impl}_3mode_analysis_{timestamp}.json"
            try:
                with open(json_file, "w") as f:
                    json.dump(self.results[impl], f, indent=2)
                logger.info("Saved comprehensive %s JSON: %s", impl, json_file)
            except Exception as e:
                logger.error("Failed to save comprehensive JSON: %s", e)

            csv_file = output_dir / f"{impl}_3mode_analysis_{timestamp}.csv"
            try:
                if self.results[impl]:
                    csv_results = [
                        {k: v for k, v in result.items() if k != "validation_warnings"}
                        for result in self.results[impl]
                    ]

                    with open(csv_file, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
                        writer.writeheader()
                        writer.writerows(csv_results)
                    logger.info("Saved comprehensive %s CSV: %s", impl, csv_file)
            except Exception as e:
                logger.error("Failed to save comprehensive CSV: %s", e)

        # Generate summary report
        self._generate_summary_report(timestamp)

        print_info("Comprehensive results saved:")
        for impl in self.implementations:
            print_info(f"  - {impl}_3mode_analysis_{timestamp}.json")
            print_info(f"  - {impl}_3mode_analysis_{timestamp}.csv")

    def _generate_summary_report(self, timestamp: str) -> None:
        """Generate markdown summary report with comprehensive analysis."""
        report_dir = Path("data/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        impl_suffix = "_".join(self.implementations)
        report_file = report_dir / f"analysis_{impl_suffix}_{timestamp}.md"

        try:
            with open(report_file, "w") as f:
                f.write(
                    "# Jetson Orin Nano: Multi-Implementation Comparative Analysis\n\n"
                )
                f.write(
                    "**Hardware:** NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super\n"
                )
                f.write(f"**Generated**: {datetime.now().isoformat()}\n")
                f.write(
                    f"**Implementations Tested**: {', '.join(self.implementations)}\n"
                )
                f.write("**L4T Version:** R36.4.4 (JetPack 6.x)\n")
                f.write("**CUDA Version:** V12.6.68\n")
                f.write(
                    f"**Power Modes Tested**: {list(self.power_manager.power_modes.values())}\n"
                )
                f.write(f"**Matrix Sizes**: {self.test_sizes}\n\n")

                # Performance summary table for each implementation
                for impl in self.implementations:
                    f.write(f"## {impl.capitalize()} Implementation Performance\n\n")
                    f.write(
                        "| Power Mode | Peak GFLOPS | Avg GFLOPS | Peak Efficiency | Avg Efficiency |\n"
                    )
                    f.write(
                        "|------------|-------------|------------|-----------------|----------------|\n"
                    )

                    for power_mode in sorted(self.power_manager.power_modes.keys()):
                        mode_results = [
                            r
                            for r in self.results[impl]
                            if r["power_mode_id"] == power_mode
                        ]
                        if mode_results:
                            peak_gflops = max(r["gflops"] for r in mode_results)
                            avg_gflops = sum(r["gflops"] for r in mode_results) / len(
                                mode_results
                            )
                            peak_efficiency = max(
                                r["efficiency_percent"] for r in mode_results
                            )
                            avg_efficiency = sum(
                                r["efficiency_percent"] for r in mode_results
                            ) / len(mode_results)

                            mode_name = self.power_manager.power_modes[power_mode]
                            f.write(
                                f"| {mode_name} | {peak_gflops:.2f} | {avg_gflops:.2f} | "
                                f"{peak_efficiency:.1f}% | {avg_efficiency:.1f}% |\n"
                            )
                    f.write("\n")

                # Comparison section if multiple implementations
                if len(self.implementations) > 1:
                    f.write("## Implementation Comparison\n\n")
                    f.write("Performance improvement of blocked over naive:\n\n")
                    f.write(
                        "| Power Mode | Matrix Size | Naive GFLOPS | Blocked GFLOPS | Improvement |\n"
                    )
                    f.write(
                        "|------------|-------------|--------------|----------------|-------------|\n"
                    )

                    for power_mode in sorted(self.power_manager.power_modes.keys()):
                        mode_name = self.power_manager.power_modes[power_mode]
                        for size in self.test_sizes:
                            naive_result = next(
                                (
                                    r
                                    for r in self.results.get("naive", [])
                                    if r["power_mode_id"] == power_mode
                                    and r["matrix_size"] == size
                                ),
                                None,
                            )
                            blocked_result = next(
                                (
                                    r
                                    for r in self.results.get("blocked", [])
                                    if r["power_mode_id"] == power_mode
                                    and r["matrix_size"] == size
                                ),
                                None,
                            )

                            if naive_result and blocked_result:
                                naive_gflops = naive_result["gflops"]
                                blocked_gflops = blocked_result["gflops"]
                                improvement = (
                                    (blocked_gflops - naive_gflops) / naive_gflops
                                ) * 100

                                f.write(
                                    f"| {mode_name} | {size}×{size} | {naive_gflops:.2f} | "
                                    f"{blocked_gflops:.2f} | {improvement:+.1f}% |\n"
                                )
                    f.write("\n")

            logger.info("Generated summary report: %s", report_file)
            print_success(f"Summary report: {report_file}")

        except Exception as e:
            logger.error("Failed to generate summary report: %s", e)


def main() -> int:
    """Main execution function with comprehensive error handling."""
    logger.info("Starting Jetson Orin Nano Multi-Implementation Benchmarking System")
    print_info("Jetson Orin Nano: Multi-Implementation Benchmarking System")
    print_info("Engineering Reference Developer Kit Super")
    print_info("=" * 60)

    try:
        # Detect available implementations
        available_impls = []
        if os.path.exists("cuda/naive_benchmark"):
            available_impls.append("naive")
        if os.path.exists("cuda/blocked_benchmark"):
            available_impls.append("blocked")
        if os.path.exists("cuda/cublas_benchmark"):
            available_impls.append("cublas")
        if os.path.exists("cuda/tensor_core_benchmark"):
            available_impls.append("tensor_core")

        if not available_impls:
            logger.error("No benchmark binaries found")
            print_error("No benchmark binaries found. Please compile first:")
            print_error("  make compile")
            return 1

        print_info(f"Detected implementations: {', '.join(available_impls)}")

        # Note: Skipping upfront sudo check to avoid hanging with passwordless sudo
        # The power mode manager will handle sudo failures gracefully when switching modes
        logger.info(
            "Proceeding with benchmark - sudo access will be checked during power mode switching"
        )

        # Execute comprehensive analysis for all available implementations
        benchmark = PowerModeComparativeBenchmark(implementations=available_impls)
        benchmark.run_comprehensive_analysis()

        print_info("")
        print_info("Next steps:")
        print_info("1. Review results in data/raw/power_modes/")
        print_info("2. Generate visualizations: make visualize")
        print_info("3. Compare implementation performance")

        logger.info("Benchmarking system completed successfully")
        return 0

    except (JetsonPowerModeError, BenchmarkValidationError) as e:
        logger.error("Application error: %s", e)
        print_error(str(e))
        return 1
    except KeyboardInterrupt:
        logger.info("Benchmarking interrupted by user")
        print_info("\nBenchmarking interrupted by user")
        return 1
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        print_error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
