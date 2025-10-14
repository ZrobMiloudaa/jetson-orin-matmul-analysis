#!/usr/bin/env python3
"""Pytest Configuration and Fixtures for Jetson Orin Nano Benchmarking System.

Copyright 2025 ByteStack Labs
SPDX-License-Identifier: MIT

This module provides comprehensive pytest configuration and fixtures for testing the complete
benchmarking system across multiple power modes (15W, 25W, MAXN_SUPER) with enterprise-grade
error handling, logging, and validation.

Target Hardware: Jetson Orin Nano Engineering Reference Developer Kit Super
Software Stack: L4T R36.4.4 (JetPack 6.x), CUDA V12.6.68

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 1.0.0
License: MIT
"""

import logging
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

# Configure logging for test execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class TestConfigurationError(Exception):
    """Raised when test configuration is invalid."""

    pass


class FixtureError(Exception):
    """Raised when fixture setup or teardown fails."""

    pass


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest for Jetson Orin Nano benchmarking tests.

    Args:
        config: Pytest configuration object.
    """
    logger.info("Configuring pytest for Jetson Orin Nano benchmarking system")

    # Add custom markers for test categorization
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring full system setup"
    )
    config.addinivalue_line(
        "markers", "performance: Performance calculation and validation tests"
    )
    config.addinivalue_line("markers", "utility: Utility function unit tests")
    config.addinivalue_line("markers", "cuda: Tests requiring CUDA functionality")
    config.addinivalue_line(
        "markers", "power_mode: Tests validating power mode functionality"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take significant time to execute"
    )

    logger.debug("Custom pytest markers configured successfully")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify collected test items based on configuration.

    Args:
        config: Pytest configuration object.
        items: List of collected test items.
    """
    logger.info(f"Processing {len(items)} collected test items")

    # Auto-mark tests based on their module and name patterns
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark performance tests
        if "performance" in item.nodeid or "gflops" in item.name.lower():
            item.add_marker(pytest.mark.performance)

        # Mark utility tests
        if "utils" in item.nodeid or "utility" in item.nodeid:
            item.add_marker(pytest.mark.utility)

        # Mark CUDA-related tests
        if "cuda" in item.name.lower() or "gpu" in item.name.lower():
            item.add_marker(pytest.mark.cuda)

        # Mark power mode tests
        if "power" in item.name.lower() or "scaling" in item.name.lower():
            item.add_marker(pytest.mark.power_mode)

        # Mark slow tests
        if any(
            keyword in item.name.lower()
            for keyword in ["complex", "scaling", "thermal"]
        ):
            item.add_marker(pytest.mark.slow)

    logger.debug("Test items marked successfully based on patterns")


@pytest.fixture(scope="session")
def test_session_config() -> dict[str, Any]:
    """Provide session-wide test configuration.

    Returns:
        Dictionary containing test session configuration.
    """
    logger.info("Setting up test session configuration")

    config = {
        "target_hardware": "Jetson Orin Nano Engineering Reference Developer Kit Super",
        "software_stack": "L4T R36.4.4 (JetPack 6.x), CUDA V12.6.68",
        "test_matrix_sizes": [64, 128, 256, 512],
        "power_modes": ["15W", "25W", "MAXN_SUPER"],
        "tolerance_gflops": 1e-6,
        "tolerance_efficiency": 1e-6,
        "max_reasonable_gflops": 10000.0,
        "max_temperature_celsius": 85.0,
        "min_temperature_celsius": 20.0,
    }

    logger.debug(f"Test session configuration: {config}")
    return config


@pytest.fixture
def sample_performance_data() -> dict[str, Any]:
    """Provide sample performance data for testing calculations.

    Returns:
        Dictionary containing sample performance metrics.

    Raises:
        FixtureError: If sample data validation fails.
    """
    logger.debug("Creating sample performance data fixture")

    try:
        data = {
            "elapsed_ms": 1.5,
            "gflops": 89.5,
            "memory_bandwidth": 12.3,
            "arithmetic_intensity": 42.67,
            "efficiency_percent": 25.4,
            "numerical_error": 1e-6,
            "power_mode_id": 1,
            "power_mode_name": "25W",
            "matrix_size": 256,
            "pre_temperature": 48.2,
            "post_temperature": 49.1,
            "avg_power_consumption": 24.8,
            "thermal_rise": 0.9,
            "hardware_model": "Jetson Orin Nano Engineering Reference Developer Kit Super",
            "l4t_version": "R36.4.4",
            "cuda_version": "V12.6.68",
        }

        # Validate critical fields
        _validate_performance_data(data)

        logger.debug("Sample performance data created successfully")
        return data

    except Exception as e:
        logger.error(f"Failed to create sample performance data: {e}")
        raise FixtureError(f"Sample performance data creation failed: {e}") from e


@pytest.fixture
def sample_multi_mode_data() -> list[dict[str, Any]]:
    """Provide sample multi-power mode performance data.

    Returns:
        List of dictionaries containing performance data for different power modes.

    Raises:
        FixtureError: If multi-mode data validation fails.
    """
    logger.debug("Creating multi-mode performance data fixture")

    try:
        data = [
            {
                "power_mode_id": 0,
                "power_mode_name": "15W",
                "matrix_size": 256,
                "gflops": 87.1,
                "efficiency_percent": 24.7,
                "avg_power_consumption": 18.3,
                "pre_temperature": 51.6,
                "post_temperature": 51.9,
            },
            {
                "power_mode_id": 1,
                "power_mode_name": "25W",
                "matrix_size": 256,
                "gflops": 120.5,
                "efficiency_percent": 25.6,
                "avg_power_consumption": 21.3,
                "pre_temperature": 51.9,
                "post_temperature": 51.9,
            },
            {
                "power_mode_id": 2,
                "power_mode_name": "MAXN_SUPER",
                "matrix_size": 256,
                "gflops": 130.3,
                "efficiency_percent": 21.7,
                "avg_power_consumption": 22.8,
                "pre_temperature": 52.5,
                "post_temperature": 52.4,
            },
        ]

        # Validate each data point
        for i, point in enumerate(data):
            try:
                _validate_performance_data(point)
            except Exception as e:
                raise FixtureError(
                    f"Multi-mode data validation failed for point {i}: {e}"
                ) from e

        logger.debug(
            f"Multi-mode performance data created with {len(data)} power modes"
        )
        return data

    except Exception as e:
        logger.error(f"Failed to create multi-mode performance data: {e}")
        raise FixtureError(f"Multi-mode data creation failed: {e}") from e


@pytest.fixture
def temp_results_dir() -> Generator[Path, None, None]:
    """Provide temporary directory for test results.

    Yields:
        Path object pointing to temporary results directory.

    Raises:
        FixtureError: If temporary directory creation fails.
    """
    logger.debug("Creating temporary results directory")

    try:
        with tempfile.TemporaryDirectory(prefix="jetson_test_") as tmpdir:
            temp_path = Path(tmpdir)

            # Create subdirectories matching expected structure
            subdirs = [
                "logs",
                "raw/power_modes",
                "reports",
                "plots/power_analysis",
            ]

            for subdir in subdirs:
                (temp_path / subdir).mkdir(parents=True, exist_ok=True)

            logger.debug(f"Temporary directory created: {temp_path}")
            yield temp_path

    except Exception as e:
        logger.error(f"Failed to create temporary results directory: {e}")
        raise FixtureError(f"Temporary directory creation failed: {e}") from e
    finally:
        logger.debug("Temporary results directory cleanup completed")


@pytest.fixture
def mock_cuda_environment(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Mock CUDA environment for CPU-only testing.

    Args:
        monkeypatch: Pytest monkeypatch fixture for mocking.

    Returns:
        Dictionary containing mock CUDA environment information.

    Raises:
        FixtureError: If CUDA environment mocking fails.
    """
    logger.debug("Setting up mock CUDA environment")

    try:
        mock_device_props = {
            "name": "Mock Jetson Orin Nano",
            "major": 8,
            "minor": 7,
            "totalGlobalMem": 8 * 1024**3,  # 8GB
            "sharedMemPerBlock": 49152,  # 48KB
            "multiProcessorCount": 32,  # Mock SM count
            "maxThreadsPerMultiProcessor": 1536,
            "clockRate": 918000,  # 918 MHz
        }

        def mock_device_count() -> int:
            """Mock CUDA device count."""
            return 1

        def mock_get_device_properties(device_id: int = 0) -> dict[str, Any]:
            """Mock CUDA device properties."""
            if device_id != 0:
                raise RuntimeError(f"Invalid device ID: {device_id}")
            return mock_device_props.copy()

        def mock_cuda_runtime_version() -> str:
            """Mock CUDA runtime version."""
            return "V12.6.68"

        def mock_driver_version() -> str:
            """Mock CUDA driver version."""
            return "540.4.0"

        # Apply mocks
        monkeypatch.setattr("cuda.device_count", mock_device_count)
        monkeypatch.setattr("cuda.get_device_properties", mock_get_device_properties)
        monkeypatch.setattr("cuda.runtime_version", mock_cuda_runtime_version)
        monkeypatch.setattr("cuda.driver_version", mock_driver_version)

        mock_env = {
            "device_count": 1,
            "device_properties": mock_device_props,
            "runtime_version": "V12.6.68",
            "driver_version": "540.4.0",
            "is_available": True,
        }

        logger.debug("Mock CUDA environment configured successfully")
        return mock_env

    except Exception as e:
        logger.error(f"Failed to setup mock CUDA environment: {e}")
        raise FixtureError(f"CUDA environment mocking failed: {e}") from e


@pytest.fixture
def mock_system_monitoring(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Mock system monitoring functionality for testing.

    Args:
        monkeypatch: Pytest monkeypatch fixture for mocking.

    Returns:
        Dictionary containing mock system monitoring functions.

    Raises:
        FixtureError: If system monitoring mocking fails.
    """
    logger.debug("Setting up mock system monitoring")

    try:
        # Mock temperature readings
        mock_temperatures = {
            "cpu-thermal": 52.0,
            "gpu-thermal": 51.0,
            "soc0-thermal": 50.0,
            "tj-thermal": 52.0,
        }

        # Mock power readings
        mock_power_readings = {
            "15W": 18.3,
            "25W": 21.3,
            "MAXN_SUPER": 22.8,
        }

        def mock_read_temperatures() -> dict[str, float]:
            """Mock temperature sensor readings."""
            return mock_temperatures.copy()

        def mock_read_power_consumption(mode: str = "25W") -> float:
            """Mock power consumption readings."""
            return mock_power_readings.get(mode, 20.0)

        def mock_get_system_info() -> dict[str, Any]:
            """Mock system information."""
            return {
                "hardware_model": "Jetson Orin Nano Engineering Reference Developer Kit Super",
                "l4t_version": "R36.4.4",
                "cuda_version": "V12.6.68",
                "memory_total": "7.4GB",
                "cpu_count": 6,
            }

        # Apply mocks
        monkeypatch.setattr("monitoring.read_temperatures", mock_read_temperatures)
        monkeypatch.setattr(
            "monitoring.read_power_consumption", mock_read_power_consumption
        )
        monkeypatch.setattr("monitoring.get_system_info", mock_get_system_info)

        mock_monitoring = {
            "temperatures": mock_temperatures,
            "power_readings": mock_power_readings,
            "read_temperatures": mock_read_temperatures,
            "read_power_consumption": mock_read_power_consumption,
            "get_system_info": mock_get_system_info,
        }

        logger.debug("Mock system monitoring configured successfully")
        return mock_monitoring

    except Exception as e:
        logger.error(f"Failed to setup mock system monitoring: {e}")
        raise FixtureError(f"System monitoring mocking failed: {e}") from e


@pytest.fixture
def matrix_test_sizes() -> list[int]:
    """Provide standard matrix sizes for testing.

    Returns:
        List of matrix dimensions for testing.
    """
    return [64, 128, 256, 512]


@pytest.fixture
def power_mode_configs() -> list[dict[str, Any]]:
    """Provide power mode configuration data for testing.

    Returns:
        List of power mode configurations.
    """
    return [
        {
            "mode_id": 0,
            "mode_name": "15W",
            "max_power": 15.0,
            "target_frequency": 614,  # MHz
            "description": "Balanced performance mode",
        },
        {
            "mode_id": 1,
            "mode_name": "25W",
            "max_power": 25.0,
            "target_frequency": 918,  # MHz
            "description": "High performance mode",
        },
        {
            "mode_id": 2,
            "mode_name": "MAXN_SUPER",
            "max_power": 30.0,
            "target_frequency": 918,  # MHz
            "description": "Maximum performance mode",
        },
    ]


@pytest.fixture
def benchmark_tolerances() -> dict[str, float]:
    """Provide tolerance values for benchmark validation.

    Returns:
        Dictionary containing tolerance values for different metrics.
    """
    return {
        "gflops_tolerance": 1e-6,
        "efficiency_tolerance": 1e-6,
        "temperature_tolerance": 2.0,  # Celsius
        "power_tolerance": 1.0,  # Watts
        "time_tolerance": 0.001,  # Milliseconds
        "relative_error_max": 0.05,  # 5% maximum relative error
    }


def _validate_performance_data(data: dict[str, Any]) -> None:
    """Validate performance data structure and values.

    Args:
        data: Performance data dictionary to validate.

    Raises:
        FixtureError: If performance data is invalid.
    """
    required_fields = ["matrix_size", "gflops", "efficiency_percent"]

    # Check required fields
    for field in required_fields:
        if field not in data:
            raise FixtureError(
                f"Required field '{field}' missing from performance data"
            )

    # Validate numeric ranges
    if data["gflops"] < 0:
        raise FixtureError(f"GFLOPS must be non-negative, got {data['gflops']}")

    if not (0 <= data["efficiency_percent"] <= 100):
        raise FixtureError(
            f"Efficiency must be 0-100%, got {data['efficiency_percent']}"
        )

    if data["matrix_size"] <= 0:
        raise FixtureError(f"Matrix size must be positive, got {data['matrix_size']}")

    # Validate optional temperature fields if present
    if "pre_temperature" in data and "post_temperature" in data:
        pre_temp = data["pre_temperature"]
        post_temp = data["post_temperature"]

        if not (20.0 <= pre_temp <= 100.0):
            raise FixtureError(f"Pre-temperature outside valid range: {pre_temp}°C")

        if not (20.0 <= post_temp <= 100.0):
            raise FixtureError(f"Post-temperature outside valid range: {post_temp}°C")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Log test results with detailed information.

    Args:
        report: Pytest test report object.
    """
    if report.when == "call":
        if report.outcome == "passed":
            logger.info(f"TEST PASSED: {report.nodeid}")
        elif report.outcome == "failed":
            logger.error(f"TEST FAILED: {report.nodeid}")
            if hasattr(report, "longrepr"):
                logger.error(f"Failure details: {report.longrepr}")
        elif report.outcome == "skipped":
            logger.warning(f"TEST SKIPPED: {report.nodeid}")


def pytest_sessionstart(session: pytest.Session) -> None:
    """Log session start information.

    Args:
        session: Pytest session object.
    """
    logger.info("=" * 70)
    logger.info("JETSON ORIN NANO BENCHMARKING SYSTEM - TEST EXECUTION START")
    logger.info("=" * 70)
    logger.info("Target: Jetson Orin Nano Engineering Reference Developer Kit Super")
    logger.info("Software: L4T R36.4.4 (JetPack 6.x), CUDA V12.6.68")
    logger.info("=" * 70)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Log session completion information.

    Args:
        session: Pytest session object.
        exitstatus: Exit status code.
    """
    logger.info("=" * 70)
    logger.info("JETSON ORIN NANO BENCHMARKING SYSTEM - TEST EXECUTION COMPLETE")

    if exitstatus == 0:
        logger.info("RESULT: All tests passed successfully")
    else:
        logger.error(f"RESULT: Test execution failed with exit code {exitstatus}")

    logger.info("=" * 70)
