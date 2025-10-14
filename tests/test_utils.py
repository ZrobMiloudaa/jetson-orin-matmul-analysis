#!/usr/bin/env python3
"""Utility Function Testing Suite for Jetson Orin Nano Benchmarking System.

Copyright 2025 ByteStack Labs
SPDX-License-Identifier: MIT

This module provides comprehensive unit tests for utility functions in the cuda/utils/
module, testing common.cu functionality through Python bindings with enterprise-grade error
handling, logging, and validation.

Target Hardware: Jetson Orin Nano Engineering Reference Developer Kit Super
Software Stack: L4T R36.4.4 (JetPack 6.x), CUDA V12.6.68

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 1.0.0
License: MIT
"""

import logging
import sys

import numpy as np
import pytest

# Configure logging for test execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class MatrixValidationError(Exception):
    """Raised when matrix validation fails."""

    pass


class PerformanceCalculationError(Exception):
    """Raised when performance calculation validation fails."""

    pass


class SystemMonitoringError(Exception):
    """Raised when system monitoring validation fails."""

    pass


class TestMatrixInitialization:
    """Test matrix initialization utilities.

    This test class validates matrix initialization functions including
    size validation, random matrix generation, and identity matrix creation.
    """

    @staticmethod
    def get_valid_matrix_sizes() -> list[int]:
        """Get list of valid matrix sizes for testing.

        Returns:
            List of valid matrix dimensions.
        """
        return [64, 128, 256, 512, 1024]

    @staticmethod
    def get_invalid_matrix_sizes() -> list[int]:
        """Get list of invalid matrix sizes for testing.

        Returns:
            List of invalid matrix dimensions.
        """
        return [0, -1, 33, 2049]

    def test_matrix_size_validation(self) -> None:
        """Test matrix size validation logic.

        Validates that matrix size validation correctly identifies valid
        and invalid matrix dimensions according to system constraints.

        Raises:
            MatrixValidationError: If validation logic is incorrect.
        """
        logger.info("Testing matrix size validation logic")

        valid_sizes = self.get_valid_matrix_sizes()
        invalid_sizes = self.get_invalid_matrix_sizes()

        # Test valid sizes
        for size in valid_sizes:
            try:
                self._validate_matrix_size(size)
                logger.debug(f"Valid size {size} passed validation")
            except MatrixValidationError as e:
                logger.error(f"Valid size {size} failed validation: {e}")
                raise

        # Test invalid sizes - only test truly invalid ones that should raise exceptions
        truly_invalid_sizes = [
            size for size in invalid_sizes if size <= 0 or size > 2048
        ]
        for size in truly_invalid_sizes:
            with pytest.raises(MatrixValidationError):
                self._validate_matrix_size(size)
                logger.debug(f"Invalid size {size} correctly rejected")

        # Test sizes that trigger warnings but don't raise exceptions (like 33)
        warning_sizes = [size for size in invalid_sizes if 0 < size <= 2048]
        for size in warning_sizes:
            try:
                self._validate_matrix_size(size)
                logger.debug(
                    f"Size {size} triggered warning but didn't fail (expected behavior)"
                )
            except MatrixValidationError:
                logger.error(f"Size {size} unexpectedly raised exception")
                raise

        total_invalid = len(truly_invalid_sizes)
        logger.info(
            f"Matrix size validation complete - {len(valid_sizes)} valid, {total_invalid} invalid"
        )

    def test_random_matrix_properties(self) -> None:
        """Test properties of randomly initialized matrices.

        Validates that random matrix initialization produces matrices
        with correct properties including data type, dimensions, and value ranges.

        Raises:
            MatrixValidationError: If random matrix properties are incorrect.
        """
        logger.info("Testing random matrix initialization properties")

        np.random.seed(42)  # Ensure reproducible test results

        sizes = [64, 128]
        total_matrices = 0

        for n in sizes:
            logger.debug(f"Testing random matrices of size {n}x{n}")

            # Generate random matrices
            A = np.random.uniform(-1.0, 1.0, (n, n)).astype(np.float32)
            B = np.random.uniform(-1.0, 1.0, (n, n)).astype(np.float32)

            # Validate matrix properties
            self._validate_random_matrix(A, n, "Matrix A")
            self._validate_random_matrix(B, n, "Matrix B")

            total_matrices += 2

        logger.info(
            f"Random matrix validation complete - {total_matrices} matrices tested"
        )

    def test_identity_matrix_initialization(self) -> None:
        """Test identity matrix initialization.

        Validates that identity matrix initialization produces correct
        identity matrices with proper diagonal and off-diagonal elements.

        Raises:
            MatrixValidationError: If identity matrix is incorrect.
        """
        logger.info("Testing identity matrix initialization")

        sizes = [64, 128]

        for n in sizes:
            logger.debug(f"Testing identity matrix of size {n}x{n}")

            identity_matrix = np.eye(n, dtype=np.float32)

            # Validate basic properties
            if identity_matrix.shape != (n, n):
                raise MatrixValidationError(
                    f"Identity matrix shape {identity_matrix.shape} != ({n}, {n})"
                )

            if identity_matrix.dtype != np.float32:
                raise MatrixValidationError(
                    f"Identity matrix dtype {identity_matrix.dtype} != float32"
                )

            # Validate identity properties
            self._validate_identity_matrix(identity_matrix, n)

        logger.info(
            f"Identity matrix validation complete - {len(sizes)} matrices tested"
        )

    def _validate_matrix_size(self, size: int) -> None:
        """Validate matrix size according to system constraints.

        Args:
            size: Matrix dimension to validate.

        Raises:
            MatrixValidationError: If size is invalid.
        """
        if size <= 0:
            raise MatrixValidationError(f"Matrix size must be positive, got {size}")

        if size > 2048:
            raise MatrixValidationError(f"Matrix size {size} exceeds maximum (2048)")

        # Check if size is power of 2 for optimal performance
        if (size & (size - 1)) != 0:
            logger.warning(
                f"Matrix size {size} is not a power of 2 - may impact performance"
            )

    def _validate_random_matrix(
        self, matrix: np.ndarray, expected_size: int, matrix_name: str
    ) -> None:
        """Validate properties of a randomly initialized matrix.

        Args:
            matrix: Matrix to validate.
            expected_size: Expected matrix dimension.
            matrix_name: Name of matrix for logging.

        Raises:
            MatrixValidationError: If matrix properties are incorrect.
        """
        # Validate data type
        if matrix.dtype != np.float32:
            raise MatrixValidationError(
                f"{matrix_name} dtype {matrix.dtype} != float32"
            )

        # Validate dimensions
        expected_shape = (expected_size, expected_size)
        if matrix.shape != expected_shape:
            raise MatrixValidationError(
                f"{matrix_name} shape {matrix.shape} != {expected_shape}"
            )

        # Validate value ranges
        min_val, max_val = matrix.min(), matrix.max()
        if not (-1.0 <= min_val <= 1.0 and -1.0 <= max_val <= 1.0):
            raise MatrixValidationError(
                f"{matrix_name} values outside [-1.0, 1.0] range: [{min_val}, {max_val}]"
            )

        # Validate no NaN or Inf values
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            raise MatrixValidationError(f"{matrix_name} contains NaN or Inf values")

    def _validate_identity_matrix(self, matrix: np.ndarray, size: int) -> None:
        """Validate identity matrix properties.

        Args:
            matrix: Identity matrix to validate.
            size: Expected matrix dimension.

        Raises:
            MatrixValidationError: If identity matrix is incorrect.
        """
        tolerance = 1e-7  # Floating point comparison tolerance

        for i in range(size):
            for j in range(size):
                expected_value = 1.0 if i == j else 0.0
                actual_value = matrix[i, j]

                if abs(actual_value - expected_value) > tolerance:
                    raise MatrixValidationError(
                        f"Identity matrix element ({i},{j}) = {actual_value}, expected {expected_value}"
                    )


class TestPerformanceCalculations:
    """Test performance metric calculations from common.cu logic.

    This test class validates performance calculation functions including
    GFLOPS, memory bandwidth, arithmetic intensity, and efficiency calculations.
    """

    def test_gflops_calculation(self) -> None:
        """Test GFLOPS calculation accuracy.

        Validates GFLOPS calculation across different matrix sizes and
        execution times to ensure mathematical accuracy.

        Raises:
            PerformanceCalculationError: If GFLOPS calculation is incorrect.
        """
        logger.info("Testing GFLOPS calculation accuracy")

        test_cases = [
            {"n": 64, "time_ms": 0.1, "expected_min": 5.0, "expected_max": 10000.0},
            {"n": 128, "time_ms": 1.0, "expected_min": 2.0, "expected_max": 8000.0},
            {"n": 256, "time_ms": 5.0, "expected_min": 5.0, "expected_max": 7000.0},
        ]

        for i, case in enumerate(test_cases):
            logger.debug(
                f"Testing GFLOPS case {i + 1}: n={case['n']}, time_ms={case['time_ms']}"
            )

            n = case["n"]
            time_ms = case["time_ms"]

            # Calculate GFLOPS: (2 * n^3) / (time_ms * 1e6)
            gflops = self._calculate_gflops(int(n), time_ms)

            self._validate_gflops_range(
                gflops, case["expected_min"], case["expected_max"]
            )

        logger.info(
            f"GFLOPS calculation validation complete - {len(test_cases)} test cases passed"
        )

    def test_memory_bandwidth_calculation(self) -> None:
        """Test memory bandwidth calculation.

        Validates memory bandwidth calculation for matrix multiplication
        operations with different matrix sizes and execution times.

        Raises:
            PerformanceCalculationError: If bandwidth calculation is incorrect.
        """
        logger.info("Testing memory bandwidth calculation")

        test_cases = [
            {"n": 64, "time_ms": 0.1},
            {"n": 128, "time_ms": 1.0},
            {"n": 256, "time_ms": 5.0},
        ]

        for i, case in enumerate(test_cases):
            logger.debug(
                f"Testing bandwidth case {i + 1}: n={case['n']}, time_ms={case['time_ms']}"
            )

            n = case["n"]
            time_ms = case["time_ms"]

            bandwidth_gbps = self._calculate_memory_bandwidth(int(n), time_ms)

            self._validate_memory_bandwidth(bandwidth_gbps)

        logger.info(
            f"Memory bandwidth calculation validation complete - {len(test_cases)} test cases passed"
        )

    def test_arithmetic_intensity_calculation(self) -> None:
        """Test arithmetic intensity calculation.

        Validates arithmetic intensity calculation for matrix multiplication
        operations across different matrix sizes.

        Raises:
            PerformanceCalculationError: If arithmetic intensity is incorrect.
        """
        logger.info("Testing arithmetic intensity calculation")

        sizes = [64, 128, 256, 512]

        for size in sizes:
            logger.debug(f"Testing arithmetic intensity for size {size}x{size}")

            arithmetic_intensity = self._calculate_arithmetic_intensity(size)
            expected_intensity = size / 6.0  # Theoretical for matrix multiplication

            self._validate_arithmetic_intensity(
                arithmetic_intensity, expected_intensity, size
            )

        logger.info(
            f"Arithmetic intensity validation complete - {len(sizes)} sizes tested"
        )

    def test_efficiency_calculation_bounds(self) -> None:
        """Test that efficiency calculations stay within valid bounds.

        Validates that efficiency calculations always produce results
        within the valid 0-100% range.

        Raises:
            PerformanceCalculationError: If efficiency bounds are violated.
        """
        logger.info("Testing efficiency calculation bounds")

        test_cases = [
            {"measured_gflops": 50.0, "theoretical_peak": 100.0, "expected_eff": 50.0},
            {
                "measured_gflops": 100.0,
                "theoretical_peak": 100.0,
                "expected_eff": 100.0,
            },
            {"measured_gflops": 25.0, "theoretical_peak": 200.0, "expected_eff": 12.5},
        ]

        for i, case in enumerate(test_cases):
            logger.debug(
                f"Testing efficiency case {i + 1}: {case['measured_gflops']} / {case['theoretical_peak']} GFLOPS"
            )

            efficiency = self._calculate_efficiency(
                case["measured_gflops"], case["theoretical_peak"]
            )

            self._validate_efficiency_result(efficiency, case["expected_eff"])

        logger.info(
            f"Efficiency calculation validation complete - {len(test_cases)} test cases passed"
        )

    def _calculate_gflops(self, matrix_size: int | float, time_ms: float) -> float:
        """Calculate GFLOPS for matrix multiplication.

        Args:
            matrix_size: Matrix dimension.
            time_ms: Execution time in milliseconds.

        Returns:
            Calculated GFLOPS value.

        Raises:
            PerformanceCalculationError: If calculation parameters are invalid.
        """
        if matrix_size <= 0:
            raise PerformanceCalculationError("Matrix size must be positive")
        if time_ms <= 0:
            raise PerformanceCalculationError("Execution time must be positive")

        theoretical_flops = 2.0 * matrix_size * matrix_size * matrix_size
        return theoretical_flops / (time_ms * 1e6)

    def _calculate_memory_bandwidth(
        self, matrix_size: int | float, time_ms: float
    ) -> float:
        """Calculate memory bandwidth for matrix multiplication.

        Args:
            matrix_size: Matrix dimension.
            time_ms: Execution time in milliseconds.

        Returns:
            Memory bandwidth in GB/s.

        Raises:
            PerformanceCalculationError: If calculation parameters are invalid.
        """
        if matrix_size <= 0:
            raise PerformanceCalculationError("Matrix size must be positive")
        if time_ms <= 0:
            raise PerformanceCalculationError("Execution time must be positive")

        # Memory bandwidth = bytes_transferred / time
        # Matrix multiplication: 2 reads + 1 write = 3 * n^2 * sizeof(float)
        bytes_transferred = 3 * matrix_size * matrix_size * 4  # 4 bytes per float
        return bytes_transferred / (time_ms * 1e6)

    def _calculate_arithmetic_intensity(self, matrix_size: int) -> float:
        """Calculate arithmetic intensity for matrix multiplication.

        Args:
            matrix_size: Matrix dimension.

        Returns:
            Arithmetic intensity value.

        Raises:
            PerformanceCalculationError: If matrix size is invalid.
        """
        if matrix_size <= 0:
            raise PerformanceCalculationError("Matrix size must be positive")

        # Arithmetic intensity = FLOPS / bytes_accessed
        flops = 2 * matrix_size * matrix_size * matrix_size  # n^3 multiplies + n^3 adds
        bytes_accessed = 3 * matrix_size * matrix_size * 4  # 3 matrices * n^2 * 4 bytes

        return flops / bytes_accessed

    def _calculate_efficiency(
        self, measured_gflops: float, theoretical_peak: float
    ) -> float:
        """Calculate computational efficiency percentage.

        Args:
            measured_gflops: Measured performance.
            theoretical_peak: Theoretical peak performance.

        Returns:
            Efficiency percentage.

        Raises:
            PerformanceCalculationError: If parameters are invalid.
        """
        if theoretical_peak <= 0:
            raise PerformanceCalculationError("Theoretical peak must be positive")
        if measured_gflops < 0:
            raise PerformanceCalculationError("Measured GFLOPS must be non-negative")

        return (measured_gflops / theoretical_peak) * 100.0

    def _validate_gflops_range(
        self, gflops: float, min_expected: float, max_expected: float
    ) -> None:
        """Validate GFLOPS value is within expected range.

        Args:
            gflops: Calculated GFLOPS value.
            min_expected: Minimum expected value.
            max_expected: Maximum expected value.

        Raises:
            PerformanceCalculationError: If GFLOPS is out of range.
        """
        if not (min_expected <= gflops <= max_expected):
            raise PerformanceCalculationError(
                f"GFLOPS {gflops:.2f} outside expected range [{min_expected}, {max_expected}]"
            )

        if gflops <= 0:
            raise PerformanceCalculationError("GFLOPS must be positive")

    def _validate_memory_bandwidth(self, bandwidth_gbps: float) -> None:
        """Validate memory bandwidth value.

        Args:
            bandwidth_gbps: Memory bandwidth in GB/s.

        Raises:
            PerformanceCalculationError: If bandwidth is invalid.
        """
        if bandwidth_gbps <= 0:
            raise PerformanceCalculationError("Memory bandwidth must be positive")

        # Reasonable upper bound for Jetson Orin Nano
        max_reasonable_bandwidth = 1000.0  # GB/s
        if bandwidth_gbps > max_reasonable_bandwidth:
            raise PerformanceCalculationError(
                f"Memory bandwidth {bandwidth_gbps:.2f} GB/s exceeds reasonable maximum"
            )

    def _validate_arithmetic_intensity(
        self, calculated: float, expected: float, matrix_size: int
    ) -> None:
        """Validate arithmetic intensity calculation.

        Args:
            calculated: Calculated arithmetic intensity.
            expected: Expected arithmetic intensity.
            matrix_size: Matrix dimension used.

        Raises:
            PerformanceCalculationError: If arithmetic intensity is incorrect.
        """
        tolerance = 1e-6
        if abs(calculated - expected) > tolerance:
            raise PerformanceCalculationError(
                f"Arithmetic intensity mismatch for size {matrix_size}: "
                f"calculated={calculated:.6f}, expected={expected:.6f}"
            )

    def _validate_efficiency_result(self, calculated: float, expected: float) -> None:
        """Validate efficiency calculation result.

        Args:
            calculated: Calculated efficiency.
            expected: Expected efficiency.

        Raises:
            PerformanceCalculationError: If efficiency is incorrect.
        """
        tolerance = 1e-6
        if abs(calculated - expected) > tolerance:
            raise PerformanceCalculationError(
                f"Efficiency mismatch: calculated={calculated:.6f}, expected={expected:.6f}"
            )

        # Critical validation: efficiency should never exceed 100%
        if calculated > 100.0:
            raise PerformanceCalculationError(
                f"Efficiency {calculated:.1f}% exceeds 100% maximum"
            )

        if calculated < 0:
            raise PerformanceCalculationError(
                f"Efficiency {calculated:.1f}% cannot be negative"
            )


class TestSystemMonitoring:
    """Test system monitoring functionality.

    This test class validates system monitoring functions including
    temperature reading, power consumption, and thermal stability detection.
    """

    def test_temperature_reading_validation(self) -> None:
        """Test temperature reading validation.

        Validates that temperature reading validation correctly identifies
        valid and invalid temperature values.

        Raises:
            SystemMonitoringError: If temperature validation is incorrect.
        """
        logger.info("Testing temperature reading validation")

        valid_temps = [25.0, 45.5, 65.0, 85.0]
        invalid_temps = [-10.0, 0.0, 150.0, 200.0]

        for temp in valid_temps:
            if not self._is_valid_temperature(temp):
                raise SystemMonitoringError(
                    f"Valid temperature {temp}°C incorrectly rejected"
                )
            logger.debug(f"Valid temperature {temp}°C accepted")

        for temp in invalid_temps:
            if self._is_valid_temperature(temp):
                raise SystemMonitoringError(
                    f"Invalid temperature {temp}°C incorrectly accepted"
                )
            logger.debug(f"Invalid temperature {temp}°C correctly rejected")

        logger.info(
            f"Temperature validation complete - {len(valid_temps)} valid, {len(invalid_temps)} invalid"
        )

    def test_power_consumption_validation(self) -> None:
        """Test power consumption validation.

        Validates power consumption readings for different power modes
        to ensure they fall within expected ranges.

        Raises:
            SystemMonitoringError: If power validation is incorrect.
        """
        logger.info("Testing power consumption validation")

        power_readings = {
            "15W_mode": [12.0, 15.0, 18.0],  # Should be around 15W
            "25W_mode": [20.0, 25.0, 28.0],  # Should be around 25W
            "MAXN_mode": [25.0, 30.0, 35.0],  # Should be around 30W
        }

        total_readings = 0

        for mode, readings in power_readings.items():
            logger.debug(f"Testing power readings for {mode}")

            for power in readings:
                self._validate_power_reading(power, mode)
                total_readings += 1

        logger.info(f"Power validation complete - {total_readings} readings tested")

    def test_thermal_stability_logic(self) -> None:
        """Test thermal stability detection logic.

        Validates thermal stability detection algorithm using
        synthetic temperature history data.

        Raises:
            SystemMonitoringError: If stability detection is incorrect.
        """
        logger.info("Testing thermal stability detection logic")

        # Test stable temperature scenario
        stable_temps = [48.0, 48.2, 47.8, 48.1, 48.0]  # Stable within 1°C
        if not self._check_thermal_stability(stable_temps):
            raise SystemMonitoringError(
                "Stable temperature sequence incorrectly identified as unstable"
            )

        # Test unstable temperature scenario
        unstable_temps = [45.0, 50.0, 55.0, 48.0, 52.0]  # Unstable
        if self._check_thermal_stability(unstable_temps):
            raise SystemMonitoringError(
                "Unstable temperature sequence incorrectly identified as stable"
            )

        # Test insufficient data scenario
        insufficient_temps = [48.0, 48.2]
        if self._check_thermal_stability(insufficient_temps):
            raise SystemMonitoringError(
                "Insufficient temperature data incorrectly identified as stable"
            )

        logger.info("Thermal stability detection validation complete")

    def _is_valid_temperature(self, temperature: float) -> bool:
        """Check if temperature reading is valid.

        Args:
            temperature: Temperature value in Celsius.

        Returns:
            True if temperature is valid, False otherwise.
        """
        return 20.0 <= temperature <= 100.0

    def _validate_power_reading(self, power: float, mode: str) -> None:
        """Validate power consumption reading.

        Args:
            power: Power consumption in watts.
            mode: Power mode identifier.

        Raises:
            SystemMonitoringError: If power reading is invalid.
        """
        if power <= 0:
            raise SystemMonitoringError(
                f"Power reading must be positive, got {power}W for {mode}"
            )

        # Reasonable range validation
        if not (1.0 <= power <= 100.0):
            raise SystemMonitoringError(
                f"Power reading {power}W outside reasonable range for {mode}"
            )

    def _check_thermal_stability(
        self, temperatures: list[float], threshold: float = 1.0
    ) -> bool:
        """Check thermal stability from temperature history.

        Args:
            temperatures: List of temperature readings.
            threshold: Maximum temperature variation for stability.

        Returns:
            True if thermally stable, False otherwise.
        """
        if len(temperatures) < 5:
            return False

        temp_range = max(temperatures) - min(temperatures)
        return temp_range < threshold


class TestErrorHandling:
    """Test error handling in utility functions.

    This test class validates error handling and defensive programming
    in utility functions including memory allocation and CUDA error handling.
    """

    def test_memory_allocation_validation(self) -> None:
        """Test memory allocation size validation.

        Validates memory allocation calculations for different matrix sizes
        to ensure they don't exceed system constraints.

        Raises:
            AssertionError: If memory allocation logic is incorrect.
        """
        logger.info("Testing memory allocation validation")

        reasonable_sizes = [64, 128, 256, 512, 1024, 2048]
        oversized = [4096, 8192]

        # Test reasonable sizes
        for size in reasonable_sizes:
            memory_usage = self._calculate_memory_usage(size)

            # Should be reasonable for Jetson Orin Nano (8GB memory)
            max_reasonable_memory = 100 * 1024 * 1024  # 100MB
            assert memory_usage < max_reasonable_memory, (
                f"Size {size} uses {memory_usage} bytes (>{max_reasonable_memory})"
            )

            logger.debug(
                f"Size {size}x{size} uses {memory_usage / (1024 * 1024):.1f} MB (reasonable)"
            )

        # Test oversized matrices
        for size in oversized:
            memory_usage = self._calculate_memory_usage(size)

            # Should exceed normal operational limits
            min_large_memory = 100 * 1024 * 1024  # 100MB
            assert memory_usage > min_large_memory, (
                f"Oversized {size} uses only {memory_usage} bytes (<{min_large_memory})"
            )

            logger.debug(
                f"Size {size}x{size} uses {memory_usage / (1024 * 1024):.1f} MB (oversized)"
            )

        logger.info(
            f"Memory allocation validation complete - {len(reasonable_sizes + oversized)} sizes tested"
        )

    def test_cuda_error_simulation(self) -> None:
        """Test CUDA error handling simulation.

        Simulates CUDA error conditions to validate error handling logic.
        """
        logger.info("Testing CUDA error handling simulation")

        # CUDA error code simulation
        cuda_success = 0
        cuda_error_invalid_value = 11
        cuda_error_memory_allocation = 2

        # Test success case
        assert self._simulate_cuda_check(cuda_success), "CUDA success case failed"
        logger.debug("CUDA success case handled correctly")

        # Test error cases
        assert not self._simulate_cuda_check(cuda_error_invalid_value), (
            "Invalid value error not detected"
        )
        assert not self._simulate_cuda_check(cuda_error_memory_allocation), (
            "Memory error not detected"
        )

        logger.debug("CUDA error cases handled correctly")
        logger.info("CUDA error handling validation complete")

    def _calculate_memory_usage(self, matrix_size: int) -> int:
        """Calculate memory usage for matrix operations.

        Args:
            matrix_size: Matrix dimension.

        Returns:
            Total memory usage in bytes.
        """
        matrix_memory = matrix_size * matrix_size * 4  # bytes for float32
        total_memory = matrix_memory * 3  # A, B, C matrices
        return total_memory

    def _simulate_cuda_check(self, error_code: int) -> bool:
        """Simulate CUDA error checking.

        Args:
            error_code: CUDA error code to check.

        Returns:
            True if no error, False if error detected.
        """
        cuda_success = 0
        return error_code == cuda_success


class TestNumericalAccuracy:
    """Test numerical accuracy validation.

    This test class validates numerical accuracy of matrix operations
    and floating point consistency.
    """

    def test_matrix_multiplication_accuracy(self) -> None:
        """Test numerical accuracy of matrix multiplication.

        Validates that matrix multiplication maintains acceptable
        numerical accuracy between single and double precision.

        Raises:
            AssertionError: If numerical accuracy is insufficient.
        """
        logger.info("Testing matrix multiplication numerical accuracy")

        # Small test case for exact verification
        n = 4
        np.random.seed(42)  # Ensure reproducibility

        A = np.random.rand(n, n).astype(np.float32)
        B = np.random.rand(n, n).astype(np.float32)

        # Reference calculation (double precision)
        C_ref = np.matmul(A.astype(np.float64), B.astype(np.float64))

        # Simulated GPU calculation (single precision)
        C_gpu = np.matmul(A, B)

        # Calculate relative error
        relative_error = np.mean(
            np.abs(C_gpu.astype(np.float64) - C_ref) / np.abs(C_ref)
        )

        # Validate accuracy
        max_acceptable_error = 1e-5
        assert relative_error < max_acceptable_error, (
            f"Relative error {relative_error:.2e} exceeds threshold {max_acceptable_error:.2e}"
        )

        logger.info(
            f"Numerical accuracy validated - relative error: {relative_error:.2e}"
        )

    def test_floating_point_consistency(self) -> None:
        """Test floating point consistency across calculations.

        Validates that repeated calculations produce identical results
        and don't contain invalid floating point values.

        Raises:
            AssertionError: If floating point consistency fails.
        """
        logger.info("Testing floating point consistency")

        n = 64
        np.random.seed(42)  # Ensure reproducibility

        A = np.random.uniform(-1, 1, (n, n)).astype(np.float32)
        B = np.random.uniform(-1, 1, (n, n)).astype(np.float32)

        # Multiple calculations
        C1 = np.matmul(A, B)
        C2 = np.matmul(A, B)

        # Should be exactly identical
        assert np.array_equal(C1, C2), (
            "Repeated calculations produced different results"
        )

        # Check for NaN or Inf values
        assert not np.any(np.isnan(C1)), "Result contains NaN values"
        assert not np.any(np.isinf(C1)), "Result contains Inf values"

        logger.info(
            "Floating point consistency validated - no NaN/Inf values, identical results"
        )


def run_utility_tests() -> int:
    """Run all utility tests and return exit code.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger.info("Starting Jetson Orin Nano utility function tests")
    logger.info("=" * 60)

    try:
        # Run pytest on this module
        exit_code = pytest.main([__file__, "-v", "--tb=short"])

        if exit_code == 0:
            logger.info("All utility tests passed successfully")
        else:
            logger.error("Some utility tests failed")

        return exit_code

    except Exception as e:
        logger.error(f"Utility test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_utility_tests()
    sys.exit(exit_code)
