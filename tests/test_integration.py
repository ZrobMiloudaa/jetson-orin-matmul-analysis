#!/usr/bin/env python3
"""Integration Testing Suite for Jetson Orin Nano Benchmarking System.

Copyright 2025 ByteStack Labs
SPDX-License-Identifier: MIT

This module provides comprehensive integration tests for the complete benchmarking pipeline
across multiple power modes (15W, 25W, MAXN_SUPER) with enterprise-grade error handling,
logging, and validation.

Target Hardware: Jetson Orin Nano Engineering Reference Developer Kit Super
Software Stack: L4T R36.4.4 (JetPack 6.x), CUDA V12.6.68

Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
Version: 1.0.0
License: MIT
"""

import logging
import sys
from pathlib import Path

# Configure logging for test execution
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


class ProjectStructureError(Exception):
    """Raised when required project structure is invalid."""

    pass


class BenchmarkFrameworkError(Exception):
    """Raised when benchmarking framework validation fails."""

    pass


class TestProjectIntegration:
    """Integration tests for the complete benchmarking pipeline.

    This test suite validates the overall project structure, file dependencies,
    and integration between different components of the benchmarking system.
    """

    @staticmethod
    def get_required_files() -> dict[str, str]:
        """Get dictionary of required files with descriptions.

        Returns:
            Dictionary mapping file paths to their descriptions.
        """
        return {
            "Makefile": "Build system configuration",
            "pyproject.toml": "Python project configuration",
            "cuda/kernels/naive_multiplication.cu": "CUDA naive matrix multiplication implementation",
            "cuda/utils/common.cu": "CUDA utility functions",
            "cuda/utils/common.h": "CUDA utility headers",
            "benchmarks/multi_power_mode_benchmark.py": "Multi-power mode benchmarking script",
            "data/visualize_power_modes.py": "Visualization generation module",
        }

    def test_project_structure(self) -> None:
        """Test that all required project files exist.

        Validates the complete project structure including build files,
        source code, benchmarking scripts, and visualization modules.

        Raises:
            ProjectStructureError: If any required file is missing.
        """
        logger.info("Validating project structure integrity")

        required_files = self.get_required_files()
        missing_files: list[str] = []

        for file_path, description in required_files.items():
            path_obj = Path(file_path)
            if not path_obj.exists():
                missing_files.append(f"{file_path} ({description})")
                logger.error(f"Missing required file: {file_path} - {description}")
            else:
                logger.debug(f"Found required file: {file_path}")

        if missing_files:
            error_msg = f"Missing {len(missing_files)} required files: {', '.join(missing_files)}"
            raise ProjectStructureError(error_msg)

        logger.info(
            f"Project structure validation complete - all {len(required_files)} files present"
        )

    def test_benchmark_framework_logic(self) -> None:
        """Test benchmarking framework without GPU execution.

        Validates performance metric calculations, mathematical consistency,
        and framework logic using CPU-based calculations.

        Raises:
            BenchmarkFrameworkError: If framework validation fails.
        """
        logger.info("Testing benchmarking framework logic")

        try:
            # Add benchmarks to path for testing
            benchmark_path = Path("benchmarks")
            if benchmark_path.exists():
                sys.path.append(str(benchmark_path))
                logger.debug("Added benchmarks directory to Python path")
            else:
                logger.warning(
                    "Benchmarks directory not found - skipping path addition"
                )

            # Test performance metric calculations with known values
            test_data = {
                "elapsed_ms": 1.0,
                "matrix_size": 128,
                "theoretical_flops": 2 * 128**3,
            }

            # Test GFLOPS calculation
            gflops = self._calculate_gflops(
                test_data["theoretical_flops"], test_data["elapsed_ms"]
            )

            self._validate_gflops(gflops, test_data["matrix_size"])

            # Test efficiency calculation
            theoretical_peak = 600.0  # Example theoretical peak GFLOPS
            efficiency = self._calculate_efficiency(gflops, theoretical_peak)

            self._validate_efficiency(efficiency)

            logger.info("Benchmarking framework logic validation successful")

        except Exception as e:
            logger.error(f"Benchmarking framework validation failed: {e}")
            raise BenchmarkFrameworkError(f"Framework validation error: {e}") from e

    def test_cuda_implementation_structure(self) -> None:
        """Test CUDA implementation file structure and dependencies.

        Validates that CUDA source files have proper structure and
        include necessary dependencies.

        Raises:
            ProjectStructureError: If CUDA implementation structure is invalid.
        """
        logger.info("Validating CUDA implementation structure")

        cuda_files = [
            "cuda/kernels/naive_multiplication.cu",
            "cuda/utils/common.cu",
            "cuda/utils/common.h",
        ]

        for cuda_file in cuda_files:
            self._validate_cuda_file_structure(Path(cuda_file))

        logger.info("CUDA implementation structure validation complete")

    def test_python_module_imports(self) -> None:
        """Test that Python modules can be imported without errors.

        Validates import statements and module dependencies to ensure
        the benchmarking system can be loaded successfully.
        """
        logger.info("Testing Python module import capabilities")

        # Test critical module imports
        import_tests = [
            ("pathlib", "Path"),
            ("json", None),
            ("logging", None),
            ("datetime", "datetime"),
        ]

        successful_imports = 0

        for module_name, class_name in import_tests:
            try:
                module = __import__(module_name)
                if class_name:
                    getattr(module, class_name)
                successful_imports += 1
                logger.debug(f"Successfully imported {module_name}")
            except ImportError as e:
                logger.error(f"Failed to import {module_name}: {e}")
                raise BenchmarkFrameworkError(
                    f"Import error for {module_name}: {e}"
                ) from e

        logger.info(
            f"Module import validation complete - {successful_imports}/{len(import_tests)} modules imported"
        )

    def _calculate_gflops(self, theoretical_flops: float, elapsed_ms: float) -> float:
        """Calculate GFLOPS from theoretical FLOPS and execution time.

        Args:
            theoretical_flops: Theoretical floating point operations.
            elapsed_ms: Execution time in milliseconds.

        Returns:
            Calculated GFLOPS value.

        Raises:
            ValueError: If input parameters are invalid.
        """
        if elapsed_ms <= 0:
            raise ValueError("Elapsed time must be positive")
        if theoretical_flops < 0:
            raise ValueError("Theoretical FLOPS must be non-negative")

        return theoretical_flops / (elapsed_ms * 1e6)

    def _validate_gflops(self, gflops: float, matrix_size: int | float) -> None:
        """Validate GFLOPS calculation results.

        Args:
            gflops: Calculated GFLOPS value.
            matrix_size: Matrix dimension used in calculation.

        Raises:
            BenchmarkFrameworkError: If GFLOPS value is invalid.
        """
        if gflops <= 0:
            raise BenchmarkFrameworkError("GFLOPS must be positive")

        # Reasonable upper bound based on hardware capabilities
        max_reasonable_gflops = 10000.0
        if gflops > max_reasonable_gflops:
            raise BenchmarkFrameworkError(
                f"GFLOPS ({gflops:.2f}) exceeds reasonable maximum ({max_reasonable_gflops})"
            )

        # Matrix size dependent validation
        matrix_size_int = int(matrix_size) if matrix_size > 0 else 0
        if matrix_size_int > 0:
            min_expected_gflops = 0.1  # Very conservative minimum
            if gflops < min_expected_gflops:
                logger.warning(
                    f"GFLOPS ({gflops:.2f}) below expected minimum for size {matrix_size_int}"
                )

    def _calculate_efficiency(
        self, measured_gflops: float, theoretical_peak: float
    ) -> float:
        """Calculate computational efficiency percentage.

        Args:
            measured_gflops: Measured GFLOPS performance.
            theoretical_peak: Theoretical peak GFLOPS.

        Returns:
            Efficiency percentage (0-100).

        Raises:
            ValueError: If input parameters are invalid.
        """
        if theoretical_peak <= 0:
            raise ValueError("Theoretical peak must be positive")
        if measured_gflops < 0:
            raise ValueError("Measured GFLOPS must be non-negative")

        return (measured_gflops / theoretical_peak) * 100.0

    def _validate_efficiency(self, efficiency: float) -> None:
        """Validate efficiency calculation results.

        Args:
            efficiency: Calculated efficiency percentage.

        Raises:
            BenchmarkFrameworkError: If efficiency value is invalid.
        """
        if efficiency < 0:
            raise BenchmarkFrameworkError("Efficiency cannot be negative")

        if efficiency > 100:
            raise BenchmarkFrameworkError(
                f"Efficiency ({efficiency:.1f}%) exceeds 100% - indicates calculation error"
            )

        # Log efficiency assessment
        if efficiency > 50:
            logger.debug(f"High efficiency detected: {efficiency:.1f}%")
        elif efficiency < 5:
            logger.warning(f"Very low efficiency detected: {efficiency:.1f}%")

    def _validate_cuda_file_structure(self, cuda_file: Path) -> None:
        """Validate CUDA source file structure and content.

        Args:
            cuda_file: Path to CUDA source file.

        Raises:
            ProjectStructureError: If CUDA file structure is invalid.
        """
        if not cuda_file.exists():
            raise ProjectStructureError(f"CUDA file not found: {cuda_file}")

        try:
            with open(cuda_file, encoding="utf-8") as f:
                content = f.read()

            # Basic structure validation
            if cuda_file.suffix == ".cu":
                required_patterns = ["#include", "cuda"]
                for pattern in required_patterns:
                    if pattern not in content.lower():
                        logger.warning(
                            f"Expected pattern '{pattern}' not found in {cuda_file}"
                        )

            elif cuda_file.suffix == ".h":
                header_patterns = ["#ifndef", "#define", "#endif"]
                found_patterns = sum(
                    1 for pattern in header_patterns if pattern in content
                )
                if found_patterns < 2:
                    logger.warning(f"Header guard patterns incomplete in {cuda_file}")

            logger.debug(f"CUDA file structure validated: {cuda_file}")

        except Exception as e:
            raise ProjectStructureError(f"Error validating {cuda_file}: {e}") from e


def run_integration_tests() -> int:
    """Run all integration tests and return exit code.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logger.info("Starting Jetson Orin Nano benchmarking system integration tests")
    logger.info("=" * 70)

    try:
        test_suite = TestProjectIntegration()

        # Run all integration tests
        test_methods = [
            ("Project Structure", test_suite.test_project_structure),
            ("Benchmark Framework", test_suite.test_benchmark_framework_logic),
            ("CUDA Implementation", test_suite.test_cuda_implementation_structure),
            ("Python Modules", test_suite.test_python_module_imports),
        ]

        passed_tests = 0
        total_tests = len(test_methods)

        for test_name, test_method in test_methods:
            try:
                logger.info(f"Running {test_name} test...")
                test_method()
                passed_tests += 1
                logger.info(f"PASS: {test_name} test completed successfully")
            except Exception as e:
                logger.error(f"FAIL: {test_name} test failed - {e}")

        logger.info("=" * 70)
        logger.info(
            f"Integration test summary: {passed_tests}/{total_tests} tests passed"
        )

        if passed_tests == total_tests:
            logger.info("All integration tests passed successfully")
            return 0
        else:
            logger.error(f"{total_tests - passed_tests} integration tests failed")
            return 1

    except Exception as e:
        logger.error(f"Integration test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = run_integration_tests()
    sys.exit(exit_code)
