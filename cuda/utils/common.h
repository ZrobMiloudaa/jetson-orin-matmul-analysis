/**
 * @file common.h
 * @brief Common utilities and constants for Jetson Orin Nano matrix multiplication analysis
 * @author Jesse Moses (@Cre4T3Tiv3)
 * @date 2025
 * @copyright ByteStack Labs - MIT License
 *
 * Provides shared functionality, hardware specifications, and performance
 * analysis utilities for CUDA matrix multiplication benchmarks on the
 * NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super.
 *
 * Target Hardware:
 * - NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
 * - CUDA Compute Capability: 8.7 (Ampere Architecture)
 * - Power Modes: 15W, 25W, MAXN SUPER
 * - Memory: 7.4GB LPDDR5 @ 68 GB/s theoretical bandwidth
 *
 * Key Features:
 * - Enterprise-grade error handling with exception safety
 * - Comprehensive performance analysis and system monitoring
 * - Multi-power mode characterization support
 * - Structured logging integration throughout
 */

#ifndef COMMON_H
#define COMMON_H

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

#include "logger.h"

int detect_current_power_mode();

/**
 * @brief CUDA error checking with comprehensive logging support
 * @param call CUDA runtime API call to validate
 *
 * Validates CUDA API calls and logs detailed error information using
 * the enterprise logging system. Throws std::runtime_error on failures
 * to enable proper exception handling rather than terminating the program.
 *
 * @throws std::runtime_error if CUDA call fails
 */
#define CUDA_CHECK(call)                                                                       \
  do {                                                                                         \
    cudaError_t err = call;                                                                    \
    if (err != cudaSuccess) {                                                                  \
      LOG_CRITICAL_F("CUDA Error: %s (code: %d) at %s:%d in %s", cudaGetErrorString(err), err, \
                     __FILE__, __LINE__, __FUNCTION__);                                        \
      PRINT_ERROR_F("CUDA operation failed: %s", cudaGetErrorString(err));                     \
      throw std::runtime_error("CUDA operation failed");                                       \
    }                                                                                          \
  } while (0)

/**
 * @brief cuBLAS error checking with comprehensive logging support
 * @param call cuBLAS API call to validate
 *
 * Validates cuBLAS operations and provides detailed error reporting
 * through the logging system. Throws exceptions instead of terminating
 * to enable graceful error recovery.
 *
 * @throws std::runtime_error if cuBLAS call fails
 */
#define CUBLAS_CHECK(call)                                                                  \
  do {                                                                                      \
    cublasStatus_t stat = call;                                                             \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                                    \
      LOG_CRITICAL_F("cuBLAS Error: status code %d at %s:%d in %s", static_cast<int>(stat), \
                     __FILE__, __LINE__, __FUNCTION__);                                     \
      PRINT_ERROR_F("cuBLAS operation failed with status code %d", static_cast<int>(stat)); \
      throw std::runtime_error("cuBLAS operation failed");                                  \
    }                                                                                       \
  } while (0)

/**
 * @brief Safe memory allocation with comprehensive error checking
 * @param ptr Pointer variable to receive allocated memory
 * @param size Size in bytes to allocate
 * @param description Human-readable description for logging
 *
 * Performs memory allocation with detailed logging and exception-based
 * error handling. Logs allocation success and throws on failure.
 *
 * @throws std::bad_alloc if allocation fails
 */
#define SAFE_MALLOC(ptr, size, description)                                                \
  do {                                                                                     \
    ptr = malloc(size);                                                                    \
    if (!ptr) {                                                                            \
      LOG_CRITICAL_F("Memory allocation failed: %s (size: %zu bytes)", description, size); \
      PRINT_ERROR_F("Memory allocation failed for %s", description);                       \
      throw std::bad_alloc();                                                              \
    }                                                                                      \
    LOG_DEBUG_F("Allocated %zu bytes for %s", size, description);                          \
  } while (0)

/**
 * @brief Performance metrics structure for comprehensive power mode analysis
 *
 * Contains detailed timing, throughput, efficiency, and system monitoring
 * data for matrix multiplication benchmarks across different power modes.
 * Designed for enterprise-grade performance analysis and reporting.
 */
struct PerformanceMetrics {
  double elapsed_time_ms;                 ///< Kernel execution time in milliseconds
  double theoretical_flops;               ///< Theoretical floating-point operations
  double measured_gflops;                 ///< Measured performance in GFLOPS
  double memory_bandwidth_gbps;           ///< Achieved memory bandwidth in GB/s
  double arithmetic_intensity;            ///< FLOPS per byte ratio (roofline analysis)
  double power_watts;                     ///< System power consumption in watts
  double temperature_celsius;             ///< GPU/SoC temperature in Celsius
  size_t bytes_transferred;               ///< Total bytes transferred to/from GPU
  double efficiency_percent;              ///< Efficiency relative to theoretical peak
  int power_mode_id;                      ///< Active power mode identifier
  const char* power_mode_name;            ///< Human-readable power mode name
  double peak_memory_bandwidth_achieved;  ///< Memory bandwidth utilization percentage
  double compute_utilization_percent;     ///< Compute resource utilization percentage

  /**
   * @brief Default constructor initializing all fields to safe defaults
   *
   * Ensures all metrics are initialized to valid values preventing
   * undefined behavior and making debugging easier.
   */
  PerformanceMetrics()
      : elapsed_time_ms(0.0),
        theoretical_flops(0.0),
        measured_gflops(0.0),
        memory_bandwidth_gbps(0.0),
        arithmetic_intensity(0.0),
        power_watts(0.0),
        temperature_celsius(0.0),
        bytes_transferred(0),
        efficiency_percent(0.0),
        power_mode_id(-1),
        power_mode_name("Unknown"),
        peak_memory_bandwidth_achieved(0.0),
        compute_utilization_percent(0.0) {}
};

/**
 * @brief Power mode specifications for Jetson Orin Nano
 *
 * Defines hardware configuration parameters for each supported power mode,
 * including frequency estimates and power consumption limits. Used for
 * accurate performance scaling analysis across different operating modes.
 */
struct PowerModeSpecs {
  int mode_id;                       ///< Numeric mode identifier (0, 1, 2)
  const char* name;                  ///< Human-readable mode name
  double max_power_watts;            ///< Maximum power consumption limit
  double estimated_gpu_freq_mhz;     ///< Estimated GPU frequency for this mode
  double estimated_memory_freq_mhz;  ///< Estimated memory frequency for this mode
};

/**
 * @brief Hardware specifications namespace for Jetson Orin Nano
 *
 * Contains verified hardware specifications based on NVIDIA documentation
 * and empirical validation. Organized in namespace to prevent global
 * namespace pollution and provide clear categorization.
 */
namespace JetsonSpecs {
// Core hardware specifications
constexpr int CUDA_CORES = 1024;              ///< Ampere CUDA cores
constexpr int COMPUTE_CAPABILITY_MAJOR = 8;   ///< CUDA compute capability major version
constexpr int COMPUTE_CAPABILITY_MINOR = 7;   ///< CUDA compute capability minor version
constexpr int STREAMING_MULTIPROCESSORS = 8;  ///< Number of streaming multiprocessors

// Performance specifications based on NVIDIA documentation and empirical measurement
constexpr double BASE_FREQ_MHZ = 612.0;    ///< Base GPU frequency (15W mode)
constexpr double BOOST_FREQ_MHZ = 1020.0;  ///< Boost GPU frequency (MAXN_SUPER mode)
constexpr double MEMORY_BW_GBPS = 68.0;    ///< LPDDR5 theoretical bandwidth
constexpr double L2_CACHE_MB = 4.0;        ///< L2 cache size
constexpr int L1_CACHE_KB_PER_SM = 128;    ///< L1 cache per SM
constexpr int SHARED_MEM_KB_PER_SM = 128;  ///< Shared memory per SM

// Performance targets based on empirical analysis and hardware specs
// FP32 Theoretical Peak = CUDA_CORES × BOOST_FREQ_MHZ × 2 (FMA) / 1000
// = 1024 × 1020 × 2 / 1000 = 2088.96 GFLOPS (MAXN_SUPER mode @ 1020 MHz)
constexpr double THEORETICAL_PEAK_GFLOPS = 2088.96;  ///< Absolute FP32 theoretical maximum (FMA)
/// Realistic achievable peak for hand-written kernels
constexpr double REALISTIC_PEAK_GFLOPS = 600.0;
/// Expected for naive implementation
constexpr double NAIVE_EXPECTED_GFLOPS = 50.0;

// Temperature and power monitoring limits for validation
constexpr double MAX_SAFE_TEMPERATURE_C = 85.0;        ///< Maximum safe operating temperature
constexpr double MIN_REASONABLE_TEMPERATURE_C = 20.0;  ///< Minimum reasonable temperature
constexpr double MAX_REASONABLE_POWER_W = 100.0;       ///< Maximum reasonable power consumption
constexpr double MIN_REASONABLE_POWER_W = 1.0;         ///< Minimum reasonable power consumption
}  // namespace JetsonSpecs

/**
 * @brief Available power mode configurations for Jetson Orin Nano
 *
 * Defines the three primary power modes supported by the Jetson Orin Nano
 * with conservative frequency estimates for empirical validation. These
 * configurations are used for performance scaling analysis.
 */
const PowerModeSpecs POWER_MODE_SPECS[] = {
    {0, "15W", 15.0, 612.0, 1600.0},  // Conservative power mode (612 MHz GPU, empirically measured)
    {1, "25W", 25.0, 918.0, 1866.0},  // Balanced power mode (918 MHz GPU, empirically measured)
    {2, "MAXN_SUPER", 30.0, 1020.0,
     1866.0}  // Maximum performance mode (1020 MHz GPU, empirically measured)
};

/// Number of available power modes
constexpr size_t NUM_POWER_MODES = sizeof(POWER_MODE_SPECS) / sizeof(PowerModeSpecs);

// ============================================================================
// Function Declarations
// ============================================================================

/**
 * @brief Get current GPU frequency from hardware
 * @return Current GPU frequency in MHz, or -1.0 if unable to read
 *
 * Reads the actual GPU frequency from the devfreq subsystem at runtime.
 * This provides measured frequency data rather than relying on estimates.
 * Returns -1.0 if the frequency file cannot be accessed (requires appropriate
 * permissions or may vary by kernel version).
 */
double get_current_gpu_frequency_mhz();

/**
 * @brief Initialize matrices with specified pattern
 * @param A First input matrix (row-major order)
 * @param B Second input matrix (row-major order)
 * @param n Matrix dimension (n x n)
 * @param type Initialization pattern ("random", "identity", "ones", "incremental")
 * @throws std::invalid_argument if parameters are invalid
 *
 * Initializes matrices with reproducible patterns for consistent benchmarking
 * across different power modes and test runs. Uses fixed seed for deterministic
 * behavior while supporting multiple initialization patterns for testing.
 */
void initialize_matrices(float* A, float* B, int n, const char* type = "random");

/**
 * @brief Reference CPU matrix multiplication implementation
 * @param A First input matrix (n x n)
 * @param B Second input matrix (n x n)
 * @param C Output matrix (n x n)
 * @param n Matrix dimension
 * @throws std::invalid_argument if matrices are null or n <= 0
 *
 * Provides ground truth reference implementation for correctness verification.
 * Uses standard O(n^3) algorithm with cache-friendly access patterns for
 * reliable accuracy comparison against GPU implementations.
 */
void cpu_matrix_multiply_reference(const float* A, const float* B, float* C, int n);

/**
 * @brief Verify numerical accuracy between GPU and CPU results
 * @param gpu_result GPU computation result
 * @param cpu_reference CPU reference result
 * @param n Matrix dimension
 * @return Mean relative error between results
 * @throws std::invalid_argument if inputs are invalid
 *
 * Computes statistical accuracy metrics including mean relative error
 * and maximum absolute error for validation of GPU implementations.
 * Essential for ensuring numerical correctness of optimized algorithms.
 */
double verify_accuracy(const float* gpu_result, const float* cpu_reference, int n);

/**
 * @brief Calculate comprehensive performance metrics
 * @param n Matrix dimension
 * @param elapsed_time_ms Kernel execution time in milliseconds
 * @param power_mode Active power mode identifier
 * @return Complete performance analysis structure
 * @throws std::invalid_argument if parameters are invalid
 *
 * Analyzes performance across multiple dimensions including throughput,
 * efficiency, memory utilization, and power consumption. Forms the core
 * of the benchmarking analysis pipeline.
 */
PerformanceMetrics calculate_performance_metrics(int n, double elapsed_time_ms, int power_mode = 0);

/**
 * @brief Generate formatted performance report using enterprise logging
 * @param metrics Performance metrics structure
 * @param algorithm_name Name of the algorithm being analyzed
 * @param n Matrix dimension
 *
 * Outputs comprehensive performance analysis with efficiency metrics,
 * bottleneck identification, and optimization recommendations using
 * structured logging for enterprise environments.
 */
void print_performance_report(const PerformanceMetrics& metrics, const char* algorithm_name, int n);

/**
 * @brief Display Jetson system information and capabilities
 *
 * Provides detailed system information including hardware specifications,
 * power mode configurations, and expected performance characteristics
 * using enterprise logging for comprehensive system documentation.
 */
void print_jetson_system_info();

/**
 * @brief Read Jetson power consumption with robust error handling
 * @return Power consumption in watts, or 0.0 if reading fails
 *
 * Attempts to read system power consumption from multiple hardware
 * monitoring interfaces with comprehensive error handling and validation.
 * Supports various power monitoring configurations across different systems.
 */
double read_jetson_power_consumption();

/**
 * @brief Read Jetson temperature with robust error handling
 * @return Temperature in Celsius, or reasonable default if reading fails
 *
 * Monitors GPU/SoC temperature from thermal zones with fallback
 * mechanisms for different system configurations. Critical for
 * thermal throttling detection and system health monitoring.
 */
double read_jetson_temperature();

/**
 * @brief Validate power mode configuration
 * @param mode_id Power mode identifier to validate
 * @return true if mode is valid and supported, false otherwise
 *
 * Verifies that the specified power mode is within valid range
 * and supported by the current system configuration. Prevents
 * invalid power mode access and provides clear error reporting.
 */
bool validate_power_mode(int mode_id);

/**
 * @brief Calculate theoretical GFLOPS for specific power mode
 * @param power_mode Power mode identifier
 * @return Theoretical peak performance in GFLOPS
 * @throws std::invalid_argument if power_mode is invalid
 *
 * Estimates theoretical peak performance based on power mode
 * frequency scaling and hardware specifications. Essential for
 * accurate efficiency calculations across different power modes.
 */
double calculate_theoretical_gflops_for_mode(int power_mode);

/**
 * @brief Calculate memory bandwidth utilization percentage
 * @param metrics Performance metrics structure
 * @return Memory bandwidth utilization as percentage of theoretical peak
 *
 * Analyzes achieved memory bandwidth relative to hardware limits
 * for identifying memory-bound operations and optimization opportunities.
 */
double calculate_memory_bandwidth_utilization(const PerformanceMetrics& metrics);

/**
 * @brief Analyze and report performance bottlenecks
 * @param metrics Performance metrics structure
 *
 * Identifies whether operations are compute-bound or memory-bound using
 * roofline analysis and provides specific optimization recommendations.
 * Core component of the performance analysis pipeline.
 */
void analyze_performance_bottleneck(const PerformanceMetrics& metrics);

/**
 * @brief Validate GPU memory requirements before allocation
 * @param n Matrix dimension
 * @return true if sufficient memory available, false otherwise
 *
 * Checks available GPU memory against required allocation size
 * to prevent out-of-memory errors during execution. Includes
 * detailed memory usage analysis and fragmentation warnings.
 */
bool validate_memory_requirements(int n);

/**
 * @brief Safe device memory allocation with comprehensive logging
 * @param ptr Pointer to device memory pointer
 * @param size Size in bytes to allocate
 * @param description Description for logging purposes
 * @throws std::runtime_error if allocation fails
 *
 * Allocates GPU memory with comprehensive error checking and logging.
 * Template function supporting type-safe allocation for any data type.
 */
template <typename T>
void safe_cuda_malloc(T** ptr, size_t size, const char* description) {
  if (!ptr) {
    LOG_ERROR_F("Invalid pointer for CUDA allocation: %s", description);
    throw std::invalid_argument("Invalid pointer for CUDA allocation");
  }

  if (size == 0) {
    LOG_WARNING_F("Zero-size allocation requested for %s", description);
    *ptr = nullptr;
    return;
  }

  try {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), size));
    LOG_DEBUG_F("Successfully allocated %zu bytes of GPU memory for %s", size, description);
  } catch (const std::exception& e) {
    LOG_ERROR_F("GPU memory allocation failed for %s: %s", description, e.what());
    PRINT_ERROR_DETAILED(
        "GPU memory allocation failed",
        std::string("GPU memory allocation failed for ") + description + ": " + e.what());
    throw;
  }
}

/**
 * @brief Safe device memory deallocation with comprehensive logging
 * @param ptr Pointer to device memory to free
 * @param description Description for logging purposes
 *
 * Safely frees GPU memory with null pointer checking and logging.
 * Template function supporting type-safe deallocation for any data type.
 */
template <typename T>
void safe_cuda_free(T* ptr, const char* description) {
  if (ptr != nullptr) {
    try {
      CUDA_CHECK(cudaFree(ptr));
      LOG_DEBUG_F("Successfully freed GPU memory for %s", description);
    } catch (const std::exception& e) {
      LOG_WARNING_F("GPU memory deallocation warning for %s: %s", description, e.what());
      // Don't rethrow - continue with cleanup
    }
  } else {
    LOG_DEBUG_F("Skipped freeing null pointer for %s", description);
  }
}

/**
 * @brief Safe host memory allocation with error checking
 * @param ptr Pointer variable to receive allocated memory
 * @param size Size in bytes to allocate
 * @param description Description for logging
 * @throws std::bad_alloc if allocation fails
 *
 * Performs host memory allocation with detailed logging and validation.
 * Provides consistent error handling across the application.
 */
template <typename T>
void safe_host_malloc(T** ptr, size_t size, const char* description) {
  if (!ptr) {
    LOG_ERROR_F("Invalid pointer for host allocation: %s", description);
    throw std::invalid_argument("Invalid pointer for host allocation");
  }

  if (size == 0) {
    LOG_WARNING_F("Zero-size host allocation requested for %s", description);
    *ptr = nullptr;
    return;
  }

  *ptr = static_cast<T*>(malloc(size));
  if (!*ptr) {
    LOG_CRITICAL_F("Host memory allocation failed: %s (size: %zu bytes)", description, size);
    PRINT_ERROR_F("Host memory allocation failed for %s", description);
    throw std::bad_alloc();
  }

  LOG_DEBUG_F("Successfully allocated %zu bytes of host memory for %s", size, description);
}

#endif  // COMMON_H
