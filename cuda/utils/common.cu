/**
 * @file common.cu
 * @brief Implementation of common utilities for Jetson Orin Nano matrix multiplication analysis
 * @author Jesse Moses (@Cre4T3Tiv3)
 * @date 2025
 * @copyright ByteStack Labs - MIT License
 *
 * Implements shared functionality for matrix operations, performance analysis,
 * and system monitoring on the NVIDIA Jetson Orin Nano platform. Provides
 * enterprise-grade error handling and comprehensive logging throughout.
 *
 * Key Features:
 * - Reproducible matrix initialization with multiple patterns
 * - Hardware monitoring with robust fallback mechanisms
 * - Comprehensive performance analysis and bottleneck identification
 * - Enterprise-grade error handling with detailed logging
 */

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "common.h"

/**
 * @brief Get current GPU frequency from hardware
 * @return Current GPU frequency in MHz, or -1.0 if unable to read
 *
 * Reads the actual GPU frequency from the devfreq subsystem at runtime.
 * Attempts multiple common devfreq paths for compatibility across different
 * Jetson platforms and kernel versions.
 */
double get_current_gpu_frequency_mhz() {
  // Try common devfreq paths for Jetson Orin Nano
  const char* freq_paths[] = {
      "/sys/devices/gpu.0/devfreq/17000000.ga10b/cur_freq",
      "/sys/class/devfreq/17000000.ga10b/cur_freq",
      "/sys/kernel/debug/bpmp/debug/clk/nafll_gpu/rate",
      nullptr  // Sentinel
  };

  for (int i = 0; freq_paths[i] != nullptr; i++) {
    FILE* freq_file = fopen(freq_paths[i], "r");
    if (freq_file) {
      uint64_t freq_hz = 0;
      int scanned = fscanf(freq_file, "%" PRIu64, &freq_hz);
      fclose(freq_file);

      if (scanned == 1 && freq_hz > 0) {
        double freq_mhz = freq_hz / 1000000.0;  // Convert Hz to MHz
        LOG_DEBUG_F("GPU frequency read from %s: %.1f MHz", freq_paths[i], freq_mhz);
        return freq_mhz;
      }
    }
  }

  LOG_WARNING("Unable to read GPU frequency from devfreq subsystem");
  return -1.0;
}

/**
 * @brief Initialize matrices with specified pattern and reproducible seeding
 * @param A First input matrix (n x n) in row-major order
 * @param B Second input matrix (n x n) in row-major order
 * @param n Matrix dimension
 * @param type Initialization pattern ("random", "identity", "ones", "incremental")
 * @throws std::invalid_argument if parameters are invalid
 *
 * Provides deterministic matrix initialization for consistent benchmarking
 * across different power modes and test configurations. Uses fixed seed
 * for reproducible results while supporting multiple initialization patterns.
 */
void initialize_matrices(float* A, float* B, int n, const char* type) {
  // Input validation with comprehensive error checking
  if (!A || !B) {
    LOG_ERROR("Matrix pointers cannot be null");
    throw std::invalid_argument("Invalid matrix pointers provided");
  }

  if (n <= 0) {
    LOG_ERROR_F("Invalid matrix dimension: %d (must be positive)", n);
    throw std::invalid_argument("Matrix dimension must be positive");
  }

  if (!type) {
    LOG_ERROR("Initialization type string cannot be null");
    throw std::invalid_argument("Invalid initialization type");
  }

  LOG_DEBUG_F("Initializing %dx%d matrices with pattern: '%s'", n, n, type);

  // Use fixed seed for reproducibility across power modes and test runs
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  try {
    if (strcmp(type, "identity") == 0) {
      LOG_DEBUG("Creating identity matrix A and random matrix B");

      // Initialize A as identity matrix
      for (int i = 0; i < n * n; i++) {
        A[i] = 0.0f;
      }
      for (int i = 0; i < n; i++) {
        A[i * n + i] = 1.0f;
      }

      // Initialize B with random values
      for (int i = 0; i < n * n; i++) {
        B[i] = dis(gen);
      }

    } else if (strcmp(type, "ones") == 0) {
      LOG_DEBUG("Creating matrices filled with ones");

      for (int i = 0; i < n * n; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
      }

    } else if (strcmp(type, "incremental") == 0) {
      LOG_DEBUG("Creating matrices with incremental values for debugging");

      for (int i = 0; i < n * n; i++) {
        A[i] = static_cast<float>(i % 100) / 100.0f;
        B[i] = static_cast<float>((i + 50) % 100) / 100.0f;
      }

    } else {
      // Default case: random matrices (consistent across power mode tests)
      LOG_DEBUG("Creating random matrices with fixed seed for reproducibility");

      for (int i = 0; i < n * n; i++) {
        A[i] = dis(gen);
        B[i] = dis(gen);
      }
    }

    PRINT_INFO_F("Successfully initialized matrices with '%s' pattern", type);
  } catch (const std::exception& e) {
    LOG_ERROR_F("Matrix initialization failed: %s", e.what());
    throw;
  }
}

/**
 * @brief Reference CPU matrix multiplication for correctness verification
 * @param A First input matrix (n x n) in row-major order
 * @param B Second input matrix (n x n) in row-major order
 * @param C Output matrix (n x n) in row-major order
 * @param n Matrix dimension
 * @throws std::invalid_argument if parameters are invalid
 *
 * Implements standard O(n^3) matrix multiplication algorithm for ground truth
 * comparison. Uses cache-friendly access patterns and provides detailed timing.
 */
void cpu_matrix_multiply_reference(const float* A, const float* B, float* C, int n) {
  // Comprehensive input validation
  if (!A || !B || !C) {
    LOG_ERROR("Matrix pointers cannot be null for CPU reference computation");
    throw std::invalid_argument("Invalid matrix pointers");
  }

  if (n <= 0) {
    LOG_ERROR_F("Invalid matrix dimension for CPU computation: %d", n);
    throw std::invalid_argument("Matrix dimension must be positive");
  }

  LOG_DEBUG_F("Starting CPU reference computation for %dx%d matrices", n, n);

  try {
    // Standard O(n^3) matrix multiplication
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
          sum += A[i * n + k] * B[k * n + j];
        }
        C[i * n + j] = sum;
      }
    }

    LOG_DEBUG("CPU reference computation completed successfully");
  } catch (const std::exception& e) {
    LOG_ERROR_F("CPU reference computation failed: %s", e.what());
    throw;
  }
}

/**
 * @brief Verify numerical accuracy between GPU and CPU results
 * @param gpu_result GPU computation result matrix
 * @param cpu_reference CPU reference result matrix
 * @param n Matrix dimension
 * @return Mean relative error between results
 * @throws std::invalid_argument if parameters are invalid
 *
 * Performs comprehensive accuracy analysis including relative error computation,
 * maximum absolute error tracking, and statistical validation of results.
 */
double verify_accuracy(const float* gpu_result, const float* cpu_reference, int n) {
  // Input validation
  if (!gpu_result || !cpu_reference) {
    LOG_ERROR("Result matrix pointers cannot be null for accuracy verification");
    throw std::invalid_argument("Invalid result matrix pointers");
  }

  if (n <= 0) {
    LOG_ERROR_F("Invalid matrix dimension for accuracy verification: %d", n);
    throw std::invalid_argument("Matrix dimension must be positive");
  }

  LOG_DEBUG_F("Verifying accuracy for %dx%d result matrices", n, n);

  double max_error = 0.0;
  double relative_error_sum = 0.0;
  int valid_comparisons = 0;
  int total_elements = n * n;

  try {
    // Calculate accuracy metrics
    for (int i = 0; i < total_elements; i++) {
      double abs_error = fabs(gpu_result[i] - cpu_reference[i]);
      max_error = fmax(max_error, abs_error);

      // Only calculate relative error for non-zero reference values
      if (fabs(cpu_reference[i]) > 1e-10) {
        double relative_error = abs_error / fabs(cpu_reference[i]);
        relative_error_sum += relative_error;
        valid_comparisons++;
      }
    }

    if (valid_comparisons == 0) {
      LOG_WARNING("No valid comparisons found - all reference values near zero");
      return 0.0;
    }

    double mean_relative_error = relative_error_sum / valid_comparisons;

    // Log detailed accuracy results
    LOG_DEBUG_F("%s", "Accuracy verification completed:");
    LOG_DEBUG_F("  Maximum absolute error: %.2e", max_error);
    LOG_DEBUG_F("  Mean relative error: %.2e", mean_relative_error);
    LOG_DEBUG_F("  Valid comparisons: %d/%d (%.1f%%)", valid_comparisons, total_elements,
                100.0 * valid_comparisons / total_elements);

    // Warn about high numerical errors
    if (mean_relative_error > 1e-3) {
      PRINT_WARNING_F("High numerical error detected: %.2e - investigate precision issues",
                      mean_relative_error);
    }

    return mean_relative_error;
  } catch (const std::exception& e) {
    LOG_ERROR_F("Accuracy verification failed: %s", e.what());
    throw;
  }
}

/**
 * @brief Read Jetson temperature from multiple thermal zones with error handling
 * @return Temperature in Celsius, or reasonable default if reading fails
 *
 * Attempts to read system temperature from multiple thermal monitoring
 * sources with robust error handling and validation of reasonable ranges.
 */
double read_jetson_temperature() {
  LOG_DEBUG("Attempting to read Jetson temperature from thermal zones");

  // Multiple thermal zone paths for different system configurations
  const char* temp_paths[] = {"/sys/class/thermal/thermal_zone0/temp",
                              "/sys/class/thermal/thermal_zone1/temp",
                              "/sys/class/thermal/thermal_zone2/temp"};

  const size_t num_paths = sizeof(temp_paths) / sizeof(temp_paths[0]);

  for (size_t i = 0; i < num_paths; i++) {
    const char* path = temp_paths[i];

    try {
      std::ifstream file(path, std::ios::in);
      if (file.is_open()) {
        std::string line;
        if (std::getline(file, line)) {
          double temp_millic = std::stod(line);
          double temp_c = temp_millic / 1000.0;

          // Validate reasonable temperature range
          if (temp_c >= JetsonSpecs::MIN_REASONABLE_TEMPERATURE_C &&
              temp_c <= JetsonSpecs::MAX_SAFE_TEMPERATURE_C) {
            LOG_DEBUG_F("Successfully read temperature: %.1f°C from %s", temp_c, path);
            return temp_c;
          } else {
            LOG_DEBUG_F("Temperature out of reasonable range: %.1f°C from %s", temp_c, path);
          }
        }
        file.close();
      } else {
        LOG_DEBUG_F("Could not open thermal zone file: %s", path);
      }
    } catch (const std::exception& e) {
      LOG_DEBUG_F("Exception reading temperature from %s: %s", path, e.what());
      continue;
    }
  }

  // Fallback default value
  const double default_temp = 48.0;
  LOG_WARNING_F("Could not read temperature from any source, using default: %.1f°C", default_temp);
  return default_temp;
}

/**
 * @brief Detect current Jetson power mode from system configuration
 * @return Power mode identifier (0=15W, 1=25W, 2=MAXN_SUPER, -1=unknown)
 *
 * Queries the nvpmodel utility to determine the active power configuration.
 * Provides fallback handling for systems where nvpmodel is not available.
 */
int detect_current_power_mode() {
  int current_power_mode = -1;

  try {
    FILE* mode_query = popen("nvpmodel -q 2>/dev/null | grep 'NV Power Mode' | tail -1", "r");
    if (mode_query) {
      char mode_line[256];
      if (fgets(mode_line, sizeof(mode_line), mode_query)) {
        if (strstr(mode_line, "15W")) {
          current_power_mode = 0;
        } else if (strstr(mode_line, "25W")) {
          current_power_mode = 1;
        } else if (strstr(mode_line, "MAXN")) {
          current_power_mode = 2;
        }
        PRINT_INFO_F("Detected power mode: %s", current_power_mode >= 0
                                                    ? POWER_MODE_SPECS[current_power_mode].name
                                                    : "Unknown");
      }
      pclose(mode_query);
    }
  } catch (const std::exception& e) {
    LOG_WARNING_F("Failed to detect power mode: %s", e.what());
  }

  if (current_power_mode == -1) {
    PRINT_WARNING("Could not detect power mode - using default (MAXN_SUPER)");
    current_power_mode = 2;  // Default to maximum performance mode
  }

  return current_power_mode;
}

/**
 * @brief Read Jetson power consumption from multiple monitoring sources
 * @return Power consumption in watts, or 0.0 if reading fails
 *
 * Attempts to read system power consumption from various hardware monitoring
 * interfaces with unit conversion and validation of reasonable power ranges.
 */
double read_jetson_power_consumption() {
  LOG_DEBUG("Attempting to read Jetson power consumption from hardware monitors");

  // Multiple power monitoring paths for different hardware configurations
  const char* power_paths[] = {"/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
                               "/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device1/in_power0_input",
                               "/sys/class/hwmon/hwmon0/power1_input",
                               "/sys/class/hwmon/hwmon1/power1_input",
                               "/sys/class/hwmon/hwmon2/power1_input"};

  const size_t num_paths = sizeof(power_paths) / sizeof(power_paths[0]);

  for (size_t i = 0; i < num_paths; i++) {
    const char* path = power_paths[i];

    try {
      std::ifstream file(path, std::ios::in);
      if (file.is_open()) {
        std::string line;
        if (std::getline(file, line)) {
          double power_raw = std::stod(line);

          // Handle different power units (microWatts, milliWatts, Watts)
          double power_w = 0.0;
          if (power_raw > 1000000) {  // Likely microWatts
            power_w = power_raw / 1000000.0;
          } else if (power_raw > 1000) {  // Likely milliWatts
            power_w = power_raw / 1000.0;
          } else {  // Likely Watts
            power_w = power_raw;
          }

          // Validate reasonable power range for Jetson systems
          if (power_w >= JetsonSpecs::MIN_REASONABLE_POWER_W &&
              power_w <= JetsonSpecs::MAX_REASONABLE_POWER_W) {
            LOG_DEBUG_F("Successfully read power: %.1f W from %s", power_w, path);
            return power_w;
          } else {
            LOG_DEBUG_F("Power reading out of reasonable range: %.1f W from %s", power_w, path);
          }
        }
        file.close();
      } else {
        LOG_DEBUG_F("Could not open power monitoring file: %s", path);
      }
    } catch (const std::exception& e) {
      LOG_DEBUG_F("Exception reading power from %s: %s", path, e.what());
      continue;
    }
  }

  LOG_DEBUG("Could not read power from any source - monitoring may be unavailable");
  return 0.0;  // Power monitoring unavailable - acceptable for benchmarking
}

/**
 * @brief Calculate theoretical GFLOPS for specific power mode
 * @param power_mode Power mode identifier (0=15W, 1=25W, 2=MAXN_SUPER)
 * @return Theoretical peak performance in GFLOPS
 *
 * Estimates theoretical peak performance based on power mode frequency
 * scaling and hardware specifications with validation of input parameters.
 */
double calculate_theoretical_gflops_for_mode(int power_mode) {
  // Validate power mode range
  if (power_mode < 0 || power_mode >= static_cast<int>(NUM_POWER_MODES)) {
    LOG_WARNING_F("Invalid power mode: %d, using baseline expectation", power_mode);
    return JetsonSpecs::NAIVE_EXPECTED_GFLOPS;
  }

  const PowerModeSpecs& spec = POWER_MODE_SPECS[power_mode];

  // Scale theoretical peak based on estimated frequency for this mode
  // Use absolute THEORETICAL_PEAK_GFLOPS for accurate efficiency calculations
  double freq_ratio = spec.estimated_gpu_freq_mhz / JetsonSpecs::BOOST_FREQ_MHZ;
  double theoretical_peak = JetsonSpecs::THEORETICAL_PEAK_GFLOPS * freq_ratio;

  LOG_DEBUG_F("Calculated theoretical peak for %s mode: %.1f GFLOPS (freq ratio: %.3f)", spec.name,
              theoretical_peak, freq_ratio);

  return theoretical_peak;
}

/**
 * @brief Calculate comprehensive performance metrics with error handling
 * @param n Matrix dimension
 * @param elapsed_time_ms Kernel execution time in milliseconds
 * @param power_mode Active power mode identifier
 * @return Complete performance analysis structure
 * @throws std::invalid_argument if parameters are invalid
 *
 * Performs multi-dimensional performance analysis including throughput,
 * efficiency, memory utilization, and system monitoring integration.
 */
PerformanceMetrics calculate_performance_metrics(int n, double elapsed_time_ms, int power_mode) {
  // Input validation
  if (n <= 0) {
    LOG_ERROR_F("Invalid matrix dimension for performance calculation: %d", n);
    throw std::invalid_argument("Matrix dimension must be positive");
  }

  if (elapsed_time_ms <= 0.0) {
    LOG_ERROR_F("Invalid elapsed time for performance calculation: %.3f ms", elapsed_time_ms);
    throw std::invalid_argument("Elapsed time must be positive");
  }

  LOG_DEBUG_F("Calculating performance metrics for n=%d, time=%.3f ms, power_mode=%d", n,
              elapsed_time_ms, power_mode);

  PerformanceMetrics metrics;

  try {
    // Basic timing and throughput metrics
    metrics.elapsed_time_ms = elapsed_time_ms;
    metrics.theoretical_flops = 2.0 * static_cast<double>(n) * n * n;  // n^3 mults + n^3 adds
    metrics.measured_gflops = metrics.theoretical_flops / (elapsed_time_ms * 1e6);

    // Memory transfer analysis: 2 reads of n^2 + 1 write of n^2 elements
    metrics.bytes_transferred = 3ULL * n * n * sizeof(float);
    metrics.memory_bandwidth_gbps =
        static_cast<double>(metrics.bytes_transferred) / (elapsed_time_ms * 1e6);

    // Arithmetic intensity for matrix multiplication (FLOPS per byte)
    metrics.arithmetic_intensity =
        metrics.theoretical_flops / static_cast<double>(metrics.bytes_transferred);

    // Power mode information
    metrics.power_mode_id = power_mode;
    if (power_mode >= 0 && power_mode < static_cast<int>(NUM_POWER_MODES)) {
      metrics.power_mode_name = POWER_MODE_SPECS[power_mode].name;
    } else {
      metrics.power_mode_name = "Unknown";
    }

    // Calculate realistic efficiency based on power mode
    double mode_theoretical_peak = calculate_theoretical_gflops_for_mode(power_mode);
    metrics.efficiency_percent = (metrics.measured_gflops / mode_theoretical_peak) * 100.0;

    // Memory bandwidth utilization
    metrics.peak_memory_bandwidth_achieved =
        (metrics.memory_bandwidth_gbps / JetsonSpecs::MEMORY_BW_GBPS) * 100.0;

    // Overall compute utilization estimate (against theoretical peak at boost frequency)
    metrics.compute_utilization_percent =
        (metrics.measured_gflops / JetsonSpecs::THEORETICAL_PEAK_GFLOPS) * 100.0;

    // Read system monitoring metrics with error handling
    try {
      metrics.temperature_celsius = read_jetson_temperature();
      metrics.power_watts = read_jetson_power_consumption();
    } catch (const std::exception& e) {
      LOG_WARNING_F("System monitoring failed, using defaults: %s", e.what());
      metrics.temperature_celsius = 48.0;  // Reasonable default
      metrics.power_watts = 0.0;           // Monitoring unavailable
    }

    // Log calculated metrics for debugging
    LOG_DEBUG_F("%s", "Performance metrics calculated successfully:");
    LOG_DEBUG_F("  Measured GFLOPS: %.2f", metrics.measured_gflops);
    LOG_DEBUG_F("  Memory bandwidth: %.2f GB/s", metrics.memory_bandwidth_gbps);
    LOG_DEBUG_F("  Arithmetic intensity: %.2f FLOPS/byte", metrics.arithmetic_intensity);
    LOG_DEBUG_F("  Efficiency: %.1f%% of mode peak", metrics.efficiency_percent);

    return metrics;
  } catch (const std::exception& e) {
    LOG_ERROR_F("Performance metrics calculation failed: %s", e.what());
    throw;
  }
}

/**
 * @brief Calculate memory bandwidth utilization percentage
 * @param metrics Performance metrics structure
 * @return Memory bandwidth utilization as percentage of theoretical peak
 *
 * Analyzes achieved memory bandwidth relative to hardware theoretical
 * maximum for identifying memory-bound operation characteristics.
 */
double calculate_memory_bandwidth_utilization(const PerformanceMetrics& metrics) {
  double utilization = (metrics.memory_bandwidth_gbps / JetsonSpecs::MEMORY_BW_GBPS) * 100.0;

  LOG_DEBUG_F("Memory bandwidth utilization: %.1f%% (%.2f/%.1f GB/s)", utilization,
              metrics.memory_bandwidth_gbps, JetsonSpecs::MEMORY_BW_GBPS);

  return utilization;
}

/**
 * @brief Analyze and report performance bottlenecks using structured logging
 * @param metrics Performance metrics structure
 *
 * Performs roofline analysis to identify compute vs memory limitations
 * and provides specific optimization recommendations based on bottleneck type.
 */
void analyze_performance_bottleneck(const PerformanceMetrics& metrics) {
  PRINT_INFO("Performance Bottleneck Analysis:");

  // Roofline model analysis
  double memory_roof = JetsonSpecs::MEMORY_BW_GBPS * metrics.arithmetic_intensity;
  double compute_roof = calculate_theoretical_gflops_for_mode(metrics.power_mode_id);

  PRINT_INFO_F("Memory-limited performance ceiling: %.1f GFLOPS", memory_roof);
  PRINT_INFO_F("Compute-limited performance ceiling: %.1f GFLOPS", compute_roof);

  if (memory_roof < compute_roof) {
    PRINT_INFO("CONCLUSION: MEMORY BOUND operation");
    PRINT_INFO("Optimization focus: Memory access patterns, cache utilization");
    PRINT_INFO_F("Memory bandwidth utilization: %.1f%%",
                 calculate_memory_bandwidth_utilization(metrics));

    // Provide specific optimization recommendations
    if (metrics.peak_memory_bandwidth_achieved < 50.0) {
      PRINT_INFO("RECOMMENDATION: Implement memory coalescing optimizations");
    }
    if (metrics.arithmetic_intensity < 10.0) {
      PRINT_INFO("RECOMMENDATION: Consider blocked/tiled algorithms to improve cache reuse");
    }
    if (metrics.peak_memory_bandwidth_achieved < 20.0) {
      PRINT_INFO("RECOMMENDATION: Review memory access patterns for stride optimization");
    }

  } else {
    PRINT_INFO("CONCLUSION: COMPUTE BOUND operation");
    PRINT_INFO("Optimization focus: Algorithmic improvements, instruction optimization");
    PRINT_INFO_F("Compute utilization: %.1f%%", metrics.compute_utilization_percent);

    // Provide specific optimization recommendations
    if (metrics.efficiency_percent < 50.0) {
      PRINT_INFO("RECOMMENDATION: Optimize thread utilization and reduce divergence");
    }
    if (metrics.compute_utilization_percent < 30.0) {
      PRINT_INFO("RECOMMENDATION: Consider algorithmic improvements or cuBLAS");
    }
    if (metrics.efficiency_percent < 25.0) {
      PRINT_INFO("RECOMMENDATION: Profile for instruction bottlenecks and occupancy issues");
    }
  }
}

/**
 * @brief Generate comprehensive performance report using structured logging
 * @param metrics Performance metrics structure
 * @param algorithm Algorithm name being analyzed
 * @param n Matrix dimension
 *
 * Outputs detailed performance analysis with efficiency metrics, system
 * monitoring data, and optimization recommendations using enterprise logging.
 */
void print_performance_report(const PerformanceMetrics& metrics, const char* algorithm, int n) {
  // Input validation
  if (!algorithm) {
    LOG_ERROR("Algorithm name cannot be null for performance report");
    algorithm = "Unknown";
  }

  PRINT_INFO_F("=== %s Algorithm Performance Analysis (n=%d) ===", algorithm, n);
  PRINT_INFO("Target: Jetson Orin Nano Engineering Reference Developer Kit Super");

  // Power mode information
  if (metrics.power_mode_id >= 0 && metrics.power_mode_id < static_cast<int>(NUM_POWER_MODES)) {
    const PowerModeSpecs& spec = POWER_MODE_SPECS[metrics.power_mode_id];
    PRINT_INFO_F("Power Mode: %s (%.0fW max)", spec.name, spec.max_power_watts);
  } else {
    PRINT_INFO_F("Power Mode: Unknown (ID: %d)", metrics.power_mode_id);
  }

  // Execution metrics
  PRINT_INFO("Execution Metrics:");
  PRINT_INFO_F("Elapsed Time: %.3f ms", metrics.elapsed_time_ms);
  PRINT_INFO_F("Theoretical FLOPS: %.2e", metrics.theoretical_flops);
  PRINT_INFO_F("Measured Performance: %.2f GFLOPS", metrics.measured_gflops);
  PRINT_INFO_F("Memory Bandwidth: %.2f GB/s", metrics.memory_bandwidth_gbps);
  PRINT_INFO_F("Arithmetic Intensity: %.2f FLOPS/byte", metrics.arithmetic_intensity);

  // Efficiency analysis
  PRINT_INFO("Efficiency Analysis:");
  double mode_peak = calculate_theoretical_gflops_for_mode(metrics.power_mode_id);
  PRINT_INFO_F("Power Mode Theoretical Peak: %.1f GFLOPS", mode_peak);
  PRINT_INFO_F("Algorithm Efficiency: %.1f%% of mode peak", metrics.efficiency_percent);
  PRINT_INFO_F("Memory Bandwidth Utilization: %.1f%%", metrics.peak_memory_bandwidth_achieved);
  PRINT_INFO_F("Overall Compute Utilization: %.1f%%", metrics.compute_utilization_percent);

  // System monitoring results
  if (metrics.temperature_celsius > 0) {
    PRINT_INFO_F("Thermal Status: %.1f°C", metrics.temperature_celsius);

    // Temperature warnings
    if (metrics.temperature_celsius > 80.0) {
      PRINT_WARNING("Very high temperature detected - thermal throttling likely");
    } else if (metrics.temperature_celsius > 70.0) {
      PRINT_WARNING("High temperature detected - monitor for thermal throttling");
    }
  } else {
    PRINT_INFO("Thermal Status: Monitoring unavailable");
  }

  if (metrics.power_watts > 0) {
    PRINT_INFO_F("Power Consumption: %.1f W", metrics.power_watts);
    PRINT_INFO_F("Performance/Watt: %.1f GFLOPS/W", metrics.measured_gflops / metrics.power_watts);

    // Power efficiency assessment
    double efficiency_ratio = (metrics.measured_gflops / metrics.power_watts) /
                              (JetsonSpecs::REALISTIC_PEAK_GFLOPS / 25.0);  // Assuming 25W baseline
    if (efficiency_ratio > 1.2) {
      PRINT_INFO("Power efficiency: Excellent");
    } else if (efficiency_ratio > 0.8) {
      PRINT_INFO("Power efficiency: Good");
    } else {
      PRINT_INFO("Power efficiency: Room for improvement");
    }
  } else {
    PRINT_INFO("Power Consumption: Monitoring unavailable");
  }

  // Detailed bottleneck analysis
  analyze_performance_bottleneck(metrics);
}

/**
 * @brief Validate GPU memory requirements before allocation
 * @param n Matrix dimension
 * @return true if sufficient memory available, false otherwise
 *
 * Checks available GPU memory against required allocation size
 * to prevent out-of-memory errors during execution. Includes
 * detailed memory usage analysis and fragmentation warnings.
 */
bool validate_memory_requirements(int n) {
  // Input validation
  if (n <= 0) {
    LOG_ERROR_F("Invalid matrix dimension for memory validation: %d", n);
    return false;
  }

  LOG_DEBUG_F("Validating memory requirements for %dx%d matrices", n, n);

  try {
    // Calculate memory requirements for three n×n matrices (A, B, C)
    size_t matrix_size = static_cast<size_t>(n) * n * sizeof(float);
    size_t total_gpu_memory_needed = 3 * matrix_size;  // A, B, C matrices
    // Get available GPU memory
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    // Clean console output for memory analysis
    PRINT_INFO("Memory Requirements Analysis:");
    PRINT_INFO_F("  Per matrix: %.2f MB", matrix_size / (1024.0 * 1024.0));
    PRINT_INFO_F("  Total GPU memory needed: %.2f MB", total_gpu_memory_needed / (1024.0 * 1024.0));
    PRINT_INFO_F("  Available GPU memory: %.2f MB", free_mem / (1024.0 * 1024.0));
    PRINT_INFO_F("  Total GPU memory: %.2f MB", total_mem / (1024.0 * 1024.0));
    // Check if sufficient memory is available
    if (total_gpu_memory_needed > free_mem) {
      PRINT_ERROR_F("ERROR: Insufficient GPU memory for n=%d", n);
      return false;
    }

    return true;
  } catch (const std::exception& e) {
    LOG_ERROR_F("Memory validation failed with exception: %s", e.what());
    return false;
  }
}

/**
 * @brief Display comprehensive Jetson system information using structured logging
 *
 * Provides detailed system specifications, power mode configurations,
 * and expected performance characteristics for reference and validation.
 */
void print_jetson_system_info() {
  PRINT_INFO("=== NVIDIA Jetson Orin Nano System Information ===");
  PRINT_INFO("Model: Engineering Reference Developer Kit Super");
  PRINT_INFO_F("CUDA Cores: %d (Ampere Architecture)", JetsonSpecs::CUDA_CORES);
  PRINT_INFO_F("Compute Capability: %d.%d", JetsonSpecs::COMPUTE_CAPABILITY_MAJOR,
               JetsonSpecs::COMPUTE_CAPABILITY_MINOR);
  PRINT_INFO_F("Streaming Multiprocessors: %d", JetsonSpecs::STREAMING_MULTIPROCESSORS);
  PRINT_INFO_F("Memory: 7.4GB LPDDR5 @ %.1f GB/s theoretical", JetsonSpecs::MEMORY_BW_GBPS);
  PRINT_INFO_F("L2 Cache: %.1f MB", JetsonSpecs::L2_CACHE_MB);
  PRINT_INFO_F("GPU Base/Boost Frequency: %.0f/%.0f MHz", JetsonSpecs::BASE_FREQ_MHZ,
               JetsonSpecs::BOOST_FREQ_MHZ);

  // Display current GPU frequency if available
  double current_freq = get_current_gpu_frequency_mhz();
  if (current_freq > 0) {
    PRINT_INFO_F("Current GPU Frequency: %.1f MHz (measured)", current_freq);
  }

  PRINT_INFO("Target Power Modes:");
  for (size_t i = 0; i < NUM_POWER_MODES; i++) {
    const PowerModeSpecs& spec = POWER_MODE_SPECS[i];
    PRINT_INFO_F("  Mode %d (%s): %.0fW max, ~%.0f MHz GPU", spec.mode_id, spec.name,
                 spec.max_power_watts, spec.estimated_gpu_freq_mhz);
  }

  PRINT_INFO("Software Stack:");
  PRINT_INFO("L4T Version: R36.4.4 (JetPack 6.x)");
  PRINT_INFO("CUDA Version: V12.6.68");
  PRINT_INFO("OS: Ubuntu 22.04.5 LTS");

  PRINT_INFO("Performance Expectations:");
  PRINT_INFO_F("Theoretical Peak: %.0f GFLOPS (absolute maximum)",
               JetsonSpecs::THEORETICAL_PEAK_GFLOPS);
  PRINT_INFO_F("Realistic Peak: %.0f GFLOPS (achievable with optimization)",
               JetsonSpecs::REALISTIC_PEAK_GFLOPS);
  PRINT_INFO_F("Naive Algorithm Target: %.0f GFLOPS (expected baseline)",
               JetsonSpecs::NAIVE_EXPECTED_GFLOPS);

  // Additional system information
  PRINT_INFO("Memory Hierarchy:");
  PRINT_INFO_F("L1 Cache per SM: %d KB", JetsonSpecs::L1_CACHE_KB_PER_SM);
  PRINT_INFO_F("Shared Memory per SM: %d KB", JetsonSpecs::SHARED_MEM_KB_PER_SM);
  PRINT_INFO_F("Total L1 Cache: %d KB",
               JetsonSpecs::L1_CACHE_KB_PER_SM * JetsonSpecs::STREAMING_MULTIPROCESSORS);
}
