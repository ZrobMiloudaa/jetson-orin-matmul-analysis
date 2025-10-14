/**
 * @file cublas_multiplication.cu
 * @brief cuBLAS-optimized matrix multiplication for Jetson Orin Nano performance analysis
 * @author Jesse Moses (@Cre4T3Tiv3)
 * @date 2025
 * @copyright ByteStack Labs - MIT License
 *
 * Implements matrix multiplication using NVIDIA's highly optimized cuBLAS library
 * (cublasSgemm) for comprehensive power mode analysis on the NVIDIA Jetson Orin
 * Nano Engineering Reference Developer Kit Super. Provides optimal performance
 * baseline across 15W, 25W, and MAXN SUPER power configurations.
 *
 * Mathematical Foundation: C[i,j] = sum(k=0 to n-1) A[i,k] * B[k,j]
 * Complexity: O(n^3) time, optimized with vendor-tuned algorithms
 *
 * Target Hardware:
 * - Jetson Orin Nano Engineering Reference Developer Kit Super
 * - SM 8.7 (Ampere Architecture)
 * - 1024 CUDA Cores, 8 Streaming Multiprocessors
 * - 7.4GB LPDDR5 @ 68 GB/s theoretical bandwidth
 *
 * cuBLAS Implementation Details:
 * - Uses cublasSgemm (single-precision general matrix multiply)
 * - Automatically optimized for Ampere architecture
 * - Leverages Tensor Cores when beneficial
 * - Vendor-tuned memory access patterns and tiling strategies
 */

#include <cublas_v2.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "../utils/common.h"
#include "../utils/logger.h"

bool ensure_directory_exists(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
    return true;
  }

  // Create directory with better error checking
  std::string cmd = "mkdir -p \"" + path + "\"";
  int result = system(cmd.c_str());
  if (result != 0) {
    std::cerr << "ERROR: Failed to create directory: " << path << " (exit code: " << result << ")"
              << std::endl;
    return false;
  }

  // Verify it was actually created
  return (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

/**
 * @brief Jetson system metrics structure with validation flags
 *
 * Encapsulates temperature and power monitoring data with validity
 * indicators for robust system state analysis and error handling.
 */
struct JetsonMetrics {
  float temperature_c;  ///< GPU/SoC temperature in Celsius
  float power_watts;    ///< System power consumption in watts
  bool power_valid;     ///< Power reading validity flag
  bool temp_valid;      ///< Temperature reading validity flag

  /**
   * @brief Default constructor with safe initialization
   */
  JetsonMetrics() : temperature_c(0.0f), power_watts(0.0f), power_valid(false), temp_valid(false) {}
};

/**
 * @brief Read comprehensive Jetson system metrics with error handling
 * @return JetsonMetrics structure with temperature and power data
 *
 * Attempts to read system temperature and power consumption from multiple
 * hardware monitoring sources with robust error handling and validation.
 * Uses namespace constants for validation ranges.
 */
JetsonMetrics read_jetson_metrics() {
  JetsonMetrics metrics;

  try {
    // Temperature reading with validation
    metrics.temperature_c = static_cast<float>(read_jetson_temperature());
    metrics.temp_valid = (metrics.temperature_c > JetsonSpecs::MIN_REASONABLE_TEMPERATURE_C &&
                          metrics.temperature_c < JetsonSpecs::MAX_SAFE_TEMPERATURE_C);

    if (!metrics.temp_valid) {
      LOG_WARNING_F("Temperature reading out of range: %.1f°C", metrics.temperature_c);
    } else {
      LOG_DEBUG_F("Valid temperature reading: %.1f°C", metrics.temperature_c);
    }

    // Power reading with validation
    metrics.power_watts = static_cast<float>(read_jetson_power_consumption());
    metrics.power_valid = (metrics.power_watts >= JetsonSpecs::MIN_REASONABLE_POWER_W &&
                           metrics.power_watts <= JetsonSpecs::MAX_REASONABLE_POWER_W);

    if (!metrics.power_valid && metrics.power_watts > 0.0f) {
      LOG_WARNING_F("Power reading out of range: %.1f W", metrics.power_watts);
    } else if (metrics.power_valid) {
      LOG_DEBUG_F("Valid power reading: %.1f W", metrics.power_watts);
    } else {
      LOG_DEBUG("Power monitoring unavailable - acceptable for this benchmark");
    }
  } catch (const std::exception& e) {
    LOG_ERROR_F("Failed to read system metrics: %s", e.what());
    metrics.temp_valid = false;
    metrics.power_valid = false;
  }

  return metrics;
}

/**
 * @brief Execute comprehensive cuBLAS matrix multiplication benchmark
 * @param n Matrix dimension (n x n)
 * @return 0 on success, negative value on error
 *
 * Performs complete benchmarking workflow using cuBLAS library including
 * memory validation, matrix initialization, cuBLAS context management,
 * optimized SGEMM execution, accuracy verification, and comprehensive
 * performance analysis with system monitoring.
 */
int benchmark_cublas_implementation(int n) {
  PRINT_INFO("========== Jetson Orin Nano: cuBLAS Matrix Multiplication Benchmark ==========");
  PRINT_INFO("Hardware: Engineering Reference Developer Kit Super");
  PRINT_INFO_F("Matrix Size: %d x %d", n, n);
  PRINT_INFO("Implementation: cuBLAS SGEMM (vendor-optimized)");
  PRINT_INFO("Power Mode Analysis: 15W/25W/MAXN SUPER");

  // Validate memory requirements before proceeding
  if (!validate_memory_requirements(n)) {
    return -1;
  }

  size_t matrix_size = static_cast<size_t>(n) * n * sizeof(float);

  // Host memory allocation with error checking
  std::unique_ptr<float[]> h_A, h_B, h_C, h_C_ref;
  try {
    h_A = std::make_unique<float[]>(n * n);
    h_B = std::make_unique<float[]>(n * n);
    h_C = std::make_unique<float[]>(n * n);
    h_C_ref = std::make_unique<float[]>(n * n);
    LOG_DEBUG("Host memory allocation completed successfully");
  } catch (const std::bad_alloc& e) {
    LOG_CRITICAL_F("Host memory allocation failed: %s", e.what());
    PRINT_ERROR("Host memory allocation failed");
    return -1;
  }

  // Device memory allocation with safe wrappers
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  try {
    safe_cuda_malloc(&d_A, matrix_size, "matrix A");
    safe_cuda_malloc(&d_B, matrix_size, "matrix B");
    safe_cuda_malloc(&d_C, matrix_size, "matrix C");
  } catch (const std::exception& e) {
    LOG_CRITICAL_F("Device memory allocation failed: %s", e.what());
    // Cleanup any partial allocations
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  PRINT_INFO("Memory allocation completed successfully");

  // Initialize matrices with reproducible values
  try {
    PRINT_INFO("Initializing matrices with reproducible random values...");
    initialize_matrices(h_A.get(), h_B.get(), n, "random");
    LOG_DEBUG("Matrix initialization completed");

    // Copy matrices to device
    PRINT_INFO("Transferring matrices to GPU...");
    CUDA_CHECK(cudaMemcpy(d_A, h_A.get(), matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.get(), matrix_size, cudaMemcpyHostToDevice));
    LOG_DEBUG("Data transfer to device completed");
  } catch (const std::exception& e) {
    LOG_CRITICAL_F("Matrix initialization or transfer failed: %s", e.what());
    PRINT_ERROR("Matrix initialization failed");
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  // Initialize cuBLAS handle
  cublasHandle_t handle = nullptr;
  try {
    PRINT_INFO("Initializing cuBLAS library...");
    CUBLAS_CHECK(cublasCreate(&handle));
    LOG_INFO("cuBLAS context created successfully");
  } catch (const std::exception& e) {
    LOG_CRITICAL_F("cuBLAS initialization failed: %s", e.what());
    PRINT_ERROR("cuBLAS initialization failed");
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  // cuBLAS SGEMM parameters
  // C = alpha * A * B + beta * C
  const float alpha = 1.0f;
  const float beta = 0.0f;

  PRINT_INFO("cuBLAS Configuration:");
  PRINT_INFO("  Operation: C = alpha * A * B + beta * C");
  PRINT_INFO_F("  alpha = %.1f, beta = %.1f", alpha, beta);
  PRINT_INFO_F("  Matrix dimensions: A(%d,%d) × B(%d,%d) = C(%d,%d)", n, n, n, n, n, n);

  // Read pre-execution metrics
  double pre_temp = read_jetson_temperature();
  double pre_power = read_jetson_power_consumption();

  // Warmup execution to eliminate cold-start effects
  try {
    PRINT_INFO("Executing warmup cuBLAS call...");
    // cuBLAS uses column-major ordering, so we compute: C = B^T * A^T = (A * B)^T
    // Then interpret result as row-major C = A * B
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_B, n, d_A, n,
                             &beta, d_C, n));
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG_DEBUG("Warmup execution completed");
  } catch (const std::exception& e) {
    LOG_ERROR_F("Warmup execution failed: %s", e.what());
    PRINT_ERROR("Warmup execution failed");
    cublasDestroy(handle);
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  // Brief pause for thermal stabilization
  usleep(1000000);  // 1 second

  // Timed execution with CUDA events
  cudaEvent_t start, stop;
  float elapsed_ms = 0.0f;

  try {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    PRINT_INFO("Executing timed cuBLAS SGEMM...");
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_B, n, d_A, n,
                             &beta, d_C, n));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    PRINT_INFO_F("cuBLAS kernel completed in %.3f ms", elapsed_ms);
    LOG_INFO_F("Timed execution: %.3f ms", elapsed_ms);
  } catch (const std::exception& e) {
    LOG_ERROR_F("Timed execution failed: %s", e.what());
    PRINT_ERROR("Timed cuBLAS execution failed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  // Read post-execution metrics
  double post_temp = read_jetson_temperature();
  double post_power = read_jetson_power_consumption();

  // Copy results back to host
  try {
    PRINT_INFO("Transferring results from GPU...");
    CUDA_CHECK(cudaMemcpy(h_C.get(), d_C, matrix_size, cudaMemcpyDeviceToHost));
    LOG_DEBUG("Result transfer completed");
  } catch (const std::exception& e) {
    LOG_ERROR_F("Result copy failed: %s", e.what());
    PRINT_ERROR("Result copy from GPU failed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  // Calculate performance metrics
  int current_power_mode = detect_current_power_mode();
  PerformanceMetrics metrics = calculate_performance_metrics(n, elapsed_ms, current_power_mode);

  // Numerical accuracy verification against CPU reference
  PRINT_INFO("Verifying correctness against CPU reference...");
  auto cpu_start = std::chrono::high_resolution_clock::now();
  cpu_matrix_multiply_reference(h_A.get(), h_B.get(), h_C_ref.get(), n);
  auto cpu_end = std::chrono::high_resolution_clock::now();

  double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
  double accuracy_error = verify_accuracy(h_C.get(), h_C_ref.get(), n);

  // Print comprehensive results
  PRINT_INFO("============================================================");
  PRINT_INFO("CUBLAS IMPLEMENTATION RESULTS");
  PRINT_INFO("============================================================");
  PRINT_INFO_F("Matrix Size: %d x %d", n, n);
  PRINT_INFO_F("Elapsed Time: %.3f ms", elapsed_ms);
  PRINT_INFO_F("Measured Performance: %.2f GFLOPS", metrics.measured_gflops);
  PRINT_INFO_F("Memory Bandwidth: %.2f GB/s", metrics.memory_bandwidth_gbps);
  PRINT_INFO_F("Arithmetic Intensity: %.2f FLOPS/byte", metrics.arithmetic_intensity);
  PRINT_INFO_F("Algorithm Efficiency: %.1f%%", metrics.efficiency_percent);
  PRINT_INFO("------------------------------------------------------------");
  PRINT_INFO_F("Power Mode: %s", metrics.power_mode_name);
  PRINT_INFO_F("Temperature: Pre=%.1f°C, Post=%.1f°C (Δ=%.1f°C)", pre_temp, post_temp,
               post_temp - pre_temp);

  if (pre_power > 0 && post_power > 0) {
    double avg_power = (pre_power + post_power) / 2.0;
    PRINT_INFO_F("Power Consumption: Avg=%.2f W", avg_power);
    PRINT_INFO_F("Power Efficiency: %.2f GFLOPS/W", metrics.measured_gflops / avg_power);
  }

  PRINT_INFO("------------------------------------------------------------");
  PRINT_INFO_F("Numerical Accuracy: Mean Relative Error = %.2e", accuracy_error);
  PRINT_INFO_F("CPU Reference Time: %.2f ms (%.2fx slower than cuBLAS)", cpu_time_ms,
               cpu_time_ms / elapsed_ms);
  PRINT_INFO("============================================================");

  // Performance analysis and bottleneck identification
  print_performance_report(metrics, "cuBLAS SGEMM", n);

  // Cleanup resources
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cublasDestroy(handle);
  safe_cuda_free(d_A, "matrix A");
  safe_cuda_free(d_B, "matrix B");
  safe_cuda_free(d_C, "matrix C");

  LOG_INFO("cuBLAS benchmark completed successfully");
  return 0;
}

/**
 * @brief Main entry point for cuBLAS matrix multiplication benchmark
 * @param argc Argument count
 * @param argv Argument vector (expects matrix size as first argument)
 * @return 0 on success, 1 on error
 *
 * Validates command-line arguments, initializes logging and hardware
 * information display, and executes the comprehensive benchmark suite.
 */
int main(int argc, char** argv) {
  // Initialize enterprise logging system
  if (!ensure_directory_exists("../data/logs")) {
    std::cerr << "ERROR: Failed to create logs directory" << std::endl;
    return 1;
  }

  if (!Logger::getInstance().initialize("../data/logs/jetson_cublas_benchmark.log", LogLevel::INFO,
                                        true)) {
    std::cerr << "WARNING: Failed to initialize file logging, continuing with console output"
              << std::endl;
  }

  LOG_INFO("========================================");
  LOG_INFO("Jetson Orin Nano cuBLAS Benchmark");
  LOG_INFO("========================================");

  // Validate command-line arguments
  if (argc != 2) {
    PRINT_ERROR("Usage: ./cublas_benchmark <matrix_size>");
    PRINT_INFO("Example: ./cublas_benchmark 512");
    LOG_ERROR("Invalid command-line arguments");
    return 1;
  }

  int n = atoi(argv[1]);
  if (n <= 0 || n > 8192) {
    PRINT_ERROR_F("Invalid matrix size: %d (must be between 1 and 8192)", n);
    LOG_ERROR_F("Invalid matrix size: %d", n);
    return 1;
  }

  LOG_INFO_F("Matrix size: %d x %d", n, n);

  // Display comprehensive system information
  try {
    print_jetson_system_info();
  } catch (const std::exception& e) {
    LOG_WARNING_F("Failed to display system info: %s", e.what());
    PRINT_WARNING("Could not display complete system information");
  }

  // Execute benchmark
  int result = benchmark_cublas_implementation(n);

  if (result == 0) {
    LOG_INFO("Benchmark completed successfully");
    PRINT_INFO("\nBenchmark completed successfully");
  } else {
    LOG_ERROR("Benchmark failed");
    PRINT_ERROR("\nBenchmark failed");
  }

  return (result == 0) ? 0 : 1;
}
