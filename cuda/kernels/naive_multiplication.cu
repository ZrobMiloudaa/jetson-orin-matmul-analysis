/**
 * @file naive_multiplication.cu
 * @brief Naive matrix multiplication implementation for Jetson Orin Nano performance analysis
 * @author Jesse Moses (@Cre4T3Tiv3)
 * @date 2025
 * @copyright ByteStack Labs - MIT License
 *
 * Implements a baseline O(n^3) matrix multiplication algorithm for comprehensive
 * power mode analysis on the NVIDIA Jetson Orin Nano Engineering Reference
 * Developer Kit Super. Provides empirical validation of performance characteristics
 * across 15W, 25W, and MAXN SUPER power configurations.
 *
 * Mathematical Foundation: C[i,j] = sum(k=0 to n-1) A[i,k] * B[k,j]
 * Complexity: O(n^3) time, O(1) auxiliary space
 *
 * Target Hardware:
 * - Jetson Orin Nano Engineering Reference Developer Kit Super
 * - SM 8.7 (Ampere Architecture)
 * - 1024 CUDA Cores, 8 Streaming Multiprocessors
 * - 7.4GB LPDDR5 @ 68 GB/s theoretical bandwidth
 */
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
 * @brief Naive matrix multiplication CUDA kernel optimized for SM 8.7
 * @param A Input matrix A (n x n) in row-major order
 * @param B Input matrix B (n x n) in row-major order
 * @param C Output matrix C (n x n) in row-major order
 * @param n Matrix dimension
 *
 * Implements straightforward matrix multiplication with loop unrolling
 * for improved instruction-level parallelism. Each thread computes one
 * element of the result matrix using a dot product calculation.
 *
 * Thread mapping: (blockIdx.y * blockDim.y + threadIdx.y, blockIdx.x * blockDim.x + threadIdx.x)
 * corresponds to matrix element C[row][col].
 *
 * Performance characteristics:
 * - Global memory access pattern: Non-coalesced for matrix B
 * - Arithmetic intensity: 2n FLOPS per 3 memory accesses (low)
 * - Expected memory-bound behavior on Jetson Orin Nano
 */
__global__ void naive_matrix_mult_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                         float* __restrict__ C, int n) {
  // Calculate global thread indices
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Bounds checking for matrix dimensions
  if (row >= n || col >= n) return;

  float sum = 0.0f;

  // Unroll inner loop for better instruction-level parallelism
  // Compiler hint for optimization on Ampere architecture
#pragma unroll 4
  for (int k = 0; k < n; k++) {
    sum += A[row * n + k] * B[k * n + col];
  }

  C[row * n + col] = sum;
}

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

// /**
//  * @brief Detect current Jetson power mode from system configuration
//  * @return Power mode identifier (0=15W, 1=25W, 2=MAXN_SUPER, -1=unknown)
//  *
//  * Queries the nvpmodel utility to determine the active power configuration.
//  * Provides fallback handling for systems where nvpmodel is not available.
//  */
// int detect_current_power_mode() {
//   int current_power_mode = -1;

//   try {
//     FILE* mode_query = popen("nvpmodel -q 2>/dev/null | grep 'NV Power Mode' | tail -1", "r");
//     if (mode_query) {
//       char mode_line[256];
//       if (fgets(mode_line, sizeof(mode_line), mode_query)) {
//         if (strstr(mode_line, "15W")) {
//           current_power_mode = 0;
//         } else if (strstr(mode_line, "25W")) {
//           current_power_mode = 1;
//         } else if (strstr(mode_line, "MAXN")) {
//           current_power_mode = 2;
//         }
//         PRINT_INFO_F("Detected power mode: %s", current_power_mode >= 0
//                                                     ? POWER_MODE_SPECS[current_power_mode].name
//                                                     : "Unknown");
//       }
//       pclose(mode_query);
//     }
//   } catch (const std::exception& e) {
//     LOG_WARNING_F("Failed to detect power mode: %s", e.what());
//   }

//   if (current_power_mode == -1) {
//     PRINT_WARNING("Could not detect power mode - using default (MAXN_SUPER)");
//     current_power_mode = 2;  // Default to maximum performance mode
//   }

//   return current_power_mode;
// }

/**
 * @brief Execute comprehensive matrix multiplication benchmark
 * @param n Matrix dimension (n x n)
 * @return 0 on success, negative value on error
 *
 * Performs complete benchmarking workflow including memory validation,
 * matrix initialization, GPU kernel execution, accuracy verification,
 * and comprehensive performance analysis with system monitoring.
 */
int benchmark_naive_implementation(int n) {
  PRINT_INFO("========== Jetson Orin Nano: Matrix Multiplication Benchmark ==========");
  PRINT_INFO("Hardware: Engineering Reference Developer Kit Super");
  PRINT_INFO_F("Matrix Size: %d x %d", n, n);
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

    CUDA_CHECK(cudaMemcpy(d_A, h_A.get(), matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.get(), matrix_size, cudaMemcpyHostToDevice));
    LOG_DEBUG("Matrix data transferred to GPU successfully");
  } catch (const std::exception& e) {
    LOG_ERROR_F("Matrix initialization failed: %s", e.what());
    PRINT_ERROR("Matrix initialization failed");
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  // Optimal kernel configuration for Jetson Orin Nano
  dim3 block_size(16, 16);
  dim3 grid_size((n + block_size.x - 1) / block_size.x, (n + block_size.y - 1) / block_size.y);

  PRINT_INFO("Kernel Configuration:");
  PRINT_INFO_F("  Block size: %dx%d = %d threads", block_size.x, block_size.y,
               block_size.x * block_size.y);
  PRINT_INFO_F("  Grid size: %dx%d = %d blocks", grid_size.x, grid_size.y,
               grid_size.x * grid_size.y);
  PRINT_INFO_F("  Total threads: %d", grid_size.x * grid_size.y * block_size.x * block_size.y);

  // Pre-benchmark system metrics
  JetsonMetrics pre_metrics = read_jetson_metrics();
  PRINT_INFO("Pre-benchmark system state:");
  if (pre_metrics.temp_valid) {
    PRINT_INFO_F("  Temperature: %.1f°C", pre_metrics.temperature_c);
  } else {
    PRINT_INFO("  Temperature: Monitoring unavailable");
  }
  if (pre_metrics.power_valid) {
    PRINT_INFO_F("  Power: %.1f W", pre_metrics.power_watts);
  } else {
    PRINT_INFO("  Power: Monitoring unavailable");
  }

  // Warmup run to ensure GPU is at full performance state
  try {
    PRINT_INFO("Executing warmup kernel...");
    naive_matrix_mult_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check for kernel launch errors
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
      LOG_ERROR_F("Warmup kernel failed: %s", cudaGetErrorString(kernel_error));
      PRINT_ERROR_F("Warmup kernel failed: %s", cudaGetErrorString(kernel_error));
      throw std::runtime_error("Kernel execution failed");
    }

    LOG_DEBUG("Warmup kernel completed successfully");
  } catch (const std::exception& e) {
    LOG_ERROR_F("Warmup execution failed: %s", e.what());
    PRINT_ERROR("Warmup execution failed");
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  // Wait for thermal and power stability
  LOG_DEBUG("Waiting for system stabilization...");
  usleep(1000000);  // 1 second stabilization

  // Timed benchmark run with CUDA events
  cudaEvent_t start, stop;
  float elapsed_ms = 0.0f;

  try {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    PRINT_INFO("Executing timed kernel...");

    CUDA_CHECK(cudaEventRecord(start));
    naive_matrix_mult_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Verify kernel execution success
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
      LOG_ERROR_F("Timed kernel failed: %s", cudaGetErrorString(kernel_error));
      PRINT_ERROR_F("Timed kernel failed: %s", cudaGetErrorString(kernel_error));
      throw std::runtime_error("Kernel execution failed");
    }

    PRINT_INFO_F("Kernel execution completed in %.3f ms", elapsed_ms);
  } catch (const std::exception& e) {
    LOG_ERROR_F("Timed execution failed: %s", e.what());
    PRINT_ERROR("Timed execution failed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  // Post-benchmark system metrics
  JetsonMetrics post_metrics = read_jetson_metrics();

  // Copy results back to host
  try {
    CUDA_CHECK(cudaMemcpy(h_C.get(), d_C, matrix_size, cudaMemcpyDeviceToHost));
    LOG_DEBUG("Results copied back to host successfully");
  } catch (const std::exception& e) {
    LOG_ERROR_F("Result copy failed: %s", e.what());
    PRINT_ERROR("Result copy failed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  // // Detect current power mode for accurate analysis
  int current_power_mode = detect_current_power_mode();

  // Calculate comprehensive performance metrics
  PerformanceMetrics metrics = calculate_performance_metrics(n, elapsed_ms, current_power_mode);

  // CPU reference for correctness verification
  PRINT_INFO("Verifying correctness against CPU reference...");
  auto cpu_start = std::chrono::high_resolution_clock::now();
  try {
    cpu_matrix_multiply_reference(h_A.get(), h_B.get(), h_C_ref.get(), n);
  } catch (const std::exception& e) {
    LOG_ERROR_F("CPU reference computation failed: %s", e.what());
    PRINT_ERROR("CPU reference computation failed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }
  auto cpu_end = std::chrono::high_resolution_clock::now();

  double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
  double accuracy_error = verify_accuracy(h_C.get(), h_C_ref.get(), n);

  // Generate comprehensive results output
  PRINT_INFO("============================================================");
  PRINT_INFO("PERFORMANCE RESULTS");
  PRINT_INFO("============================================================");

  print_performance_report(metrics, "Naive", n);

  PRINT_INFO("Detailed Analysis:");
  PRINT_INFO_F("Execution time: %.3f ms", elapsed_ms);
  PRINT_INFO_F("Performance: %.2f GFLOPS", metrics.measured_gflops);

  // Realistic efficiency context
  PRINT_INFO("Performance Context:");
  PRINT_INFO("This is a NAIVE O(n^3) implementation - optimization potential is significant");
  PRINT_INFO("Expected improvements with blocked algorithms: 2-4x performance");
  PRINT_INFO("Expected improvements with cuBLAS: 5-10x performance");

  PRINT_INFO("CPU Comparison:");
  PRINT_INFO_F("CPU Reference Time: %.3f ms", cpu_time_ms);
  PRINT_INFO_F("GPU Speedup: %.1fx over CPU", cpu_time_ms / elapsed_ms);
  PRINT_INFO_F("Numerical Accuracy (Mean Rel. Error): %.2e", accuracy_error);

  PRINT_INFO("System Impact:");
  if (pre_metrics.temp_valid && post_metrics.temp_valid) {
    float temp_delta = post_metrics.temperature_c - pre_metrics.temperature_c;
    PRINT_INFO_F("Temperature change: %.1f°C -> %.1f°C (Delta = %.1f°C)", pre_metrics.temperature_c,
                 post_metrics.temperature_c, temp_delta);
  }

  if (pre_metrics.power_valid && post_metrics.power_valid) {
    double avg_power = (pre_metrics.power_watts + post_metrics.power_watts) / 2.0;
    PRINT_INFO_F("Average power consumption: %.1f W", avg_power);
    if (avg_power > 0.0) {
      PRINT_INFO_F("Energy efficiency: %.1f GFLOPS/W", metrics.measured_gflops / avg_power);
    }
  }

  // Performance analysis with realistic expectations
  PRINT_INFO("Optimization Roadmap:");
  if (metrics.memory_bandwidth_gbps < JetsonSpecs::MEMORY_BW_GBPS * 0.1) {
    PRINT_INFO("PRIORITY: Memory access optimization (currently using <10% of bandwidth)");
    PRINT_INFO("Consider: Blocked matrix multiplication, memory coalescing");
  }

  if (metrics.efficiency_percent < 20.0) {
    PRINT_INFO("OPPORTUNITY: Significant algorithmic improvements possible");
    PRINT_INFO("Consider: Tiled algorithms, shared memory utilization");
  }

  PRINT_INFO("Week 2 target: 2-4x improvement with cache blocking");
  PRINT_INFO("Week 3+ target: Approach cuBLAS performance levels");

  // Cleanup resources
  try {
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    LOG_DEBUG("Resource cleanup completed successfully");
  } catch (const std::exception& e) {
    LOG_WARNING_F("Cleanup encountered issues: %s", e.what());
  }

  PRINT_INFO("Benchmark completed successfully");
  return 0;
}

/**
 * @brief Main function with enhanced system validation and error handling
 * @param argc Argument count
 * @param argv Argument vector
 * @return 0 on success, non-zero on error
 *
 * Orchestrates the complete benchmarking workflow with system validation,
 * CUDA device verification, and comprehensive error handling throughout.
 */
int main(int argc, char** argv) {
  // Initialize logging system
  // if (!Logger::getInstance().initialize("jetson_matrix_benchmark.log", LogLevel::INFO, true)) {
  //   std::cerr << "Failed to initialize logging system" << std::endl;
  //   return -1;
  // }

  // Ensure logs directory exists and initialize logging
  if (!ensure_directory_exists("../data/logs")) {
    std::cerr << "ERROR: Failed to create results/logs directory" << std::endl;
    return -1;
  }

  if (!Logger::getInstance().initialize("../data/logs/jetson_matrix_benchmark.log", LogLevel::INFO,
                                        true)) {
    std::cerr << "ERROR: Failed to initialize logging system" << std::endl;
    return -1;
  }

  PRINT_INFO("Jetson Orin Nano: Matrix Multiplication Benchmarking");
  PRINT_INFO("3-Power Mode Analysis - Naive Implementation");
  PRINT_INFO("Mathematical Foundation: O(n^3) Empirical Validation");
  PRINT_INFO("Accurate performance baselines");

  try {
    // Print system information
    print_jetson_system_info();

    // Check CUDA device capabilities
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
      LOG_CRITICAL("No CUDA devices found!");
      PRINT_ERROR("No CUDA devices found!");
      return -1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    PRINT_INFO("CUDA Device Verification:");
    PRINT_INFO_F("Device: %s", prop.name);
    PRINT_INFO_F("Compute Capability: %d.%d", prop.major, prop.minor);
    PRINT_INFO_F("Global Memory: %.2f GB", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    PRINT_INFO_F("Max Threads per Block: %d", prop.maxThreadsPerBlock);
    PRINT_INFO_F("Multiprocessor Count: %d", prop.multiProcessorCount);
    PRINT_INFO_F("Max Shared Memory per Block: %zu KB", prop.sharedMemPerBlock / 1024);

    // Validate that we're on the correct hardware
    if (prop.major != JetsonSpecs::COMPUTE_CAPABILITY_MAJOR ||
        prop.minor != JetsonSpecs::COMPUTE_CAPABILITY_MINOR) {
      PRINT_WARNING_F("Expected SM %d.%d (Jetson Orin), found SM %d.%d",
                      JetsonSpecs::COMPUTE_CAPABILITY_MAJOR, JetsonSpecs::COMPUTE_CAPABILITY_MINOR,
                      prop.major, prop.minor);
    }

    // Parse command line arguments with validation
    if (argc < 2) {
      PRINT_INFO_F("Usage: %s <matrix_size> [matrix_size2] [matrix_size3] ...", argv[0]);
      PRINT_INFO_F("Example: %s 64 128 256", argv[0]);
      PRINT_INFO("Running default test sizes for verification...");

      int default_sizes[] = {64, 128};
      int num_defaults = sizeof(default_sizes) / sizeof(default_sizes[0]);

      for (int i = 0; i < num_defaults; i++) {
        if (benchmark_naive_implementation(default_sizes[i]) != 0) {
          PRINT_ERROR_F("Benchmark failed for matrix size %d", default_sizes[i]);
          return -1;
        }
        if (i < num_defaults - 1) {
          PRINT_INFO("-------------------------------------------------------------");
        }
      }
    } else {
      for (int i = 1; i < argc; i++) {
        int n = atoi(argv[i]);

        if (n <= 0) {
          PRINT_ERROR_F("Invalid matrix size: %d (must be positive)", n);
          continue;
        }

        if (n > 4096) {
          PRINT_ERROR_F("Matrix size %d too large (max 4096 for memory safety)", n);
          continue;
        }

        if (benchmark_naive_implementation(n) != 0) {
          PRINT_ERROR_F("Benchmark failed for matrix size %d", n);
          return -1;
        }

        if (i < argc - 1) {
          PRINT_INFO("-------------------------------------------------------------");
        }
      }
    }

    PRINT_INFO("=================================================================");
    PRINT_INFO("BENCHMARK SUITE COMPLETED");
    PRINT_INFO("=================================================================");
    PRINT_INFO("Next steps:");
    PRINT_INFO(
        "1. Run comprehensive 3-mode analysis: make full-analysis OR sudo python3 "
        "benchmarks/multi_power_mode_benchmark.py");
    PRINT_INFO(
        "2. Generate visualizations: make visualize OR python3 results/visualize_power_modes.py");
    PRINT_INFO("3. Review power-performance characterization results");
    PRINT_INFO("4. Week 2: Implement blocked matrix multiplication for cache optimization");
  } catch (const std::exception& e) {
    LOG_CRITICAL_F("Unhandled exception in main: %s", e.what());
    PRINT_ERROR_F("Unhandled exception: %s", e.what());
    return -1;
  } catch (...) {
    LOG_CRITICAL("Unknown exception occurred");
    PRINT_ERROR("Unknown exception occurred");
    return -1;
  }

  Logger::getInstance().shutdown();
  return 0;
}
