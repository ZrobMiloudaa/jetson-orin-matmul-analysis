/**
 * @file tensor_core_multiplication.cu
 * @brief FP16 Tensor Core matrix multiplication for Jetson Orin Nano performance analysis
 * @author Jesse Moses (@Cre4T3Tiv3)
 * @date 2025
 * @copyright ByteStack Labs - MIT License
 *
 * Implements matrix multiplication using NVIDIA Tensor Cores (WMMA API) with FP16
 * precision for maximum performance on the Jetson Orin Nano Engineering Reference
 * Developer Kit Super. Provides optimal mixed-precision computation across 15W, 25W,
 * and MAXN SUPER power configurations.
 *
 * Mathematical Foundation: C[i,j] = sum(k=0 to n-1) A[i,k] * B[k,j]
 * Implementation: FP16 Tensor Core WMMA with FP32 accumulation
 * Complexity: O(n^3) time with hardware acceleration
 *
 * Target Hardware:
 * - Jetson Orin Nano Engineering Reference Developer Kit Super
 * - SM 8.7 (Ampere Architecture with Tensor Cores)
 * - 1024 CUDA Cores + Tensor Core acceleration
 * - 7.4GB LPDDR5 @ 68 GB/s theoretical bandwidth
 *
 * Tensor Core Details:
 * - WMMA tile size: 16×16×16 (M×N×K)
 * - Input precision: FP16
 * - Accumulation precision: FP32
 * - Theoretical speedup: 8-16x over FP32 CUDA cores
 */

#include <mma.h>
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

using nvcuda::wmma::accumulator;
using nvcuda::wmma::fill_fragment;
using nvcuda::wmma::fragment;
using nvcuda::wmma::load_matrix_sync;
using nvcuda::wmma::matrix_a;
using nvcuda::wmma::matrix_b;
using nvcuda::wmma::mem_row_major;
using nvcuda::wmma::mma_sync;
using nvcuda::wmma::row_major;
using nvcuda::wmma::store_matrix_sync;

// WMMA tile dimensions for Ampere Tensor Cores
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

bool ensure_directory_exists(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
    return true;
  }

  std::string cmd = "mkdir -p \"" + path + "\"";
  int result = system(cmd.c_str());
  if (result != 0) {
    std::cerr << "ERROR: Failed to create directory: " << path << " (exit code: " << result << ")"
              << std::endl;
    return false;
  }

  return (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

/**
 * @brief Tensor Core matrix multiplication kernel using WMMA API
 * @param A Input matrix A (n x n) in row-major order (FP32, converted to FP16)
 * @param B Input matrix B (n x n) in row-major order (FP32, converted to FP16)
 * @param C Output matrix C (n x n) in row-major order (FP32)
 * @param n Matrix dimension (must be multiple of 16)
 *
 * Uses Tensor Cores via WMMA API for FP16 matrix multiplication with FP32 accumulation.
 * Each warp computes a 16×16 output tile using 16×16×16 WMMA operations.
 *
 * Performance characteristics:
 * - Utilizes Tensor Cores for 8-16x FP16 acceleration
 * - Mixed precision: FP16 compute, FP32 accumulate
 * - Optimized memory access patterns for Ampere architecture
 */
__global__ void tensor_core_wmma_kernel(const half* __restrict__ A, const half* __restrict__ B,
                                        float* __restrict__ C, int n) {
  // Warp and block index
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare WMMA fragments
  fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
  fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
  fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

  // Initialize accumulator to zero
  fill_fragment(acc_frag, 0.0f);

  // Bounds check
  if (warpM * WMMA_M >= n || warpN * WMMA_N >= n) return;

  // Compute matrix multiplication using Tensor Cores
  for (int k = 0; k < n; k += WMMA_K) {
    int aRow = warpM * WMMA_M;
    int aCol = k;
    int bRow = k;
    int bCol = warpN * WMMA_N;

    // Bounds check for partial tiles
    if (aRow < n && aCol < n && bRow < n && bCol < n) {
      // Load A and B fragments from global memory
      load_matrix_sync(a_frag, A + aRow * n + aCol, n);
      load_matrix_sync(b_frag, B + bRow * n + bCol, n);

      // Perform Tensor Core matrix multiply-accumulate
      mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Store result to global memory
  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;
  if (cRow < n && cCol < n) {
    store_matrix_sync(C + cRow * n + cCol, acc_frag, n, mem_row_major);
  }
}

/**
 * @brief Convert FP32 matrix to FP16 for Tensor Core computation
 * @param fp32_data Input FP32 data
 * @param fp16_data Output FP16 data
 * @param size Number of elements
 */
__global__ void convert_fp32_to_fp16(const float* fp32_data, half* fp16_data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    fp16_data[idx] = __float2half(fp32_data[idx]);
  }
}

/**
 * @brief Execute comprehensive Tensor Core matrix multiplication benchmark
 * @param n Matrix dimension (n x n)
 * @return 0 on success, negative value on error
 */
int benchmark_tensor_core_implementation(int n) {
  PRINT_INFO(
      "========== Jetson Orin Nano: Tensor Core Matrix Multiplication Benchmark "
      "==========");
  PRINT_INFO("Hardware: Engineering Reference Developer Kit Super");
  PRINT_INFO_F("Matrix Size: %d x %d", n, n);
  PRINT_INFO("Implementation: FP16 Tensor Cores (WMMA) with FP32 accumulation");
  PRINT_INFO("Power Mode Analysis: 15W/25W/MAXN SUPER");

  // Tensor Cores require dimensions to be multiples of 16
  if (n % 16 != 0) {
    PRINT_WARNING_F("Matrix size %d not multiple of 16, will be padded", n);
  }

  int padded_n = ((n + 15) / 16) * 16;  // Round up to next multiple of 16

  if (!validate_memory_requirements(padded_n)) {
    return -1;
  }

  size_t matrix_size = static_cast<size_t>(n) * n * sizeof(float);
  size_t padded_matrix_size = static_cast<size_t>(padded_n) * padded_n * sizeof(float);
  size_t fp16_matrix_size = static_cast<size_t>(padded_n) * padded_n * sizeof(half);

  // Host memory allocation
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

  // Device memory allocation (FP32 for input, FP16 for computation, FP32 for output)
  float *d_A_fp32 = nullptr, *d_B_fp32 = nullptr, *d_C = nullptr;
  half *d_A_fp16 = nullptr, *d_B_fp16 = nullptr;

  try {
    safe_cuda_malloc(&d_A_fp32, padded_matrix_size, "matrix A FP32");
    safe_cuda_malloc(&d_B_fp32, padded_matrix_size, "matrix B FP32");
    safe_cuda_malloc(&d_C, padded_matrix_size, "matrix C FP32");
    safe_cuda_malloc(&d_A_fp16, fp16_matrix_size, "matrix A FP16");
    safe_cuda_malloc(&d_B_fp16, fp16_matrix_size, "matrix B FP16");
  } catch (const std::exception& e) {
    LOG_CRITICAL_F("Device memory allocation failed: %s", e.what());
    safe_cuda_free(d_A_fp32, "matrix A FP32");
    safe_cuda_free(d_B_fp32, "matrix B FP32");
    safe_cuda_free(d_C, "matrix C FP32");
    safe_cuda_free(d_A_fp16, "matrix A FP16");
    safe_cuda_free(d_B_fp16, "matrix B FP16");
    return -1;
  }

  PRINT_INFO("Memory allocation completed successfully");

  // Initialize matrices with reproducible values
  try {
    PRINT_INFO("Initializing matrices with reproducible random values...");
    initialize_matrices(h_A.get(), h_B.get(), n, "random");
    LOG_DEBUG("Matrix initialization completed");

    // Copy to device and pad if necessary
    PRINT_INFO("Transferring matrices to GPU...");
    if (n == padded_n) {
      CUDA_CHECK(cudaMemcpy(d_A_fp32, h_A.get(), matrix_size, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_B_fp32, h_B.get(), matrix_size, cudaMemcpyHostToDevice));
    } else {
      // Pad matrices to multiple of 16
      CUDA_CHECK(cudaMemset(d_A_fp32, 0, padded_matrix_size));
      CUDA_CHECK(cudaMemset(d_B_fp32, 0, padded_matrix_size));
      for (int i = 0; i < n; i++) {
        CUDA_CHECK(cudaMemcpy(d_A_fp32 + i * padded_n, h_A.get() + i * n, n * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B_fp32 + i * padded_n, h_B.get() + i * n, n * sizeof(float),
                              cudaMemcpyHostToDevice));
      }
    }

    // Convert FP32 to FP16 for Tensor Core computation
    int convert_threads = 256;
    int convert_blocks = (padded_n * padded_n + convert_threads - 1) / convert_threads;
    convert_fp32_to_fp16<<<convert_blocks, convert_threads>>>(d_A_fp32, d_A_fp16,
                                                              padded_n * padded_n);
    convert_fp32_to_fp16<<<convert_blocks, convert_threads>>>(d_B_fp32, d_B_fp16,
                                                              padded_n * padded_n);
    CUDA_CHECK(cudaDeviceSynchronize());

    LOG_DEBUG("FP32 to FP16 conversion completed");
  } catch (const std::exception& e) {
    LOG_CRITICAL_F("Matrix initialization or transfer failed: %s", e.what());
    PRINT_ERROR("Matrix initialization failed");
    safe_cuda_free(d_A_fp32, "matrix A FP32");
    safe_cuda_free(d_B_fp32, "matrix B FP32");
    safe_cuda_free(d_C, "matrix C FP32");
    safe_cuda_free(d_A_fp16, "matrix A FP16");
    safe_cuda_free(d_B_fp16, "matrix B FP16");
    return -1;
  }

  // Configure kernel launch parameters
  // Each warp handles one 16×16 tile
  dim3 block_size(128);  // 4 warps per block
  dim3 grid_size((padded_n / WMMA_M + 3) / 4, (padded_n / WMMA_N + 0) / 1);

  PRINT_INFO("Kernel Configuration:");
  PRINT_INFO_F("  WMMA tile size: %dx%dx%d", WMMA_M, WMMA_N, WMMA_K);
  PRINT_INFO_F("  Block size: %d threads (4 warps)", block_size.x);
  PRINT_INFO_F("  Grid size: %dx%d = %d blocks", grid_size.x, grid_size.y,
               grid_size.x * grid_size.y);
  PRINT_INFO_F("  Padded matrix size: %dx%d (original: %dx%d)", padded_n, padded_n, n, n);

  double pre_temp = read_jetson_temperature();
  double pre_power = read_jetson_power_consumption();

  // Warmup execution
  try {
    PRINT_INFO("Executing warmup kernel...");
    tensor_core_wmma_kernel<<<grid_size, block_size>>>(d_A_fp16, d_B_fp16, d_C, padded_n);
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    PRINT_ERROR("Warmup execution failed");
    safe_cuda_free(d_A_fp32, "matrix A FP32");
    safe_cuda_free(d_B_fp32, "matrix B FP32");
    safe_cuda_free(d_C, "matrix C FP32");
    safe_cuda_free(d_A_fp16, "matrix A FP16");
    safe_cuda_free(d_B_fp16, "matrix B FP16");
    return -1;
  }

  usleep(1000000);  // 1 second

  // Timed execution
  cudaEvent_t start, stop;
  float elapsed_ms = 0.0f;

  try {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    PRINT_INFO("Executing timed Tensor Core kernel...");
    CUDA_CHECK(cudaEventRecord(start));
    tensor_core_wmma_kernel<<<grid_size, block_size>>>(d_A_fp16, d_B_fp16, d_C, padded_n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    PRINT_INFO_F("Tensor Core kernel completed in %.3f ms", elapsed_ms);
  } catch (const std::exception& e) {
    PRINT_ERROR("Timed execution failed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    safe_cuda_free(d_A_fp32, "matrix A FP32");
    safe_cuda_free(d_B_fp32, "matrix B FP32");
    safe_cuda_free(d_C, "matrix C FP32");
    safe_cuda_free(d_A_fp16, "matrix A FP16");
    safe_cuda_free(d_B_fp16, "matrix B FP16");
    return -1;
  }

  double post_temp = read_jetson_temperature();
  double post_power = read_jetson_power_consumption();

  // Copy results back (extract non-padded region)
  try {
    PRINT_INFO("Transferring results from GPU...");
    if (n == padded_n) {
      CUDA_CHECK(cudaMemcpy(h_C.get(), d_C, matrix_size, cudaMemcpyDeviceToHost));
    } else {
      for (int i = 0; i < n; i++) {
        CUDA_CHECK(cudaMemcpy(h_C.get() + i * n, d_C + i * padded_n, n * sizeof(float),
                              cudaMemcpyDeviceToHost));
      }
    }
    LOG_DEBUG("Result transfer completed");
  } catch (const std::exception& e) {
    LOG_ERROR_F("Result copy failed: %s", e.what());
    PRINT_ERROR("Result copy from GPU failed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    safe_cuda_free(d_A_fp32, "matrix A FP32");
    safe_cuda_free(d_B_fp32, "matrix B FP32");
    safe_cuda_free(d_C, "matrix C FP32");
    safe_cuda_free(d_A_fp16, "matrix A FP16");
    safe_cuda_free(d_B_fp16, "matrix B FP16");
    return -1;
  }

  // Calculate performance metrics (based on original matrix size, not padded)
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
  PRINT_INFO("TENSOR CORE (FP16) IMPLEMENTATION RESULTS");
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
  PRINT_INFO_F("CPU Reference Time: %.2f ms (%.2fx slower than Tensor Cores)", cpu_time_ms,
               cpu_time_ms / elapsed_ms);
  PRINT_INFO("============================================================");

  // Performance analysis
  print_performance_report(metrics, "Tensor Core WMMA (FP16)", n);

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  safe_cuda_free(d_A_fp32, "matrix A FP32");
  safe_cuda_free(d_B_fp32, "matrix B FP32");
  safe_cuda_free(d_C, "matrix C FP32");
  safe_cuda_free(d_A_fp16, "matrix A FP16");
  safe_cuda_free(d_B_fp16, "matrix B FP16");

  LOG_INFO("Tensor Core benchmark completed successfully");
  return 0;
}

/**
 * @brief Main entry point for Tensor Core matrix multiplication benchmark
 */
int main(int argc, char** argv) {
  // Initialize logging
  if (!ensure_directory_exists("../data/logs")) {
    std::cerr << "ERROR: Failed to create logs directory" << std::endl;
    return 1;
  }

  if (!Logger::getInstance().initialize("../data/logs/jetson_tensor_core_benchmark.log",
                                        LogLevel::INFO, true)) {
    std::cerr << "WARNING: Failed to initialize file logging, continuing with console output"
              << std::endl;
  }

  LOG_INFO("========================================");
  LOG_INFO("Jetson Orin Nano Tensor Core Benchmark");
  LOG_INFO("========================================");

  // Validate command-line arguments
  if (argc != 2) {
    PRINT_ERROR("Usage: ./tensor_core_benchmark <matrix_size>");
    PRINT_INFO("Example: ./tensor_core_benchmark 512");
    PRINT_INFO("Note: Size will be padded to multiple of 16 for Tensor Cores");
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

  // Display system information
  try {
    print_jetson_system_info();
  } catch (const std::exception& e) {
    LOG_WARNING_F("Failed to display system info: %s", e.what());
    PRINT_WARNING("Could not display complete system information");
  }

  // Execute benchmark
  int result = benchmark_tensor_core_implementation(n);

  if (result == 0) {
    LOG_INFO("Benchmark completed successfully");
    PRINT_INFO("\nBenchmark completed successfully");
  } else {
    LOG_ERROR("Benchmark failed");
    PRINT_ERROR("\nBenchmark failed");
  }

  return (result == 0) ? 0 : 1;
}
