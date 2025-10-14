/**
 * @file blocked_multiplication.cu
 * @brief Cache-blocked matrix multiplication for Jetson Orin Nano (Week 2)
 * @author Jesse Moses (@Cre4T3Tiv3)
 * @date 2025
 * @copyright ByteStack Labs - MIT License
 */

#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../utils/common.h"
#include "../utils/logger.h"

#define TILE_SIZE 32

bool ensure_directory_exists(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
    return true;
  }
  std::string cmd = "mkdir -p \"" + path + "\"";
  int result = system(cmd.c_str());
  return (result == 0 && stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

__global__ void blocked_matrix_mult_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                           float* __restrict__ C, int n) {
  __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;

  float sum = 0.0f;
  int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

  for (int t = 0; t < numTiles; t++) {
    int a_col = t * TILE_SIZE + tx;
    As[ty][tx] = (row < n && a_col < n) ? A[row * n + a_col] : 0.0f;

    int b_row = t * TILE_SIZE + ty;
    Bs[ty][tx] = (b_row < n && col < n) ? B[b_row * n + col] : 0.0f;

    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
      sum += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }

  if (row < n && col < n) {
    C[row * n + col] = sum;
  }
}

int benchmark_blocked_implementation(int n) {
  PRINT_INFO("========== Jetson Orin Nano: Blocked Matrix Multiplication Benchmark ==========");
  PRINT_INFO("Hardware: Engineering Reference Developer Kit Super");
  PRINT_INFO_F("Matrix Size: %d x %d", n, n);
  PRINT_INFO("Implementation: Cache-blocked with shared memory");
  PRINT_INFO_F("Tile Size: %d x %d", TILE_SIZE, TILE_SIZE);

  if (!validate_memory_requirements(n)) return -1;

  size_t matrix_size = static_cast<size_t>(n) * n * sizeof(float);

  std::unique_ptr<float[]> h_A, h_B, h_C, h_C_ref;
  try {
    h_A = std::make_unique<float[]>(n * n);
    h_B = std::make_unique<float[]>(n * n);
    h_C = std::make_unique<float[]>(n * n);
    h_C_ref = std::make_unique<float[]>(n * n);
  } catch (const std::bad_alloc& e) {
    PRINT_ERROR("Host memory allocation failed");
    return -1;
  }

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  try {
    safe_cuda_malloc(&d_A, matrix_size, "matrix A");
    safe_cuda_malloc(&d_B, matrix_size, "matrix B");
    safe_cuda_malloc(&d_C, matrix_size, "matrix C");
  } catch (const std::exception& e) {
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  PRINT_INFO("Memory allocation completed");

  try {
    PRINT_INFO("Initializing matrices with reproducible values...");
    initialize_matrices(h_A.get(), h_B.get(), n, "random");
    CUDA_CHECK(cudaMemcpy(d_A, h_A.get(), matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.get(), matrix_size, cudaMemcpyHostToDevice));
  } catch (const std::exception& e) {
    PRINT_ERROR("Matrix initialization failed");
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 grid_size((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

  PRINT_INFO("Kernel Configuration:");
  PRINT_INFO_F("  Block size: %dx%d = %d threads", block_size.x, block_size.y,
               block_size.x * block_size.y);
  PRINT_INFO_F("  Grid size: %dx%d = %d blocks", grid_size.x, grid_size.y,
               grid_size.x * grid_size.y);

  double pre_temp = read_jetson_temperature();
  double pre_power = read_jetson_power_consumption();

  try {
    PRINT_INFO("Executing warmup kernel...");
    blocked_matrix_mult_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    PRINT_ERROR("Warmup execution failed");
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  usleep(1000000);

  cudaEvent_t start, stop;
  float elapsed_ms = 0.0f;

  try {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    PRINT_INFO("Executing timed kernel...");
    CUDA_CHECK(cudaEventRecord(start));
    blocked_matrix_mult_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    PRINT_INFO_F("Kernel completed in %.3f ms", elapsed_ms);
  } catch (const std::exception& e) {
    PRINT_ERROR("Timed execution failed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  double post_temp = read_jetson_temperature();
  double post_power = read_jetson_power_consumption();

  try {
    CUDA_CHECK(cudaMemcpy(h_C.get(), d_C, matrix_size, cudaMemcpyDeviceToHost));
  } catch (const std::exception& e) {
    PRINT_ERROR("Result copy failed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    safe_cuda_free(d_A, "matrix A");
    safe_cuda_free(d_B, "matrix B");
    safe_cuda_free(d_C, "matrix C");
    return -1;
  }

  int current_power_mode = detect_current_power_mode();
  PerformanceMetrics metrics = calculate_performance_metrics(n, elapsed_ms, current_power_mode);

  PRINT_INFO("Verifying correctness against CPU reference...");
  auto cpu_start = std::chrono::high_resolution_clock::now();
  cpu_matrix_multiply_reference(h_A.get(), h_B.get(), h_C_ref.get(), n);
  auto cpu_end = std::chrono::high_resolution_clock::now();

  double cpu_time_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
  double accuracy_error = verify_accuracy(h_C.get(), h_C_ref.get(), n);

  PRINT_INFO("============================================================");
  PRINT_INFO("BLOCKED IMPLEMENTATION RESULTS");
  PRINT_INFO("============================================================");

  print_performance_report(metrics, "Blocked", n);

  PRINT_INFO("Implementation Details:");
  PRINT_INFO_F("Tile size: %dx%d", TILE_SIZE, TILE_SIZE);
  PRINT_INFO_F("Execution time: %.3f ms", elapsed_ms);
  PRINT_INFO_F("Performance: %.2f GFLOPS", metrics.measured_gflops);
  PRINT_INFO("CPU Comparison:");
  PRINT_INFO_F("CPU time: %.3f ms", cpu_time_ms);
  PRINT_INFO_F("GPU speedup: %.1fx over CPU", cpu_time_ms / elapsed_ms);
  PRINT_INFO_F("Numerical accuracy: %.2e", accuracy_error);

  if (pre_temp > 0 && post_temp > 0) {
    PRINT_INFO_F("Temperature: %.1f°C -> %.1f°C (delta=%.1f°C)", pre_temp, post_temp,
                 post_temp - pre_temp);
  }

  if (pre_power > 0 && post_power > 0) {
    double avg_power = (pre_power + post_power) / 2.0;
    PRINT_INFO_F("Average power: %.1f W", avg_power);
    PRINT_INFO_F("Energy efficiency: %.1f GFLOPS/W", metrics.measured_gflops / avg_power);
  }

  safe_cuda_free(d_A, "matrix A");
  safe_cuda_free(d_B, "matrix B");
  safe_cuda_free(d_C, "matrix C");
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  PRINT_INFO("Blocked benchmark completed successfully");
  return 0;
}

int main(int argc, char** argv) {
  if (!ensure_directory_exists("../data/logs")) {
    std::cerr << "ERROR: Failed to create results/logs directory" << std::endl;
    return -1;
  }

  if (!Logger::getInstance().initialize("../data/logs/jetson_blocked_benchmark.log", LogLevel::INFO,
                                        true)) {
    std::cerr << "ERROR: Failed to initialize logging" << std::endl;
    return -1;
  }

  PRINT_INFO("Jetson Orin Nano: Blocked Matrix Multiplication (Week 2)");
  PRINT_INFO("Cache-optimized implementation with shared memory");

  try {
    print_jetson_system_info();

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
      PRINT_ERROR("No CUDA devices found");
      return -1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    PRINT_INFO("CUDA Device:");
    PRINT_INFO_F("  Name: %s", prop.name);
    PRINT_INFO_F("  Compute: %d.%d", prop.major, prop.minor);

    if (argc < 2) {
      PRINT_INFO("Running default test sizes...");
      int default_sizes[] = {64, 128, 256, 512, 1024};
      for (int i = 0; i < 5; i++) {
        if (benchmark_blocked_implementation(default_sizes[i]) != 0) {
          PRINT_ERROR_F("Benchmark failed for size %d", default_sizes[i]);
          return -1;
        }
        if (i < 4) PRINT_INFO("-------------------------------------------------------------");
      }
    } else {
      for (int i = 1; i < argc; i++) {
        int n = atoi(argv[i]);
        if (n <= 0 || n > 4096) {
          PRINT_ERROR_F("Invalid matrix size: %d", n);
          continue;
        }
        if (benchmark_blocked_implementation(n) != 0) {
          PRINT_ERROR_F("Benchmark failed for size %d", n);
          return -1;
        }
        if (i < argc - 1)
          PRINT_INFO("-------------------------------------------------------------");
      }
    }

    PRINT_INFO("=================================================================");
    PRINT_INFO("BLOCKED IMPLEMENTATION BENCHMARK COMPLETE");
    PRINT_INFO("=================================================================");
  } catch (const std::exception& e) {
    PRINT_ERROR_F("Fatal error: %s", e.what());
    return -1;
  }

  Logger::getInstance().shutdown();
  return 0;
}
