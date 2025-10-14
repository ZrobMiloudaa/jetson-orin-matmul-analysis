<p align="center">
  <a href="https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis">
    <img src="https://raw.githubusercontent.com/Cre4T3Tiv3/jetson-orin-matmul-analysis/main/docs/assets/jetson_orin_nano_matmul_power_benchmarks_social_preview_v1.0.0.png" alt="Jetson Orin Nano Power-Performance Benchmarks v1.0.0" width="640"/>
  </a>
</p>

<h1 align="center">Jetson Orin Nano Matrix Multiplication:<br>Power-Performance Analysis</h1>

<p align="center">
  <strong>Scientific benchmarking framework for analyzing CUDA matrix multiplication across 4 implementations, 3 power modes, and 5 matrix sizes on NVIDIA Jetson Orin Nano.</strong>
</p>

<!-- Version & Status Badges -->
<p align="center">
  <a href="https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis/releases/tag/v1.0.0">
    <img src="https://img.shields.io/badge/version-v1.0.0-brightgreen" alt="Version: v1.0.0">
  </a>
  <a href="https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis/actions/workflows/ci.yml">
    <img src="https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI Build Status">
  </a>
</p>

<!-- Hardware Platform Badges -->
<p align="center">
  <a href="https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/">
    <img src="https://img.shields.io/badge/Hardware-Jetson%20Orin%20Nano-76B900?logo=nvidia&logoColor=white" alt="Hardware: Jetson Orin Nano">
  </a>
  <a href="https://developer.nvidia.com/cuda-toolkit">
    <img src="https://img.shields.io/badge/CUDA_Toolkit-12.6+-76B900?logo=nvidia&logoColor=white" alt="CUDA Toolkit: 12.6+">
  </a>
  <a href="https://developer.nvidia.com/embedded/jetpack">
    <img src="https://img.shields.io/badge/JetPack-6.x-76B900?logo=nvidia&logoColor=white" alt="JetPack: 6.x">
  </a>
  <a href="https://developer.nvidia.com/embedded/linux-tegra">
    <img src="https://img.shields.io/badge/L4T-R36.4+-76B900" alt="L4T: R36.4+">
  </a>
</p>

<!-- NVIDIA Technologies Badges -->
<p align="center">
  <a href="https://developer.nvidia.com/cublas">
    <img src="https://img.shields.io/badge/cuBLAS-Optimized-76B900?logo=nvidia&logoColor=white" alt="cuBLAS Optimized">
  </a>
  <a href="https://developer.nvidia.com/tensor-cores">
    <img src="https://img.shields.io/badge/Tensor_Cores-SM_8.7-76B900?logo=nvidia&logoColor=white" alt="Tensor Cores: SM 8.7">
  </a>
  <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma">
    <img src="https://img.shields.io/badge/WMMA_API-Mixed_Precision-76B900?logo=nvidia&logoColor=white" alt="WMMA API: Mixed Precision">
  </a>
</p>

<!-- Programming Languages Badges -->
<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-3776AB?logo=python&logoColor=white" alt="Python: 3.10 | 3.11 | 3.12">
  </a>
  <a href="https://en.cppreference.com/w/cpp/14">
    <img src="https://img.shields.io/badge/C++_Standard-14-00599C?logo=cplusplus&logoColor=white" alt="C++ Standard: 14">
  </a>
  <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html">
    <img src="https://img.shields.io/badge/CUDA_C++-12.6+-76B900?logo=nvidia&logoColor=white" alt="CUDA C++: 12.6+">
  </a>
</p>

<!-- Code Quality & Testing Badges -->
<p align="center">
  <a href="https://docs.astral.sh/ruff/">
    <img src="https://img.shields.io/badge/linter-Ruff-261230?logo=ruff&logoColor=white" alt="Linter: Ruff">
  </a>
  <a href="https://mypy-lang.org/">
    <img src="https://img.shields.io/badge/type_checker-mypy-1F5082" alt="Type Checker: mypy">
  </a>
  <a href="https://docs.pytest.org/">
    <img src="https://img.shields.io/badge/testing-pytest-0A9EDC?logo=pytest&logoColor=white" alt="Testing: pytest">
  </a>
  <a href="https://clang.llvm.org/docs/ClangFormat.html">
    <img src="https://img.shields.io/badge/formatter-clang--format-262D3A?logo=llvm&logoColor=white" alt="Formatter: clang-format">
  </a>
  <a href="https://www.shellcheck.net/">
    <img src="https://img.shields.io/badge/shell-shellcheck-4EAA25?logo=gnu-bash&logoColor=white" alt="Shell: shellcheck">
  </a>
</p>

<!-- License & Community Badges -->
<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="Contributions Welcome">
  </a>
  <a href="https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis/stargazers">
    <img src="https://img.shields.io/github/stars/Cre4T3Tiv3/jetson-orin-matmul-analysis?style=social" alt="GitHub Stars">
  </a>
</p>

<!-- Author & Attribution Badges -->
<p align="center">
  <a href="https://orcid.org/0009-0006-0322-7974">
    <img src="https://img.shields.io/badge/ORCID-0009--0006--0322--7974-A6CE39?logo=orcid&logoColor=white" alt="ORCID: 0009-0006-0322-7974">
  </a>
  <a href="https://bytestacklabs.com">
    <img src="https://img.shields.io/badge/Made%20by-ByteStack%20Labs-2ea44f" alt="ByteStack Labs">
  </a>
  <a href="https://www.linkedin.com/in/jlmoses/">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
</p>

---

<!-- Table of Contents -->
## Table of Contents

- [Key Findings](#-key-findings)
- [What Makes This Different](#-what-makes-this-different)
- [Performance Comparison](#-performance-comparison)
- [Key Insights for ML Engineers](#-key-insights-for-ml-engineers)
  - [Power Budget Optimization](#power-budget-optimization)
  - [Precision Trade-offs](#precision-trade-offs)
- [Project Motivation](#-project-motivation)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Run Benchmarks](#run-benchmarks)
- [Technical Validation](#-technical-validation)
- [Project Structure](#-project-structure)
- [Implementations Explained](#-implementations-explained)
- [Production Deployment Recommendations](#-production-deployment-recommendations)
  - [Battery-Powered Devices](#for-battery-powered-devices-robots-drones-agvs)
  - [Plugged-In Edge Devices](#for-plugged-in-edge-devices-smart-cameras-industrial-iot)
  - [ML Training/Fine-Tuning](#for-ml-trainingfine-tuning-on-edge-federated-learning)
- [Learning Resources](#-learning-resources)
- [Visualization Gallery](#-visualization-gallery)
- [Testing](#-testing)
- [Development Workflow](#-development-workflow)
- [Data Management & Reproducibility](#-data-management--reproducibility)
- [Results & Publications](#-results--publications)
- [Contributing](#-contributing)
- [License & Attribution](#license--attribution)
- [Author & Contact](#author--contact)

---

## Key Findings

**Peak Performance:**
- cuBLAS: 1,282 GFLOPS (61% of theoretical peak)
- Tensor Cores (TF32): 952 GFLOPS (10.0× vs naive at 1024×1024)
- Power Efficiency: 25W mode achieves 90% of MAXN performance at 88% power consumption

**Validation:**
- Theoretical peak: 2,089 GFLOPS (FP32 @ 1020 MHz)
- Memory bandwidth: 68 GB/s (LPDDR5)
- Measurement accuracy: 99.5%
- Numerical accuracy: < 1e-5 (FP32), < 0.01 (TF32)

---

## What Makes This Different

**Multi-Dimensional Analysis:** 60 validated data points across 4 implementations, 3 power modes, and 5 matrix sizes. Real-time GPU frequency measurement with thermal and power consumption tracking.

**Scientific Rigor:** All metrics validated against theoretical limits. Fully reproducible with automated pipelines. 99.5% measurement accuracy.

---

## Performance Comparison

| Implementation | Peak GFLOPS | Efficiency* | Speedup vs Naive† | Best Use Case |
|----------------|-------------|-------------|-------------------|---------------|
| **Naive**      | 95          | 4.6%        | 1.0×              | Educational baseline |
| **Blocked**    | 150         | 7.2%        | 1.6×              | Cache optimization study |
| **cuBLAS**     | **1,282**   | **61.4%**   | **13.5×**         | **Production workloads** |
| **Tensor Core**| 952         | 45.6%       | 10.0×             | ML inference (TF32) |

<sub>*Efficiency relative to hardware theoretical peak (2,089 GFLOPS @ 1020 MHz MAXN_SUPER)</sub>
<sub>†Speedup calculated at 1024×1024 matrix size</sub>

### Methodology

Speedup calculated at 1024×1024 matrix size (representative of transformer attention and dense layers). Naive baseline: 95.23 GFLOPS. Alternative peak-vs-peak comparison (naive peaks at 512×512: 157 GFLOPS) yields 8.2× and 6.1× speedups.

---

## Key Insights for ML Engineers

### Power Budget Optimization

25W mode provides optimal performance-per-watt:

```
15W:   792 GFLOPS @ 24W  = 33 GFLOPS/W
25W: 1,150 GFLOPS @ 22W  = 52 GFLOPS/W
MAXN: 1,282 GFLOPS @ 25W  = 51 GFLOPS/W
```

For battery-powered deployments, 25W mode delivers 90% of MAXN performance at 88% power consumption.

### Precision Trade-offs

Tensor Cores (TF32): 10.0× faster than naive with 0.00972 max error vs < 1e-6 for FP32. Suitable for neural networks; not for high-precision scientific computing.

BERT inference example (1024×1024):
- FP32 (cuBLAS): 1,282 GFLOPS, < 1e-6 error
- TF32 (Tensor Cores): 952 GFLOPS, 0.00972 error

---

## Project Motivation

Edge ML deployment commonly assumes maximum power yields maximum performance. Benchmarking reveals this wastes power: MAXN mode consumes 20% more power for only 10% performance gain over 25W mode. At scale (1000+ devices), suboptimal power configuration wastes significant energy and reduces battery life.

This project provides validated benchmarks for power-aware ML deployment on Jetson Orin Nano hardware.

---

## Quick Start

### Prerequisites

- NVIDIA Jetson Orin Nano with JetPack 6.x
- CUDA 12.6+
- Python 3.10+ with UV package manager
- Sudo access (for power mode switching)

### Installation

```bash
# Clone repository
git clone https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis.git
cd jetson-orin-matmul-analysis

# One-command setup: creates venv, installs deps, compiles CUDA
make quick-start
```

### Optional: Passwordless Sudo Setup

Configure passwordless sudo for uninterrupted benchmarking:

```bash
sudo ./scripts/setup_passwordless_sudo.sh
```

Grants sudo access to: nvpmodel (power modes), tegrastats (telemetry), collect_system_specs.sh.
Remove with: `sudo ./scripts/remove_passwordless_sudo.sh`

See [`SUDO_SETUP_GUIDE.md`](SUDO_SETUP_GUIDE.md) for details.

### Run Benchmarks

```bash
# Quick functionality test (single power mode)
make test-quick

# Full power mode analysis (~15 minutes)
sudo make full-analysis

# Generate visualizations
make visualize
```

### Example Output

```
====================================================================
Matrix Multiplication Benchmark - cuBLAS Implementation
====================================================================
Matrix Size: 1024 x 1024
Power Mode: MAXN_SUPER (30W max)
Current GPU Frequency: 612.5 MHz (measured)

Performance Results:
  Elapsed Time: 1.689 ms
  Measured Performance: 1271.53 GFLOPS
  Memory Bandwidth: 7.45 GB/s
  Efficiency: 60.9% of theoretical peak (2089 GFLOPS @ 1020 MHz)
  Numerical Error: 0.000000
```

---

## Technical Validation

Validation suite ensures 99.5% accuracy on GFLOPS/bandwidth calculations. All measurements verified against theoretical hardware limits. GPU frequency measured at runtime; thermal monitoring included.

```bash
make test-all          # Complete test suite
make verify-accuracy   # Numerical accuracy validation
```

---

## Project Structure

```
jetson-matrix-benchmarks/
├── cuda/
│   ├── kernels/                    # CUDA implementations
│   │   ├── naive_multiplication.cu      # O(n³) baseline
│   │   ├── blocked_multiplication.cu    # Tiled with shared memory
│   │   ├── cublas_multiplication.cu     # Vendor-optimized
│   │   └── tensor_core_multiplication.cu # WMMA API
│   └── utils/
│       ├── common.h                # Shared utilities header
│       ├── common.cu               # Performance analysis, monitoring
│       └── logger.h                # Enterprise-grade logging
├── benchmarks/
│   └── multi_power_mode_benchmark.py # Orchestrator
├── data/
│   ├── raw/power_modes/           # Benchmark results (JSON/CSV)
│   ├── reports/                   # Analysis markdown reports
│   └── plots/                     # Visualizations
├── tests/                         # Pytest suite
├── scripts/                       # Helper scripts
└── Makefile                       # Automated build/test/benchmark
```

---

## Implementations Explained

### 1. Naive Implementation
**Algorithm:** Triple-nested loop with global memory access
```cpp
C[i][j] += A[i][k] * B[k][j]  // Direct global memory reads
```
**Performance:** 17-104 GFLOPS
**Use Case:** Educational baseline, demonstrates GPU underutilization

### 2. Blocked (Tiled) Implementation
**Algorithm:** Tiled matrix multiplication with shared memory
```cpp
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];
// Load tiles -> compute -> write back
```
**Performance:** 18-158 GFLOPS (+52% vs naive)
**Use Case:** Teaching cache optimization techniques

### 3. cuBLAS Implementation
**Algorithm:** NVIDIA's vendor-optimized library
```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, ...);
```
**Performance:** 19-1,282 GFLOPS (up to 61% of theoretical peak)
**Use Case:** Production deployments, maximum performance

### 4. Tensor Core Implementation
**Algorithm:** Warp Matrix Multiply-Accumulate (WMMA) API
```cpp
wmma::load_matrix_sync(a_frag, A + ..., 16);  // Load 16×16 tile
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // Matrix multiply
```
**Performance:** 32-952 GFLOPS with TF32 precision
**Use Case:** ML inference acceleration, acceptable precision loss

---

## Production Deployment Recommendations

### Battery-Powered Devices
**Use 25W mode:** 1,150 GFLOPS at 52 GFLOPS/W. Delivers 90% of MAXN performance at 88% power. Extends mission time ~2 hours on 8-hour battery. No active cooling required.

### Plugged-In Edge Devices
**Use MAXN mode:** 1,282 GFLOPS at 61% efficiency. Maximum throughput for real-time processing (4K@30fps). Requires cooling infrastructure.

### ML Training/Fine-Tuning
**Use Tensor Cores (TF32) at 25W:** 952 GFLOPS, 10.0× vs naive. Suitable for federated learning and transformer inference. Max error 0.00972 (acceptable for neural networks).
- Neural network training where 1% precision loss is acceptable

**Trade-offs:**
- Not suitable for high-precision scientific computing
- Reduced mantissa (19-bit vs 23-bit FP32) may affect some workloads

---

### Power Mode Selection Decision Tree

**Is power supply constrained?**

→ **YES (Battery-Powered)**
  - Need maximum battery life?
    - **YES** → Use **15W mode** (33 GFLOPS/W, longest runtime)
    - **NO** → Use **25W mode** (52 GFLOPS/W, balanced)

→ **NO (Plugged-In)**
  - Is latency critical?
    - **YES** → Use **MAXN mode** (1,282 GFLOPS, maximum throughput)
    - **NO** → Use **25W mode** (1,150 GFLOPS, 90% of MAXN at 88% power)

---

### Performance Validation Checklist

Before production deployment, validate on your hardware:

- [ ] Run `make full-analysis` to benchmark all power modes
- [ ] Measure actual GPU frequency (varies per unit)
- [ ] Profile your specific workload
- [ ] Monitor thermal behavior over 1-hour sustained load
- [ ] Calculate actual GFLOPS/watt for your models
- [ ] Validate numerical accuracy for your precision requirements
- [ ] Test thermal throttling under worst-case ambient temperature

Hardware units vary due to silicon lottery, thermal paste quality, and ambient conditions.

---

## Learning Resources

**Topics:** CUDA kernel optimization (naive -> blocked -> vendor libraries), Tensor Core programming (WMMA API, SM 8.7), power-performance trade-offs on edge devices, scientific benchmarking methodology, hardware-in-the-loop validation.

**Audience:** ML engineers (edge deployment), CUDA developers (optimization patterns), researchers (power-constrained computing), students (GPU programming fundamentals).

---

## Visualization Gallery

<details>
<summary><b>Sample visualizations</b></summary>

**Multi-Implementation Comparison:** Performance scaling across power modes and matrix sizes
**Power Efficiency Analysis:** GFLOPS-per-watt across 15W/25W/MAXN modes
**Speedup Heatmap:** Relative performance gains between implementations

</details>

Run `make visualize` to generate plots.

---

## Testing

**Coverage:** Unit tests (matrix operations, performance calculations), integration tests (end-to-end pipelines), validation tests (numerical accuracy, theoretical limits), hardware tests (platform detection, thermal monitoring).

```bash
make test-all           # Complete suite
make test-unit          # Unit tests only
make test-integration   # Integration tests
make test-performance   # Performance validation
make test-coverage      # With coverage report
```

---

## Development Workflow

### Code Quality

All code is automatically checked for quality issues in CI/CD. Run these commands locally before committing:

```bash
# Run all quality checks (Python + CUDA + Shell)
make check-all

# Auto-fix formatting issues
make fix-all

# Individual checks
make lint-python    # Ruff + mypy
make lint-cuda      # cpplint
make lint-shell     # shellcheck
make format-cuda    # clang-format (uses .clang-format config)
```

**Install Git Hooks (Recommended):**
```bash
# Auto-run checks before commits
./scripts/install_git_hooks.sh
```

**See [.github/LINTING_GUIDE.md](.github/LINTING_GUIDE.md) for detailed setup, CI/CD integration, and troubleshooting.**

### Complete Pipeline
```bash
# Setup -> Quality Checks -> Compile -> Test -> Benchmark -> Visualize
make complete-pipeline
```

---

## Data Management & Reproducibility

Track performance across changes and validate regressions.

### Quick Workflow
```bash
make archive-data           # Archive current baseline
vim cuda/kernels/blocked_multiplication.cu  # Make changes
make rerun-and-compare      # Re-run and compare
```

### Commands

| Command | Purpose |
|---------|---------|
| `make archive-data` | Archive current benchmark data with timestamp |
| `make rerun-and-compare` | Archive -> clean -> rerun -> compare |
| `make compare-last` | Compare current results with most recent archive |
| `make list-archives` | List all archived benchmark runs |

### Use Cases

**Regression Testing:** Archive before optimization, compare after
**Hardware Validation:** Compare passive vs active cooling, thermal behavior
**Software Validation:** Verify CUDA/driver upgrades don't regress performance
**Environmental Analysis:** Quantify thermal throttling under different ambient conditions

### Comparison Report

Reports include per-implementation comparison (GFLOPS, power efficiency, percentage changes), speedup analysis, status indicators ([OK] Similar <1%, [+] Improved, [!] Regressed), automatic regression detection.

```markdown
## CUBLAS Implementation
| Power Mode | Baseline | New Run | Change | Status |
|------------|----------|---------|--------|--------|
| 25W        | 1152 GFLOPS @ 23W | 1150 GFLOPS @ 23W | -2.0 GFLOPS (-0.2%) | [OK] Similar |
```

### Archive Directory Structure

Each archive creates a timestamped snapshot:
```
data/archive/baseline_20251005_143022/
├── raw/power_modes/           # JSON benchmark results
├── plots/                     # Generated visualizations
├── reports/                   # Markdown analysis reports
├── logs/                      # Execution logs
└── metadata/                  # Git commit, timestamp, file list
```

### CI/CD Integration

```bash
make archive-data
make complete-pipeline
make compare-last > comparison_report.md

# Fail CI if performance regressed
if grep -q "\[!\] Regressed" comparison_report.md; then
    echo "Performance regression detected!"
    exit 1
fi
```

### Best Practices

Baseline before major changes (kernel optimizations, library upgrades, hardware modifications). Maintain consistent test conditions (ambient temperature, power supply, background processes). Document significant changes in archive metadata.

```bash
make archive-data
cat >> data/archive/baseline_*/metadata/archive_info.md << EOF
## Optimization Notes
- Increased tile size from 16 to 32
- Expected improvement: 10-15%
EOF
```

---

## Results & Publications

### Reference Baseline (v1.0.0)

Validated reference results from author's hardware (Jetson Orin Nano, Oct 2025):

**Benchmark Data** (`data/raw/power_modes/`): `{implementation}_3mode_analysis_*.{json,csv}` (combined 3-power-mode results), `{implementation}_mode_{0,1,2}_{power}_*.{json,csv}` (individual power mode data). Implementations: naive, blocked, cublas, tensor_core.

**Visualizations** (`data/plots/`): Power analysis charts, cross-implementation comparison, speedup heatmaps.

**Reports** (`data/reports/`): Cross-implementation performance analysis, detailed benchmark metrics.

These artifacts serve as reference comparison (validate your hardware), output examples (see framework output), and scientific reproducibility (documented evidence for citations). Run `make full-analysis` to generate your own results in the same directories.

### Citation

If you use this work in production deployments, research, or derivative projects, please cite:

#### BibTeX
```bibtex
@software{moses2025jetson,
  author = {Moses, Jesse},
  orcid = {0009-0006-0322-7974},
  title = {Jetson Orin Nano Power-Performance Analysis:
           Scientific Benchmarking Framework for Edge AI Deployment},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis},
  version = {1.0.0},
  note = {Validated analysis of CUDA matrix multiplication across 4 implementations,
          3 power modes, and 5 matrix sizes with 99.5\% measurement accuracy}
}
```

#### APA Style
```
Moses, J. L. (2025). Jetson Orin Nano Power-Performance Analysis:
  Scientific Benchmarking Framework for Edge AI Deployment (Version 1.0.0) [Computer software].
  GitHub. https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis
```

#### IEEE Style
```
J. L. Moses, "Jetson Orin Nano Power-Performance Analysis: Scientific Benchmarking
Framework for Edge AI Deployment," 2025. [Online]. Available:
https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis. [Accessed: Oct. 10, 2025].
```

This repository includes `CITATION.cff` for automated citation management via GitHub's "Cite this repository" feature.

---

## Contributing

Contributions welcome. Project follows automated CI/CD testing, code quality enforcement (Ruff, cpplint, mypy), comprehensive documentation, scientific rigor in benchmarking.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Good first issues:** FP16 precision benchmarking, dynamic tile size selection, Jupyter notebook tutorial, Docker container for reproducibility.

---

---

## License & Attribution

**Jetson Orin Nano Matrix Multiplication: Power-Performance Analysis** is licensed under the **MIT** License.
See [`LICENSE`](LICENSE) for complete terms.

### Citation
```bibtex
@software{moses2025jetson,
  author = {Moses, Jesse},
  orcid = {0009-0006-0322-7974},
  title = {Jetson Orin Nano Matrix Multiplication: Power-Performance Analysis},
  url = {https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis},
  version = {1.0.0},
  year = {2025},
  organization = {ByteStack Labs}
}
```

See `CITATION.cff` for additional citation formats.

---

## Author & Contact

**Jetson Orin Nano Matrix Multiplication: Power-Performance Analysis** was built by [Jesse Moses (@Cre4T3Tiv3)](https://github.com/Cre4T3Tiv3) at [ByteStack Labs](https://bytestacklabs.com).

### Connect & Contribute
- **Bug Reports**: [GitHub Issues](https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis/discussions)
- **Direct Contact**: [ByteStack Labs](https://bytestacklabs.com)
- **Show Support**: [Star this repository](https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis/stargazers)

---

> **Question for the Edge AI Community**: Could power-aware benchmarking methodology be the missing foundation for sustainable edge deployment at scale?

---

<p align="center">
  <strong>Made with rigorous analysis by ByteStack Labs</strong><br>
  <em>Bringing scientific rigor to edge AI optimization</em>
</p>
