# Contributing to Jetson Orin Nano Matrix Benchmarks

Contributions welcome. This project targets edge AI optimization and CUDA performance analysis on Jetson hardware.

---

## Ways to Contribute

### 1. **Community Benchmark Submissions**
Share your benchmark results from different hardware configurations:
- Different Jetson models (AGX Orin, Xavier, etc.)
- Alternative power modes or thermal conditions
- Custom workloads beyond matrix multiplication
- Real-world ML model inference benchmarks

**How to submit:** Open an issue using the "Community Benchmark" template.

### 2. **Bug Reports**
Found a bug? Help us improve:
- Use the "Bug Report" issue template
- Include system specifications (JetPack version, L4T, CUDA)
- Provide reproduction steps
- Attach error logs or screenshots

### 3. **Feature Requests**
Suggest new features or improvements:
- Use the "Feature Request" issue template
- Explain the use case and expected benefit
- Discuss feasibility and implementation approach

### 4. **Code Contributions**
Improve implementations or add new features:
- CUDA kernel optimizations
- New precision modes (FP16, INT8, mixed precision)
- Additional benchmark implementations
- Visualization improvements
- Documentation improvements

### 5. **Documentation**
Improve clarity and accessibility:
- Fix typos or unclear explanations
- Add usage examples
- Translate documentation (if multilingual support is added)
- Create tutorials or blog posts

---

## Getting Started

### Prerequisites

- **Hardware:** NVIDIA Jetson Orin Nano (or compatible Jetson device)
- **Software:**
  - JetPack 6.x or later
  - CUDA 12.6+
  - Python 3.10+
  - UV package manager
  - Git

### Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/CONTRIBUTOR_USERNAME/jetson-orin-matmul-analysis.git
cd jetson-orin-matmul-analysis

# Add upstream remote
git remote add upstream https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis.git
```

### Setup Development Environment

```bash
# Run automated setup
make quick-start

# This will:
# 1. Create Python virtual environment with UV
# 2. Install dependencies (pandas, matplotlib, seaborn, pytest, ruff, mypy)
# 3. Compile all CUDA implementations
# 4. Run test suite to verify setup
```

### Configure Passwordless Sudo (Recommended for Development)

For uninterrupted benchmarking during development:

```bash
# Run the one-time setup (grants passwordless sudo for 3 specific commands)
sudo ./scripts/setup_passwordless_sudo.sh

# Benefits:
# - No password interruptions during 15-minute benchmark runs
# - Secure (only nvpmodel, tegrastats, collect_system_specs.sh whitelisted)
# - Easy removal: sudo ./scripts/remove_passwordless_sudo.sh
```

See [`SUDO_SETUP_GUIDE.md`](SUDO_SETUP_GUIDE.md) for detailed information.

### Verify Your Setup

```bash
# Run all quality checks
make check-all

# Run complete test suite
make test-all

# Validate project structure
make validate-structure
make validate-hardware

# Test passwordless sudo (if configured)
make system-specs      # Should not prompt for password
```

---

## Development Workflow

### 1. Create a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 2. Make Your Changes

**For CUDA Code:**
```bash
# Edit files in cuda/kernels/ or cuda/utils/

# Compile and test
make force-compile
make test-quick

# Run specific implementation
cd cuda && ./your_benchmark 1024
```

**For Python Code:**
```bash
# Edit files in benchmarks/, data/, or tests/

# Run formatting
make format-python

# Run linting
make lint-python

# Run tests
make test-unit
```

### 3. Follow Code Quality Standards

**Before committing, ensure:**

```bash
# All code quality checks pass
make check-all

# All tests pass
make test-all

# Code is properly formatted
make fix-all
```

**Quality Standards:**
- **Python:** Ruff (linting + formatting) + mypy (type checking)
- **CUDA/C++:** clang-format + cpplint
- **Shell:** shellcheck
- **Tests:** pytest with >90% coverage (ideal)
- **Documentation:** Clear docstrings and comments

**See [.github/LINTING_GUIDE.md](.github/LINTING_GUIDE.md) for detailed linting setup, CI/CD integration, and troubleshooting common issues.**

### 4. Commit Your Changes

**Follow Conventional Commits:**

```bash
# Format: <type>(<scope>): <description>

# Examples:
git commit -m "feat(cuda): add FP16 matrix multiplication kernel"
git commit -m "fix(benchmark): correct efficiency calculation for 15W mode"
git commit -m "docs(readme): update installation instructions for JetPack 6.1"
git commit -m "test(performance): add validation for memory bandwidth limits"
git commit -m "refactor(visualize): improve power mode comparison plotting"
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code restructuring without behavior change
- `perf`: Performance improvements
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
# Use the PR template and provide:
# - Clear description of changes
# - Issue number (if applicable)
# - Testing performed
# - Screenshots (if UI changes)
```

---

## Pull Request Guidelines

### PR Checklist

Before submitting, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass (`make test-all`)
- [ ] Code quality checks pass (`make check-all`)
- [ ] New code has appropriate test coverage
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow conventional format
- [ ] PR description clearly explains changes
- [ ] Branch is up-to-date with `main`

### PR Review Process

1. **Automated Checks:** CI/CD runs linting, tests, and compilation
2. **Maintainer Review:** Code review by project maintainers
3. **Community Feedback:** Open discussion and suggestions
4. **Revisions:** Address feedback and push updates
5. **Approval:** Once approved, PR will be merged

**Response Time:**
- Initial response: Within 48 hours
- Full review: Within 1 week
- Merge: After approval + passing CI

---

## Testing Requirements

### Unit Tests

```bash
# Run all unit tests
make test-unit

# Run specific test file
python -m pytest tests/test_performance.py -v

# Run with coverage
make test-coverage
```

### Integration Tests

```bash
# Run integration tests
make test-integration

# Test complete pipeline
make complete-pipeline
```

### CUDA Functionality Tests

```bash
# Quick functionality test
make test-quick

# Test specific implementation
cd cuda && ./naive_benchmark 512
cd cuda && ./blocked_benchmark 512
cd cuda && ./cublas_benchmark 512
cd cuda && ./tensor_core_benchmark 512
```

### Performance Validation

```bash
# Validate benchmark accuracy
python validate_benchmark_accuracy.py

# Run full power mode analysis (requires sudo)
sudo make full-analysis

# Generate visualizations
make visualize
```

---

## Code Style Guide

### Python Style

**Follow PEP 8 with Ruff enforcement:**

```python
# Good: Type hints, clear naming, docstrings
def calculate_gflops(matrix_size: int, elapsed_ms: float) -> float:
    """
    Calculate GFLOPS for matrix multiplication.

    Args:
        matrix_size: Dimension of square matrix (N×N)
        elapsed_ms: Execution time in milliseconds

    Returns:
        Performance in GFLOPS (billions of FLOPs per second)
    """
    flops = 2 * matrix_size ** 3
    return flops / (elapsed_ms * 1e6)

# Bad: No types, unclear naming, no docstring
def calc(n, t):
    return 2 * n ** 3 / (t * 1e6)
```

**Best Practices:**
- Use type hints for function signatures
- Write clear docstrings (Google style)
- Keep functions focused and small (<50 lines)
- Use descriptive variable names
- Add comments for complex logic

### CUDA/C++ Style

**Follow project `.clang-format` configuration:**

```cpp
// Good: Clear structure, comments, error checking
__global__ void matrixMultiplyKernel(const float *A, const float *B,
                                      float *C, int N) {
    // Calculate global thread position
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (row >= N || col >= N) {
        return;
    }

    // Compute dot product
    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// Bad: Unclear, no comments, no checks
__global__ void mm(float *a, float *b, float *c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0;
    for (int k = 0; k < n; k++) s += a[i*n+k] * b[k*n+j];
    c[i*n+j] = s;
}
```

**Best Practices:**
- Use descriptive kernel names
- Add boundary checks
- Improve memory access patterns
- Document optimization techniques
- Include performance comments

---

## Bug Report Template

When reporting bugs, include:

1. **System Information:**
   - Jetson model and revision
   - JetPack version
   - L4T version
   - CUDA version
   - Python version

2. **Reproduction Steps:**
   - Exact commands run
   - Configuration used
   - Power mode settings

3. **Expected vs Actual Behavior:**
   - What you expected to happen
   - What actually happened

4. **Error Output:**
   - Complete error messages
   - Stack traces
   - Relevant logs

5. **Environment:**
   - Thermal conditions
   - Other running processes
   - Available memory

---

## Feature Request Template

When requesting features, explain:

1. **Problem Statement:**
   - What problem does this solve?
   - Who benefits from this feature?

2. **Proposed Solution:**
   - Describe the feature
   - How should it work?

3. **Alternatives Considered:**
   - Other approaches you've thought about

4. **Use Case:**
   - Real-world scenario where this helps

5. **Implementation Complexity:**
   - Rough estimate of effort
   - Required expertise

---

## Recognition

### Contributors Hall of Fame

Contributors are recognized in:
- Project README (Contributors section)
- Release notes
- Annual contributor acknowledgments

### Contribution Types Recognized

- **Code contributions**
- **Documentation improvements**
- **Testing and validation**
- **Bug reports and fixes**
- **Feature suggestions**
- **Community support**

---

## Getting Help

### Communication Channels

- **GitHub Issues:** Bug reports, feature requests
- **GitHub Discussions:** General questions, ideas, showcase
- **Email:** jesse@bytestacklabs.com (for private inquiries)

### Response Times

- **Critical bugs:** Within 24 hours
- **General issues:** Within 48 hours
- **Feature requests:** Within 1 week
- **PRs:** Initial response within 48 hours

---

## Code of Conduct

### Our Standards

We are committed to providing a welcoming and inclusive environment:

**Positive behaviors:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what's best for the community
- Showing empathy towards others

**Unacceptable behaviors:**
- Harassment or discriminatory language
- Trolling or inflammatory comments
- Personal or political attacks
- Publishing others' private information
- Other conduct inappropriate in professional settings

### Enforcement

Instances of unacceptable behavior may be reported to jesse@bytestacklabs.com.

---

## License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

## Acknowledgment

Contributions improve the project's utility for edge AI deployment. Code, documentation, and feedback all advance the work.

---

**Last Updated:** October 2025
**Maintainer:** Jesse Moses ([@Cre4T3Tiv3](https://github.com/Cre4T3Tiv3))
**Organization:** ByteStack Labs
**ORCID:** [0009-0006-0322-7974](https://orcid.org/0009-0006-0322-7974)
