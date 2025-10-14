# Code Quality & Linting Guide

This document explains the automated code quality checks used in this project to maintain clean, consistent, and reliable code.

## Overview

All code is automatically checked for quality issues using multiple linters before being merged. The CI/CD pipeline enforces these standards on every push and pull request.

## Quick Start

```bash
# Check all code (Python, CUDA/C++, Shell)
make check-all

# Fix auto-fixable issues
make fix-all

# Run individual linters
make lint-python    # Ruff + mypy
make lint-cuda      # cpplint
make lint-shell     # shellcheck
```

## Linters Used

### 1. Python: Ruff + mypy

**Ruff** is a fast Python linter that replaces multiple tools (black, isort, flake8, pyupgrade).

**mypy** provides static type checking.

**What they check:**
- Code formatting and style (PEP 8)
- Import ordering
- Unused imports and variables
- Type annotations
- Security issues (replaces Bandit)
- Code complexity

**Configuration:** `pyproject.toml`

**Commands:**
```bash
make lint-python     # Check Python code
make format-python   # Auto-fix Python formatting
```

**Example errors:**
```
benchmarks/multi_power_mode_benchmark.py:45:1: E501 Line too long (102 > 88 characters)
data/visualize_power_modes.py:120:5: F841 Local variable 'unused_var' is assigned to but never used
tests/test_performance.py:30:15: error: Argument 1 to "calculate_gflops" has incompatible type "str"; expected "float"
```

### 2. CUDA/C++: cpplint

**cpplint** checks C++ code style based on Google's C++ style guide.

**What it checks:**
- Line length (≤100 characters)
- Indentation and whitespace
- Header guards
- Include order
- Comment style
- Naming conventions

**Configuration:** `.cpplint`

**Commands:**
```bash
make lint-cuda       # Check CUDA/C++ code
make format-cuda     # Auto-fix CUDA/C++ formatting (clang-format)
```

**Example errors:**
```
./cuda/utils/logger.h:77: Lines should be <= 100 characters long [whitespace/line_length] [2]
./cuda/kernels/naive_multiplication.cu:45: Missing space before ( in if( [whitespace/parens] [5]
```

### 3. Shell: shellcheck

**shellcheck** analyzes shell scripts for common issues and best practices.

**What it checks:**
- Variable quoting and bracing
- Command substitution
- Array usage
- Error handling
- POSIX compliance

**Configuration:** `.shellcheckrc`

**Commands:**
```bash
make lint-shell      # Check shell scripts
```

**Example errors:**
```
scripts/archive_and_rerun.sh:30: SC2292 Prefer [[ ]] over [ ] for tests in Bash
scripts/setup_passwordless_sudo.sh:77: SC2250 Prefer putting braces around variable references
```

## CI/CD Integration

### Workflows

Both GitHub Actions workflows enforce code quality:

**`.github/workflows/ci.yml`** (Main CI Pipeline):
```yaml
jobs:
  code-quality:
    name: Code Quality & Linting
    steps:
      - name: Run all quality checks (Python + CUDA + Shell)
        run: make check-all
```

**`.github/workflows/validate.yml`** (Validation):
```yaml
jobs:
  validation:
    steps:
      - name: Run all quality checks (Python + CUDA + Shell)
        run: make check-all
```

### Enforcement

- **Blocking:** Code quality checks run FIRST and block all other jobs if they fail
- **Required:** The `code-quality` job must pass before tests run
- **Automated:** Runs on every push and pull request

## Local Development Setup

### Pre-commit Hook (Recommended)

Automatically check code before committing:

```bash
# Create .git/hooks/pre-commit
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook: Run code quality checks

set -e

echo "Running code quality checks..."
make check-all

if [ $? -ne 0 ]; then
    echo ""
    echo "[!] Code quality checks failed!"
    echo "Fix issues with: make fix-all"
    echo "Or commit with --no-verify to skip (not recommended)"
    exit 1
fi

echo "[*] All quality checks passed"
EOF

chmod +x .git/hooks/pre-commit
```

### VS Code Integration

Add to `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "ruff",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```

### Make Targets Reference

| Command | Description |
|---------|-------------|
| `make check-all` | Run all linters (Python + CUDA + Shell) |
| `make lint` | Alias for check-all |
| `make fix-all` | Auto-fix all auto-fixable issues |
| `make format` | Alias for fix-all |
| `make lint-python` | Check Python code only |
| `make lint-cuda` | Check CUDA/C++ code only |
| `make lint-shell` | Check shell scripts only |
| `make format-python` | Format Python code |
| `make format-cuda` | Format CUDA/C++ code |
| `make format-shell` | Format shell scripts |

## Common Issues & Solutions

### Python: Line too long

**Error:**
```
E501 Line too long (120 > 88 characters)
```

**Fix:**
```python
# Before
result = some_function(very_long_argument_1, very_long_argument_2, very_long_argument_3)

# After
result = some_function(
    very_long_argument_1,
    very_long_argument_2,
    very_long_argument_3,
)
```

### Shell: Missing braces around variables

**Error:**
```
SC2250: Prefer putting braces around variable references
```

**Fix:**
```bash
# Before
echo "Path: $MY_PATH"

# After
echo "Path: ${MY_PATH}"
```

### CUDA: Line too long

**Error:**
```
Lines should be <= 100 characters long
```

**Fix:**
```cpp
// Before
std::cerr << "ERROR: Cannot write to file: " << filename << " - permission denied" << std::endl;

// After
std::cerr << "ERROR: Cannot write to file: " << filename
          << " - permission denied" << std::endl;
```

## Bypassing Checks (Not Recommended)

In rare cases where you need to bypass a check:

### Python (Ruff)
```python
# noqa: E501  # Line too long
very_long_line_that_cannot_be_broken()
```

### Shell (shellcheck)
```bash
# shellcheck disable=SC2250
echo "Path: $MY_PATH"
```

### CUDA (cpplint)
```cpp
// NOLINT(whitespace/line_length)
very_long_line_that_cannot_be_broken();
```

**Important:** Always add a comment explaining WHY you're bypassing the check.

## Best Practices

1. **Run `make check-all` before committing**
2. **Use `make fix-all` to auto-fix issues**
3. **Install pre-commit hook for automatic checking**
4. **Don't bypass checks without good reason**
5. **Keep configuration files up to date**

## Configuration Files

- **Python:** `pyproject.toml` (Ruff + mypy config)
- **CUDA/C++:** `.cpplint` (cpplint config), `.clang-format` (formatting)
- **Shell:** `.shellcheckrc` (shellcheck config)
- **Editor:** `.editorconfig` (cross-editor settings)

## Getting Help

- **Ruff:** https://docs.astral.sh/ruff/
- **mypy:** https://mypy.readthedocs.io/
- **cpplint:** https://github.com/cpplint/cpplint
- **shellcheck:** https://www.shellcheck.net/wiki/
- **Project Issues:** https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis/issues

## Summary

All code is checked automatically in CI/CD. Key commands:
- Use `make check-all` before committing
- Use `make fix-all` to auto-fix issues
- Install pre-commit hook for convenience
- Keep code clean and consistent
