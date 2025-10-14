# ------------------------------------------------------------------------------
# Jetson Orin Nano Matrix Multiplication Analysis
# Copyright 2025 ByteStack Labs - MIT License
# 
# Target: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
# ------------------------------------------------------------------------------

.PHONY: help setup-env compile test system-specs full-analysis visualize complete-pipeline
.PHONY: clean debug-cuda quick-start verify-accuracy validation-checklist force-compile
.PHONY: lint format style-check lint-python lint-cuda lint-shell
.PHONY: format-python format-cuda format-shell format-shell-safe fix-all check-all
.PHONY: check-cuda-format show-clang-format-diff verify-clang-format-config
.PHONY: verify-shellcheck-config clean-lint install-cuda-tools install-shell-tools info
.PHONY: test-quick test-unit test-integration test-performance test-utils test-all
.PHONY: test-coverage clean-test clean-all validate-structure validate-hardware validate-data ci
.PHONY: archive-data rerun-and-compare list-archives compare-last

.DEFAULT_GOAL := help

# Terminal Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
CYAN := \033[0;36m
MAGENTA := \033[0;35m
NC := \033[0m

# Project Configuration
PROJECT_NAME := jetson-matrix-benchmarks
VERSION := 1.0.0
PYTHON_VERSION := 3.10
AUTHOR := Jesse Moses (@Cre4T3Tiv3) - jesse@bytestacklabs.com
TARGET_HARDWARE := Jetson Orin Nano Engineering Reference Developer Kit Super

# CUDA Configuration
NVCC := nvcc
NVCC_FLAGS := -O3 -arch=sm_87 -std=c++14 -lineinfo -maxrregcount=64 --use_fast_math
CUDA_LIBS := -lcublas -lcurand
INCLUDES := -I/usr/local/cuda/include

# Environment
VENV_DIR := .venv
VENV_BIN := $(VENV_DIR)/bin
PYTHON := $(VENV_BIN)/python
UV := uv

# Directories
IMPL_DIR := cuda
BENCH_DIR := benchmarks
RESULTS_DIR := data
TEST_DIR := tests

# Build Targets
CUDA_TARGET := $(IMPL_DIR)/naive_benchmark
BLOCKED_TARGET := $(IMPL_DIR)/blocked_benchmark
CUBLAS_TARGET := $(IMPL_DIR)/cublas_benchmark
TENSOR_CORE_TARGET := $(IMPL_DIR)/tensor_core_benchmark
CUDA_SOURCES := $(IMPL_DIR)/kernels/naive_multiplication.cu $(IMPL_DIR)/utils/common.cu $(IMPL_DIR)/utils/common.h

# File Discovery (project-scoped)
PYTHON_FILES := $(shell find . -name "*.py" -not -path "./.venv/*" -not -path "./build/*" -not -path "./__pycache__/*" -not -path "./.git/*" -not -path "./.ruff_cache/*" 2>/dev/null)
CUDA_FILES := $(shell find ./cuda ./benchmarks -name "*.cu" -o -name "*.cuh" 2>/dev/null)
CPP_FILES := $(shell find ./cuda ./benchmarks -name "*.cpp" -o -name "*.h" -o -name "*.hpp" 2>/dev/null)
SHELL_FILES := $(shell find ./scripts -name "*.sh" -executable 2>/dev/null)

export PYTHONPATH := $(PWD)

help: ## Show all available commands organized by category
	@echo "$(BLUE)Jetson Orin Nano Matrix Multiplication Analysis$(NC)"
	@echo "$(YELLOW)Target: $(TARGET_HARDWARE)$(NC)"
	@echo "$(YELLOW)Version: $(VERSION)$(NC)"
	@echo ""
	@echo "$(YELLOW)Essential Commands:$(NC)"
	@echo "  $(GREEN)make quick-start$(NC)         - Complete setup and verification"
	@echo "  $(GREEN)make complete-pipeline$(NC)   - Full benchmark pipeline with quality checks"
	@echo "  $(GREEN)make test-all$(NC)            - Run complete test suite"
	@echo "  $(GREEN)make help$(NC)                - Show this help menu"
	@echo ""
	@echo "$(YELLOW)Environment & Setup:$(NC)"
	@echo "  $(GREEN)make setup-env$(NC)           - Create Python environment with tools"
	@echo "  $(GREEN)make compile$(NC)             - Compile CUDA implementation"
	@echo "  $(GREEN)make force-compile$(NC)       - Force recompilation of CUDA code"
	@echo "  $(GREEN)make debug-cuda$(NC)          - Check CUDA environment and libraries"
	@echo "  $(GREEN)make validate-structure$(NC)  - Validate project directory structure"
	@echo "  $(GREEN)make validate-hardware$(NC)   - Validate hardware compatibility"
	@echo ""
	@echo "$(YELLOW)Testing & Validation:$(NC)"
	@echo "  $(GREEN)make test-all$(NC)            - Run complete test suite (unit + functionality)"
	@echo "  $(GREEN)make test-unit$(NC)           - Run pytest unit tests"
	@echo "  $(GREEN)make test-quick$(NC)          - Quick CUDA functionality test"
	@echo "  $(GREEN)make test-integration$(NC)    - Run integration tests"
	@echo "  $(GREEN)make test-performance$(NC)    - Run performance calculation tests"
	@echo "  $(GREEN)make test-utils$(NC)          - Run utility function tests"
	@echo "  $(GREEN)make test-coverage$(NC)       - Run tests with coverage analysis"
	@echo "  $(GREEN)make verify-accuracy$(NC)     - Comprehensive numerical accuracy verification"
	@echo "  $(GREEN)make system-specs$(NC)        - Document hardware specifications"
	@echo "  $(GREEN)make validation-checklist$(NC) - Validate benchmark completion"
	@echo ""
	@echo "$(YELLOW)Code Quality & Formatting:$(NC)"
	@echo "  $(GREEN)make check-all$(NC)           - Run all linting and style checks"
	@echo "  $(GREEN)make fix-all$(NC)             - Auto-fix all formatting issues"
	@echo "  $(GREEN)make lint$(NC)                - Lint all file types"
	@echo "  $(GREEN)make format$(NC)              - Format all code files"
	@echo "  $(GREEN)make lint-python$(NC)         - Lint Python files with Ruff and mypy"
	@echo "  $(GREEN)make lint-cuda$(NC)           - Lint CUDA and C++ files"
	@echo "  $(GREEN)make lint-shell$(NC)          - Lint shell scripts"
	@echo "  $(GREEN)make format-python$(NC)       - Format Python files with Ruff"
	@echo "  $(GREEN)make format-cuda$(NC)         - Format CUDA/C++ files with clang-format"
	@echo "  $(GREEN)make format-shell$(NC)        - Format shell scripts"
	@echo "  $(GREEN)make format-shell-safe$(NC)   - Safe shell formatting with validation"
	@echo ""
	@echo "$(YELLOW)Analysis & Benchmarking:$(NC)"
	@echo "  $(GREEN)make full-analysis$(NC)       - Run 3-power mode benchmarking"
	@echo "  $(GREEN)make visualize$(NC)           - Generate performance plots and reports"
	@echo ""
	@echo "$(YELLOW)Code Quality Verification:$(NC)"
	@echo "  $(GREEN)make check-cuda-format$(NC)   - Check CUDA/C++ formatting compliance"
	@echo "  $(GREEN)make show-clang-format-diff$(NC) - Show formatting differences"
	@echo "  $(GREEN)make verify-clang-format-config$(NC) - Verify .clang-format config"
	@echo "  $(GREEN)make verify-shellcheck-config$(NC) - Verify shellcheck setup"
	@echo ""
	@echo "$(YELLOW)Utilities & Maintenance:$(NC)"
	@echo "  $(GREEN)make clean$(NC)               - Remove build artifacts"
	@echo "  $(GREEN)make clean-test$(NC)          - Remove test artifacts and coverage files"
	@echo "  $(GREEN)make clean-lint$(NC)          - Remove linting cache files"
	@echo "  $(GREEN)make clean-all$(NC)           - Remove all artifacts (build, test, lint)"
	@echo "  $(GREEN)make install-cuda-tools$(NC)  - Install CUDA development tools"
	@echo "  $(GREEN)make install-shell-tools$(NC) - Install shell script tools"
	@echo "  $(GREEN)make info$(NC)                - Show project information"
	@echo ""
	@echo "$(YELLOW)Data Management:$(NC)"
	@echo "  $(GREEN)make archive-data$(NC)        - Archive current benchmark data"
	@echo "  $(GREEN)make rerun-and-compare$(NC)   - Archive + rerun + compare results"
	@echo "  $(GREEN)make list-archives$(NC)       - List all archived benchmark runs"
	@echo "  $(GREEN)make compare-last$(NC)        - Compare current vs last archive"
	@echo ""
	@echo "$(YELLOW)Recommended Workflow:$(NC)"
	@echo "  1. $(GREEN)make quick-start$(NC)      - Initial setup"
	@echo "  2. $(GREEN)make test-all$(NC)         - Verify all functionality"
	@echo "  3. $(GREEN)make check-all$(NC)        - Verify code quality"
	@echo "  4. $(GREEN)make complete-pipeline$(NC) - Full analysis pipeline"

# === VALIDATION ===
validate-data: ## Validate benchmark JSON data
	@echo "$(BLUE)Validating benchmark data...$(NC)"
	@if [ -f "scripts/validate_results.py" ]; then \
		python3 scripts/validate_results.py; \
	else \
		echo "$(YELLOW)Note: validate_results.py not found. Checking for JSON files...$(NC)"; \
		if ls data/raw/power_modes/*.json 1> /dev/null 2>&1; then \
			echo "$(GREEN)Benchmark data files found - validation skipped$(NC)"; \
		else \
			echo "$(YELLOW)No benchmark data found$(NC)"; \
		fi; \
	fi
	@echo "$(GREEN)Data validation complete$(NC)"

ci: format-check lint type-check validate-data ## Run all CI checks
	@echo "$(GREEN)All CI checks passed$(NC)"

validate-structure: ## Validate project directory structure
	@echo "$(CYAN)Validating project structure...$(NC)"
	@test -d $(IMPL_DIR) || (echo "$(RED)Missing $(IMPL_DIR)/ directory$(NC)" && exit 1)
	@test -f $(IMPL_DIR)/kernels/naive_multiplication.cu || (echo "$(RED)Missing CUDA source: $(IMPL_DIR)/kernels/naive_multiplication.cu$(NC)" && exit 1)
	@test -d $(IMPL_DIR)/utils || (echo "$(RED)Missing $(IMPL_DIR)/utils/ directory$(NC)" && exit 1)
	@test -f $(IMPL_DIR)/utils/common.cu || (echo "$(RED)Missing CUDA utility: $(IMPL_DIR)/utils/common.cu$(NC)" && exit 1)
	@test -f $(IMPL_DIR)/utils/common.h || (echo "$(RED)Missing header: $(IMPL_DIR)/utils/common.h$(NC)" && exit 1)
	@test -d $(BENCH_DIR) || (echo "$(RED)Missing $(BENCH_DIR)/ directory$(NC)" && exit 1)
	@test -d $(RESULTS_DIR) || (echo "$(RED)Missing $(RESULTS_DIR)/ directory$(NC)" && exit 1)
	@test -d $(TEST_DIR) || (echo "$(RED)Missing $(TEST_DIR)/ directory$(NC)" && exit 1)
	@test -d scripts || (echo "$(RED)Missing scripts/ directory$(NC)" && exit 1)
	@test -f scripts/collect_system_specs.sh || (echo "$(RED)Missing system specs script$(NC)" && exit 1)
	@test -f .clang-format || (echo "$(RED)Missing .clang-format configuration$(NC)" && exit 1)
	@test -f .shellcheckrc || (echo "$(RED)Missing .shellcheckrc configuration$(NC)" && exit 1)
	@echo "$(GREEN)Project structure validation passed$(NC)"

validate-hardware: ## Validate hardware compatibility
	@echo "$(CYAN)Validating hardware compatibility...$(NC)"
	@echo "$(BLUE)Detecting system architecture...$(NC)"
	@if ! uname -m | grep -q "aarch64"; then \
		echo "$(YELLOW)Warning: Not running on ARM64 architecture$(NC)"; \
		echo "$(YELLOW)Expected: aarch64, Found: $$(uname -m)$(NC)"; \
		echo "$(YELLOW)This project is optimized for Jetson Orin Nano$(NC)"; \
		if [ "$$FORCE_HARDWARE" = "1" ]; then \
			echo "$(YELLOW)Continuing due to FORCE_HARDWARE=1$(NC)"; \
		else \
			echo "$(RED)Use FORCE_HARDWARE=1 make <target> to override$(NC)"; \
			exit 1; \
		fi; \
	else \
		echo "$(GREEN)✓ ARM64 architecture detected$(NC)"; \
	fi
	@echo "$(BLUE)Checking for Jetson platform...$(NC)"
	@if [ -f /etc/nv_tegra_release ]; then \
		echo "$(GREEN)✓ Jetson platform detected$(NC)"; \
		echo "$(BLUE)Tegra release: $$(cat /etc/nv_tegra_release)$(NC)"; \
	else \
		echo "$(YELLOW)Warning: Jetson platform not detected$(NC)"; \
		echo "$(YELLOW)Missing /etc/nv_tegra_release$(NC)"; \
		if [ "$$FORCE_HARDWARE" = "1" ]; then \
			echo "$(YELLOW)Continuing due to FORCE_HARDWARE=1$(NC)"; \
		else \
			echo "$(RED)Use FORCE_HARDWARE=1 make <target> to override$(NC)"; \
			exit 1; \
		fi; \
	fi
	@echo "$(BLUE)Checking CUDA compute capability...$(NC)"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		if nvidia-smi -L | grep -q "Orin"; then \
			echo "$(GREEN)✓ Orin GPU detected$(NC)"; \
		else \
			echo "$(YELLOW)Warning: Orin GPU not detected in nvidia-smi$(NC)"; \
			echo "$(YELLOW)GPU detected: $$(nvidia-smi -L)$(NC)"; \
			if [ "$$FORCE_HARDWARE" = "1" ]; then \
				echo "$(YELLOW)Continuing due to FORCE_HARDWARE=1$(NC)"; \
			else \
				echo "$(RED)Use FORCE_HARDWARE=1 make <target> to override$(NC)"; \
				exit 1; \
			fi; \
		fi; \
	else \
		echo "$(YELLOW)Warning: nvidia-smi not available$(NC)"; \
		echo "$(YELLOW)Cannot verify GPU model$(NC)"; \
		if [ "$$FORCE_HARDWARE" = "1" ]; then \
			echo "$(YELLOW)Continuing due to FORCE_HARDWARE=1$(NC)"; \
		else \
			echo "$(RED)Use FORCE_HARDWARE=1 make <target> to override$(NC)"; \
			exit 1; \
		fi; \
	fi
	@echo "$(GREEN)Hardware validation completed$(NC)"

# === ENVIRONMENT SETUP ===
setup-env: validate-structure ## Create UV Python environment with Ruff and essential tools
	@echo "$(BLUE)Setting up Python environment with UV and Ruff...$(NC)"
	@# Check if venv exists and is functional (skip UV requirement if already set up)
	@if [ -d "$(VENV_DIR)" ] && [ -f "$(VENV_DIR)/.packages_installed" ]; then \
		echo "$(GREEN)Virtual environment already exists: $(VENV_DIR)$(NC)"; \
		echo "$(GREEN)Packages already installed$(NC)"; \
		echo "$(GREEN)UV environment with Ruff ready: $(VENV_DIR)$(NC)"; \
	else \
		if ! command -v uv > /dev/null 2>&1; then \
			echo "$(RED)ERROR: UV not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh$(NC)"; \
			exit 1; \
		fi; \
		if [ -d "$(VENV_DIR)" ]; then \
			echo "$(GREEN)Virtual environment already exists: $(VENV_DIR)$(NC)"; \
		else \
			echo "$(YELLOW)Creating virtual environment...$(NC)"; \
			$(UV) venv $(VENV_DIR) --python $(PYTHON_VERSION); \
		fi; \
		if [ ! -f "$(VENV_DIR)/.packages_installed" ]; then \
			echo "$(YELLOW)Installing all dependencies from pyproject.toml...$(NC)"; \
			$(UV) pip install --no-cache-dir -e ".[dev]"; \
			touch $(VENV_DIR)/.packages_installed; \
			echo "$(GREEN)All packages installed successfully from pyproject.toml$(NC)"; \
		else \
			echo "$(GREEN)Packages already installed$(NC)"; \
		fi; \
	fi
	@echo "$(GREEN)UV environment with Ruff ready: $(VENV_DIR)$(NC)"

# === CODE QUALITY ===
lint: lint-python lint-cuda lint-shell ## Run all linters
	@echo "$(GREEN)All linting checks completed$(NC)"

format: format-python format-cuda format-shell ## Format all code files
	@echo "$(GREEN)All formatting completed$(NC)"

check-all: setup-env lint ## Run all quality checks
	@echo "$(MAGENTA)========================================$(NC)"
	@echo "$(MAGENTA)CODE QUALITY CHECK COMPLETE$(NC)"  
	@echo "$(MAGENTA)========================================$(NC)"
	@echo "$(GREEN)All files passed quality checks$(NC)"

fix-all: setup-env format ## Auto-fix all formatting issues
	@echo "$(MAGENTA)========================================$(NC)"
	@echo "$(MAGENTA)AUTO-FIX COMPLETE$(NC)"
	@echo "$(MAGENTA)========================================$(NC)"
	@echo "$(GREEN)All auto-fixable issues resolved$(NC)"

# === PYTHON LINTING WITH RUFF ===
lint-python: setup-env ## Lint Python files with Ruff and mypy
	@echo "$(CYAN)Linting Python files with Ruff...$(NC)"
	@if [ -z "$(PYTHON_FILES)" ]; then \
		echo "$(YELLOW)No Python files found in project directories$(NC)"; \
	else \
		echo "$(YELLOW)Found Python files: $(PYTHON_FILES)$(NC)"; \
		echo "$(BLUE)Running Ruff check...$(NC)"; \
		$(PYTHON) -m ruff check --exclude=".venv,__pycache__,.git,.ruff_cache,.mypy_cache,.pytest_cache,build" . || echo "$(YELLOW)Warning: Ruff found issues$(NC)"; \
		echo "$(BLUE)Running mypy type checking...$(NC)"; \
		$(PYTHON) -m mypy $(PYTHON_FILES) --ignore-missing-imports --exclude '.venv|__pycache__|.git|.ruff_cache|.mypy_cache|.pytest_cache|build' || echo "$(YELLOW)Warning: mypy found issues$(NC)"; \
	fi

format-python: setup-env ## Format Python files with Ruff  
	@echo "$(CYAN)Formatting Python files with Ruff...$(NC)"
	@if [ -z "$(PYTHON_FILES)" ]; then \
		echo "$(YELLOW)No Python files found in project directories$(NC)"; \
	else \
		echo "$(YELLOW)Formatting Python files...$(NC)"; \
		echo "$(BLUE)Running Ruff format...$(NC)"; \
		$(PYTHON) -m ruff format --exclude=".venv,__pycache__,.git,.ruff_cache,.mypy_cache,.pytest_cache,build" .; \
		echo "$(BLUE)Running Ruff check --fix...$(NC)"; \
		$(PYTHON) -m ruff check --fix --exclude=".venv,__pycache__,.git,.ruff_cache,.mypy_cache,.pytest_cache,build" .; \
		echo "$(GREEN)Python formatting completed$(NC)"; \
	fi

# === CUDA/C++ LINTING ===
lint-cuda: setup-env ## Lint only project CUDA and C++ files
	@echo "$(CYAN)Linting project CUDA/C++ files...$(NC)"
	@if [ -z "$(CUDA_FILES)$(CPP_FILES)" ]; then \
		echo "$(YELLOW)No CUDA/C++ files found in project directories$(NC)"; \
	else \
		if [ -n "$(CUDA_FILES)" ]; then \
			echo "$(YELLOW)Found CUDA files: $(CUDA_FILES)$(NC)"; \
		fi; \
		if [ -n "$(CPP_FILES)" ]; then \
			echo "$(YELLOW)Found C++ files: $(CPP_FILES)$(NC)"; \
		fi; \
		echo "$(BLUE)Running cpplint on project files only...$(NC)"; \
		for file in $(CUDA_FILES) $(CPP_FILES); do \
			if [ -f "$$file" ]; then \
				echo "Checking: $$file"; \
				$(PYTHON) -m cpplint --config=.cpplint "$$file" || echo "$(YELLOW)Warning: Issues found in $$file$(NC)"; \
			fi; \
		done; \
	fi

# === CUDA/C++ FORMATTING WITH EXPLICIT CONFIG SUPPORT ===
format-cuda: ## Format CUDA/C++ files with clang-format using .clang-format config
	@echo "$(CYAN)CUDA/C++ formatting with project config...$(NC)"
	@if [ ! -f ".clang-format" ]; then \
		echo "$(YELLOW)Warning: .clang-format config file not found$(NC)"; \
		echo "$(YELLOW)Using clang-format default settings$(NC)"; \
	else \
		echo "$(GREEN)Using .clang-format configuration file$(NC)"; \
	fi
	@if ! command -v clang-format >/dev/null 2>&1; then \
		echo "$(YELLOW)clang-format not found. Attempting to install...$(NC)"; \
		if command -v apt >/dev/null 2>&1; then \
			echo "$(BLUE)Installing clang-format via apt...$(NC)"; \
			sudo apt update && sudo apt install -y clang-format; \
		elif command -v yum >/dev/null 2>&1; then \
			echo "$(BLUE)Installing clang-format via yum...$(NC)"; \
			sudo yum install -y clang-tools-extra; \
		elif command -v brew >/dev/null 2>&1; then \
			echo "$(BLUE)Installing clang-format via brew...$(NC)"; \
			brew install clang-format; \
		else \
			echo "$(RED)Cannot auto-install clang-format. Please install manually:$(NC)"; \
			echo "  Ubuntu/Debian: sudo apt install clang-format"; \
			echo "  CentOS/RHEL:   sudo yum install clang-tools-extra"; \
			echo "  macOS:         brew install clang-format"; \
			exit 1; \
		fi; \
	fi
	@if command -v clang-format >/dev/null 2>&1; then \
		echo "$(BLUE)clang-format version: $$(clang-format --version)$(NC)"; \
		for file in $(CUDA_FILES) $(CPP_FILES); do \
			if [ -f "$$file" ]; then \
				echo "Formatting: $$file"; \
				clang-format -i --style=file "$$file"; \
			fi; \
		done; \
		echo "$(GREEN)CUDA/C++ formatting completed$(NC)"; \
	else \
		echo "$(RED)clang-format installation failed$(NC)"; \
		exit 1; \
	fi

check-cuda-format: ## Check if CUDA/C++ files conform to .clang-format style
	@echo "$(CYAN)Checking CUDA/C++ formatting against .clang-format...$(NC)"
	@if [ ! -f ".clang-format" ]; then \
		echo "$(RED)Error: .clang-format config file not found$(NC)"; \
		exit 1; \
	fi
	@if ! command -v clang-format >/dev/null 2>&1; then \
		echo "$(YELLOW)clang-format not available - skipping format check$(NC)"; \
	else \
		echo "$(GREEN)Using .clang-format configuration$(NC)"; \
		format_issues=0; \
		for file in $(CUDA_FILES) $(CPP_FILES); do \
			if [ -f "$$file" ]; then \
				echo "Checking: $$file"; \
				if ! clang-format --style=file "$$file" | diff -u "$$file" - >/dev/null; then \
					echo "$(RED)  Formatting issues found in $$file$(NC)"; \
					format_issues=$$((format_issues + 1)); \
				else \
					echo "$(GREEN)  $$file conforms to .clang-format style$(NC)"; \
				fi; \
			fi; \
		done; \
		if [ $$format_issues -gt 0 ]; then \
			echo "$(RED)Found $$format_issues files with formatting issues$(NC)"; \
			echo "Run 'make format-cuda' to fix them"; \
			exit 1; \
		else \
			echo "$(GREEN)All CUDA/C++ files conform to .clang-format style$(NC)"; \
		fi; \
	fi

show-clang-format-diff: ## Show formatting differences without applying changes
	@echo "$(CYAN)Showing formatting differences...$(NC)"
	@if [ ! -f ".clang-format" ]; then \
		echo "$(RED)Error: .clang-format config file not found$(NC)"; \
		exit 1; \
	fi
	@if ! command -v clang-format >/dev/null 2>&1; then \
		echo "$(RED)clang-format not available$(NC)"; \
		exit 1; \
	fi
	@for file in $(CUDA_FILES) $(CPP_FILES); do \
		if [ -f "$$file" ]; then \
			echo "$(YELLOW)=== Formatting diff for $$file ===$(NC)"; \
			if clang-format --style=file "$$file" | diff -u "$$file" - || echo "$(GREEN)No formatting changes needed for $$file$(NC)"; then \
				true; \
			fi; \
			echo ""; \
		fi; \
	done

verify-clang-format-config: ## Verify .clang-format configuration is valid
	@echo "$(CYAN)Verifying .clang-format configuration...$(NC)"
	@if [ ! -f ".clang-format" ]; then \
		echo "$(RED)Error: .clang-format config file not found$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN).clang-format file found$(NC)"
	@echo "$(YELLOW)Configuration contents:$(NC)"
	@cat .clang-format
	@echo ""
	@if command -v clang-format >/dev/null 2>&1; then \
		echo "$(BLUE)Testing configuration validity...$(NC)"; \
		if echo "int main(){return 0;}" | clang-format --style=file > /dev/null 2>&1; then \
			echo "$(GREEN).clang-format configuration is valid$(NC)"; \
		else \
			echo "$(RED).clang-format configuration has errors$(NC)"; \
			exit 1; \
		fi; \
	else \
		echo "$(YELLOW)clang-format not available - cannot validate config$(NC)"; \
	fi

# === SHELL SCRIPT LINTING ===
lint-shell: setup-env ## Lint shell scripts in scripts/ directory
	@echo "$(CYAN)Linting shell scripts in scripts/ directory...$(NC)"
	@if [ -z "$(SHELL_FILES)" ]; then \
		echo "$(YELLOW)No shell scripts found in scripts/ directory$(NC)"; \
	else \
		echo "$(YELLOW)Found shell files: $(SHELL_FILES)$(NC)"; \
		if command -v shellcheck >/dev/null 2>&1; then \
			echo "$(BLUE)Running system shellcheck...$(NC)"; \
			for file in $(SHELL_FILES); do \
				if [ -f "$$file" ]; then \
					echo "Checking: $$file"; \
					shellcheck --rcfile=.shellcheckrc "$$file" || echo "$(YELLOW)Warning: Issues found in $$file$(NC)"; \
				fi; \
			done; \
		elif [ -f "$(VENV_BIN)/shellcheck" ]; then \
			echo "$(BLUE)Running venv shellcheck...$(NC)"; \
			for file in $(SHELL_FILES); do \
				if [ -f "$$file" ]; then \
					echo "Checking: $$file"; \
					$(VENV_BIN)/shellcheck --rcfile=.shellcheckrc "$$file" || echo "$(YELLOW)Warning: Issues found in $$file$(NC)"; \
				fi; \
			done; \
		else \
			echo "$(YELLOW)shellcheck not available - skipping shell script linting$(NC)"; \
			echo "$(YELLOW)Install with: sudo apt install shellcheck$(NC)"; \
		fi; \
	fi

format-shell: ## Format shell scripts in scripts/ directory
	@echo "$(CYAN)Formatting shell scripts in scripts/ directory...$(NC)"
	@if [ -z "$(SHELL_FILES)" ]; then \
		echo "$(YELLOW)No shell scripts found in scripts/ directory$(NC)"; \
	else \
		echo "$(YELLOW)Processing shell files: $(SHELL_FILES)$(NC)"; \
		for file in $(SHELL_FILES); do \
			if [ -f "$$file" ]; then \
				echo "Processing: $$file"; \
				if ! head -1 "$$file" | grep -q "^#!/"; then \
					echo "$(YELLOW)Warning: $$file missing shebang$(NC)"; \
				fi; \
				chmod +x "$$file"; \
				if [ -n "$$(tail -c1 "$$file")" ]; then \
					echo "" >> "$$file"; \
				fi; \
			fi; \
		done; \
		echo "$(GREEN)Shell script formatting completed$(NC)"; \
	fi

verify-shellcheck-config: ## Verify shellcheck setup
	@echo "$(CYAN)Verifying shellcheck configuration...$(NC)"
	@if [ ! -f ".shellcheckrc" ]; then \
		echo "$(RED).shellcheckrc not found$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN).shellcheckrc found$(NC)"
	@if command -v shellcheck >/dev/null 2>&1; then \
		echo "$(GREEN)shellcheck available: $$(which shellcheck)$(NC)"; \
	elif [ -f "$(VENV_BIN)/shellcheck" ]; then \
		echo "$(GREEN)venv shellcheck available$(NC)"; \
	else \
		echo "$(RED)shellcheck not available$(NC)"; \
	fi

format-shell-safe: verify-shellcheck-config format-shell ## Safe shell formatting with validation
	@echo "$(GREEN)Shell formatting with validation completed$(NC)"

# === COMPILATION ===
compile: validate-structure ## Compile all implementations (naive, blocked, cuBLAS, Tensor Core)
	@if [ -f "$(CUDA_TARGET)" ] && [ -f "$(BLOCKED_TARGET)" ] && [ -f "$(CUBLAS_TARGET)" ] && [ -f "$(TENSOR_CORE_TARGET)" ]; then \
		echo "$(GREEN)All implementations already compiled$(NC)"; \
	else \
		$(MAKE) force-compile; \
	fi

force-compile: validate-structure ## Force recompilation of all CUDA code
	@echo "$(BLUE)Compiling CUDA Implementations...$(NC)"
	@echo "$(YELLOW)1. Compiling naive implementation...$(NC)"
	@if [ ! -f "$(IMPL_DIR)/kernels/naive_multiplication.cu" ]; then \
		echo "$(RED)ERROR: naive_multiplication.cu not found$(NC)"; \
		exit 1; \
	fi
	cd $(IMPL_DIR) && $(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o naive_benchmark kernels/naive_multiplication.cu utils/common.cu $(CUDA_LIBS)
	@echo "$(GREEN)✓ Naive compilation successful$(NC)"
	@echo "$(YELLOW)2. Compiling blocked implementation...$(NC)"
	@if [ ! -f "$(IMPL_DIR)/kernels/blocked_multiplication.cu" ]; then \
		echo "$(RED)ERROR: blocked_multiplication.cu not found$(NC)"; \
		exit 1; \
	fi
	cd $(IMPL_DIR) && $(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o blocked_benchmark kernels/blocked_multiplication.cu utils/common.cu $(CUDA_LIBS)
	@echo "$(GREEN)✓ Blocked compilation successful$(NC)"
	@echo "$(YELLOW)3. Compiling cuBLAS implementation...$(NC)"
	@if [ ! -f "$(IMPL_DIR)/kernels/cublas_multiplication.cu" ]; then \
		echo "$(RED)ERROR: cublas_multiplication.cu not found$(NC)"; \
		exit 1; \
	fi
	cd $(IMPL_DIR) && $(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o cublas_benchmark kernels/cublas_multiplication.cu utils/common.cu $(CUDA_LIBS)
	@echo "$(GREEN)✓ cuBLAS compilation successful$(NC)"
	@echo "$(YELLOW)4. Compiling Tensor Core implementation...$(NC)"
	@if [ ! -f "$(IMPL_DIR)/kernels/tensor_core_multiplication.cu" ]; then \
		echo "$(RED)ERROR: tensor_core_multiplication.cu not found$(NC)"; \
		exit 1; \
	fi
	cd $(IMPL_DIR) && $(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o tensor_core_benchmark kernels/tensor_core_multiplication.cu utils/common.cu $(CUDA_LIBS)
	@echo "$(GREEN)✓ Tensor Core compilation successful$(NC)"
	@echo "$(GREEN)All implementations compiled successfully$(NC)"

# === TESTING & ANALYSIS ===
test: test-quick ## Run quick functionality test (alias for test-quick)

test-quick: ## Quick CUDA functionality test
	@echo "$(CYAN)Testing Basic CUDA Functionality...$(NC)"
	@if [ ! -f "$(CUDA_TARGET)" ]; then \
		echo "$(RED)Binary not found. Compiling...$(NC)"; \
		$(MAKE) compile; \
	fi
	cd $(IMPL_DIR) && ./naive_benchmark 64
	@echo "$(GREEN)Quick test completed$(NC)"

test-unit: setup-env ## Run unit tests with pytest
	@echo "$(CYAN)Running Unit Tests...$(NC)"
	@if [ ! -d "$(TEST_DIR)" ]; then \
		echo "$(RED)Tests directory not found$(NC)"; \
		exit 1; \
	fi
	$(PYTHON) -m pytest $(TEST_DIR)/ -v --tb=short
	@echo "$(GREEN)Unit tests completed$(NC)"

test-integration: setup-env compile ## Run integration tests
	@echo "$(CYAN)Running Integration Tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR)/test_integration.py -v --tb=short
	@echo "$(GREEN)Integration tests completed$(NC)"

test-performance: setup-env ## Run performance calculation tests
	@echo "$(CYAN)Running Performance Tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR)/test_performance.py -v --tb=short
	@echo "$(GREEN)Performance tests completed$(NC)"

test-utils: setup-env ## Run utility function tests
	@echo "$(CYAN)Running Utility Tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR)/test_utils.py -v --tb=short
	@echo "$(GREEN)Utility tests completed$(NC)"

test-all: setup-env compile test-unit test-quick ## Run all tests (unit + functionality)
	@echo "$(MAGENTA)========================================$(NC)"
	@echo "$(MAGENTA)ALL TESTS COMPLETED$(NC)"
	@echo "$(MAGENTA)========================================$(NC)"
	@echo "$(GREEN)Test Suite Summary:$(NC)"
	@echo "  ✓ Unit tests (pytest)"
	@echo "  ✓ Integration tests"
	@echo "  ✓ Performance calculation tests"
	@echo "  ✓ Utility function tests"
	@echo "  ✓ CUDA functionality test"

test-coverage: setup-env ## Run tests with coverage report
	@echo "$(CYAN)Running Tests with Coverage Analysis...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR)/ --cov=benchmarks --cov=data --cov-report=html --cov-report=term-missing -v
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

verify-accuracy: setup-env compile ## Verify numerical accuracy with detailed testing
	@echo "$(CYAN)Verifying Numerical Accuracy...$(NC)"
	@echo "$(BLUE)1. Running CUDA functionality test...$(NC)"
	cd $(IMPL_DIR) && ./naive_benchmark 64
	@echo "$(BLUE)2. Running numerical accuracy tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR)/test_utils.py::TestNumericalAccuracy -v --tb=short
	@echo "$(GREEN)Numerical accuracy verification completed$(NC)"

system-specs: ## Generate hardware specifications
	@echo "$(BLUE)Generating System Specifications...$(NC)"
	@echo "$(YELLOW)Note: This command requires sudo for system telemetry$(NC)"
	@echo "$(YELLOW)You will be prompted for your password once$(NC)"
	@sudo -v  # Request sudo credentials upfront
	@if [ ! -f "scripts/collect_system_specs.sh" ]; then \
		echo "$(RED)ERROR: collect_system_specs.sh not found$(NC)"; \
		exit 1; \
	fi
	@if [ ! -x "scripts/collect_system_specs.sh" ]; then \
		echo "$(YELLOW)Making script executable...$(NC)"; \
		chmod +x scripts/collect_system_specs.sh; \
	fi
	@echo "$(YELLOW)Running system specification collection...$(NC)"
	@sudo scripts/collect_system_specs.sh
	@if [ $$? -eq 0 ]; then \
		echo "$(GREEN)System specifications documented successfully$(NC)"; \
	else \
		echo "$(RED)WARNING: System specs collection completed with warnings$(NC)"; \
		echo "$(YELLOW)Check the generated report for details$(NC)"; \
	fi

full-analysis: setup-env validate-hardware ## Run 3-power mode analysis
	@echo "$(MAGENTA)Running 3-Power Mode Analysis...$(NC)"
	@echo "$(YELLOW)Note: This command requires sudo for power mode switching$(NC)"
	@echo "$(YELLOW)If passwordless sudo is configured, this will run without prompts$(NC)"
	@echo "$(YELLOW)Otherwise, you will be prompted for your password when needed$(NC)"
	@if [ ! -f "$(CUDA_TARGET)" ]; then \
		$(MAKE) compile; \
	fi
	@$(PYTHON) $(BENCH_DIR)/multi_power_mode_benchmark.py

visualize: setup-env ## Generate performance visualizations
	@echo "$(MAGENTA)Generating Visualizations...$(NC)"
	$(PYTHON) $(RESULTS_DIR)/visualize_power_modes.py
	@echo "$(GREEN)Plots generated$(NC)"

# === WORKFLOWS ===
quick-start: setup-env compile test-all verify-accuracy ## Complete setup and verification
	@echo "$(MAGENTA)========================================$(NC)"
	@echo "$(MAGENTA)QUICK START COMPLETED$(NC)"
	@echo "$(MAGENTA)========================================$(NC)"
	@echo ""
	@echo "$(GREEN)Environment and Implementation Ready!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next Steps:$(NC)"
	@echo "• Run quality checks: $(GREEN)make check-all$(NC)"
	@echo "• Run full analysis: $(GREEN)make full-analysis$(NC)"
	@echo "• Complete pipeline: $(GREEN)make complete-pipeline$(NC)"

complete-pipeline: setup-env check-all compile test-all verify-accuracy system-specs full-analysis visualize ## Complete benchmark pipeline
	@echo "$(MAGENTA)========================================$(NC)"
	@echo "$(MAGENTA)BENCHMARK ANALYSIS COMPLETE$(NC)"
	@echo "$(MAGENTA)========================================$(NC)"
	@echo ""
	@echo "$(GREEN)Deliverables:$(NC)"
	@echo "• Code quality verified with Ruff-based linting"
	@echo "• Complete test suite validation"
	@echo "• Accurate baseline performance characterization"
	@echo "• 3-power mode analysis"
	@echo "• System specifications for reproducibility"
	@echo "• Performance visualizations"
	@$(MAKE) validation-checklist

validation-checklist: ## Validate benchmark completion
	@echo "$(BLUE)========================================$(NC)"
	@echo "$(BLUE)BENCHMARK VALIDATION CHECKLIST$(NC)"
	@echo "$(BLUE)========================================$(NC)"
	@echo ""
	@echo "$(YELLOW)Requirements Check:$(NC)"
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "  ✓ Development environment ready"; \
	else \
		echo "  ✗ Environment setup needed"; \
	fi
	@if [ -f "$(CUDA_TARGET)" ]; then \
		echo "  ✓ CUDA implementation compiled"; \
	else \
		echo "  ✗ Compilation needed"; \
	fi
	@if [ -d "$(TEST_DIR)" ] && [ -f "$(TEST_DIR)/conftest.py" ]; then \
		echo "  ✓ Test suite available"; \
		if $(PYTHON) -m pytest $(TEST_DIR)/ --collect-only -q >/dev/null 2>&1; then \
			echo "  ✓ Tests are discoverable and valid"; \
		else \
			echo "  ⚠ Test collection issues detected"; \
		fi; \
	else \
		echo "  ✗ Test suite setup needed"; \
	fi
	@if ls data/reports/jetson_system_specifications_*.md >/dev/null 2>&1; then \
		echo "  ✓ System specifications documented"; \
	else \
		echo "  ✗ System specs needed"; \
	fi
	@if ls data/raw/power_modes/jetson_orin_nano_3mode_analysis_*.json >/dev/null 2>&1; then \
		echo "  ✓ 3-power mode analysis completed"; \
	else \
		echo "  ✗ Analysis needed"; \
	fi
	@if ls data/plots/power_analysis/*.png >/dev/null 2>&1; then \
		echo "  ✓ Visualizations generated"; \
	else \
		echo "  ✗ Visualizations needed"; \
	fi

# === UTILITIES ===
clean-test: ## Remove test artifacts and coverage files
	@echo "$(YELLOW)Cleaning test artifacts...$(NC)"
	@find . -maxdepth 3 -type d -name ".pytest_cache" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || echo "$(YELLOW)Warning: Some pytest cache files could not be removed$(NC)"
	@find . -maxdepth 3 -type f -name ".coverage" -not -path "./.venv/*" -delete 2>/dev/null || echo "$(YELLOW)Warning: Some coverage files could not be removed$(NC)"
	@find . -maxdepth 3 -type d -name "htmlcov" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || echo "$(YELLOW)Warning: Some HTML coverage directories could not be removed$(NC)"
	@find . -maxdepth 3 -type f -name "coverage.xml" -not -path "./.venv/*" -delete 2>/dev/null || echo "$(YELLOW)Warning: Some coverage XML files could not be removed$(NC)"
	@echo "$(GREEN)Test cleanup completed$(NC)"

clean-lint: ## Remove linting cache files (project scope only)
	@echo "$(YELLOW)Cleaning project linting cache files...$(NC)"
	@find . -maxdepth 3 -type d -name ".ruff_cache" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || echo "$(YELLOW)Warning: Some Ruff cache files could not be removed$(NC)"
	@find . -maxdepth 3 -type d -name ".mypy_cache" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || echo "$(YELLOW)Warning: Some mypy cache files could not be removed$(NC)"
	@find . -maxdepth 3 -type d -name ".pytest_cache" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || echo "$(YELLOW)Warning: Some pytest cache files could not be removed$(NC)"
	@find . -maxdepth 3 -type f -name ".coverage" -not -path "./.venv/*" -delete 2>/dev/null || echo "$(YELLOW)Warning: Some coverage files could not be removed$(NC)"
	@find . -maxdepth 3 -type d -name "htmlcov" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || echo "$(YELLOW)Warning: Some HTML coverage directories could not be removed$(NC)"
	@echo "$(GREEN)Project linting cleanup completed$(NC)"

clean: ## Remove build artifacts (project scope only)
	@echo "$(YELLOW)Cleaning project build artifacts...$(NC)"
	@if [ -f "$(CUDA_TARGET)" ] && [ -f "$(BLOCKED_TARGET)" ] && [ -f "$(CUBLAS_TARGET)" ] && [ -f "$(TENSOR_CORE_TARGET)" ]; then \
		echo "$(YELLOW)Removing compiled binaries: $(CUDA_TARGET) $(BLOCKED_TARGET) $(CUBLAS_TARGET) $(TENSOR_CORE_TARGET)$(NC)"; \
		rm -f $(CUDA_TARGET) $(BLOCKED_TARGET) $(CUBLAS_TARGET) $(TENSOR_CORE_TARGET); \
	else \
		if [ -f "$(CUDA_TARGET)" ]; then \
			echo "$(YELLOW)Removing compiled binary: $(CUDA_TARGET)$(NC)"; \
			rm -f $(CUDA_TARGET); \
		fi; \
		if [ -f "$(BLOCKED_TARGET)" ]; then \
			echo "$(YELLOW)Removing compiled binary: $(BLOCKED_TARGET)$(NC)"; \
			rm -f $(BLOCKED_TARGET); \
		fi; \
		if [ -f "$(CUBLAS_TARGET)" ]; then \
			echo "$(YELLOW)Removing compiled binary: $(CUBLAS_TARGET)$(NC)"; \
			rm -f $(CUBLAS_TARGET); \
		fi; \
		if [ -f "$(TENSOR_CORE_TARGET)" ]; then \
			echo "$(YELLOW)Removing compiled binary: $(TENSOR_CORE_TARGET)$(NC)"; \
			rm -f $(TENSOR_CORE_TARGET); \
		fi; \
	fi
	@cd $(IMPL_DIR) && rm -f *.o 2>/dev/null || echo "$(YELLOW)Warning: Some object files could not be removed$(NC)"
	@find . -maxdepth 3 -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || echo "$(YELLOW)Warning: Some Python cache directories could not be removed$(NC)"
	@find . -maxdepth 3 -type f -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || echo "$(YELLOW)Warning: Some Python cache files could not be removed$(NC)"
	@find . -maxdepth 2 -type d -name "*.egg-info" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || echo "$(YELLOW)Warning: Some egg-info directories could not be removed$(NC)"
	@echo "$(GREEN)Project cleanup completed$(NC)"

clean-all: clean clean-test clean-lint ## Remove all artifacts (build, test, lint)
	@echo "$(GREEN)Complete cleanup finished$(NC)"

install-cuda-tools: ## Install CUDA development tools
	@echo "$(BLUE)Installing CUDA development tools...$(NC)"
	@if command -v apt >/dev/null 2>&1; then \
		echo "Installing via apt..."; \
		sudo apt update; \
		sudo apt install -y clang-format clang-tidy cppcheck; \
	elif command -v yum >/dev/null 2>&1; then \
		echo "Installing via yum..."; \
		sudo yum install -y clang-tools-extra cppcheck; \
	else \
		echo "$(YELLOW)Please install manually for your system:$(NC)"; \
		echo "  clang-format: C++ code formatter"; \
		echo "  clang-tidy: C++ static analyzer"; \
		echo "  cppcheck: Additional static analysis"; \
	fi

debug-cuda: ## Check CUDA environment
	@echo "$(YELLOW)=== CUDA Environment Check ===$(NC)"
	@echo "NVCC version:"
	@if which nvcc >/dev/null 2>&1; then \
		nvcc --version; \
	else \
		echo "$(RED)NVCC not found in PATH$(NC)"; \
		if [ -f /usr/local/cuda/bin/nvcc ]; then \
			echo "$(YELLOW)Found at: /usr/local/cuda/bin/nvcc$(NC)"; \
			/usr/local/cuda/bin/nvcc --version; \
		fi; \
	fi
	@echo ""
	@echo "GPU detection:"
	@if nvidia-smi -L >/dev/null 2>&1; then \
		nvidia-smi -L; \
	else \
		echo "nvidia-smi not available (normal for integrated GPU)"; \
		if ls /dev/nvidia* >/dev/null 2>&1; then \
			echo "NVIDIA devices found:"; \
			ls -la /dev/nvidia*; \
		fi; \
	fi
	@echo ""
	@echo "CUDA libraries:"
	@if [ -d /usr/local/cuda/lib64/ ]; then \
		ls -la /usr/local/cuda/lib64/ | grep -E "(cublas|curand)" 2>/dev/null || echo "Standard CUDA libraries not found in expected location"; \
	else \
		echo "CUDA lib64 directory not found at standard location"; \
	fi

install-shell-tools: ## Install shell script development tools  
	@echo "$(BLUE)Installing shell script tools...$(NC)"
	@if [ ! -f "$(VENV_DIR)/.packages_installed" ]; then \
		echo "$(RED)Python environment not set up. Run 'make setup-env' first$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)shellcheck-py already installed via setup-env$(NC)"
	@if command -v shfmt >/dev/null 2>&1; then \
		echo "$(GREEN)shfmt already available$(NC)"; \
	else \
		echo "$(YELLOW)Optional: Install shfmt for advanced shell formatting$(NC)"; \
		echo "  go install mvdan.cc/sh/v3/cmd/shfmt@latest"; \
	fi

info: ## Show project information
	@echo "$(BLUE)Project:$(NC)     $(PROJECT_NAME)"
	@echo "$(BLUE)Version:$(NC)     $(VERSION)"
	@echo "$(BLUE)Hardware:$(NC)    $(TARGET_HARDWARE)"
	@echo "$(BLUE)Author:$(NC)      $(AUTHOR)"

# === DATA MANAGEMENT ===
archive-data: ## Archive current benchmark data with timestamp
	@echo "$(BLUE)Archiving current benchmark data...$(NC)"
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S) && \
	ARCHIVE_DIR="$(RESULTS_DIR)/archive/baseline_$$TIMESTAMP" && \
	mkdir -p "$$ARCHIVE_DIR/raw" "$$ARCHIVE_DIR/plots" "$$ARCHIVE_DIR/reports" "$$ARCHIVE_DIR/logs" "$$ARCHIVE_DIR/metadata" && \
	cp -r $(RESULTS_DIR)/raw/* "$$ARCHIVE_DIR/raw/" 2>/dev/null || true && \
	cp -r $(RESULTS_DIR)/plots/* "$$ARCHIVE_DIR/plots/" 2>/dev/null || true && \
	cp -r $(RESULTS_DIR)/reports/* "$$ARCHIVE_DIR/reports/" 2>/dev/null || true && \
	cp -r $(RESULTS_DIR)/logs/* "$$ARCHIVE_DIR/logs/" 2>/dev/null || true && \
	echo "# Benchmark Archive: $$TIMESTAMP" > "$$ARCHIVE_DIR/metadata/archive_info.md" && \
	echo "" >> "$$ARCHIVE_DIR/metadata/archive_info.md" && \
	echo "**Created:** $$(date)" >> "$$ARCHIVE_DIR/metadata/archive_info.md" && \
	echo "**Git Commit:** $$(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')" >> "$$ARCHIVE_DIR/metadata/archive_info.md" && \
	echo "**Git Branch:** $$(git branch --show-current 2>/dev/null || echo 'N/A')" >> "$$ARCHIVE_DIR/metadata/archive_info.md" && \
	echo "" >> "$$ARCHIVE_DIR/metadata/archive_info.md" && \
	echo "## Files Archived" >> "$$ARCHIVE_DIR/metadata/archive_info.md" && \
	find "$$ARCHIVE_DIR" -type f | sort >> "$$ARCHIVE_DIR/metadata/archive_info.md" && \
	echo "$(GREEN)✓ Data archived to: $$ARCHIVE_DIR$(NC)"

rerun-and-compare: ## Archive current data, re-run pipeline, and compare results
	@if [ ! -d "$(RESULTS_DIR)/raw/power_modes" ] || [ -z "$$(ls -A $(RESULTS_DIR)/raw/power_modes 2>/dev/null)" ]; then \
		echo "$(RED)Error: No current benchmark data to archive$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Starting archive + rerun + compare workflow...$(NC)"
	@./scripts/archive_and_rerun.sh

list-archives: ## List all archived benchmark runs
	@echo "$(BLUE)Archived Benchmark Runs:$(NC)"
	@if [ -d "$(RESULTS_DIR)/archive" ]; then \
		for dir in $(RESULTS_DIR)/archive/*/; do \
			if [ -f "$$dir/metadata/archive_info.md" ] || [ -f "$$dir/metadata/run_info.md" ]; then \
				echo "$(GREEN)$$dir$(NC)"; \
				if [ -f "$$dir/metadata/archive_info.md" ]; then \
					grep "Created:" "$$dir/metadata/archive_info.md" || true; \
				fi; \
				if [ -f "$$dir/metadata/run_info.md" ]; then \
					grep "Timestamp:" "$$dir/metadata/run_info.md" || true; \
				fi; \
				echo ""; \
			fi; \
		done; \
	else \
		echo "$(YELLOW)No archives found. Create one with: make archive-data$(NC)"; \
	fi

compare-last: ## Compare current results with most recent archive
	@echo "$(BLUE)Comparing current results with last archive...$(NC)"
	@LAST_ARCHIVE=$$(ls -td $(RESULTS_DIR)/archive/*/ 2>/dev/null | head -1); \
	if [ -z "$$LAST_ARCHIVE" ]; then \
		echo "$(RED)No archived runs found$(NC)"; \
		exit 1; \
	fi; \
	echo "$(YELLOW)Baseline: $$LAST_ARCHIVE$(NC)"; \
	echo "$(YELLOW)Current:  $(RESULTS_DIR)/raw/power_modes$(NC)"; \
	$(PYTHON) scripts/compare_benchmark_runs.py "$$LAST_ARCHIVE/raw/power_modes" "$(RESULTS_DIR)/raw/power_modes"