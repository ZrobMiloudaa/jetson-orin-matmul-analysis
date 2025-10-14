# Quick Reference Card

**Jetson Orin Nano Matrix Multiplication Benchmarking v1.0.0**

---

## Initial Setup (One-Time)

```bash
# Clone and setup
git clone https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis.git
cd jetson-orin-matmul-analysis
make quick-start

# Optional: Configure passwordless sudo (recommended)
sudo ./scripts/setup_passwordless_sudo.sh
```

---

## Common Commands

### Running Benchmarks
```bash
make test-quick          # Quick test (64×64 matrix, ~5 seconds)
make full-analysis       # Full 3-mode analysis (~15 minutes)
make visualize           # Generate plots from existing data
make system-specs        # Document hardware specs
```

### Testing & Validation
```bash
make test-all            # Complete test suite (27 tests)
make test-unit           # Unit tests only
make verify-accuracy     # Numerical accuracy validation
```

### Code Quality
```bash
make check-all           # Run all linters (Ruff, mypy, cpplint, shellcheck)
make fix-all             # Auto-fix formatting issues
make lint-python         # Python linting only
make lint-cuda           # CUDA linting only
```

### Data Management
```bash
make archive-data        # Archive current results with timestamp
make list-archives       # List all archived runs
make compare-last        # Compare current vs most recent archive
```

---

## Sudo Requirements

### With Passwordless Setup (Recommended)
```bash
# One-time configuration
sudo ./scripts/setup_passwordless_sudo.sh

# Then these work without password:
make system-specs
make full-analysis
```

### Without Passwordless Setup
```bash
# Password prompt once at start, then runs uninterrupted
make system-specs
make full-analysis
```

### Remove Passwordless Sudo
```bash
sudo ./scripts/remove_passwordless_sudo.sh
```

---

## Important Files

| File/Directory | Purpose |
|----------------|---------|
| `Makefile` | All automation commands |
| `SUDO_SETUP_GUIDE.md` | Detailed sudo configuration guide |
| `data/raw/power_modes/` | Benchmark results (JSON/CSV) |
| `data/plots/` | Generated visualizations |
| `data/reports/` | Analysis reports |
| `cuda/` | CUDA implementations |
| `tests/` | Test suite |

---

## Power Modes

| Mode | Power | GPU Freq | Use Case |
|------|-------|----------|----------|
| 15W | 15W max | ~612 MHz | Battery-powered, extended runtime |
| 25W | 25W max | ~918 MHz | **Sweet spot** (90% of max perf, 88% power) |
| MAXN | 30W max | ~1020 MHz | Plugged-in, maximum throughput |

---

## Performance Summary

### Matrix Size: 1024×1024 @ MAXN Mode

| Implementation | GFLOPS | Efficiency | Speedup vs Naive |
|----------------|--------|------------|------------------|
| Naive | 95 | 4.6% | 1.0× (baseline) |
| Blocked | 150 | 7.2% | 1.6× |
| cuBLAS | 1,282 | 61% | 13.5× |
| Tensor Core (TF32) | 952 | 46% | 10.0× |

---

## Troubleshooting

### "UV not installed"
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### "CUDA compilation failed"
```bash
make debug-cuda          # Check CUDA environment
which nvcc               # Verify CUDA in PATH
```

### "Permission denied" for power modes
```bash
# Option 1: Run with sudo
sudo make full-analysis

# Option 2: Configure passwordless sudo
sudo ./scripts/setup_passwordless_sudo.sh
```

### "Tests failing"
```bash
make test-all -v        # Verbose output
.venv/bin/python -m pytest tests/ -vv --tb=long
```

---

## Documentation Links

- **Main README:** `README.md` - Project overview and key findings
- **Contributing:** `CONTRIBUTING.md` - How to contribute
- **Sudo Setup:** `SUDO_SETUP_GUIDE.md` - Passwordless sudo configuration
- **Quick Reference:** `QUICK_REFERENCE.md` - Command cheat sheet
- **Citation:** `CITATION.cff` - Academic citation metadata

---

## Getting Help

1. Check documentation: Start with `README.md` and `QUICK_REFERENCE.md`
2. Review logs: `data/logs/jetson_benchmark.log`
3. Search issues: [GitHub Issues](https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis/issues)
4. Report bugs: Use "Bug Report" issue template
5. Ask questions: Open a "Discussion" on GitHub

---

## Research & Citation

```bibtex
@software{moses2025jetson,
  author = {Moses, Jesse},
  orcid = {0009-0006-0322-7974},
  title = {Jetson Orin Nano Matrix Multiplication: Power-Performance Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis},
  version = {1.0.0}
}
```

---

## Key Findings (TL;DR)

1. **25W mode is the sweet spot:** 90% of max performance at 88% power consumption
2. **cuBLAS is 13.5× faster** than naive at 1024×1024 (MAXN mode)
3. **Tensor Cores with TF32:** 10× speedup with 99.0% accuracy (suitable for ML)
4. **Memory bandwidth limited:** Most implementations <10% bandwidth utilization
5. **Thermal stable:** No throttling observed during sustained benchmarks

---

**Last Updated:** 2025-10-13
**Project Version:** 1.0.0
**Hardware:** Jetson Orin Nano Engineering Reference Developer Kit Super
**Software:** JetPack 6.x, CUDA 12.6, L4T R36.4+
