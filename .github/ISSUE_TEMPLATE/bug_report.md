---
name: Bug Report
about: Report a bug or unexpected behavior in the benchmarking framework
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
**A clear and concise description of what the bug is.**

## To Reproduce
Steps to reproduce the behavior:
1. Run command '...'
2. With configuration '...'
3. See error

## Expected Behavior
**What you expected to happen.**

## Actual Behavior
**What actually happened.**

## Error Output
```
Paste error messages or stack traces here
```

## Environment
**Please complete the following information:**

### Hardware
- Device: [e.g., Jetson Orin Nano Engineering Reference Developer Kit Super]
- Power Mode: [e.g., 15W / 25W / MAXN]
- Thermal Condition: [e.g., ambient temperature, active cooling]

### Software
- JetPack Version: [e.g., 6.0]
- L4T Version: [e.g., R36.4.4]
- CUDA Version: [run `nvcc --version`]
- Python Version: [run `python --version`]
- UV Version: [run `uv --version`]

### Benchmark Configuration
- Matrix Sizes: [e.g., 64, 128, 256, 512, 1024]
- Implementations: [e.g., naive, blocked, cuBLAS, tensor_core]
- Power Modes: [e.g., all / specific modes]

## Compilation Output
**If compilation failed, paste the output:**
```bash
# Run: make force-compile 2>&1 | tee compile.log
# Paste output here
```

## Test Results
**Did tests pass before encountering the bug?**
```bash
# Run: make test-all
# Paste results here
```

## Screenshots
**If applicable, add screenshots to help explain your problem.**

## Additional Context
**Add any other context about the problem here.**

## Attempted Solutions
**What have you tried to fix this?**

## Checklist
- [ ] I have read the [README.md](../README.md)
- [ ] I have run `make validate-structure` successfully
- [ ] I have run `make validate-hardware` successfully
- [ ] I have checked existing issues for duplicates
- [ ] I can reproduce this bug consistently
- [ ] I have included all relevant error messages
