---
name: Community Benchmark Results
about: Share your benchmark results from running this framework on your hardware
title: '[RESULTS] Jetson [Model] - [Your Configuration]'
labels: community-benchmark, data
assignees: ''
---

##  Thank You for Sharing Your Results!

Your contribution helps build a community database of validated benchmarks across different hardware configurations. This data is invaluable for understanding real-world performance characteristics.

---

## Hardware Configuration

### Device Information
- **Jetson Model:** [e.g., Orin Nano, Orin NX, AGX Orin]
- **SKU/Variant:** [e.g., 8GB, 16GB]
- **Cooling Solution:** [e.g., passive heatsink, active fan, custom cooling]
- **Ambient Temperature:** [e.g., 22°C]
- **Power Supply:** [e.g., barrel jack, USB-C PD]

### Software Stack
- **JetPack Version:** [run: `cat /etc/nv_tegra_release`]
- **L4T Version:** [e.g., R36.4.4]
- **CUDA Version:** [run: `nvcc --version`]
- **cuBLAS Version:** [run: `dpkg -l | grep cublas`]

---

## Benchmark Results

### cuBLAS Implementation

**Power Mode: 15W**
| Matrix Size | GFLOPS | Efficiency (%) | Time (ms) | GPU Freq (MHz) |
|-------------|--------|----------------|-----------|----------------|
| 64          |        |                |           |                |
| 128         |        |                |           |                |
| 256         |        |                |           |                |
| 512         |        |                |           |                |
| 1024        |        |                |           |                |

**Power Mode: 25W**
| Matrix Size | GFLOPS | Efficiency (%) | Time (ms) | GPU Freq (MHz) |
|-------------|--------|----------------|-----------|----------------|
| 64          |        |                |           |                |
| 128         |        |                |           |                |
| 256         |        |                |           |                |
| 512         |        |                |           |                |
| 1024        |        |                |           |                |

**Power Mode: MAXN**
| Matrix Size | GFLOPS | Efficiency (%) | Time (ms) | GPU Freq (MHz) |
|-------------|--------|----------------|-----------|----------------|
| 64          |        |                |           |                |
| 128         |        |                |           |                |
| 256         |        |                |           |                |
| 512         |        |                |           |                |
| 1024        |        |                |           |                |

### Tensor Core Implementation (Optional)

**Power Mode: 25W (Recommended for TF32)**
| Matrix Size | GFLOPS | Accuracy Error | Time (ms) | GPU Freq (MHz) |
|-------------|--------|----------------|-----------|----------------|
| 64          |        |                |           |                |
| 128         |        |                |           |                |
| 256         |        |                |           |                |
| 512         |        |                |           |                |
| 1024        |        |                |           |                |

---

## Power Efficiency Analysis

**GFLOPS/Watt Comparison:**
| Power Mode | Average GFLOPS | Power Consumption (W) | GFLOPS/Watt |
|------------|----------------|----------------------|-------------|
| 15W        |                |                      |             |
| 25W        |                |                      |             |
| MAXN       |                |                      |             |

**Sweet Spot Identified:** [e.g., 25W mode delivers 90% of MAXN performance at 88% power]

---

## Thermal Observations

**Temperature Readings:**
- **Pre-Benchmark (Idle):** ___ °C
- **Peak Temperature (Under Load):** ___ °C
- **Post-Benchmark (Cooldown):** ___ °C
- **Thermal Throttling Observed:** Yes / No

**Notes on Cooling:**
[Describe your cooling setup and any thermal management observations]

---

## System Specifications

**Run this command and paste output:**
```bash
sudo ./scripts/collect_system_specs.sh
```

<details>
<summary>System Specs Output (click to expand)</summary>

```
Paste full output here
```

</details>

---

## Validation Results

**Numerical Accuracy:**
```bash
# Run: make verify-accuracy
# Paste results here
```

- **GFLOPS Calculation Accuracy:** [e.g., 99.5%]
- **Measurements Exceeding Theoretical Limits:** [e.g., 0]
- **Numerical Error (FP32):** [e.g., < 1e-5]

---

## Interesting Findings

**Did you discover anything unexpected?**
- [ ] Power mode behaved differently than expected
- [ ] Thermal throttling occurred earlier/later than anticipated
- [ ] Performance varied significantly from published results
- [ ] Sweet spot was at different power mode
- [ ] Other: _____

**Details:**
[Describe any surprising or noteworthy observations]

---

## Comparison to Published Results

**How do your results compare to the main repository findings?**
- [ ] Very similar (±5%)
- [ ] Somewhat different (5-15% variance)
- [ ] Significantly different (>15% variance)

**If significantly different, potential reasons:**
- [ ] Different hardware variant (e.g., 8GB vs 16GB)
- [ ] Different cooling solution
- [ ] Different software stack version
- [ ] Environmental factors (temperature, altitude)
- [ ] Other: _____

---

## Raw Data Files (Optional)

**If you'd like to share raw benchmark data:**

<details>
<summary>JSON Output Files</summary>

```json
Paste contents of data/raw/power_modes/*.json here
or attach files to this issue
```

</details>

---

## Use Case Context

**What are you using this benchmarking framework for?**
- [ ] Production edge AI deployment planning
- [ ] Academic research
- [ ] Personal learning/experimentation
- [ ] Robotics platform optimization
- [ ] IoT device power management
- [ ] Other: _____

**Project Details (optional):**
[Brief description of your project or research]

---

## Acknowledgments

**Would you like to be acknowledged if this data is included in future publications or reports?**
- [ ] Yes, acknowledge as: [Your Name / Organization]
- [ ] Yes, but keep anonymous
- [ ] No acknowledgment needed

---

## Additional Notes

**Any other observations, comments, or context you'd like to share?**

---

## Reproducibility

**Can others reproduce your results with the same hardware?**
- [ ] Yes, I followed standard procedure (`make full-analysis`)
- [ ] Yes, but with modifications (explain below)
- [ ] Unsure

**Modifications to benchmark procedure (if any):**
```bash
# List any commands run differently or configuration changes
```

---

## Checklist

- [ ] I have run `make validate-hardware` successfully
- [ ] I have provided hardware and software version information
- [ ] I have shared at least one complete power mode benchmark
- [ ] I have noted any thermal throttling or unusual behavior
- [ ] I have run validation tests and shared accuracy results
- [ ] I have described my cooling solution
- [ ] I understand this data may be used in aggregated community reports (with proper attribution if requested)

---

**Thank you for contributing to the community knowledge base!** 

Your data helps other engineers make informed decisions about power-performance trade-offs in edge AI deployments.
