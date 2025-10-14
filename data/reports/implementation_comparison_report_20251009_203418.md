# Jetson Orin Nano: Naive vs Blocked Implementation Analysis

**Hardware:** NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
**Report Date:** 2025-10-09 20:34:18
**Implementations:** Naive (O(n³) baseline) vs Blocked (32×32 tiled)

## Executive Summary

**Naive Peak Performance:** 157.00 GFLOPS
**Blocked Peak Performance:** 171.62 GFLOPS
**Overall Improvement:** +9.3%

## Performance by Matrix Size

| Size | Naive (GFLOPS) | Blocked (GFLOPS) | Improvement |
|------|----------------|------------------|-------------|
| 64×64 | 19.96 | 28.90 | +44.8% |
| 128×128 | 67.82 | 97.42 | +43.6% |
| 256×256 | 113.46 | 127.22 | +12.1% |
| 512×512 | 131.61 | 145.54 | +10.6% |
| 1024×1024 | 83.13 | 123.44 | +48.5% |

## Performance by Power Mode (512×512)

| Mode | Naive (GFLOPS) | Blocked (GFLOPS) | Speedup |
|------|----------------|------------------|----------|
| 15W | 97.24 | 109.58 | 1.13x |
| 25W | 140.58 | 155.43 | 1.11x |
| MAXN_SUPER | 157.00 | 171.62 | 1.09x |

## Key Findings

1. **Consistent Improvement:** Blocked implementation shows measurable gains across all configurations
2. **Cache Optimization:** 32×32 tiling effectively utilizes on-chip memory hierarchy
3. **Power Mode Scaling:** Optimization benefits preserved across all power modes
4. **Practical Impact:** ~10% average improvement translates to meaningful throughput gains

## Technical Details

**Naive Implementation:**
- Direct O(n³) algorithm
- Global memory access pattern
- Minimal memory reuse

**Blocked Implementation:**
- 32×32 tile size for optimal cache fit
- Shared memory utilization (2KB per block)
- Transposed B matrix storage for coalesced access
- Improved data locality and reuse

