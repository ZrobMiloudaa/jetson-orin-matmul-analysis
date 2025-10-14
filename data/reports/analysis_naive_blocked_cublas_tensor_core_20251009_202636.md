# Jetson Orin Nano: Multi-Implementation Comparative Analysis

**Hardware:** NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
**Generated**: 2025-10-09T20:26:36.881850
**Implementations Tested**: naive, blocked, cublas, tensor_core
**L4T Version:** R36.4.4 (JetPack 6.x)
**CUDA Version:** V12.6.68
**Power Modes Tested**: ['15W', '25W', 'MAXN_SUPER']
**Matrix Sizes**: [64, 128, 256, 512, 1024]

## Naive Implementation Performance

| Power Mode | Peak GFLOPS | Avg GFLOPS | Peak Efficiency | Avg Efficiency |
|------------|-------------|------------|-----------------|----------------|
| 15W | 97.24 | 65.16 | 7.8% | 5.2% |
| 25W | 140.58 | 88.10 | 7.5% | 4.7% |
| MAXN_SUPER | 157.00 | 96.32 | 7.5% | 4.6% |

## Blocked Implementation Performance

| Power Mode | Peak GFLOPS | Avg GFLOPS | Peak Efficiency | Avg Efficiency |
|------------|-------------|------------|-----------------|----------------|
| 15W | 109.58 | 80.64 | 8.7% | 6.4% |
| 25W | 155.43 | 110.00 | 8.3% | 5.8% |
| MAXN_SUPER | 171.62 | 122.88 | 8.2% | 5.9% |

## Cublas Implementation Performance

| Power Mode | Peak GFLOPS | Avg GFLOPS | Peak Efficiency | Avg Efficiency |
|------------|-------------|------------|-----------------|----------------|
| 15W | 792.27 | 406.30 | 63.2% | 32.4% |
| 25W | 1149.62 | 572.51 | 61.1% | 30.4% |
| MAXN_SUPER | 1282.49 | 634.40 | 61.4% | 30.4% |

## Tensor_core Implementation Performance

| Power Mode | Peak GFLOPS | Avg GFLOPS | Peak Efficiency | Avg Efficiency |
|------------|-------------|------------|-----------------|----------------|
| 15W | 580.14 | 359.44 | 46.3% | 28.7% |
| 25W | 852.28 | 522.81 | 45.3% | 27.8% |
| MAXN_SUPER | 951.60 | 572.80 | 45.6% | 27.4% |

## Implementation Comparison

Performance improvement of blocked over naive:

| Power Mode | Matrix Size | Naive GFLOPS | Blocked GFLOPS | Improvement |
|------------|-------------|--------------|----------------|-------------|
| 15W | 64×64 | 17.12 | 23.64 | +38.1% |
| 15W | 128×128 | 55.63 | 78.25 | +40.7% |
| 15W | 256×256 | 87.67 | 98.19 | +12.0% |
| 15W | 512×512 | 97.24 | 109.58 | +12.7% |
| 15W | 1024×1024 | 68.15 | 93.52 | +37.2% |
| 25W | 64×64 | 21.03 | 30.74 | +46.2% |
| 25W | 128×128 | 72.02 | 103.04 | +43.1% |
| 25W | 256×256 | 120.87 | 134.33 | +11.1% |
| 25W | 512×512 | 140.58 | 155.43 | +10.6% |
| 25W | 1024×1024 | 86.01 | 126.46 | +47.0% |
| MAXN_SUPER | 64×64 | 21.73 | 32.32 | +48.7% |
| MAXN_SUPER | 128×128 | 75.81 | 110.98 | +46.4% |
| MAXN_SUPER | 256×256 | 131.85 | 149.14 | +13.1% |
| MAXN_SUPER | 512×512 | 157.00 | 171.62 | +9.3% |
| MAXN_SUPER | 1024×1024 | 95.23 | 150.34 | +57.9% |

