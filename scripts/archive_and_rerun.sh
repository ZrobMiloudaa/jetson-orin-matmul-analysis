#!/bin/bash
# Archive current benchmark data and re-run pipeline for comparison
# Usage: ./scripts/archive_and_rerun.sh [optional-run-name]
#
# Copyright 2025 ByteStack Labs
# SPDX-License-Identifier: MIT
# Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
# Version: 1.0.0

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
ARCHIVE_DIR="${DATA_DIR}/archive"

# Generate timestamp and run identifier
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${1:-run_${TIMESTAMP}}"
ARCHIVE_PATH="${ARCHIVE_DIR}/${RUN_NAME}"

echo -e "${BLUE}=== Jetson Benchmark Data Archive & Re-run Pipeline ===${NC}"
echo -e "${YELLOW}Archive: ${ARCHIVE_PATH}${NC}"
echo ""

# Step 1: Verify current data exists
echo -e "${BLUE}[1/7] Verifying current data...${NC}"
if [[ ! -d "${DATA_DIR}/raw/power_modes" ]]; then
    echo -e "${RED}Error: No benchmark data found in ${DATA_DIR}/raw/power_modes${NC}"
    exit 1
fi

DATA_COUNT=$(find "${DATA_DIR}/raw/power_modes" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l) || true
if [[ "${DATA_COUNT}" -eq 0 ]]; then
    echo -e "${RED}Error: No JSON data files found${NC}"
    exit 1
fi
echo -e "${GREEN}[*] Found ${DATA_COUNT} benchmark data files${NC}"

# Step 2: Create archive directory structure
echo -e "\n${BLUE}[2/7] Creating archive directory...${NC}"
mkdir -p "${ARCHIVE_PATH}"/{raw,plots,reports,logs,metadata}
echo -e "${GREEN}[*] Archive directory created${NC}"

# Step 3: Copy data to archive
echo -e "\n${BLUE}[3/7] Archiving current data...${NC}"
cp -r "${DATA_DIR}/raw"/* "${ARCHIVE_PATH}/raw/" 2>/dev/null || true
cp -r "${DATA_DIR}/plots"/* "${ARCHIVE_PATH}/plots/" 2>/dev/null || true
cp -r "${DATA_DIR}/reports"/* "${ARCHIVE_PATH}/reports/" 2>/dev/null || true
cp -r "${DATA_DIR}/logs"/* "${ARCHIVE_PATH}/logs/" 2>/dev/null || true
echo -e "${GREEN}[*] Data archived${NC}"

# Step 4: Extract key metrics from current run
echo -e "\n${BLUE}[4/7] Extracting metrics from current run...${NC}"

# Find cuBLAS data file (using find instead of ls for better handling)
CUBLAS_FILE=$(find "${ARCHIVE_PATH}/raw/power_modes" -maxdepth 1 -name "cublas_*_analysis_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-) || true
NAIVE_FILE=$(find "${ARCHIVE_PATH}/raw/power_modes" -maxdepth 1 -name "naive_*_analysis_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-) || true
TENSOR_FILE=$(find "${ARCHIVE_PATH}/raw/power_modes" -maxdepth 1 -name "tensor_core_*_analysis_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-) || true

if [[ -n "${CUBLAS_FILE}" ]]; then
    CUBLAS_MAXN=$(python3 -c "import json; data=json.load(open('${CUBLAS_FILE}')); print(f\"{data['results']['MAXN_SUPER']['matrix_1024']['gflops']:.2f}\")" 2>/dev/null || echo "N/A")
    CUBLAS_25W=$(python3 -c "import json; data=json.load(open('${CUBLAS_FILE}')); print(f\"{data['results']['25W']['matrix_1024']['gflops']:.2f}\")" 2>/dev/null || echo "N/A")
fi

if [[ -n "${NAIVE_FILE}" ]]; then
    NAIVE_MAXN=$(python3 -c "import json; data=json.load(open('${NAIVE_FILE}')); print(f\"{data['results']['MAXN_SUPER']['matrix_1024']['gflops']:.2f}\")" 2>/dev/null || echo "N/A")
fi

# Step 5: Create archive metadata
echo -e "\n${BLUE}[5/7] Creating archive metadata...${NC}"

# Capture command outputs separately to avoid SC2312 warnings
TIMESTAMP_STR=$(date) || true
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "N/A") || true
GIT_BRANCH=$(git branch --show-current 2>/dev/null || echo "N/A") || true
HOSTNAME_STR=$(hostname) || true
SPEEDUP_CALC=$(python3 -c "try: print(f'{float('${CUBLAS_MAXN:-0}') / float('${NAIVE_MAXN:-1}'):.1f}×')
except: print('N/A')" 2>/dev/null) || true
ARCHIVE_FILES=$(find "${ARCHIVE_PATH}" -type f | sort) || true

cat > "${ARCHIVE_PATH}/metadata/run_info.md" << EOF
# Benchmark Run: ${RUN_NAME}

**Timestamp:** ${TIMESTAMP_STR}
**Git Commit:** ${GIT_COMMIT}
**Git Branch:** ${GIT_BRANCH}
**Hostname:** ${HOSTNAME_STR}

## Key Metrics Summary

### cuBLAS Performance
- MAXN Mode: ${CUBLAS_MAXN:-N/A} GFLOPS
- 25W Mode: ${CUBLAS_25W:-N/A} GFLOPS

### Baseline Comparison
- Naive MAXN: ${NAIVE_MAXN:-N/A} GFLOPS
- Speedup: ${SPEEDUP_CALC}

## Files Archived
\`\`\`
${ARCHIVE_FILES}
\`\`\`

## Notes
This archive was created before re-running the benchmark pipeline.
Use for comparison and regression analysis.
EOF

echo -e "${GREEN}[*] Metadata created${NC}"

# Step 6: Clean current data directory
echo -e "\n${BLUE}[6/7] Cleaning current data directory...${NC}"
echo -e "${YELLOW}The following will be deleted:${NC}"
echo "  - ${DATA_DIR}/raw/power_modes/*.json"
echo "  - ${DATA_DIR}/plots/**/*.png"
echo "  - ${DATA_DIR}/reports/*.md"
echo "  - ${DATA_DIR}/logs/*.log"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! ${REPLY} =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Aborted. Data preserved.${NC}"
    exit 0
fi

rm -f "${DATA_DIR}/raw/power_modes"/*.json
rm -f "${DATA_DIR}/plots"/**/*.png
rm -f "${DATA_DIR}/reports"/*.md
rm -f "${DATA_DIR}/logs"/*.log
echo -e "${GREEN}[*] Data directory cleaned${NC}"

# Step 7: Re-run pipeline
echo -e "\n${BLUE}[7/7] Re-running complete pipeline...${NC}"
echo -e "${YELLOW}Executing: make complete-pipeline${NC}"
echo ""

cd "${PROJECT_ROOT}"
make complete-pipeline

# Final summary
echo -e "\n${GREEN}=== Archive & Re-run Complete ===${NC}"
echo -e "${BLUE}Archive location:${NC} ${ARCHIVE_PATH}"
echo -e "${BLUE}New data location:${NC} ${DATA_DIR}/raw"
echo ""
echo -e "${YELLOW}To compare results:${NC}"
echo "  python scripts/compare_benchmark_runs.py \\"
echo "    ${ARCHIVE_PATH}/raw/power_modes \\"
echo "    ${DATA_DIR}/raw/power_modes"
echo ""
echo -e "${YELLOW}To view archive metadata:${NC}"
echo "  cat ${ARCHIVE_PATH}/metadata/run_info.md"
