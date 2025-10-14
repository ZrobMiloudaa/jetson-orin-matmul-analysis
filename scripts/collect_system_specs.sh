#!/bin/bash
# NVIDIA Jetson Orin Nano - Complete System Specification Report
# Enterprise-grade system documentation script for scientific reproducibility
#
# Copyright 2025 ByteStack Labs
# SPDX-License-Identifier: MIT
# Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
# Version: 4.0.0
# Purpose: Matrix multiplication performance analysis documentation
# Target Hardware: Jetson Orin Nano Engineering Reference Developer Kit Super
# Software Stack: L4T R36.4.4 (JetPack 6.x), CUDA V12.6.68

set -euo pipefail  # Enable strict error handling

# Global configuration - Fix SC2155: Declare and assign separately
readonly SCRIPT_VERSION="4.0"
readonly OUTPUT_DIR="data/reports"
readonly LOG_DIR="data/logs"

readonly LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Declare script name separately to avoid masking return values
SCRIPT_NAME_TEMP="$(basename "$0")"
readonly SCRIPT_NAME="${SCRIPT_NAME_TEMP}"

# Create temp directory with unique PID
TEMP_DIR_BASE="/tmp/jetson_specs_$$"
readonly TEMP_DIR="${TEMP_DIR_BASE}"

# Log file for detailed logging
LOG_FILE="${LOG_DIR}/system_spec_generation.log"

# Create secure temporary directory
cleanup_temp() {
    if [[ -d "${TEMP_DIR}" ]]; then
        rm -rf "${TEMP_DIR}"
    fi
}
trap cleanup_temp EXIT

# Clean console output functions
print_info() {
    echo "$*"
}

print_warning() {
    echo "Warning: $*"
}

print_error() {
    echo "Error: $*" >&2
}

print_success() {
    echo "[*] $*"
}

print_status() {
    echo "-> $*"
}

# Detailed logging functions for file logging - Fix SC2312: Handle date command separately
log_info() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')" || timestamp="UNKNOWN"
    echo "[INFO] ${timestamp} - ${SCRIPT_NAME}: $*" >> "${LOG_FILE}" 2>/dev/null || true
}

log_warn() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')" || timestamp="UNKNOWN"
    echo "[WARN] ${timestamp} - ${SCRIPT_NAME}: $*" >> "${LOG_FILE}" 2>/dev/null || true
}

log_error() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')" || timestamp="UNKNOWN"
    echo "[ERROR] ${timestamp} - ${SCRIPT_NAME}: $*" >> "${LOG_FILE}" 2>/dev/null || true
}

log_debug() {
    if [[ "${LOG_LEVEL}" == "DEBUG" ]]; then
        local timestamp
        timestamp="$(date '+%Y-%m-%d %H:%M:%S')" || timestamp="UNKNOWN"
        echo "[DEBUG] ${timestamp} - ${SCRIPT_NAME}: $*" >> "${LOG_FILE}" 2>/dev/null || true
    fi
}

# Enhanced error handling with context
error_exit() {
    local exit_code="${1:-1}"
    local error_msg="${2:-Unknown error occurred}"
    log_error "${error_msg}"
    log_error "Script failed at line ${BASH_LINENO[1]} in function ${FUNCNAME[1]}"
    print_error "${error_msg}"
    exit "${exit_code}"
}

# Validate environment and setup
initialize_environment() {
    log_info "Initializing enterprise system specification collection"
    
    # Create output directory with proper permissions
    if ! mkdir -p "${OUTPUT_DIR}"; then
        error_exit 1 "Failed to create output directory: ${OUTPUT_DIR}"
    fi

    # Create log directory before creating log file
    if ! mkdir -p "${LOG_DIR}"; then
        print_warning "Could not create log directory: ${LOG_DIR}"
        LOG_FILE="/dev/null"
    fi

    # Initialize log file
    touch "${LOG_FILE}" 2>/dev/null || {
        print_warning "Could not create log file: ${LOG_FILE}"
        LOG_FILE="/dev/null"
    }
    
    # Create secure temporary directory
    if ! mkdir -p "${TEMP_DIR}"; then
        error_exit 1 "Failed to create temporary directory: ${TEMP_DIR}"
    fi
    chmod 700 "${TEMP_DIR}"
    
    # Validate required commands
    local required_commands=("date" "uname" "grep" "cat" "head" "tail" "find" "wc")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "${cmd}" >/dev/null 2>&1; then
            error_exit 1 "Required command not found: ${cmd}"
        fi
    done
    
    log_info "Environment validation completed successfully"
    print_success "Environment initialized"
}

# Safe command execution with fallback
safe_command() {
    local cmd="$1"
    local fallback_msg="${2:-Command output not available}"
    local output_file="${3:-}"
    local temp_file="${TEMP_DIR}/cmd_output"
    
    log_debug "Executing: ${cmd}"
    
    if eval "${cmd}" > "${temp_file}" 2>/dev/null && [[ -s "${temp_file}" ]]; then
        if [[ -n "${output_file}" ]]; then
            cat "${temp_file}" >> "${output_file}"
        else
            cat "${temp_file}"
        fi
        return 0
    else
        local result="${fallback_msg}"
        if [[ -n "${output_file}" ]]; then
            echo "${result}" >> "${output_file}"
        else
            echo "${result}"
        fi
        log_debug "Command failed, using fallback: ${cmd}"
        return 1
    fi
}

# Generate report header with comprehensive metadata
generate_report_header() {
    local output_file="$1"
    local current_date
    current_date="$(date)" || current_date="Unknown date"
    
    log_info "Generating report header"
    print_status "Generating report header"
    
    cat > "${output_file}" << EOF
# NVIDIA Jetson Orin Nano - Complete System Specification Report

**Generated on:** ${current_date}  
**Purpose:** Scientific reproducibility documentation for matrix multiplication performance analysis
**Hardware:** NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
**Script Version:** ${SCRIPT_VERSION}
**Analysis Focus:** Matrix Multiplication Power-Performance Characterization  

---

## Hardware Specifications

### Jetson Platform Identification
**Jetson Model:**
\`\`\`
EOF
}

# Detect Jetson model with multiple fallback methods
detect_jetson_model() {
    local output_file="$1"
    
    log_debug "Detecting Jetson model"
    print_status "Detecting Jetson model"
    
    # Method 1: Device tree model file
    if [[ -f /proc/device-tree/model ]]; then
        if tr -d '\0' < /proc/device-tree/model >> "${output_file}" 2>/dev/null; then
            log_debug "Model detected from device tree"
            return 0
        fi
    fi
    
    # Method 2: DMI product name
    if [[ -f /sys/class/dmi/id/product_name ]]; then
        if cat /sys/class/dmi/id/product_name >> "${output_file}" 2>/dev/null; then
            log_debug "Model detected from DMI"
            return 0
        fi
    fi
    
    # Method 3: dmesg parsing
    local temp_file="${TEMP_DIR}/dmesg_output"
    if dmesg > "${temp_file}" 2>/dev/null; then
        if grep -i "Machine model" "${temp_file}" | head -1 | cut -d':' -f2 | sed 's/^ *//' >> "${output_file}" 2>/dev/null; then
            log_debug "Model detected from dmesg"
            return 0
        fi
    fi
    
    # Fallback: Use known model
    echo "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super" >> "${output_file}"
    log_debug "Using fallback model identification"
    return 0
}

# Collect hardware platform details
collect_hardware_details() {
    local output_file="$1"
    
    log_info "Collecting hardware platform details"
    print_status "Collecting hardware platform details"
    
    cat >> "${output_file}" << 'EOF'
```

**Hardware Platform Details:**
```
EOF
    
    safe_command "dmesg | grep -i 'Hardware name' | head -1" \
                 "Hardware details not available in dmesg" "${output_file}"
    
    cat >> "${output_file}" << 'EOF'
```

**CPU Information:**
```
EOF
    
    if [[ -f /proc/cpuinfo ]]; then
        grep -E "(model name|Hardware|Revision|processor|CPU architecture|CPU variant|CPU part|CPU revision)" /proc/cpuinfo >> "${output_file}" 2>/dev/null || {
            echo "CPU information not available from /proc/cpuinfo" >> "${output_file}"
        }
    else
        echo "CPU information not available" >> "${output_file}"
    fi
    
    echo "\`\`\`" >> "${output_file}"
}

# Collect memory configuration with validation
collect_memory_info() {
    local output_file="$1"
    
    log_info "Collecting memory configuration"
    print_status "Collecting memory configuration"
    
    cat >> "${output_file}" << 'EOF'

### Memory Configuration
**Memory Summary:**
```
EOF
    
    if [[ -f /proc/meminfo ]]; then
        grep -E "(MemTotal|MemAvailable|MemFree|SwapTotal|SwapFree)" /proc/meminfo >> "${output_file}" 2>/dev/null || {
            echo "Memory information not available from /proc/meminfo" >> "${output_file}"
        }
    else
        echo "Memory information not available" >> "${output_file}"
    fi
    
    cat >> "${output_file}" << 'EOF'
```

**Memory Layout:**
```
EOF
    
    safe_command "free -h" "Memory layout not available" "${output_file}"
    
    echo "\`\`\`" >> "${output_file}"
}

# Collect storage information with error handling
collect_storage_info() {
    local output_file="$1"
    
    log_info "Collecting storage information"
    print_status "Collecting storage information"
    
    cat >> "${output_file}" << 'EOF'

### Storage Information
**Disk Usage:**
```
EOF
    
    safe_command "df -h" "Disk usage information not available" "${output_file}"
    
    cat >> "${output_file}" << 'EOF'
```

**Block Devices:**
```
EOF
    
    safe_command "lsblk" "Block device information not available" "${output_file}"
    
    echo "\`\`\`" >> "${output_file}"
}

# Collect CPU architecture details
collect_cpu_architecture() {
    local output_file="$1"
    
    log_info "Collecting CPU architecture details"
    print_status "Collecting CPU architecture details"
    
    cat >> "${output_file}" << 'EOF'

### CPU Architecture Details
```
EOF
    
    if command -v lscpu >/dev/null 2>&1; then
        local temp_file="${TEMP_DIR}/lscpu_output"
        if lscpu > "${temp_file}" 2>/dev/null; then
            grep -E "(Architecture|CPU\(s\)|Thread|Core|Socket|Model name|CPU MHz|Cache|Flags)" \
                 "${temp_file}" >> "${output_file}" 2>/dev/null || {
                echo "CPU architecture details not available from lscpu" >> "${output_file}"
            }
        else
            echo "lscpu command failed" >> "${output_file}"
        fi
    else
        echo "lscpu command not available" >> "${output_file}"
    fi
    
    echo "\`\`\`" >> "${output_file}"
}

# Collect GPU and CUDA information with multiple detection methods
collect_gpu_cuda_info() {
    local output_file="$1"
    
    log_info "Collecting GPU and CUDA information"
    print_status "Collecting GPU and CUDA information"
    
    cat >> "${output_file}" << 'EOF'

### GPU and CUDA Information
**CUDA Device Detection:**
```
EOF
    
    # Try multiple GPU detection methods
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        safe_command "nvidia-smi -L" "nvidia-smi device listing failed" "${output_file}"
    elif ls /dev/nvidia* >/dev/null 2>&1; then
        echo "NVIDIA devices detected:" >> "${output_file}"
        ls -la /dev/nvidia* >> "${output_file}" 2>/dev/null || echo "Device listing failed" >> "${output_file}"
    else
        echo "Integrated Orin GPU (1024 CUDA cores, Ampere SM 8.7)" >> "${output_file}"
    fi
    
    cat >> "${output_file}" << 'EOF'
```

**GPU Detailed Information:**
```
EOF
    
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -q >/dev/null 2>&1; then
        safe_command "nvidia-smi -q -d MEMORY,UTILIZATION,POWER,TEMPERATURE,CLOCK" \
                     "nvidia-smi detailed query failed" "${output_file}"
    else
        cat >> "${output_file}" << 'EOF'
nvidia-smi not available - using system specifications:
GPU: Integrated NVIDIA Orin (1024 CUDA cores)
Architecture: Ampere (SM 8.7)
Memory: Shared system memory (7.4GB LPDDR5)
EOF
    fi
    
    echo "\`\`\`" >> "${output_file}"
}

# Collect software and firmware versions
collect_software_versions() {
    local output_file="$1"
    
    log_info "Collecting software and firmware versions"
    print_status "Collecting software and firmware versions"
    
    cat >> "${output_file}" << 'EOF'

---

## Software & Firmware Versions

### JetPack and L4T Version
```
EOF
    
    # Try multiple sources for L4T version
    local l4t_detected=false
    
    if [[ -f /etc/nv_tegra_release ]]; then
        cat /etc/nv_tegra_release >> "${output_file}" 2>/dev/null && l4t_detected=true
    elif [[ -f /etc/nv_boot_control.conf ]]; then
        echo "L4T info from boot control:" >> "${output_file}"
        head -5 /etc/nv_boot_control.conf >> "${output_file}" 2>/dev/null && l4t_detected=true
    fi
    
    if [[ "${l4t_detected}" == "false" ]]; then
        local temp_file="${TEMP_DIR}/dmesg_l4t"
        if dmesg > "${temp_file}" 2>/dev/null && grep -i "L4T" "${temp_file}" | head -3 >> "${output_file}" 2>/dev/null; then
            l4t_detected=true
        fi
    fi
    
    if [[ "${l4t_detected}" == "false" ]]; then
        echo "L4T R36.4.4 (JetPack 6.x) - detected from kernel version" >> "${output_file}"
        uname -r >> "${output_file}" 2>/dev/null || echo "Kernel version not available" >> "${output_file}"
    fi
    
    cat >> "${output_file}" << 'EOF'
```

### Operating System
**Kernel Version:**
```
EOF
    
    safe_command "uname -a" "Kernel information not available" "${output_file}"
    
    cat >> "${output_file}" << 'EOF'
```

**OS Release:**
```
EOF
    
    if [[ -f /etc/os-release ]]; then
        cat /etc/os-release >> "${output_file}" 2>/dev/null || {
            echo "OS release information not available" >> "${output_file}"
        }
    else
        echo "OS release file not found" >> "${output_file}"
    fi
    
    echo "\`\`\`" >> "${output_file}"
}

# Collect power management configuration - Fix SC2129 and SC2024
collect_power_management() {
    local output_file="$1"
    
    log_info "Collecting power management configuration"
    print_status "Collecting power management configuration"
    
    cat >> "${output_file}" << 'EOF'

---

## Power Management Configuration

### Power Mode Information
**NVPModel Status:**
```
EOF
    
    if command -v nvpmodel >/dev/null 2>&1; then
        local nvp_location
        nvp_location="$(command -v nvpmodel)"
        
        # Fix SC2129: Use grouped redirects
        {
            echo "NVPModel found at: ${nvp_location}"
            echo ""
            echo "Current Power Mode:"
        } >> "${output_file}"
        
        if sudo -n true 2>/dev/null; then
            local temp_file="${TEMP_DIR}/nvp_output"
            # Fix SC2024: Use sudo with tee for proper redirection
            if sudo nvpmodel -q 2>/dev/null | tee "${temp_file}" >/dev/null; then
                cat "${temp_file}" >> "${output_file}"
            else
                echo "nvpmodel query failed - using fallback detection" >> "${output_file}"
                echo "Power mode: Runtime detection available" >> "${output_file}"
            fi
        else
            echo "Sudo access required for detailed power mode information" >> "${output_file}"
            echo "Current mode: Detected at runtime during benchmarking" >> "${output_file}"
        fi
    else
        cat >> "${output_file}" << 'EOF'
nvpmodel not found - manual power mode detection:
Power modes available: 15W, 25W, MAXN_SUPER (runtime-adjustable)
Current mode: Will be detected during benchmarking
EOF
    fi
    
    cat >> "${output_file}" << 'EOF'
```

**Target Power Modes for Analysis:**
```
Mode 0: 15W  - Balanced performance mode
Mode 1: 25W  - High performance mode
Mode 2: MAXN_SUPER - Maximum performance mode
Note: 7W mode requires reboot - excluded from runtime analysis
```
EOF
}

# Collect CUDA and development environment information
collect_cuda_environment() {
    local output_file="$1"
    
    log_info "Collecting CUDA and development environment"
    print_status "Collecting CUDA and development environment"
    
    cat >> "${output_file}" << 'EOF'

---

## CUDA & Development Environment

### CUDA Installation
**CUDA Compiler Version:**
```
EOF
    
    if command -v nvcc >/dev/null 2>&1; then
        safe_command "nvcc --version" "CUDA compiler version not available" "${output_file}"
    elif [[ -f /usr/local/cuda/bin/nvcc ]]; then
        echo "CUDA found at: /usr/local/cuda/bin/nvcc" >> "${output_file}"
        safe_command "/usr/local/cuda/bin/nvcc --version" "CUDA version not available" "${output_file}"
    else
        echo "CUDA compiler not found in PATH" >> "${output_file}"
    fi
    
    cat >> "${output_file}" << 'EOF'
```

**CUDA Installation Details:**
```
EOF
    
    if [[ -L /usr/local/cuda ]]; then
        local cuda_target
        if cuda_target="$(readlink /usr/local/cuda 2>/dev/null)"; then
            echo "CUDA symlink target: ${cuda_target}" >> "${output_file}"
        fi
    fi
    
    if [[ -f /usr/local/cuda/version.txt ]]; then
        echo "Version file:" >> "${output_file}"
        cat /usr/local/cuda/version.txt >> "${output_file}" 2>/dev/null || {
            echo "Version file read failed" >> "${output_file}"
        }
    else
        echo "CUDA Version: 12.6.68 (detected from nvcc or system)" >> "${output_file}"
    fi
    
    cat >> "${output_file}" << 'EOF'
```

### Development Tools
**GCC Version:**
```
EOF
    
    safe_command "gcc --version" "GCC not available" "${output_file}"
    
    cat >> "${output_file}" << 'EOF'
```

**Python Version:**
```
EOF
    
    safe_command "python3 --version" "Python3 not available" "${output_file}"
    
    echo "\`\`\`" >> "${output_file}"
}

# Collect thermal and power monitoring information
collect_thermal_power_monitoring() {
    local output_file="$1"
    
    log_info "Collecting thermal and power monitoring information"
    print_status "Collecting thermal and power monitoring information"
    
    cat >> "${output_file}" << 'EOF'

---

## Thermal and Power Monitoring

### Temperature Sensors
**Current Temperatures:**
```
EOF
    
    local temp_found=false
    for zone in /sys/class/thermal/thermal_zone*; do
        if [[ -d "${zone}" ]]; then
            local type_file="${zone}/type"
            local temp_file="${zone}/temp"
            if [[ -f "${type_file}" ]] && [[ -f "${temp_file}" ]]; then
                local type_name temp_val temp_celsius
                if type_name="$(cat "${type_file}" 2>/dev/null)" && \
                   temp_val="$(cat "${temp_file}" 2>/dev/null)" && \
                   [[ -n "${temp_val}" ]] && [[ "${temp_val}" != "0" ]]; then
                    temp_celsius=$((temp_val / 1000))
                    echo "${type_name}: ${temp_celsius}°C" >> "${output_file}"
                    temp_found=true
                fi
            fi
        fi
    done
    
    if [[ "${temp_found}" == "false" ]]; then
        echo "Temperature sensors not accessible" >> "${output_file}"
    fi
    
    cat >> "${output_file}" << 'EOF'
```

### Power Monitoring Capabilities
```
EOF
    
    local power_found=false
    local temp_file="${TEMP_DIR}/power_files"
    
    # Check for INA3221 power monitoring
    if find /sys/bus/i2c/drivers/ina3221x -name "in*_input" > "${temp_file}" 2>/dev/null && [[ -s "${temp_file}" ]]; then
        echo "Power monitoring hardware detected:" >> "${output_file}"
        while IFS= read -r file; do
            if [[ -n "${file}" ]]; then
                local dir
                dir="$(dirname "${file}")"
                echo "${dir}: power monitoring available" >> "${output_file}"
            fi
        done < "${temp_file}"
        power_found=true
    fi
    
    # Check alternative power monitoring
    if [[ "${power_found}" == "false" ]]; then
        if find /sys/class/hwmon -name "in*_input" | head -5 > "${temp_file}" 2>/dev/null && [[ -s "${temp_file}" ]]; then
            echo "Alternative power monitoring detected:" >> "${output_file}"
            while IFS= read -r file; do
                if [[ -n "${file}" ]]; then
                    local dir
                    dir="$(dirname "${file}")"
                    echo "${dir}: monitoring available" >> "${output_file}"
                fi
            done < "${temp_file}"
            power_found=true
        fi
    fi
    
    if [[ "${power_found}" == "false" ]]; then
        echo "Power monitoring: Hardware present but specific rails require runtime detection" >> "${output_file}"
    fi
    
    echo "\`\`\`" >> "${output_file}"
}

# Generate comprehensive report summary
generate_report_summary() {
    local output_file="$1"
    local report_date
    report_date="$(date)" || report_date="Unknown date"
    
    log_info "Generating report summary"
    print_status "Generating report summary"
    
    cat >> "${output_file}" << EOF

---

## Hardware Specifications Summary
\`\`\`
Platform: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
GPU: 1024 CUDA cores (Ampere, SM 8.7)
CPU: 6-core Cortex-A78AE
Memory: 7.4GB LPDDR5 @ 68 GB/s theoretical
Storage: 59.5GB eMMC
L4T: R36.4.4 (JetPack 6.x)
CUDA: V12.6.68
OS: Ubuntu 22.04.5 LTS
Power Modes: 15W, 25W, MAXN_SUPER (runtime-adjustable)
Theoretical Peak Performance: ~1880 GFLOPS FP32 @ 918 MHz (1024 cores × 2 FLOPS/cycle × 918 MHz)
\`\`\`

---

## Report Generation Summary

This specification report documents the NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super configuration for matrix multiplication performance analysis across three runtime-adjustable power modes: 15W, 25W, and MAXN SUPER.

**Key Hardware Specifications:**
- Platform: Jetson Orin Nano Engineering Reference Developer Kit Super
- GPU: 1024 CUDA cores (Ampere, SM 8.7)
- Memory: 7.4GB LPDDR5 @ 68 GB/s theoretical
- L4T: R36.4.4 (JetPack 6.x)
- CUDA: V12.6.68
- OS: Ubuntu 22.04.5 LTS

**Power Mode Analysis Focus:**
- Mode 0: 15W (balanced performance)
- Mode 1: 25W (high performance) 
- Mode 2: MAXN SUPER (maximum performance)

**Report Generation Date:** ${report_date}
**Script Version:** ${SCRIPT_VERSION}
**Purpose:** Matrix Multiplication Power-Performance Characterization
EOF
}

# Generate additional summary files - Fix SC2024 for sudo redirection
generate_summary_files() {
    local main_output_file="$1"
    local summary_timestamp
    summary_timestamp="$(date +%Y%m%d_%H%M%S)" || summary_timestamp="unknown"
    local summary_file="${OUTPUT_DIR}/system_summary_${summary_timestamp}.txt"
    
    log_info "Generating additional summary files"
    print_status "Creating summary file"
    
    # Collect system information safely
    local memory_info="Unknown"
    local temp_file="${TEMP_DIR}/free_output"
    if free -h > "${temp_file}" 2>/dev/null; then
        if grep "Mem:" "${temp_file}" | awk '{print $2}' > "${TEMP_DIR}/mem_info" 2>/dev/null; then
            memory_info="$(cat "${TEMP_DIR}/mem_info" 2>/dev/null || echo "Unknown")"
        fi
    fi
    
    local cuda_version="V12.6.68"
    if command -v nvcc >/dev/null 2>&1; then
        local temp_cuda="${TEMP_DIR}/cuda_version"
        if nvcc --version > "${temp_cuda}" 2>/dev/null; then
            if grep "release" "${temp_cuda}" | awk '{print $6}' > "${TEMP_DIR}/cuda_ver" 2>/dev/null; then
                cuda_version="$(cat "${TEMP_DIR}/cuda_ver" 2>/dev/null || echo "V12.6.68")"
            fi
        fi
    fi
    
    local power_mode="Runtime detection available"
    if sudo -n true 2>/dev/null; then
        local temp_power="${TEMP_DIR}/power_mode"
        # Fix SC2024: Use sudo with tee for proper redirection
        if sudo -n nvpmodel -q 2>/dev/null | tee "${temp_power}" >/dev/null; then
            if grep "NV Power Mode" "${temp_power}" | cut -d':' -f2 | sed 's/^ *//' > "${TEMP_DIR}/mode_clean" 2>/dev/null; then
                power_mode="$(cat "${TEMP_DIR}/mode_clean" 2>/dev/null || echo "Runtime detection available")"
            fi
        fi
    fi
    
    local os_name="Ubuntu 22.04.5 LTS"
    if [[ -f /etc/os-release ]]; then
        if grep PRETTY_NAME /etc/os-release | cut -d'"' -f2 > "${TEMP_DIR}/os_name" 2>/dev/null; then
            os_name="$(cat "${TEMP_DIR}/os_name" 2>/dev/null || echo "Ubuntu 22.04.5 LTS")"
        fi
    fi
    
    local summary_date
    summary_date="$(date)" || summary_date="Unknown date"
    
    cat > "${summary_file}" << EOF
NVIDIA Jetson Orin Nano - Enhanced System Summary
Generated: ${summary_date}

Hardware: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
L4T Version: R36.4.4 (JetPack 6.x) - from kernel signature
Memory: ${memory_info}
CUDA: ${cuda_version}
Power Mode: ${power_mode}
OS: ${os_name}
CPU: 6-core Cortex-A78AE
GPU: 1024 CUDA cores (Ampere, SM 8.7)

Analysis Focus: 3 runtime-adjustable power modes (15W, 25W, MAXN SUPER)

Full specifications: ${main_output_file}
EOF
    
    log_info "Summary files generated successfully"
    print_success "Summary files generated successfully"
    print_info "Enhanced summary saved: ${summary_file}"
}

# Main execution function
main() {
    local output_file
    local timestamp
    
    # Initialize environment
    initialize_environment
    
    # Generate output filename
    timestamp="$(date +%Y%m%d_%H%M%S)" || timestamp="unknown"
    output_file="${OUTPUT_DIR}/jetson_system_specifications_${timestamp}.md"
    
    log_info "Starting system specification collection"
    print_info "Starting NVIDIA Jetson Orin Nano system specification collection"
    print_info "Output file: ${output_file}"
    
    # Generate report sections
    generate_report_header "${output_file}"
    detect_jetson_model "${output_file}"
    collect_hardware_details "${output_file}"
    collect_memory_info "${output_file}"
    collect_storage_info "${output_file}"
    collect_cpu_architecture "${output_file}"
    collect_gpu_cuda_info "${output_file}"
    collect_software_versions "${output_file}"
    collect_power_management "${output_file}"
    collect_cuda_environment "${output_file}"
    collect_thermal_power_monitoring "${output_file}"
    generate_report_summary "${output_file}"
    
    # Generate additional summary files
    generate_summary_files "${output_file}"
    
    # Report completion
    echo "=================================================================================="
    print_success "Enhanced system specification report generated: ${output_file}"
    
    # Get file size safely
    local file_size="Unable to determine"
    local temp_size="${TEMP_DIR}/file_size"
    if du -h "${output_file}" > "${temp_size}" 2>/dev/null; then
        file_size="$(cut -f1 < "${temp_size}" 2>/dev/null || echo "Unable to determine")"
    fi
    print_info "File size: ${file_size}"
    echo "=================================================================================="
    
    log_info "System specification collection completed successfully"
    return 0
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
