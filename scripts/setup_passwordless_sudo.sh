#!/bin/bash
# setup_passwordless_sudo.sh
# Automated setup script for passwordless sudo configuration
# Jetson Orin Nano Matrix Multiplication Benchmarking Project
#
# Copyright 2025 ByteStack Labs
# SPDX-License-Identifier: MIT
# Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
# Version: 1.0.0

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
CURRENT_USER="${SUDO_USER:-${USER}}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUDOERS_FILE="/etc/sudoers.d/jetson-benchmarking"
BACKUP_DIR="${HOME}/.config/jetson-benchmarking/backups"

# Commands that require passwordless sudo
NVPMODEL_PATH="/usr/sbin/nvpmodel"
TEGRASTATS_PATH="/usr/bin/tegrastats"
COLLECT_SPECS_SCRIPT="${PROJECT_ROOT}/scripts/collect_system_specs.sh"

echo -e "${MAGENTA}=================================================================${NC}"
echo -e "${MAGENTA}Jetson Benchmarking: Passwordless Sudo Setup${NC}"
echo -e "${MAGENTA}=================================================================${NC}"
echo ""

# Check if running as root or with sudo
if [[ ${EUID} -ne 0 ]]; then
    echo -e "${RED}ERROR: This script must be run with sudo${NC}"
    echo -e "${YELLOW}Usage: sudo ./scripts/setup_passwordless_sudo.sh${NC}"
    exit 1
fi

# Validate we're on a Jetson platform
echo -e "${CYAN}-> Validating Jetson platform...${NC}"
if [[ ! -f /etc/nv_tegra_release ]]; then
    echo -e "${RED}ERROR: Not running on NVIDIA Jetson platform${NC}"
    echo -e "${YELLOW}This script is designed for Jetson devices only${NC}"
    exit 1
fi
echo -e "${GREEN}[*] Jetson platform detected${NC}"

# Validate required binaries exist
echo -e "${CYAN}-> Validating required binaries...${NC}"

if [[ ! -f "${NVPMODEL_PATH}" ]]; then
    echo -e "${RED}ERROR: nvpmodel not found at ${NVPMODEL_PATH}${NC}"
    exit 1
fi
echo -e "${GREEN}[*] Found nvpmodel: ${NVPMODEL_PATH}${NC}"

if [[ ! -f "${TEGRASTATS_PATH}" ]]; then
    echo -e "${RED}ERROR: tegrastats not found at ${TEGRASTATS_PATH}${NC}"
    exit 1
fi
echo -e "${GREEN}[*] Found tegrastats: ${TEGRASTATS_PATH}${NC}"

if [[ ! -f "${COLLECT_SPECS_SCRIPT}" ]]; then
    echo -e "${RED}ERROR: collect_system_specs.sh not found at ${COLLECT_SPECS_SCRIPT}${NC}"
    exit 1
fi
echo -e "${GREEN}[*] Found collect_system_specs.sh: ${COLLECT_SPECS_SCRIPT}${NC}"

# Display configuration summary
echo ""
echo -e "${BLUE}Configuration Summary:${NC}"
echo -e "  User: ${GREEN}${CURRENT_USER}${NC}"
echo -e "  Project Root: ${GREEN}${PROJECT_ROOT}${NC}"
echo -e "  Sudoers File: ${GREEN}${SUDOERS_FILE}${NC}"
echo ""
echo -e "${YELLOW}The following commands will be granted passwordless sudo:${NC}"
echo -e "  1. ${CYAN}${NVPMODEL_PATH}${NC} (power mode switching)"
echo -e "  2. ${CYAN}${TEGRASTATS_PATH}${NC} (system telemetry)"
echo -e "  3. ${CYAN}${COLLECT_SPECS_SCRIPT}${NC} (hardware spec collection)"
echo ""

# Prompt for confirmation
read -rp "$(echo -e "${YELLOW}Do you want to proceed? [y/N]: ${NC}")" confirm
if [[ ! "${confirm}" =~ ^[Yy]$ ]]; then
    echo -e "${RED}Setup cancelled by user${NC}"
    exit 0
fi

# Create backup directory
echo ""
echo -e "${CYAN}-> Creating backup directory...${NC}"
mkdir -p "${BACKUP_DIR}"
chown "${CURRENT_USER}:${CURRENT_USER}" "${BACKUP_DIR}"
echo -e "${GREEN}[*] Backup directory: ${BACKUP_DIR}${NC}"

# Backup existing sudoers file if present
if [[ -f "${SUDOERS_FILE}" ]]; then
    BACKUP_FILE="${BACKUP_DIR}/jetson-benchmarking.$(date +%Y%m%d_%H%M%S).bak"
    echo -e "${CYAN}-> Backing up existing sudoers file...${NC}"
    cp "${SUDOERS_FILE}" "${BACKUP_FILE}"
    chown "${CURRENT_USER}:${CURRENT_USER}" "${BACKUP_FILE}"
    echo -e "${GREEN}[*] Backup saved: ${BACKUP_FILE}${NC}"
fi

# Generate sudoers content
echo -e "${CYAN}-> Generating sudoers configuration...${NC}"
SUDOERS_CONTENT="# Jetson Orin Nano Matrix Multiplication Benchmarking
# Passwordless sudo configuration for benchmark automation
# Generated: $(date '+%Y-%m-%d %H:%M:%S')
# User: ${CURRENT_USER}
# Project: ${PROJECT_ROOT}
#
# Security Note: This configuration grants passwordless sudo ONLY for
# specific power management and telemetry commands required for benchmarking.
# These commands cannot modify system files or access sensitive data.
#
# Commands whitelisted:
# - nvpmodel: Power mode switching (15W/25W/MAXN)
# - tegrastats: Hardware telemetry monitoring
# - collect_system_specs.sh: System specification collection
#
# To remove: sudo rm ${SUDOERS_FILE}

# Power mode management
${CURRENT_USER} ALL=(ALL) NOPASSWD: ${NVPMODEL_PATH}

# System telemetry
${CURRENT_USER} ALL=(ALL) NOPASSWD: ${TEGRASTATS_PATH}

# Hardware specification collection
${CURRENT_USER} ALL=(ALL) NOPASSWD: ${COLLECT_SPECS_SCRIPT}
"

# Write sudoers file
echo -e "${CYAN}-> Writing sudoers configuration...${NC}"
echo "${SUDOERS_CONTENT}" > "${SUDOERS_FILE}"

# Set correct permissions (sudoers files must be 0440)
chmod 0440 "${SUDOERS_FILE}"

# Validate sudoers syntax
echo -e "${CYAN}-> Validating sudoers syntax...${NC}"
if visudo -c -f "${SUDOERS_FILE}" >/dev/null 2>&1; then
    echo -e "${GREEN}[*] Sudoers syntax valid${NC}"
else
    echo -e "${RED}ERROR: Invalid sudoers syntax${NC}"
    echo -e "${YELLOW}Rolling back changes...${NC}"
    rm -f "${SUDOERS_FILE}"
    if [[ -f "${BACKUP_FILE}" ]]; then
        cp "${BACKUP_FILE}" "${SUDOERS_FILE}"
        chmod 0440 "${SUDOERS_FILE}"
        echo -e "${YELLOW}Restored backup${NC}"
    fi
    exit 1
fi

# Test the configuration
echo ""
echo -e "${CYAN}-> Testing passwordless sudo configuration...${NC}"

# Test nvpmodel
if sudo -n -u "${CURRENT_USER}" sudo -n "${NVPMODEL_PATH}" -q >/dev/null 2>&1; then
    echo -e "${GREEN}[*] nvpmodel: passwordless sudo working${NC}"
else
    echo -e "${YELLOW}[!] nvpmodel: Could not verify (may require relogin)${NC}"
fi

# Test tegrastats
if sudo -n -u "${CURRENT_USER}" sudo -n "${TEGRASTATS_PATH}" --help >/dev/null 2>&1; then
    echo -e "${GREEN}[*] tegrastats: passwordless sudo working${NC}"
else
    echo -e "${YELLOW}[!] tegrastats: Could not verify (may require relogin)${NC}"
fi

# Test collect_system_specs.sh
if sudo -n -u "${CURRENT_USER}" sudo -n test -x "${COLLECT_SPECS_SCRIPT}"; then
    echo -e "${GREEN}[*] collect_system_specs.sh: passwordless sudo working${NC}"
else
    echo -e "${YELLOW}[!] collect_system_specs.sh: Could not verify (may require relogin)${NC}"
fi

# Create uninstall script
UNINSTALL_SCRIPT="${PROJECT_ROOT}/scripts/remove_passwordless_sudo.sh"
echo -e "${CYAN}-> Creating uninstall script...${NC}"

cat > "${UNINSTALL_SCRIPT}" << 'UNINSTALL_EOF'
#!/bin/bash
# remove_passwordless_sudo.sh
# Remove passwordless sudo configuration for Jetson benchmarking

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SUDOERS_FILE="/etc/sudoers.d/jetson-benchmarking"

if [[ ${EUID} -ne 0 ]]; then
    echo -e "${RED}ERROR: This script must be run with sudo${NC}"
    echo -e "${YELLOW}Usage: sudo ./scripts/remove_passwordless_sudo.sh${NC}"
    exit 1
fi

if [[ ! -f "${SUDOERS_FILE}" ]]; then
    echo -e "${YELLOW}Passwordless sudo configuration not found${NC}"
    echo -e "${YELLOW}Nothing to remove${NC}"
    exit 0
fi

echo -e "${YELLOW}This will remove passwordless sudo for Jetson benchmarking${NC}"
read -rp "$(echo -e "${YELLOW}Are you sure? [y/N]: ${NC}")" confirm

if [[ "${confirm}" =~ ^[Yy]$ ]]; then
    rm -f "${SUDOERS_FILE}"
    echo -e "${GREEN}[*] Passwordless sudo configuration removed${NC}"
    echo -e "${YELLOW}Note: You will need to enter your password for:${NC}"
    echo -e "  - make system-specs"
    echo -e "  - make full-analysis"
else
    echo -e "${RED}Removal cancelled${NC}"
fi
UNINSTALL_EOF

chmod +x "${UNINSTALL_SCRIPT}"
chown "${CURRENT_USER}:${CURRENT_USER}" "${UNINSTALL_SCRIPT}"
echo -e "${GREEN}[*] Uninstall script: ${UNINSTALL_SCRIPT}${NC}"

# Success summary
echo ""
echo -e "${MAGENTA}=================================================================${NC}"
echo -e "${GREEN}[*] Passwordless Sudo Setup Complete${NC}"
echo -e "${MAGENTA}=================================================================${NC}"
echo ""
echo -e "${CYAN}Configuration Details:${NC}"
echo -e "  Sudoers file: ${GREEN}${SUDOERS_FILE}${NC}"
echo -e "  Backup directory: ${GREEN}${BACKUP_DIR}${NC}"
echo -e "  Uninstall script: ${GREEN}${UNINSTALL_SCRIPT}${NC}"
echo ""
echo -e "${CYAN}Commands now available without password:${NC}"
echo -e "  ${GREEN}make system-specs${NC}      # Hardware specification collection"
echo -e "  ${GREEN}make full-analysis${NC}     # 3-power mode benchmarking"
echo ""
echo -e "${YELLOW}Verification:${NC}"
echo -e "  Run: ${CYAN}sudo nvpmodel -q${NC}"
echo -e "  Expected: No password prompt"
echo ""
echo -e "${YELLOW}To remove this configuration:${NC}"
echo -e "  Run: ${CYAN}sudo ./scripts/remove_passwordless_sudo.sh${NC}"
echo ""
echo -e "${YELLOW}Note: You may need to start a new terminal session or run:${NC}"
echo -e "  ${CYAN}sudo -k${NC}  # Clear sudo cache"
echo ""
echo -e "${GREEN}Setup completed successfully!${NC}"
