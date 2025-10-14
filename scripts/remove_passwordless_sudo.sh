#!/bin/bash
# remove_passwordless_sudo.sh
# Remove passwordless sudo configuration for Jetson benchmarking
#
# Copyright 2025 ByteStack Labs
# SPDX-License-Identifier: MIT
# Author: Jesse Moses (@Cre4T3Tiv3) <jesse@bytestacklabs.com>
# Version: 1.0.0

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
