#!/bin/bash
# install_git_hooks.sh
# Install Git hooks for automatic code quality checks
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

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="${PROJECT_ROOT}/.git/hooks"

echo -e "${MAGENTA}=================================================================${NC}"
echo -e "${MAGENTA}Git Hooks Installation${NC}"
echo -e "${MAGENTA}=================================================================${NC}"
echo ""

# Check if we're in a git repository
if [[ ! -d "${PROJECT_ROOT}/.git" ]]; then
    echo -e "${RED}ERROR: Not in a git repository${NC}"
    echo -e "${YELLOW}This script must be run from within a git repository${NC}"
    exit 1
fi

echo -e "${BLUE}Installing Git hooks to: ${HOOKS_DIR}${NC}"
echo ""

# Create pre-commit hook
echo -e "${CYAN}-> Creating pre-commit hook...${NC}"
cat > "${HOOKS_DIR}/pre-commit" << 'HOOK_EOF'
#!/bin/bash
# Pre-commit hook: Run code quality checks before committing

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}[Pre-commit Hook] Running code quality checks...${NC}"
echo ""

# Check if make is available
if ! command -v make &> /dev/null; then
    echo -e "${RED}ERROR: make command not found${NC}"
    exit 1
fi

# Run code quality checks
if ! make check-all; then
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  [!] CODE QUALITY CHECKS FAILED                           ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Please fix the issues above before committing.${NC}"
    echo ""
    echo -e "${YELLOW}Quick fixes:${NC}"
    echo -e "  ${GREEN}make fix-all${NC}        # Auto-fix formatting issues"
    echo -e "  ${GREEN}make check-all${NC}      # Re-run all checks"
    echo ""
    echo -e "${YELLOW}To commit anyway (not recommended):${NC}"
    echo -e "  ${GREEN}git commit --no-verify${NC}"
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  [*] ALL CODE QUALITY CHECKS PASSED                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

exit 0
HOOK_EOF

chmod +x "${HOOKS_DIR}/pre-commit"
echo -e "${GREEN}[*] Pre-commit hook installed${NC}"

# Create pre-push hook
echo -e "${CYAN}-> Creating pre-push hook...${NC}"
cat > "${HOOKS_DIR}/pre-push" << 'HOOK_EOF'
#!/bin/bash
# Pre-push hook: Run tests before pushing

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}[Pre-push Hook] Running tests...${NC}"
echo ""

# Check if make is available
if ! command -v make &> /dev/null; then
    echo -e "${RED}ERROR: make command not found${NC}"
    exit 1
fi

# Run unit tests
if ! make test-unit; then
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  [!] TESTS FAILED                                         ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Please fix failing tests before pushing.${NC}"
    echo ""
    echo -e "${YELLOW}To push anyway (not recommended):${NC}"
    echo -e "  ${GREEN}git push --no-verify${NC}"
    echo ""
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  [*] ALL TESTS PASSED                                     ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

exit 0
HOOK_EOF

chmod +x "${HOOKS_DIR}/pre-push"
echo -e "${GREEN}[*] Pre-push hook installed${NC}"

# Create commit-msg hook for conventional commits
echo -e "${CYAN}-> Creating commit-msg hook...${NC}"
cat > "${HOOKS_DIR}/commit-msg" << 'HOOK_EOF'
#!/bin/bash
# Commit-msg hook: Validate commit message format (Conventional Commits)

set -e

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

commit_msg_file=$1
commit_msg=$(cat "${commit_msg_file}")

# Conventional Commits pattern
# Format: type(scope): description
# Examples:
#   feat(cuda): add FP16 kernel
#   fix(benchmark): correct timing
#   docs: update README
pattern='^(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\([a-z0-9-]+\))?: .{1,72}$'

if ! echo "${commit_msg}" | grep -qE "${pattern}"; then
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  [!] INVALID COMMIT MESSAGE FORMAT                        ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Your commit message:${NC}"
    echo -e "  ${commit_msg}"
    echo ""
    echo -e "${YELLOW}Expected format (Conventional Commits):${NC}"
    echo -e "  ${GREEN}type(scope): description${NC}"
    echo ""
    echo -e "${YELLOW}Valid types:${NC}"
    echo -e "  feat, fix, docs, style, refactor, test, chore, perf, ci, build, revert"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  ${GREEN}feat(cuda): add FP16 matrix multiplication kernel${NC}"
    echo -e "  ${GREEN}fix(benchmark): correct efficiency calculation${NC}"
    echo -e "  ${GREEN}docs: update installation instructions${NC}"
    echo ""
    echo -e "${YELLOW}To bypass this check (not recommended):${NC}"
    echo -e "  ${GREEN}git commit --no-verify${NC}"
    echo ""
    exit 1
fi

exit 0
HOOK_EOF

chmod +x "${HOOKS_DIR}/commit-msg"
echo -e "${GREEN}[*] Commit-msg hook installed${NC}"

# Success summary
echo ""
echo -e "${MAGENTA}=================================================================${NC}"
echo -e "${GREEN}[*] Git Hooks Installation Complete${NC}"
echo -e "${MAGENTA}=================================================================${NC}"
echo ""
echo -e "${CYAN}Installed hooks:${NC}"
echo -e "  ${GREEN}pre-commit${NC}   - Runs code quality checks (make check-all)"
echo -e "  ${GREEN}pre-push${NC}     - Runs unit tests (make test-unit)"
echo -e "  ${GREEN}commit-msg${NC}   - Validates Conventional Commits format"
echo ""
echo -e "${CYAN}What happens now:${NC}"
echo -e "  [*] Before each commit: Code quality checks run automatically"
echo -e "  [*] Before each push: Unit tests run automatically"
echo -e "  [*] Invalid commit messages will be rejected"
echo ""
echo -e "${YELLOW}To bypass hooks (use sparingly):${NC}"
echo -e "  ${GREEN}git commit --no-verify${NC}"
echo -e "  ${GREEN}git push --no-verify${NC}"
echo ""
echo -e "${YELLOW}To uninstall hooks:${NC}"
echo -e "  ${GREEN}rm ${HOOKS_DIR}/pre-commit${NC}"
echo -e "  ${GREEN}rm ${HOOKS_DIR}/pre-push${NC}"
echo -e "  ${GREEN}rm ${HOOKS_DIR}/commit-msg${NC}"
echo ""
echo -e "${GREEN}Happy coding!${NC}"
echo ""
