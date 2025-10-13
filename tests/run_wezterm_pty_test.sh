#!/bin/bash
###
# WezTerm PTY Test Runner
#
# Runs terminal detection tests in a real WezTerm PTY session with TTY=true.
# This ensures tests run in an environment identical to actual user usage.
#
# Usage:
#   ./tests/run_wezterm_pty_test.sh [test_script]
#
# Examples:
#   ./tests/run_wezterm_pty_test.sh tests/test_theme_detection.py
#   ./tests/run_wezterm_pty_test.sh tests/colors_test.py
###

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default test script
TEST_SCRIPT="${1:-tests/test_theme_detection.py}"

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}WezTerm PTY Test Runner${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Check if WezTerm CLI is available
if ! command -v wezterm &> /dev/null; then
    echo -e "${RED}ERROR: wezterm command not found${NC}"
    echo "Please install WezTerm: https://wezfurlong.org/wezterm/"
    exit 1
fi

# Check if test script exists
if [ ! -f "$TEST_SCRIPT" ]; then
    echo -e "${RED}ERROR: Test script not found: $TEST_SCRIPT${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} WezTerm CLI found: $(which wezterm)"
echo -e "${GREEN}✓${NC} Test script: $TEST_SCRIPT"
echo ""

# Get WezTerm version
WEZTERM_VERSION=$(wezterm --version 2>&1 | head -n1)
echo -e "WezTerm version: ${YELLOW}$WEZTERM_VERSION${NC}"
echo ""

echo -e "${BLUE}Running test in WezTerm PTY session...${NC}"
echo -e "${BLUE}─────────────────────────────────────────────────────────${NC}"
echo ""

# Run the test in a WezTerm PTY session
# This creates a real PTY with TTY=true, TERM_PROGRAM=WezTerm, etc.
wezterm cli spawn --cwd "$(pwd)" -- uv run python "$TEST_SCRIPT"

echo ""
echo -e "${BLUE}─────────────────────────────────────────────────────────${NC}"
echo -e "${GREEN}✓${NC} Test completed"
echo ""

# Show environment info
echo -e "${BLUE}Test Environment:${NC}"
echo "  - Real PTY: yes (via wezterm cli spawn)"
echo "  - TTY: true"
echo "  - TERM_PROGRAM: WezTerm"
echo "  - OSC 11 query: available"
echo ""
