#!/bin/bash
#
# Setup script for testing SABRE + Remote MCP Server integration
#
# This script:
# 1. Installs dependencies
# 2. Configures SABRE to use the test MCP server
# 3. Provides instructions for running the test
#

set -e

echo "============================================================"
echo "SABRE + Remote MCP Server Test Setup"
echo "============================================================"
echo ""

# Install aiohttp for the MCP server
echo "1. Installing dependencies..."
if command -v uv &> /dev/null; then
    echo "   Using uv to install aiohttp..."
    uv pip install aiohttp
else
    echo "   Using pip to install aiohttp..."
    pip install aiohttp
fi
echo "   ✓ Dependencies installed"
echo ""

# Backup existing MCP config if it exists
CONFIG_DIR="$HOME/.config/sabre"
CONFIG_FILE="$CONFIG_DIR/mcp.yaml"

echo "2. Configuring SABRE MCP settings..."
mkdir -p "$CONFIG_DIR"

if [ -f "$CONFIG_FILE" ]; then
    BACKUP_FILE="$CONFIG_FILE.backup.$(date +%Y%m%d_%H%M%S)"
    echo "   Backing up existing config to: $BACKUP_FILE"
    cp "$CONFIG_FILE" "$BACKUP_FILE"
fi

# Copy test config
cp mcp_test.yaml "$CONFIG_FILE"
echo "   ✓ MCP config installed to: $CONFIG_FILE"
echo ""

echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Start the MCP server (in a separate terminal):"
echo "   cd $(pwd)"
echo "   python simple_mcp_server.py"
echo ""
echo "2. Test the server (optional verification):"
echo "   chmod +x test_mcp_server.sh"
echo "   ./test_mcp_server.sh"
echo ""
echo "3. Start SABRE:"
echo "   uv run sabre"
echo ""
echo "4. In SABRE, try these commands:"
echo "   - Use the echo tool to say hello"
echo "   - Calculate 42 + 58 using the calculator"
echo "   - What tools are available from remote_test?"
echo ""
echo "Check SABRE logs for:"
echo "   'Connected to MCP server: remote_test (sse)'"
echo "   'Discovered 2 tools from remote_test'"
echo ""
