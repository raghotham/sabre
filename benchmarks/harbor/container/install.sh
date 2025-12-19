#!/bin/bash
#
# SABRE Agent Installation Script for Harbor
#
# Installs SABRE and its dependencies in the Harbor container
#

set -euo pipefail

echo "==================================================================="
echo "Installing SABRE for Harbor"
echo "==================================================================="
echo ""

# Install system dependencies
echo "[1/5] Installing system dependencies..."
apt-get update -qq > /dev/null 2>&1
apt-get install -y -qq curl git > /dev/null 2>&1
echo "✓ System dependencies installed"

# Install uv
echo "[2/5] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
echo "✓ uv installed"

# Copy SABRE source from host (mounted at /logs/agent)
echo "[3/5] Installing SABRE..."
if [ -d "/logs/agent/sabre_source" ]; then
    # SABRE source was copied to logs_dir by agent.py
    cp -r /logs/agent/sabre_source /tmp/sabre
    cd /tmp/sabre
    uv sync > /dev/null 2>&1
    echo "✓ SABRE installed from host"
else
    echo "✗ SABRE source not found at /logs/agent/sabre_source"
    exit 1
fi

# Install Playwright for Web helper
echo "[4/5] Installing Playwright..."
# Install in SABRE's uv environment (not via uvx)
cd /tmp/sabre
uv run playwright install chromium --only-shell --with-deps > /dev/null 2>&1
echo "✓ Playwright installed"

echo "[5/5] Verifying installation..."
uv --version
echo ""
echo "==================================================================="
echo "SABRE installation complete!"
echo "==================================================================="
