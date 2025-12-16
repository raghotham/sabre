#!/bin/bash
#
# SABRE Agent Installation Script for Harbor
#
# This script installs uv (universal Python package installer).
# The agent.py is a uv script with inline dependencies.
#

set -euo pipefail

echo "==================================================================="
echo "SABRE Agent Setup for Harbor"
echo "==================================================================="
echo ""

# Install curl if not available
echo "[1/3] Installing curl..."
if ! command -v curl &> /dev/null; then
    apt-get update -qq > /dev/null 2>&1
    apt-get install -y -qq curl > /dev/null 2>&1
    echo "✓ curl installed"
else
    echo "✓ curl already installed"
fi

# Install uv
echo "[2/3] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
echo "✓ uv installed"

echo "[3/3] Verifying installation..."
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo ""
echo "==================================================================="
echo "SABRE agent setup complete!"
echo ""
echo "Note: This agent communicates with a SABRE server running on the host."
echo "      Ensure the server is running at: http://host.docker.internal:8011"
echo "      Start with: uv run sabre-server"
echo "==================================================================="
