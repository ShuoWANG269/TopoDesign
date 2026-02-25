#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "Installing dependencies for RL Topology Optimization with uv..."

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed or not in PATH."
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

if [[ ! -f "requirements.lock" ]]; then
    echo "Error: requirements.lock not found in ${SCRIPT_DIR}"
    echo "Expected file: ${SCRIPT_DIR}/requirements.lock"
    exit 1
fi

echo "Creating/updating virtual environment (.venv) with Python 3.9..."
if ! uv venv --python 3.9 --allow-existing .venv; then
    echo "Error: failed to create .venv with Python 3.9."
    echo "Please install Python 3.9 on this machine and rerun."
    exit 1
fi

echo "Syncing exact locked dependencies from requirements.lock..."
uv pip sync --python .venv/bin/python requirements.lock

echo ""
echo "Installation complete."
echo "Verification:"
echo "  .venv/bin/python -V"
echo "  uv pip freeze --python .venv/bin/python"
echo "Optional smoke test:"
echo "  .venv/bin/python test_basic.py"
