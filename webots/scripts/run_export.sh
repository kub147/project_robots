#!/bin/bash
# run_export.sh — Quick export a scene for Webots
# Usage:
#   ./webots/scripts/run_export.sh Douro_Vineyards_512
#   ./webots/scripts/run_export.sh Porto_City_512 --start 52 55 --goal 462 455

cd "$(dirname "$0")/../.."  # navigate to project root
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x ".venv/bin/python" ] && [ -z "$PYTHON_BIN_OVERRIDE" ]; then
  PYTHON_BIN=".venv/bin/python"
fi
"$PYTHON_BIN" webots/scripts/export_webots.py "$@"
