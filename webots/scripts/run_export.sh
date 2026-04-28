#!/bin/bash
# run_export.sh — Quick export a scene for Webots
# Usage:
#   ./webots/scripts/run_export.sh Douro_Vineyards_512
#   ./webots/scripts/run_export.sh Porto_City_512 --start 52 55 --goal 462 455

cd "$(dirname "$0")/../.."  # navigate to project root
python3 webots/scripts/export_webots.py "$@"
