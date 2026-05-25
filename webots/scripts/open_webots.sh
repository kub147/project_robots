#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WEBOTS_BIN="${WEBOTS_BIN:-/Applications/Webots.app/Contents/MacOS/webots}"
WORLD_FILE="$PROJECT_ROOT/webots/worlds/terrain_navigation.wbt"
WORLD_DIR="$PROJECT_ROOT/webots/worlds"
RUN_WORLD_FILE="$WORLD_FILE"

if [ ! -x "$WEBOTS_BIN" ]; then
  echo "Webots binary not found at: $WEBOTS_BIN"
  echo "Install Webots or set WEBOTS_BIN=/path/to/webots"
  exit 1
fi

if [ ! -f "$WORLD_FILE" ]; then
  echo "World file not found: $WORLD_FILE"
  exit 1
fi

rm -f "$WORLD_DIR/.terrain_navigation.wbproj" "$WORLD_DIR/.terrain_navigation.jpg"
rm -rf "$HOME/Library/Caches/Cyberbotics/Webots"

if [ "$#" -gt 0 ]; then
  RUN_WORLD_FILE="$WORLD_DIR/.terrain_navigation_runtime.wbt"
  python3 - "$WORLD_FILE" "$RUN_WORLD_FILE" "$@" <<'PY'
import json
import re
import sys

source_path, output_path, *extra_args = sys.argv[1:]
source = open(source_path, encoding="utf-8").read()
extra = "\n".join(f"    {json.dumps(arg)}" for arg in extra_args)

pattern = r'(controllerArgs\s*\[\n)(.*?)(\n\s*\])'
match = re.search(pattern, source, flags=re.DOTALL)
if not match:
    raise SystemExit("Could not find controllerArgs block in the Webots world.")

replacement = f"{match.group(1)}{match.group(2)}\n{extra}{match.group(3)}"
updated = source[:match.start()] + replacement + source[match.end():]
open(output_path, "w", encoding="utf-8").write(updated)
print(f"Using temporary world with controller args: {' '.join(extra_args)}")
PY
  trap 'rm -f "$RUN_WORLD_FILE"' EXIT
fi

"$WEBOTS_BIN" --mode=realtime --stdout --stderr "$RUN_WORLD_FILE" &
WEBOTS_PID=$!

if command -v osascript >/dev/null 2>&1; then
  (
    sleep 3
    osascript <<'OSA' >/dev/null 2>&1 || true
tell application "Webots" to activate
delay 0.5
tell application "System Events"
  tell process "webots"
    click menu item "Top View" of menu "Change View" of menu item "Change View" of menu "View" of menu bar 1
  end tell
end tell
OSA
  ) &
fi

wait "$WEBOTS_PID"
