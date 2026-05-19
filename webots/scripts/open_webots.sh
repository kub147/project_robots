#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WEBOTS_BIN="${WEBOTS_BIN:-/Applications/Webots.app/Contents/MacOS/webots}"
WORLD_FILE="$PROJECT_ROOT/webots/worlds/terrain_navigation.wbt"
WORLD_DIR="$PROJECT_ROOT/webots/worlds"

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

"$WEBOTS_BIN" --mode=realtime --stdout --stderr "$WORLD_FILE" &
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
