#!/bin/zsh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEBOTS_BIN="/Applications/Webots.app/Contents/MacOS/webots"
STAGING_DIR="/tmp/project_robots_webots_smoke"
WORLD_FILE="$STAGING_DIR/worlds/smoke_test.wbt"

if [[ ! -x "$WEBOTS_BIN" ]]; then
  echo "Webots executable not found at: $WEBOTS_BIN"
  exit 1
fi

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR/worlds"
cp "$PROJECT_DIR/webots/worlds/smoke_test.wbt" "$WORLD_FILE"

echo "Opening Webots smoke test:"
echo "$WORLD_FILE"

pkill -f "/Applications/Webots.app/Contents/MacOS/webots" 2>/dev/null || true
sleep 1

exec "$WEBOTS_BIN" --clear-cache "$WORLD_FILE"
