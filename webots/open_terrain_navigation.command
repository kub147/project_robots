#!/bin/zsh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEBOTS_BIN="/Applications/Webots.app/Contents/MacOS/webots"
STAGING_DIR="/tmp/project_robots_webots"
WORLD_FILE="$STAGING_DIR/worlds/terrain_navigation.wbt"

if [[ ! -x "$WEBOTS_BIN" ]]; then
  echo "Webots executable not found at: $WEBOTS_BIN"
  exit 1
fi

if [[ ! -f "$PROJECT_DIR/webots/worlds/terrain_navigation.wbt" ]]; then
  echo "World file not found at: $PROJECT_DIR/webots/worlds/terrain_navigation.wbt"
  exit 1
fi

rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR/worlds/textures" "$STAGING_DIR/controllers"

cp "$PROJECT_DIR/webots/worlds/terrain_navigation.wbt" "$STAGING_DIR/worlds/terrain_navigation.wbt"
cp "$PROJECT_DIR/webots/worlds/smoke_test.wbt" "$STAGING_DIR/worlds/smoke_test.wbt"
cp "$PROJECT_DIR/webots/worlds/textures/current_display.png" "$STAGING_DIR/worlds/textures/current_display.png"

echo "Opening Webots world:"
echo "$WORLD_FILE"

pkill -f "/Applications/Webots.app/Contents/MacOS/webots" 2>/dev/null || true
sleep 1

exec "$WEBOTS_BIN" --clear-cache "$WORLD_FILE"
