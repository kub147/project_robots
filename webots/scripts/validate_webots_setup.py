#!/usr/bin/env python3
import configparser
import json
import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEBOTS_DIR = PROJECT_ROOT / "webots"
WORLD_FILE = WEBOTS_DIR / "worlds" / "terrain_navigation.wbt"
TEXTURE_FILE = WEBOTS_DIR / "worlds" / "textures" / "current_display.png"
SATELLITE_TEXTURE_FILE = WEBOTS_DIR / "worlds" / "textures" / "current_satellite.png"
CONTROLLER_FILE = WEBOTS_DIR / "controllers" / "path_follower" / "path_follower.py"
CONFIG_FILE = WEBOTS_DIR / "controllers" / "path_follower" / "config.ini"
PATHS_DIR = WEBOTS_DIR / "paths"


def fail(message):
    print(f"[FAIL] {message}")
    return False


def ok(message):
    print(f"[ OK ] {message}")
    return True


def main():
    checks = []

    checks.append(ok(f"project root: {PROJECT_ROOT}"))
    checks.append(ok(f"python: {sys.executable}"))

    checks.append(ok(f"world exists: {WORLD_FILE}") if WORLD_FILE.exists() else fail(f"missing world: {WORLD_FILE}"))
    checks.append(ok(f"class texture exists: {TEXTURE_FILE}") if TEXTURE_FILE.exists() else fail(f"missing class texture: {TEXTURE_FILE}"))
    checks.append(ok(f"satellite texture exists: {SATELLITE_TEXTURE_FILE}") if SATELLITE_TEXTURE_FILE.exists() else fail(f"missing satellite texture: {SATELLITE_TEXTURE_FILE}"))
    checks.append(ok(f"controller exists: {CONTROLLER_FILE}") if CONTROLLER_FILE.exists() else fail(f"missing controller: {CONTROLLER_FILE}"))

    if CONFIG_FILE.exists():
        parser = configparser.ConfigParser()
        parser.read(CONFIG_FILE)
        command = parser.get("python", "COMMAND", fallback="")
        if command and Path(command).exists():
            checks.append(ok(f"controller python configured: {command}"))
        else:
            checks.append(fail(f"invalid controller python command in {CONFIG_FILE}: {command}"))
    else:
        checks.append(fail(f"missing controller config: {CONFIG_FILE}"))

    if WORLD_FILE.exists():
        world_text = WORLD_FILE.read_text()
        checks.append(ok('world uses controller "path_follower"') if 'controller "path_follower"' in world_text else fail('world does not use controller "path_follower"'))
        checks.append(ok("world has supervisor TRUE") if "supervisor TRUE" in world_text else fail("world robot is not a Supervisor"))
        checks.append(ok("world references current_satellite.png") if "textures/current_satellite.png" in world_text else fail("world does not reference current_satellite.png"))

        scene = None
        start = None
        goal = None
        scene_match = re.search(r'"--scene"\s+"([^"]+)"', world_text)
        start_match = re.search(r'"--start"\s+"(\d+)"\s+"(\d+)"', world_text)
        goal_match = re.search(r'"--goal"\s+"(\d+)"\s+"(\d+)"', world_text)
        if scene_match and start_match and goal_match:
            scene = scene_match.group(1)
            start = tuple(map(int, start_match.groups()))
            goal = tuple(map(int, goal_match.groups()))
            path_file = PATHS_DIR / f"{scene}_path_{start[0]}_{start[1]}_{goal[0]}_{goal[1]}.json"
            if path_file.exists():
                with path_file.open() as f:
                    data = json.load(f)
                checks.append(ok(f"path exists: {path_file.name} ({len(data['path'])} waypoints)"))
            else:
                checks.append(fail(f"missing path for world args: {path_file}"))
        else:
            checks.append(fail("could not parse --scene/--start/--goal from world controllerArgs"))

    webots_bin = Path("/Applications/Webots.app/Contents/MacOS/webots")
    checks.append(ok(f"Webots app found: {webots_bin}") if webots_bin.exists() else fail(f"Webots app not found: {webots_bin}"))

    print()
    if all(checks):
        print("Webots setup looks ready.")
        return 0

    print("Webots setup has issues. Fix the [FAIL] lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
