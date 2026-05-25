#!/usr/bin/env python3
"""
export_webots.py — Export Python planner outputs for Webots.

Reads cost_map, display_map, terrain_map, and a planned path from the
output/ directory and produces:
  1. webots/worlds/textures/<scene>_display.png   (ground texture)
  2. webots/paths/<scene>_<start>_<goal>.json      (waypoints + metadata)

Usage:
  python webots/scripts/export_webots.py <scene_name> [--start row col] [--goal row col]
  python webots/scripts/export_webots.py Douro_Vineyards_512
  python webots/scripts/export_webots.py Porto_City_512 --start 52 55 --goal 462 455

If --start/--goal are omitted, the script looks for a matching Q-table file
in output/ to infer the start and goal.
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

# Add project root to path so we can import from navigation.py if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
WEBOTS_DIR = os.path.join(PROJECT_ROOT, "webots")
TEXTURES_DIR = os.path.join(WEBOTS_DIR, "worlds", "textures")
PATHS_DIR = os.path.join(WEBOTS_DIR, "paths")
TARGET_SIZE = 512
CELL_SIZE = 10.0  # meters per pixel (Sentinel-2 resolution)


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [TEXTURES_DIR, PATHS_DIR]:
        os.makedirs(d, exist_ok=True)


def load_npy(name):
    """Load a .npy file from output/."""
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(path):
        print(f"  ⚠  Warning: {path} not found, skipping.")
        return None
    return np.load(path)


def find_path_from_qtable(scene_name):
    """
    Scan output/ for Q-table files matching the scene name.
    Q-table filenames have the format: <scene>_qtable_<sr>_<sc>_<gr>_<gc>.npy
    Returns list of (start, goal) tuples.
    """
    matches = []
    prefix = f"{scene_name}_qtable_"
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith(prefix) and f.endswith(".npy"):
            parts = f.replace(prefix, "").replace(".npy", "").split("_")
            if len(parts) == 4:
                sr, sc, gr, gc = map(int, parts)
                matches.append(((sr, sc), (gr, gc)))
    return matches


def grid_to_world(row, col, height, width):
    """Convert grid (row, col) to Webots (x, z) centered on origin."""
    x = col * CELL_SIZE - (width * CELL_SIZE / 2.0)
    z = row * CELL_SIZE - (height * CELL_SIZE / 2.0)
    return (x, z)


def world_to_grid(x, z, height, width):
    """Convert Webots (x, z) back to grid (row, col)."""
    col = (x + width * CELL_SIZE / 2.0) / CELL_SIZE
    row = (z + height * CELL_SIZE / 2.0) / CELL_SIZE
    return (int(round(row)), int(round(col)))


def export_display_texture(scene_name, display_map):
    """Save the display map as a PNG texture for Webots."""
    # display_map is shape (H, W, 3), dtype uint8
    img = Image.fromarray(display_map, mode="RGB")
    # Flip vertically because Webots texture origin is bottom-left vs image top-left
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    # Save scene-specific texture
    path = os.path.join(TEXTURES_DIR, f"{scene_name}_display.png")
    img.save(path)
    print(f"  ✓  Saved ground texture: {path}")
    # Also save as current_display.png for the .wbt file to reference
    fixed_path = os.path.join(TEXTURES_DIR, "current_display.png")
    img.save(fixed_path)
    print(f"  ✓  Saved as current_display.png (used by .wbt)")
    return path


def export_cost_texture(scene_name, cost_map):
    """Optional: save a colorized cost map texture."""
    # Normalize cost to 0-255
    norm = np.clip((cost_map - 1.0) / 49.0, 0.0, 1.0)  # 1..50 → 0..1
    gray = (norm * 255).astype(np.uint8)
    img = Image.fromarray(gray, mode="L")
    # Colorize with a colormap (inferno-like)
    from matplotlib import colormaps
    cmap = colormaps["magma_r"]
    rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    img_colored = Image.fromarray(rgb, mode="RGB")
    img_colored = img_colored.transpose(Image.FLIP_TOP_BOTTOM)
    path = os.path.join(TEXTURES_DIR, f"{scene_name}_cost.png")
    img_colored.save(path)
    print(f"  ✓  Saved cost texture: {path}")
    return path


def build_road_biased_cost_map(terrain_map, fallback_cost_map):
    """Build a Webots-demo planning cost that strongly prefers road-like cells."""
    if terrain_map is None:
        return fallback_cost_map

    return np.select(
        [
            terrain_map == 2,  # road / urban
            terrain_map == 3,  # barren / open / path-like
            terrain_map == 4,  # meadow
            terrain_map == 5,  # forest
            terrain_map == 1,  # wetland / river edge
            terrain_map == 0,  # water
        ],
        [
            1.0,
            2.3,
            8.0,
            25.0,
            80.0,
            160.0,
        ],
        default=12.0,
    ).astype(np.float32)


def export_waypoints(scene_name, path_cells, start, goal, display_map, cost_map, terrain_map):
    """
    Export the planned path as a JSON waypoint file.

    Args:
        scene_name: Name of the scene (e.g. "Douro_Vineyards_512")
        path_cells: List of (row, col) tuples forming the path
        start: (row, col) start
        goal: (row, col) goal
        display_map: RGB display map for reference
        cost_map: cost map for metrics
        terrain_map: terrain classification map
    """
    height, width = cost_map.shape

    # Build waypoints list with grid and world coordinates
    waypoints = []
    for i, (row, col) in enumerate(path_cells):
        wx, wz = grid_to_world(row, col, height, width)
        waypoints.append({
            "index": i,
            "row": int(row),
            "col": int(col),
            "x_m": round(wx, 2),
            "z_m": round(wz, 2),
        })

    # Compute metrics
    total_distance_px = 0.0
    total_cost = 0.0
    for a, b in zip(path_cells[:-1], path_cells[1:]):
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        step = np.sqrt(dr * dr + dc * dc)
        total_distance_px += step
        total_cost += float(cost_map[b]) * step

    data = {
        "meta": {
            "scene": scene_name,
            "map_height_px": height,
            "map_width_px": width,
            "cell_size_m": CELL_SIZE,
            "map_center_x_m": 0.0,
            "map_center_z_m": 0.0,
            "map_extent_min_x_m": -(width * CELL_SIZE / 2.0),
            "map_extent_min_z_m": -(height * CELL_SIZE / 2.0),
            "map_extent_max_x_m": width * CELL_SIZE / 2.0,
            "map_extent_max_z_m": height * CELL_SIZE / 2.0,
        },
        "start": {
            "row": int(start[0]),
            "col": int(start[1]),
            "x_m": grid_to_world(start[0], start[1], height, width)[0],
            "z_m": grid_to_world(start[0], start[1], height, width)[1],
        },
        "goal": {
            "row": int(goal[0]),
            "col": int(goal[1]),
            "x_m": grid_to_world(goal[0], goal[1], height, width)[0],
            "z_m": grid_to_world(goal[0], goal[1], height, width)[1],
        },
        "path": waypoints,
        "metrics": {
            "num_waypoints": len(waypoints),
            "distance_px": round(total_distance_px, 1),
            "distance_m": round(total_distance_px * CELL_SIZE, 1),
            "cumulative_cost": round(total_cost, 1),
        },
    }

    # Determine filename from start/goal
    filename = f"{scene_name}_path_{start[0]}_{start[1]}_{goal[0]}_{goal[1]}.json"
    path = os.path.join(PATHS_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓  Saved waypoints: {path}")
    return path


def downsample_path(path_cells, max_points=200):
    """
    Downsample a path to at most max_points for manageable waypoints.
    Uses uniform decimation.
    """
    if len(path_cells) <= max_points:
        return path_cells

    indices = np.linspace(0, len(path_cells) - 1, max_points, dtype=int)
    return [path_cells[i] for i in indices]


def export_satellite_texture_with_path(scene, start, goal, size, mode):
    """Regenerate the Webots ground texture with the exported path drawn on it."""
    from webots.scripts.export_satellite_texture import (
        build_webots_texture,
        draw_path_overlay,
        load_path_data,
        save_alignment_preview,
        save_texture,
        to_webots_texture_orientation,
    )

    scene_path = os.path.join(PROJECT_ROOT, "scenes", f"{scene}.tif")
    if not os.path.exists(scene_path):
        print(f"  ⚠  Scene source not found, skipping path texture: {scene_path}")
        return

    image = build_webots_texture(scene, scene_path, size, mode)
    image = to_webots_texture_orientation(image)
    path_data = load_path_data(scene, start, goal)
    if path_data is None:
        print("  ⚠  No path data found, skipping path texture overlay.")
        return

    draw_path_overlay(image, path_data, size)
    scene_output = os.path.join(TEXTURES_DIR, f"{scene}_satellite_{size}.png")
    current_output = os.path.join(TEXTURES_DIR, "current_satellite.png")
    save_texture(image, scene_output, "PNG")
    save_texture(image, current_output, "PNG")
    preview_output = save_alignment_preview(image, scene, size, start, goal)
    print(f"  ✓  Saved path ground texture: {current_output}")
    if preview_output:
        print(f"  ✓  Saved alignment preview: {preview_output}")


def main():
    parser = argparse.ArgumentParser(description="Export Python planner data for Webots")
    parser.add_argument("scene", help="Scene name (e.g., Douro_Vineyards_512)")
    parser.add_argument("--start", type=int, nargs=2, metavar=("row", "col"),
                        help="Start grid position (row col)")
    parser.add_argument("--goal", type=int, nargs=2, metavar=("row", "col"),
                        help="Goal grid position (row col)")
    parser.add_argument("--path", type=str, default=None,
                        help="Path to a CSV/JSON path file (optional). If not given, looks for Q-table.")
    parser.add_argument("--downsample", type=int, default=200,
                        help="Maximum number of waypoints (default: 200)")
    parser.add_argument("--cost-texture", action="store_true",
                        help="Also export cost map as texture")
    parser.add_argument("--road-biased", action="store_true",
                        help="Prefer road/open urban cells more strongly for the Webots visual route")
    parser.add_argument("--no-path-texture", action="store_true",
                        help="Do not redraw current_satellite.png with the path on the ground")
    parser.add_argument("--texture-size", type=int, default=2048,
                        help="Size of the Webots satellite texture generated with the path (default: 2048)")
    parser.add_argument("--texture-mode", choices=("hybrid", "satellite", "terrain"), default="satellite",
                        help="Texture style used when drawing the ground path")
    args = parser.parse_args()

    ensure_dirs()
    scene = args.scene

    print(f"\nExporting {scene} for Webots...\n")

    # 1. Load maps
    display_map = load_npy(f"{scene}_display_map.npy")
    cost_map = load_npy(f"{scene}_cost_map.npy")
    terrain_map = load_npy(f"{scene}_terrain_map.npy")

    if display_map is None and cost_map is None:
        print("  ✗  No display or cost map found. Run main.py first.")
        sys.exit(1)

    # Build display map from cost if not available
    if display_map is None and cost_map is not None:
        print("  ℹ  No display map found, generating from cost map...")
        from navigation import build_display_map, infer_terrain_from_cost
        if terrain_map is None:
            terrain_map = infer_terrain_from_cost(cost_map)
        display_map = build_display_map(terrain_map, cost_map)

    # 2. Export textures
    export_display_texture(scene, display_map)
    if args.cost_texture and cost_map is not None:
        export_cost_texture(scene, cost_map)

    # 3. Find or build the path
    if args.start and args.goal:
        start = tuple(args.start)
        goal = tuple(args.goal)
    else:
        # Try to infer from Q-table files
        qtable_paths = find_path_from_qtable(scene)
        if not qtable_paths:
            print("  ✗  No Q-table files found and no --start/--goal given.")
            print("     Provide --start and --goal or run navigation.py first.")
            sys.exit(1)
        if len(qtable_paths) > 1:
            print(f"  ℹ  Found {len(qtable_paths)} Q-table files. Using the first one.")
            for i, (s, g) in enumerate(qtable_paths):
                print(f"      {i}: start={s} goal={g}")
        start, goal = qtable_paths[0]
        print(f"  ℹ  Using start={start} goal={goal} from Q-table filename.")

    # 4. Reconstruct a path
    # Priority: 1) provided path file, 2) compute A* now
    path_cells = None

    if args.path:
        # Load from external file
        ext = os.path.splitext(args.path)[1].lower()
        if ext == ".json":
            with open(args.path) as f:
                path_data = json.load(f)
            path_cells = [(wp["row"], wp["col"]) for wp in path_data.get("path", path_data.get("waypoints", []))]
        elif ext == ".csv":
            import csv
            path_cells = []
            with open(args.path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    path_cells.append((int(row["row"]), int(row["col"])))
        print(f"  ℹ  Loaded {len(path_cells)} waypoints from {args.path}")

    if path_cells is None and cost_map is not None:
        # Compute A* path on the fly
        print("  ℹ  Computing A* path...")
        try:
            from navigation import GridPlanner
            planning_cost_map = cost_map
            if args.road_biased:
                print("  ℹ  Using road-biased planning costs for Webots visualization...")
                planning_cost_map = build_road_biased_cost_map(terrain_map, cost_map)
            planner = GridPlanner(cost_map.shape[0], cost_map.shape[1])
            path_cells = planner.plan(planning_cost_map, start, goal, use_heuristic=True)
            if path_cells is None:
                print("  ✗  A* could not find a path. Try different start/goal.")
                sys.exit(1)
            print(f"  ✓  A* found path with {len(path_cells)} cells")
        except ImportError:
            print("  ✗  Could not import GridPlanner from navigation.py")
            sys.exit(1)

    if not path_cells:
        print("  ✗  No path available.")
        sys.exit(1)

    # 5. Downsample for manageable waypoints
    path_cells = downsample_path(path_cells, max_points=args.downsample)

    # 6. Export waypoints
    export_waypoints(scene, path_cells, start, goal, display_map, cost_map, terrain_map)
    if not args.no_path_texture:
        export_satellite_texture_with_path(scene, start, goal, args.texture_size, args.texture_mode)

    print(f"\nDone! You can now open Webots with:")
    print(f"  webots/worlds/terrain_navigation.wbt")
    print(f"(Make sure to update the scene name at the top of the controller)\n")


if __name__ == "__main__":
    main()
