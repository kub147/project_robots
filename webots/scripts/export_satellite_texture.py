#!/usr/bin/env python3
"""Export a Webots texture aligned with the planner grid."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENES_DIR = PROJECT_ROOT / "scenes"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEXTURES_DIR = PROJECT_ROOT / "webots" / "worlds" / "textures"
PATHS_DIR = PROJECT_ROOT / "webots" / "paths"
TARGET_GRID_SIZE = 512
PATH_COLOR = (255, 0, 255)
PATH_OUTLINE_COLOR = (20, 0, 20)


def stretch_band(band):
    band = np.nan_to_num(band.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.float32)

    low, high = np.percentile(valid, [2, 98])
    if high <= low:
        high = low + 1.0

    stretched = np.clip((band - low) / (high - low), 0.0, 1.0)
    return stretched


def build_rgb(scene_path, output_size):
    with rasterio.open(scene_path) as src:
        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)

    rgb = np.dstack([stretch_band(red), stretch_band(green), stretch_band(blue)])
    rgb = np.power(rgb, 0.82)
    image = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")

    # Match main.py: all navigation data is first normalized to a 512 x 512
    # planner grid. Upscaling happens only after this step, so the displayed
    # texture and row/col waypoints share the same distortion and extents.
    image = image.resize((TARGET_GRID_SIZE, TARGET_GRID_SIZE), Image.Resampling.LANCZOS)
    image = image.resize((output_size, output_size), Image.Resampling.LANCZOS)
    image = ImageEnhance.Color(image).enhance(1.18)
    image = ImageEnhance.Contrast(image).enhance(1.08)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=80, threshold=3))
    return image


def build_terrain_overlay(scene, output_size):
    display_path = OUTPUT_DIR / f"{scene}_display_map.npy"
    if not display_path.exists():
        return None

    display = np.load(display_path)
    image = Image.fromarray(display.astype(np.uint8), mode="RGB")
    return image.resize((output_size, output_size), Image.Resampling.NEAREST)


def build_webots_texture(scene, scene_path, output_size, mode):
    satellite = build_rgb(scene_path, output_size)
    satellite = ImageEnhance.Color(satellite).enhance(1.08)
    satellite = ImageEnhance.Contrast(satellite).enhance(1.18)
    satellite = ImageEnhance.Sharpness(satellite).enhance(1.9)

    terrain = build_terrain_overlay(scene, output_size)
    if mode == "terrain":
        if terrain is None:
            raise SystemExit(f"Terrain display map not found: {OUTPUT_DIR / f'{scene}_display_map.npy'}")
        return terrain

    if mode == "hybrid" and terrain is not None:
        image = Image.blend(satellite, terrain, 0.28)
        image = ImageEnhance.Contrast(image).enhance(1.10)
        image = ImageEnhance.Sharpness(image).enhance(1.5)
        return image

    return satellite


def to_webots_texture_orientation(image):
    # Webots maps image texture origin differently from the Python/Pygame grid.
    # Existing Webots exporters flip the grid display texture vertically; doing
    # the same here keeps the satellite layer aligned with exported waypoints.
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def find_path_file(scene, start=None, goal=None):
    if start and goal:
        exact = PATHS_DIR / f"{scene}_path_{start[0]}_{start[1]}_{goal[0]}_{goal[1]}.json"
        if exact.exists():
            return exact

    matches = sorted(PATHS_DIR.glob(f"{scene}_path_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def path_points_for_image(data, output_size):
    scale = output_size / TARGET_GRID_SIZE

    def point_from_grid(row, col):
        x = col * scale
        y = (TARGET_GRID_SIZE - 1 - row) * scale
        return x, y

    return [point_from_grid(point["row"], point["col"]) for point in data["path"]]


def draw_path_overlay(image, data, output_size, line_width=None):
    draw = ImageDraw.Draw(image)
    points = path_points_for_image(data, output_size)
    if len(points) < 2:
        return image

    width = line_width or max(6, output_size // 220)
    outline_width = width + max(3, output_size // 512)
    draw.line(points, fill=PATH_OUTLINE_COLOR, width=outline_width, joint="curve")
    draw.line(points, fill=PATH_COLOR, width=width, joint="curve")
    return image


def load_path_data(scene, start=None, goal=None):
    path_file = find_path_file(scene, start, goal)
    if path_file is None:
        return None

    with path_file.open() as f:
        return json.load(f)


def save_alignment_preview(image, scene, output_size, start=None, goal=None):
    data = load_path_data(scene, start, goal)
    if data is None:
        return None

    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    scale = output_size / TARGET_GRID_SIZE

    def point_from_grid(row, col):
        x = col * scale
        y = (TARGET_GRID_SIZE - 1 - row) * scale
        return x, y

    draw_path_overlay(preview, data, output_size)

    sx, sy = point_from_grid(data["start"]["row"], data["start"]["col"])
    gx, gy = point_from_grid(data["goal"]["row"], data["goal"]["col"])
    radius = max(16, output_size // 48)
    draw.ellipse((sx - radius, sy - radius, sx + radius, sy + radius), fill=(0, 220, 55), outline=(255, 255, 255), width=max(2, output_size // 512))
    draw.ellipse((gx - radius, gy - radius, gx + radius, gy + radius), fill=(255, 40, 30), outline=(255, 255, 255), width=max(2, output_size // 512))

    preview_path = TEXTURES_DIR / f"{scene}_webots_alignment_preview.png"
    save_texture(preview, preview_path, "PNG")
    return preview_path


def save_texture(image, path, image_format):
    path = Path(path)
    suffix = path.suffix
    save_kwargs = {}

    if image_format == "JPEG":
        save_kwargs = {"quality": 94, "optimize": True, "progressive": False, "subsampling": 1}
    elif image_format == "PNG":
        save_kwargs = {"compress_level": 6}

    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        image.save(tmp_path, format=image_format, **save_kwargs)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Create a high-resolution RGB texture for Webots.")
    parser.add_argument("scene", help="Scene name without .tif, e.g. Porto_City_512")
    parser.add_argument("--size", type=int, default=2048, help="Output texture size in pixels.")
    parser.add_argument(
        "--format",
        choices=("jpg", "jpeg", "png"),
        default="png",
        help="Output image format. PNG is the default for Webots texture quality.",
    )
    parser.add_argument(
        "--mode",
        choices=("hybrid", "satellite", "terrain"),
        default="satellite",
        help="Texture style. Hybrid blends satellite detail with the classified terrain map for readability.",
    )
    parser.add_argument("--start", type=int, nargs=2, metavar=("row", "col"), help="Optional start cell for alignment preview.")
    parser.add_argument("--goal", type=int, nargs=2, metavar=("row", "col"), help="Optional goal cell for alignment preview.")
    parser.add_argument(
        "--draw-path",
        action="store_true",
        help="Draw the selected planner path directly onto the exported ground texture.",
    )
    args = parser.parse_args()

    scene_path = SCENES_DIR / f"{args.scene}.tif"
    if not scene_path.exists():
        raise SystemExit(f"Scene not found: {scene_path}")

    TEXTURES_DIR.mkdir(parents=True, exist_ok=True)
    image = build_webots_texture(args.scene, scene_path, args.size, args.mode)
    image = to_webots_texture_orientation(image)
    if args.draw_path:
        path_data = load_path_data(args.scene, args.start, args.goal)
        if path_data is None:
            raise SystemExit("No exported path found for --draw-path. Run export_webots.py first or pass matching --start/--goal.")
        draw_path_overlay(image, path_data, args.size)

    extension = "jpg" if args.format in ("jpg", "jpeg") else "png"
    image_format = "JPEG" if extension == "jpg" else "PNG"
    scene_output = TEXTURES_DIR / f"{args.scene}_satellite_{args.size}.{extension}"
    current_output = TEXTURES_DIR / f"current_satellite.{extension}"
    save_texture(image, scene_output, image_format)
    save_texture(image, current_output, image_format)
    preview_output = save_alignment_preview(image, args.scene, args.size, args.start, args.goal)

    print(f"Saved {scene_output}")
    print(f"Saved {current_output}")
    if preview_output:
        print(f"Saved {preview_output}")


if __name__ == "__main__":
    main()
