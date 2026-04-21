"""
Week 1 — Sentinel-2 Terrain Processing Pipeline (Optimized)
Project: Autonomous Terrain Navigation Using Sentinel-2 (CC3046)
Group: Jakub Wilk, Łukasz Furmanek, Noa Santos [cite: 7, 8, 9]

This script processes Sentinel-2 GeoTIFFs to create 512x512 cost maps.
It handles 5-band or 6-band inputs safely and produces a richer terrain map
that is easier to interpret inside the navigation demo.
"""

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.ndimage import zoom

# ─────────────────────────────────────────────
#  CONFIGURATION (Proposal Section 3.1 & 4)
# ─────────────────────────────────────────────

SCENES_DIR = "scenes"  # Input folder for GEE .tif files [cite: 101]
OUTPUT_DIR = "output"  # Output folder for results [cite: 101]
TARGET_SIZE = 512  # Standardized grid size (512x512)

# Base classification thresholds from proposal [cite: 39, 40, 41]
NDVI_FOREST_THRESHOLD = 0.5  # NDVI > 0.5: Forest [cite: 39]
NDVI_MEADOW_MIN = 0.2  # 0.2 < NDVI <= 0.5: Meadow [cite: 40]
NDWI_WATER_THRESHOLD = 0.3  # NDWI > 0.3: Open Water [cite: 41]
NDWI_WET_THRESHOLD = 0.12
NDVI_BARREN_MAX = 0.12
NDVI_SHRUB_MAX = 0.38

# Terrain classes used for analysis and UI rendering.
TERRAIN_WATER = 0
TERRAIN_WETLAND = 1
TERRAIN_ROAD = 2
TERRAIN_BARREN = 3
TERRAIN_MEADOW = 4
TERRAIN_FOREST = 5

# Band mapping based on your GEE export [cite: 38, 101]
BAND_RED = 1  # B4
BAND_GREEN = 2  # B3
BAND_BLUE = 3  # B2
BAND_NIR = 4  # B8
BAND_NDVI = 5  # Precomputed NDVI from GEE

# ─────────────────────────────────────────────
#  TERRAIN VISUALIZATION DEFS
# ─────────────────────────────────────────────

TERRAIN_COLORS = {
    TERRAIN_WATER: "#2C7FB8",
    TERRAIN_WETLAND: "#63C4C9",
    TERRAIN_ROAD: "#D8C9A7",
    TERRAIN_BARREN: "#B38B59",
    TERRAIN_MEADOW: "#8CCB5E",
    TERRAIN_FOREST: "#2E6B2E",
}
TERRAIN_LABELS = {
    TERRAIN_WATER: "Open Water / River (cost 35-50)",
    TERRAIN_WETLAND: "Wet Soil / River Edge (cost 10-25)",
    TERRAIN_ROAD: "Road / Urban / Bare Path (cost 1.0-1.8)",
    TERRAIN_BARREN: "Dry Soil / Vineyard Rows (cost 1.6-3.2)",
    TERRAIN_MEADOW: "Meadow / Sparse Vegetation (cost 2.0-5.5)",
    TERRAIN_FOREST: "Dense Vegetation / Forest (cost 5.0-9.5)",
}


# ─────────────────────────────────────────────
#  CORE FUNCTIONS
# ─────────────────────────────────────────────

def load_scene(filepath):
    """Load GeoTIFF and return as float32 array[cite: 31]."""
    with rasterio.open(filepath) as src:
        data = src.read().astype(np.float32)
    return np.nan_to_num(data)


def extract_indices(data):
    """
    Safely extract NDVI and NDWI.
    Fixes IndexError by checking band count.
    """
    num_bands = data.shape[0]

    # Extract NDVI (5th band if exists, otherwise compute)
    if num_bands >= 5:
        ndvi = data[4]
    else:
        red, nir = data[BAND_RED - 1], data[BAND_NIR - 1]
        ndvi = (nir - red) / (nir + red + 1e-8)

    # Extract NDWI (6th band if exists, otherwise compute from Green/NIR)
    if num_bands >= 6:
        ndwi = data[5]
    else:
        green, nir = data[BAND_GREEN - 1], data[BAND_NIR - 1]
        ndwi = (green - nir) / (green + nir + 1e-8)

    return ndvi, ndwi


def classify_terrain(ndvi, ndwi):
    """Terrain classes tuned for more informative rendering and cost shaping."""
    terrain = np.full(ndvi.shape, TERRAIN_ROAD, dtype=np.uint8)

    # Water and wet margins come first so they override vegetation labels.
    terrain[ndwi > NDWI_WET_THRESHOLD] = TERRAIN_WETLAND
    terrain[ndwi > NDWI_WATER_THRESHOLD] = TERRAIN_WATER

    barren_mask = (ndvi >= 0.02) & (ndvi <= NDVI_BARREN_MAX) & (ndwi <= NDWI_WET_THRESHOLD)
    meadow_mask = (ndvi > NDVI_BARREN_MAX) & (ndvi <= NDVI_SHRUB_MAX) & (ndwi <= NDWI_WET_THRESHOLD)
    forest_mask = ndvi > NDVI_SHRUB_MAX

    terrain[barren_mask] = TERRAIN_BARREN
    terrain[meadow_mask] = TERRAIN_MEADOW
    terrain[forest_mask] = TERRAIN_FOREST
    terrain[(ndvi < 0.02) & (ndwi <= NDWI_WET_THRESHOLD)] = TERRAIN_ROAD
    return terrain


def build_continuous_cost_map(ndvi, ndwi, terrain):
    """
    Generate a continuous cost map with clearer terrain separation.

    The proposal's main costs are preserved directionally:
    roads stay cheapest, meadows moderate, forests expensive, water extreme.
    We add sub-ranges so the UI does not collapse almost everything into one tone.
    """
    cost_map = np.ones_like(ndvi, dtype=np.float32)

    road_mask = terrain == TERRAIN_ROAD
    barren_mask = terrain == TERRAIN_BARREN
    meadow_mask = terrain == TERRAIN_MEADOW
    forest_mask = terrain == TERRAIN_FOREST
    wetland_mask = terrain == TERRAIN_WETLAND
    water_mask = terrain == TERRAIN_WATER

    # Roads / urban patches: keep near the global minimum.
    cost_map[road_mask] = 1.0 + np.clip(np.abs(ndvi[road_mask]) * 2.0, 0.0, 0.8)

    # Dry soil / vineyard rows: slightly worse than roads, but still navigable.
    barren_ndvi = np.clip((ndvi[barren_mask] - 0.02) / max(1e-6, NDVI_BARREN_MAX - 0.02), 0.0, 1.0)
    cost_map[barren_mask] = 1.6 + barren_ndvi * 1.6

    # Meadows: medium cost with smooth gradient.
    meadow_ndvi = np.clip(
        (ndvi[meadow_mask] - NDVI_BARREN_MAX) / max(1e-6, NDVI_SHRUB_MAX - NDVI_BARREN_MAX),
        0.0,
        1.0,
    )
    cost_map[meadow_mask] = 2.0 + meadow_ndvi * 3.5

    # Forest: higher cost and a bit more penalty when moisture is high.
    forest_ndvi = np.clip(
        (ndvi[forest_mask] - NDVI_SHRUB_MAX) / max(1e-6, 0.8 - NDVI_SHRUB_MAX),
        0.0,
        1.0,
    )
    forest_wetness = np.clip(ndwi[forest_mask] + 0.1, 0.0, 0.5)
    cost_map[forest_mask] = 5.0 + forest_ndvi * 3.5 + forest_wetness * 2.0

    # Wetlands / river banks: expensive but not fully blocked.
    wetland_strength = np.clip(
        (ndwi[wetland_mask] - NDWI_WET_THRESHOLD) / max(1e-6, NDWI_WATER_THRESHOLD - NDWI_WET_THRESHOLD),
        0.0,
        1.0,
    )
    cost_map[wetland_mask] = 10.0 + wetland_strength * 15.0

    # Open water: very high penalty, close to impassable.
    water_strength = np.clip((ndwi[water_mask] - NDWI_WATER_THRESHOLD) / max(1e-6, 0.6 - NDWI_WATER_THRESHOLD), 0.0, 1.0)
    cost_map[water_mask] = 35.0 + water_strength * 15.0

    return np.clip(cost_map, 1.0, 50.0)


def build_display_rgb(terrain, cost_map):
    """Create a terrain-first RGB layer with subtle cost-based shading."""
    rgb = np.zeros((*terrain.shape, 3), dtype=np.uint8)
    for terrain_id, hex_color in TERRAIN_COLORS.items():
        color = np.array(mcolors.to_rgb(hex_color), dtype=np.float32) * 255.0
        mask = terrain == terrain_id
        rgb[mask] = color.astype(np.uint8)

    # Slightly darken harder cells while preserving semantic terrain colors.
    shading = 0.78 + 0.22 * (1.0 - np.clip((cost_map - 1.0) / 49.0, 0.0, 1.0))
    rgb = np.clip(rgb.astype(np.float32) * shading[..., None], 0, 255).astype(np.uint8)
    return rgb


def resize_to_target(array, target=TARGET_SIZE):
    """Forces the array into a square TARGET_SIZE x TARGET_SIZE shape."""
    # We calculate zoom factors for both axes independently
    zoom_factors = (target / array.shape[0], target / array.shape[1])
    return zoom(array, zoom_factors, order=1)


# ─────────────────────────────────────────────
#  VISUALIZATION (2x2 Layout)
# ─────────────────────────────────────────────

def plot_2x2_analysis(name, ndvi, ndwi, terrain, cost_map, display_rgb, save_path):
    """4-panel validation figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Sentinel-2 Analysis: {name}\nResolution: {TARGET_SIZE}x{TARGET_SIZE} (10m/px)",
                 fontsize=16, fontweight='bold')

    # NDVI
    im0 = axes[0, 0].imshow(ndvi, cmap="RdYlGn", vmin=-0.1, vmax=0.8)
    axes[0, 0].set_title("NDVI (Vegetation Gradient)", fontsize=12)
    plt.colorbar(im0, ax=axes[0, 0], label="Vegetation Density")

    # NDWI
    im1 = axes[0, 1].imshow(ndwi, cmap="Blues", vmin=-0.2, vmax=0.6)
    axes[0, 1].set_title("NDWI (Water Gradient)", fontsize=12)
    plt.colorbar(im1, ax=axes[0, 1], label="Water Saturation")

    # Classification
    ordered_colors = [TERRAIN_COLORS[i] for i in sorted(TERRAIN_COLORS)]
    cmap_terrain = mcolors.ListedColormap(ordered_colors)
    axes[1, 0].imshow(terrain, cmap=cmap_terrain)
    axes[1, 0].set_title("Terrain Classes", fontsize=12)
    legend_elements = [Patch(facecolor=TERRAIN_COLORS[k], label=TERRAIN_LABELS[k]) for k in sorted(TERRAIN_LABELS)]
    axes[1, 0].legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.8)

    # Terrain-aware cost map preview.
    axes[1, 1].imshow(display_rgb)
    im3 = axes[1, 1].imshow(cost_map, cmap="magma_r", alpha=0.28, vmin=1.0, vmax=50.0)
    axes[1, 1].set_title("Terrain + Traversal Cost Preview", fontsize=12)
    plt.colorbar(im3, ax=axes[1, 1], label="Traversal Cost (1 to 50)")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([]);
            ax.set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    print(f"    Visual analysis saved: {save_path}")
    plt.show()


# ─────────────────────────────────────────────
#  MAIN EXECUTION
# ─────────────────────────────────────────────

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    files = [f for f in os.listdir(SCENES_DIR) if f.endswith('.tif')]
    if not files:
        print(f"Error: Place your Sentinel-2 .tif files in '{SCENES_DIR}/'")
        return

    print(f"Found {len(files)} scene(s). Processing...")

    for file in files:
        path = os.path.join(SCENES_DIR, file)
        name = os.path.splitext(file)[0]

        # 1. Load and extract
        data = load_scene(path)
        raw_ndvi, raw_ndwi = extract_indices(data)

        # 2. Resize
        ndvi = resize_to_target(raw_ndvi)
        ndwi = resize_to_target(raw_ndwi)

        # 3. Classify and compute nuanced costs
        terrain = classify_terrain(ndvi, ndwi)
        cost_map = build_continuous_cost_map(ndvi, ndwi, terrain)
        display_rgb = build_display_rgb(terrain, cost_map)

        # 4. Save cost grid for Week 2 pathfinding [cite: 44, 101]
        npy_path = os.path.join(OUTPUT_DIR, f"{name}_cost_map.npy")
        np.save(npy_path, cost_map)
        terrain_path = os.path.join(OUTPUT_DIR, f"{name}_terrain_map.npy")
        np.save(terrain_path, terrain)
        display_path = os.path.join(OUTPUT_DIR, f"{name}_display_map.npy")
        np.save(display_path, display_rgb)

        # 5. Visualize
        viz_path = os.path.join(OUTPUT_DIR, f"{name}_pipeline_2x2.png")
        plot_2x2_analysis(name, ndvi, ndwi, terrain, cost_map, display_rgb, viz_path)


if __name__ == "__main__":
    main()
