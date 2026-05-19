# Autonomous Terrain Navigation Using Sentinel-2

Project for `Introduction to Intelligent Robotics (CC3046)`.

Authors:
- Jakub Wilk
- Lukasz Furmanek
- Noa Santos

## Overview

This project focuses on autonomous robot navigation over real terrain derived from `Sentinel-2` satellite imagery. The current system converts satellite scenes into navigable cost maps, lets the user choose a start point `A` and goal point `B`, and compares several planning approaches:

- `Dijkstra`
- `A*`
- `Local-sensing A* with dynamic replanning`
- `Q-learning`

The current implementation is a `Python 2D grid simulation`. It is the algorithmic foundation for a later integration into `Webots`, where the robot is expected to use this planning logic together with reinforcement learning to follow the planned route in simulation.

## Current Goal

The project currently solves this problem:

1. load a satellite-based terrain map,
2. convert it into a traversal cost map,
3. click point `A` and point `B`,
4. compute a good path,
5. visualize how the robot would move through the terrain,
6. compare classical planning with learning-based behavior.

## Satellite Processing

The terrain model is built from multispectral `GeoTIFF` scenes stored in `scenes/`.

Each scene is resized to `512 x 512` and processed using two standard remote sensing indices:

### NDVI

```text
NDVI = (NIR - RED) / (NIR + RED + epsilon)
```

Used to estimate vegetation density.

### NDWI

```text
NDWI = (GREEN - NIR) / (GREEN + NIR + epsilon)
```

Used to estimate water and moisture.

## Terrain Classes and Cost Map

The original proposal used a simple class structure. The current implementation extends it to make the map more useful visually and more informative for planning.

Current terrain classes:
- `Water`
- `Wetland`
- `Road`
- `Barren`
- `Meadow`
- `Forest`

Current cost ranges:
- `Road`: `1.0 - 1.8`
- `Barren`: `1.6 - 3.2`
- `Meadow`: `2.0 - 5.5`
- `Forest`: `5.0 - 9.5`
- `Wetland`: `10 - 25`
- `Water`: `35 - 50`

This means the robot naturally prefers roads and open ground, is more reluctant to cross dense vegetation or wet terrain, and treats water as almost impassable.

Path cost is evaluated as:

```text
path_cost += cell_cost * move_length
```

where diagonal motion uses `sqrt(2)` and straight motion uses `1`.

## Implemented Algorithms

### Dijkstra

Global baseline with full map knowledge. It provides the minimum-cost path and serves as the main optimal reference.

### A*

Global baseline with Euclidean heuristic:

```text
h(a, b) = sqrt((a_row - b_row)^2 + (a_col - b_col)^2)
```

It should match Dijkstra in path quality while being more efficient.

### Local-Sensing A*

This is the main proposal algorithm. The robot initially knows only local terrain and replans when newly discovered cells interfere with the current route.

### Q-Learning

The project also includes reinforcement learning with the standard update rule:

```text
Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
```

In the current version, Q-learning is guided by the global `A*` route and trained segment by segment so that it works reliably on large `512 x 512` maps.

## Current Implementation

### `main.py`

Responsible for:
- loading scenes,
- computing `NDVI` and `NDWI`,
- building terrain classes,
- generating the traversal cost map,
- generating colored display layers,
- saving outputs to `output/`.

Generated outputs include:
- `*_cost_map.npy`
- `*_terrain_map.npy`
- `*_display_map.npy`
- `*_pipeline_2x2.png`

### `navigation.py`

Responsible for:
- interactive `A/B` selection,
- path computation,
- route comparison across algorithms,
- robot animation in `Pygame`,
- saving learned `Q-table` weights.

## What Is Already Done

The following parts are already working:

- Sentinel-2 preprocessing pipeline
- NDVI and NDWI extraction
- terrain classification
- continuous traversal cost map
- improved terrain visualization for the interactive map
- Dijkstra baseline
- global A*
- local-sensing A* with dynamic replanning
- interactive point selection
- animated route visualization
- Q-learning with saved weights

So at this point, the project already has a working planning and simulation core.

## How To Run

Install dependencies:

```bash
pip install numpy rasterio matplotlib pygame scipy
```

Generate maps:

```bash
python main.py
```

Run the interactive navigation demo:

```bash
python navigation.py
```

Then click the start and goal points on the displayed map.

Run the Webots simulation:

```bash
./webots/scripts/open_webots.sh
```

This opens `webots/worlds/terrain_navigation.wbt`, loads the exported waypoint
JSON, and runs the `path_follower` controller inside Webots. It does not use the
Pygame 2D demo window.

To export a route that better follows road-like areas in the satellite view:

```bash
python3 webots/scripts/export_webots.py Porto_City_512 \
  --start 52 55 \
  --goal 462 455 \
  --road-biased
```

To regenerate the more realistic satellite texture used by Webots:

```bash
python3 webots/scripts/export_satellite_texture.py Porto_City_512 \
  --size 2048 \
  --format png \
  --mode satellite \
  --start 52 55 \
  --goal 462 455
```

The Webots world keeps the real map extent at `5120 m x 5120 m`
(`512 px x 10 m`). The texture can be upscaled for presentation quality without
changing the robot coordinates or the planned path. The Webots world references
`webots/worlds/textures/current_satellite.png`. The exporter applies the same
planner-grid orientation used by the Webots waypoint export, then writes
`webots/worlds/textures/Porto_City_512_webots_alignment_preview.png` with the
path drawn over the texture. Use that preview to confirm the map and robot route
still agree before opening Webots.

## Next Steps

After the current stage, the main things still to do are:

### 1. Validate the cost map and terrain model

The first next step is to verify whether the current terrain classes and cost matrix are really the best choice. The current setup works well, but it still needs systematic validation:

- check whether the current terrain penalties are realistic,
- test whether some class boundaries should be adjusted,
- confirm whether the current cost ranges produce sensible robot behavior,
- evaluate whether the current map representation is the right one for later robot control.

### 2. Improve efficiency

The next important step is to make the overall solution as efficient as possible:

- reduce planning time on large scenes,
- analyze why local replanning becomes expensive on difficult maps,
- tune parameters for more stable performance,
- improve the computational side of the pipeline before transferring it into robot simulation.

### 3. Build the bridge toward Webots

Based on the discussion with the professor, this project should later be connected to `Webots`.

That means the current grid-based planner is not the final end product. It should become the high-level navigation layer that provides a route or guidance signal for a robot inside Webots.

The next conceptual step is therefore:

- use the current map and planning pipeline as a high-level path generator,
- transfer the route or waypoint structure into Webots,
- let the simulated robot follow that route,
- connect reinforcement learning to the robot's motion behavior so that it learns to follow the path or terrain line more robustly.

In other words, the current system already solves the planning side. The later Webots stage should solve the control and embodied robot side.

### 4. Strengthen the experimental part

The remaining work should also include proper evaluation:

- compare global planning against local sensing on several maps,
- measure path cost, path length, runtime, and replanning count,
- test whether the learned policy behaves consistently on different terrains,
- prepare cleaner result summaries for the report and presentation.

## Current Stage of the Project

Right now, the project is in a good intermediate state:

- the terrain processing pipeline is working,
- the planning algorithms are working,
- reinforcement learning is already connected,
- the interactive demo is working,
- the next major step is validation, optimization, and later transfer to Webots.

So the core system is already built. What remains is refining it, proving that the terrain model is good, improving efficiency, and then using it as the basis for robot behavior in Webots.
