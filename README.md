# Autonomous Terrain Navigation Using Sentinel-2 Satellite Imagery

Project developed for `Introduction to Intelligent Robotics (CC3046)`.

Authors:
- Jakub Wilk
- Lukasz Furmanek
- Noa Santos

## Project Overview

This project studies autonomous navigation in unstructured outdoor terrain derived from `Sentinel-2` satellite imagery. The main goal is to move a robot from point `A` to point `B` while minimizing traversal cost across different terrain types such as roads, meadows, forests, wet soil, and water.

The project combines:
- remote sensing and satellite image processing,
- terrain classification from spectral indices,
- graph-based path planning on an `8-connected` grid,
- local sensing with dynamic replanning,
- reinforcement learning for route learning.

The current implementation uses a `Python-based 2D grid simulation` rather than Webots. This was an explicit design choice already justified and approved in the original proposal.

## Core Idea

Each satellite scene is converted into a `512 x 512` navigable grid. Every pixel becomes one traversable cell with an associated movement cost. Different planning strategies are then compared on the same terrain:

- `Dijkstra` as the optimal global baseline,
- `A*` as the efficient heuristic global baseline,
- `Local-sensing A*` as the main proposal contribution,
- `Q-learning` as a learned navigation policy on the same map.

## Input Data

The project uses `GeoTIFF` scenes stored in the `scenes/` directory.

At the moment, the repository contains scenes such as:
- `Porto_City_512`
- `Alentejo_Savanna_512`
- `Serra_Estrela_Mountain_512`
- `Porto_Export_Fixed_Types`
- `Douro_Vineyards_512`

Each scene is resized to `512 x 512` pixels. Assuming `10 m / pixel`, one map represents an area of roughly `5.12 km x 5.12 km`.

## Spectral Indices Used

The terrain model is built from two standard remote sensing indices.

### NDVI

`NDVI` measures vegetation density:

```text
NDVI = (NIR - RED) / (NIR + RED + epsilon)
```

where:
- `NIR` is the near-infrared band,
- `RED` is the red band,
- `epsilon` is a small constant to prevent division by zero.

Interpretation:
- low values usually correspond to roads, urban terrain, or bare soil,
- medium values correspond to grass or sparse vegetation,
- high values correspond to dense vegetation or forest.

### NDWI

`NDWI` is used to detect water and moisture:

```text
NDWI = (GREEN - NIR) / (GREEN + NIR + epsilon)
```

where:
- `GREEN` is the green band,
- `NIR` is the near-infrared band.

Interpretation:
- high values indicate water or very wet terrain,
- intermediate values indicate wet riverbanks, moist soil, or marshy terrain.

## Terrain Classification

The original proposal assumed four broad terrain classes:
- `forest`,
- `meadow`,
- `road/urban`,
- `water`.

The current implementation extends this to make the navigation view more informative and to create a more expressive cost map.

Current terrain classes:
- `Water`
- `Wetland`
- `Road`
- `Barren`
- `Meadow`
- `Forest`

Approximate classification logic:
- `NDWI > 0.30` -> `Water`
- `0.12 < NDWI <= 0.30` -> `Wetland`
- very low `NDVI` and low `NDWI` -> `Road`
- low vegetation -> `Barren`
- medium vegetation -> `Meadow`
- high vegetation -> `Forest`

This does not contradict the proposal. It refines it so the visual output and the movement model better distinguish terrain quality.

## Cost Map

Each grid cell receives a traversal cost. The cost map is continuous rather than purely discrete, so the planner can differentiate between moderately good and clearly bad terrain within the same broad class.

Current cost ranges:
- `Road`: `1.0 - 1.8`
- `Barren`: `1.6 - 3.2`
- `Meadow`: `2.0 - 5.5`
- `Forest`: `5.0 - 9.5`
- `Wetland`: `10 - 25`
- `Water`: `35 - 50`

Interpretation:
- low cost means preferred terrain,
- high cost means difficult or undesirable terrain,
- water is almost impassable but still represented with a finite cost to support controlled experiments.

Path cost is computed using both terrain cost and step length:

```text
path_cost += cell_cost * move_length
```

where:
- `move_length = 1` for horizontal and vertical motion,
- `move_length = sqrt(2)` for diagonal motion.

## Robot Model

The robot is modeled as a point agent moving on the grid:
- state: `(row, col)`
- action space: 8-connected motion
- sensor: circular local sensing radius
- goal: fixed destination `(row, col)`

In the local sensing setup, unknown space is initially treated optimistically:

```text
unknown_cost = 1.0
```

This matches the exploration strategy described in the proposal.

## Implemented Algorithms

### 1. Dijkstra

Global baseline using full knowledge of the map.

Properties:
- cost-optimal,
- no heuristic,
- slower on large maps than `A*`.

### 2. A*

Global baseline with full map knowledge and Euclidean heuristic:

```text
h(a, b) = sqrt((a_row - b_row)^2 + (a_col - b_col)^2)
```

Properties:
- returns the same optimal solution as `Dijkstra` when the heuristic is admissible,
- usually evaluates fewer cells,
- serves as the main global reference.

### 3. Local-Sensing A* with Dynamic Replanning

This is the main algorithm proposed in the project.

Execution flow:
1. the robot initially knows only local terrain,
2. unknown cells are treated as cheap,
3. an initial `A*` path is planned,
4. as the robot moves, it reveals terrain inside the sensing radius,
5. replanning is triggered when newly discovered terrain affects the remaining planned route.

The current implementation includes an important stability improvement:
- replanning is not triggered by every sensed update,
- it only triggers when changed cells actually intersect the remaining route.

### 4. Q-Learning

The project also includes a reinforcement learning component that learns a movement policy directly on the terrain cost map.

State:
- current position `(row, col)`

Actions:
- 8 neighboring moves

Update rule:

```text
Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
```

where:
- `alpha` is the learning rate,
- `gamma` is the discount factor,
- `reward` depends on terrain cost, progress toward the goal, and goal completion.

In the current version, RL is guided by the global `A*` route:
- the `A*` path is used as a guidance prior,
- long routes are split into waypoint segments,
- `Q-learning` is trained segment by segment,
- the final route is stitched into a complete path.

This was chosen for pragmatic reasons: it makes the demo stable and keeps RL usable on large `512 x 512` maps.

## Current Code Structure

### `main.py`

Responsible for:
- loading satellite scenes,
- computing `NDVI` and `NDWI`,
- classifying terrain,
- building the continuous cost map,
- creating the terrain-colored display layer,
- saving processed outputs into `output/`.

Generated files:
- `*_cost_map.npy`
- `*_terrain_map.npy`
- `*_display_map.npy`
- `*_pipeline_2x2.png`

### `navigation.py`

Responsible for:
- interactive point `A/B` selection,
- running `Dijkstra`, `A*`, `Local-sensing A*`, and `Q-learning`,
- visualizing routes in `Pygame`,
- animating robot movement,
- saving learned `Q-table` weights to `*.npy`.

## Current Project Status

The following components are already implemented:

### Completed

- Sentinel-2 terrain preprocessing pipeline,
- safe `NDVI` and `NDWI` extraction,
- `512 x 512` cost map generation,
- extended 6-class terrain classification,
- terrain-aware RGB display layers for `Pygame`,
- `Dijkstra`,
- global `A*`,
- `Local-sensing A*` with dynamic replanning,
- interactive `A/B` point selection,
- route animation and multi-algorithm visualization,
- segmented `Q-learning`,
- `Q-table` weight saving,
- testing on several real scenes.

### What Already Works

At the current stage, the system can:
- generate cost maps from satellite imagery,
- open a scene in `Pygame`,
- let the user click a start and goal point,
- compare `Dijkstra`, `A*`, `Local sensing`, and `Q-learning`,
- display the planned route on the terrain map,
- report route distance, traversal cost, and replanning counts.

## Observations from Current Tests

For short and simple routes:
- `Dijkstra` and `A*` usually return exactly the same result, which is expected,
- `Local sensing` can converge to the same final path when newly sensed terrain does not invalidate the global optimum,
- `Q-learning` often stays close to `A*`, especially on easy routes or with strong guidance.

For long and difficult routes, such as `Douro_Vineyards_512`:
- `Local sensing` can produce significantly longer and more expensive paths,
- replanning counts can become very large,
- `Q-learning` is usually close to the global baseline, but not always identical.

This behavior is reasonable and useful for the experimental part of the project.

## How to Run the Project

### Requirements

The project currently uses:
- `Python 3`
- `numpy`
- `rasterio`
- `matplotlib`
- `pygame`
- `scipy`

Example installation:

```bash
pip install numpy rasterio matplotlib pygame scipy
```

### Generate Cost Maps

```bash
python main.py
```

This script:
- processes all `.tif` files in `scenes/`,
- saves terrain and cost data to `output/`,
- generates `pipeline_2x2` visual summaries.

### Run Navigation

```bash
python navigation.py
```

Then:
1. click the start point,
2. click the goal point,
3. wait for path computation,
4. observe the route comparison and robot animation.

## What Still Needs To Be Done

Based on the proposal and the current implementation, the next major steps are:

### Experiments and Metrics

- run systematic comparisons of `Dijkstra`, `A*`, and `Local-sensing A*`,
- save results to tables or CSV files,
- automatically compute:
  - `path length`,
  - `path cost`,
  - `path cost ratio`,
  - `computation time`,
  - `number of replannings`,
  - `success rate`.

### Noise Robustness

- inject noise into `NDVI` and `NDWI`,
- test different noise levels,
- compare degradation against the noise-free baseline.

### Sensor Radius Sweep

- evaluate different sensing radii, for example `5`, `10`, `20`, `40`,
- measure their effect on:
  - replanning count,
  - path cost,
  - computation time.

### Result Analysis

- save final path overlays as PNG files,
- generate bar charts and comparison plots,
- prepare per-scene and per-biome summaries.

### RL Evaluation

- test weaker `Q-learning` guidance from `A*`,
- compare guided RL against less-guided RL,
- evaluate whether the current tabular formulation should later be replaced by a more compact state representation.

### Final Delivery Preparation

- prepare `instructions.txt`,
- formalize the experimental protocol,
- compile final tables and figures for the paper and presentation.

## Project Stage

The project is currently at the following stage:

- the satellite preprocessing pipeline is working,
- the simulation environment is working,
- all main demonstration algorithms are implemented,
- the checkpoint demo is already viable,
- the project is now transitioning from implementation into structured experimental evaluation.

In practical terms:
- the engineering foundation is largely done,
- the most important next step is automated experimentation, result collection, and report preparation.

## Notes

The current implementation is intentionally pragmatic. It is optimized for a stable, explainable demo and for readable terrain differentiation in the interface. Some design choices, especially in `Q-learning`, deliberately favor stability on large scenes over fully unconstrained RL behavior.

This makes the current version well suited for:
- parameter tuning,
- controlled comparative experiments,
- checkpoint demonstration,
- paper and presentation preparation.
