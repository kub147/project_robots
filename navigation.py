"""
Interactive terrain navigation on Sentinel-2 cost maps.

This module keeps the proposal's A* local-sensing setup and adds a practical
demo mode for the checkpoint: the user clicks start and goal on the map, then
the script computes baseline paths and trains a Q-learning policy to learn a
good route on the same terrain.
"""

import math
import os
import sys
import time
import heapq

import numpy as np
import pygame


OUTPUT_DIR = "output"
WINDOW_SIZE = 860
PANEL_WIDTH = 520
UNKNOWN_COST = 1.0
MAX_CELL_COST = 50.0
LOCAL_SENSOR_RADIUS = 20
LOCAL_MAX_STEPS = 6000
LOCAL_REPLAN_COST_DELTA = 0.6

# 8-connected motion model.
ACTIONS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]
M_SQRT2 = math.sqrt(2)
ACTION_COSTS = [1.0, 1.0, 1.0, 1.0, M_SQRT2, M_SQRT2, M_SQRT2, M_SQRT2]

# Terrain classes aligned with main.py.
TERRAIN_NAMES = {
    0: "Water",
    1: "Wetland",
    2: "Road",
    3: "Barren",
    4: "Meadow",
    5: "Forest",
}
TERRAIN_COLORS = {
    0: (44, 127, 184),
    1: (99, 196, 201),
    2: (216, 201, 167),
    3: (179, 139, 89),
    4: (140, 203, 94),
    5: (46, 107, 46),
}

# Q-learning hyperparameters.
QL_EPISODES = 220
QL_ALPHA = 0.22
QL_GAMMA = 0.94
QL_EPSILON_START = 0.40
QL_EPSILON_END = 0.05
QL_MAX_STEPS = 2500
QL_GOAL_REWARD = 150.0
QL_INVALID_MOVE_PENALTY = -12.0
QL_REPEAT_PENALTY = -1.0
QL_PROGRESS_WEIGHT = 2.4
QL_COST_WEIGHT = 0.55
QL_ASTAR_BONUS = 1.4
QL_STAGNATION_LIMIT = 180
QL_EXPERT_RATE_START = 0.35
QL_EXPERT_RATE_END = 0.05


def load_cost_map(scene_name):
    path = os.path.join(OUTPUT_DIR, f"{scene_name}_cost_map.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cost map: {path}")
    return np.load(path).astype(np.float32)


def load_terrain_map(scene_name):
    path = os.path.join(OUTPUT_DIR, f"{scene_name}_terrain_map.npy")
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.uint8)


def load_display_map(scene_name):
    path = os.path.join(OUTPUT_DIR, f"{scene_name}_display_map.npy")
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.uint8)


def infer_terrain_from_cost(cost_map):
    terrain = np.full(cost_map.shape, 2, dtype=np.uint8)
    terrain[cost_map >= 35.0] = 0
    terrain[(cost_map >= 10.0) & (cost_map < 35.0)] = 1
    terrain[(cost_map >= 5.0) & (cost_map < 10.0)] = 5
    terrain[(cost_map >= 2.0) & (cost_map < 5.0)] = 4
    terrain[(cost_map >= 1.5) & (cost_map < 2.0)] = 3
    return terrain


def build_display_map(terrain_map, cost_map):
    rgb = np.zeros((*terrain_map.shape, 3), dtype=np.uint8)
    for terrain_id, color in TERRAIN_COLORS.items():
        rgb[terrain_map == terrain_id] = np.array(color, dtype=np.uint8)

    shading = 0.80 + 0.20 * (1.0 - np.clip((cost_map - 1.0) / (MAX_CELL_COST - 1.0), 0.0, 1.0))
    rgb = np.clip(rgb.astype(np.float32) * shading[..., None], 0, 255).astype(np.uint8)
    return rgb


def normalize_to_surface(cost_map, display_map=None):
    if display_map is not None:
        return display_map

    terrain_map = infer_terrain_from_cost(cost_map)
    return build_display_map(terrain_map, cost_map)


def compute_scale(height, width):
    return max(1, WINDOW_SIZE // max(height, width))


def cell_to_screen(cell, scale):
    row, col = cell
    return int(col * scale + scale / 2), int(row * scale + scale / 2)


def draw_path(screen, path, color, scale, width=2):
    if not path or len(path) < 2:
        return
    pts = [cell_to_screen(point, scale) for point in path]
    pygame.draw.lines(screen, color, False, pts, width)


def path_metrics(path, cost_map):
    if not path or len(path) < 2:
        return {
            "steps": 0,
            "distance_px": 0.0,
            "distance_m": 0.0,
            "cost": float("inf"),
            "cell_cost": float("inf"),
        }

    total_distance = 0.0
    total_cost = 0.0
    total_cell_cost = 0.0
    for current, nxt in zip(path[:-1], path[1:]):
        dr = nxt[0] - current[0]
        dc = nxt[1] - current[1]
        step_distance = M_SQRT2 if dr and dc else 1.0
        total_distance += step_distance
        step_cost = float(cost_map[nxt])
        total_cell_cost += step_cost
        total_cost += step_cost * step_distance

    return {
        "steps": len(path) - 1,
        "distance_px": total_distance,
        "distance_m": total_distance * 10.0,
        "cost": float(total_cost),
        "cell_cost": float(total_cell_cost),
    }


def terrain_histogram(path, terrain_map):
    counts = {name: 0 for name in TERRAIN_NAMES.values()}
    for row, col in path:
        counts[TERRAIN_NAMES[int(terrain_map[row, col])]] += 1
    return counts


def format_histogram(hist):
    return ", ".join(f"{key}:{value}" for key, value in hist.items())


class GridPlanner:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    @staticmethod
    def heuristic(a, b):
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        return max(dr, dc) + (M_SQRT2 - 1.0) * min(dr, dc)

    def _neighbors(self, node):
        row, col = node
        for (dr, dc), move_cost in zip(ACTIONS, ACTION_COSTS):
            nr = row + dr
            nc = col + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                yield (nr, nc), move_cost

    def plan(self, cost_map, start, goal, use_heuristic=True, max_iter=800000):
        frontier = [(self.heuristic(start, goal) if use_heuristic else 0.0, 0.0, start)]
        parents = {start: None}
        g_cost = {start: 0.0}
        iterations = 0

        while frontier:
            iterations += 1
            if iterations > max_iter:
                return None

            _, current_g, current = heapq.heappop(frontier)
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = parents[current]
                return path[::-1]

            if current_g > g_cost.get(current, float("inf")):
                continue

            for nxt, move_cost in self._neighbors(current):
                tentative = current_g + float(cost_map[nxt]) * move_cost
                if tentative < g_cost.get(nxt, float("inf")):
                    g_cost[nxt] = tentative
                    parents[nxt] = current
                    heuristic_cost = self.heuristic(nxt, goal) if use_heuristic else 0.0
                    heapq.heappush(frontier, (tentative + heuristic_cost, tentative, nxt))

        return None


class QLearningNavigator:
    def __init__(self, cost_map, start, goal, guidance_path=None):
        self.cost_map = cost_map
        self.height, self.width = cost_map.shape
        self.start = start
        self.goal = goal
        self.q_table = np.zeros((self.height, self.width, len(ACTIONS)), dtype=np.float32)
        self._goal = goal
        self.guidance_mask = np.zeros((self.height, self.width), dtype=bool)
        self.expert_action = np.full((self.height, self.width), -1, dtype=np.int8)
        if guidance_path:
            for row, col in guidance_path:
                r0 = max(0, row - 2)
                r1 = min(self.height, row + 3)
                c0 = max(0, col - 2)
                c1 = min(self.width, col + 3)
                self.guidance_mask[r0:r1, c0:c1] = True
            for current, nxt in zip(guidance_path[:-1], guidance_path[1:]):
                delta = (nxt[0] - current[0], nxt[1] - current[1])
                if delta in ACTIONS:
                    action_idx = ACTIONS.index(delta)
                    self.expert_action[current[0], current[1]] = action_idx
                    self.q_table[current[0], current[1], action_idx] = 4.0

    @staticmethod
    def _fast_dist(r1, c1, r2, c2):
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        return max(dr, dc) + (M_SQRT2 - 1.0) * min(dr, dc)

    def choose_action(self, state, epsilon, expert_rate):
        expert_idx = int(self.expert_action[state[0], state[1]])
        if expert_idx >= 0 and np.random.random() < expert_rate:
            return expert_idx
        if np.random.random() < epsilon:
            return np.random.randint(len(ACTIONS))
        return int(np.argmax(self.q_table[state[0], state[1]]))

    def next_state(self, state, action_idx):
        dr, dc = ACTIONS[action_idx]
        nr = state[0] + dr
        nc = state[1] + dc
        if nr < 0 or nr >= self.height or nc < 0 or nc >= self.width:
            return state, QL_INVALID_MOVE_PENALTY, False

        nxt = (nr, nc)
        sr, sc = state
        gr, gc = self._goal
        old_dist = self._fast_dist(sr, sc, gr, gc)
        new_dist = self._fast_dist(nr, nc, gr, gc)

        reward = -float(self.cost_map[nxt]) * QL_COST_WEIGHT
        reward += (old_dist - new_dist) * QL_PROGRESS_WEIGHT
        if self.guidance_mask[nxt]:
            reward += QL_ASTAR_BONUS
        done = nxt == self.goal
        if done:
            reward += QL_GOAL_REWARD
        return nxt, reward, done

    def train(self, episodes=QL_EPISODES, max_steps=QL_MAX_STEPS):
        epsilon_values = np.linspace(QL_EPSILON_START, QL_EPSILON_END, episodes)
        expert_values = np.linspace(QL_EXPERT_RATE_START, QL_EXPERT_RATE_END, episodes)
        rewards = []
        success_count = 0
        best_path = None
        best_cost = float("inf")

        h, w = self.height, self.width
        goal = self.goal
        q_table = self.q_table
        cost_map = self.cost_map
        guidance_mask = self.guidance_mask
        actions = ACTIONS
        n_actions = len(actions)
        goali, goalj = goal

        for episode in range(episodes):
            state = self.start
            si, sj = state
            visited = np.zeros((h, w), dtype=bool)
            visited[si, sj] = True
            total_reward = 0.0
            path = [state]
            stagnation = 0
            epsilon = float(epsilon_values[episode])
            expert_rate = float(expert_values[episode])

            for _ in range(max_steps):
                action_idx = self.choose_action(state, epsilon, expert_rate)
                dr, dc = actions[action_idx]
                ni = si + dr
                nj = sj + dc

                if ni < 0 or ni >= h or nj < 0 or nj >= w:
                    reward = QL_INVALID_MOVE_PENALTY
                    nxt = state
                    done = False
                else:
                    nxt = (ni, nj)
                    old_dist = self._fast_dist(si, sj, goali, goalj)
                    new_dist = self._fast_dist(ni, nj, goali, goalj)
                    reward = -float(cost_map[ni, nj]) * QL_COST_WEIGHT
                    reward += (old_dist - new_dist) * QL_PROGRESS_WEIGHT
                    if guidance_mask[ni, nj]:
                        reward += QL_ASTAR_BONUS
                    done = nxt == goal
                    if done:
                        reward += QL_GOAL_REWARD

                if visited[ni, nj] and not done:
                    reward += QL_REPEAT_PENALTY
                    stagnation += 1
                else:
                    stagnation = 0

                current_q = q_table[si, sj, action_idx]
                best_future = max(q_table[ni, nj])
                q_table[si, sj, action_idx] = current_q + QL_ALPHA * (reward + QL_GAMMA * best_future - current_q)

                state = nxt
                si, sj = ni, nj
                path.append(state)
                visited[ni, nj] = True
                total_reward += reward

                if done:
                    success_count += 1
                    candidate_cost = path_metrics(path, cost_map)["cost"]
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_path = path[:]
                    break

                if stagnation >= QL_STAGNATION_LIMIT:
                    break

            rewards.append(total_reward)

        greedy_path = self.rollout()
        if greedy_path:
            greedy_cost = path_metrics(greedy_path, self.cost_map)["cost"]
            if greedy_cost < best_cost:
                best_path = greedy_path
                best_cost = greedy_cost

        return {
            "rewards": rewards,
            "success_rate": success_count / max(1, episodes),
            "best_path": best_path,
            "q_table": q_table,
            "q_min": float(q_table.min()),
            "q_max": float(q_table.max()),
            "q_mean": float(q_table.mean()),
        }

    def rollout(self, max_steps=QL_MAX_STEPS):
        state = self.start
        path = [state]
        h, w = self.height, self.width
        goal = self.goal
        seen_arr = np.zeros((h, w), dtype=bool)
        seen_arr[state[0], state[1]] = True

        for _ in range(max_steps):
            if state == goal:
                return path

            q_values = self.q_table[state[0], state[1]]
            action_order = np.argsort(q_values)[::-1]
            moved = False

            for action_idx in action_order:
                dr, dc = ACTIONS[int(action_idx)]
                nr = state[0] + dr
                nc = state[1] + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if seen_arr[nr, nc] and (nr, nc) != goal:
                        continue
                    state = (nr, nc)
                    path.append(state)
                    seen_arr[nr, nc] = True
                    moved = True
                    break

            if not moved:
                return None

        return path if path[-1] == goal else None


class LocalSensingNavigator:
    def __init__(self, global_map, start, goal, sensor_radius=LOCAL_SENSOR_RADIUS):
        self.global_map = global_map
        self.height, self.width = global_map.shape
        self.start = start
        self.goal = goal
        self.sensor_radius = sensor_radius
        self.planner = GridPlanner(self.height, self.width)

    def reveal(self, known_map, position):
        row, col = position
        rr, cc = np.ogrid[:self.height, :self.width]
        mask = (rr - row) ** 2 + (cc - col) ** 2 <= self.sensor_radius ** 2
        before = known_map[mask].copy()
        known_map[mask] = self.global_map[mask]
        changed_local = np.abs(known_map[mask] - before) > LOCAL_REPLAN_COST_DELTA
        changed_mask = np.zeros_like(known_map, dtype=bool)
        changed_mask[mask] = changed_local
        return bool(np.any(changed_local)), changed_mask

    def run(self, max_steps=LOCAL_MAX_STEPS):
        known_map = np.ones_like(self.global_map, dtype=np.float32) * UNKNOWN_COST
        current = self.start
        history = [current]
        replans = 0
        visible_mask = np.zeros_like(self.global_map, dtype=bool)

        self.reveal(known_map, current)
        current_path = self.planner.plan(known_map, current, self.goal, use_heuristic=True)

        for _ in range(max_steps):
            if current == self.goal:
                return {
                    "path": history,
                    "known_map": known_map,
                    "replans": replans,
                    "visible_mask": visible_mask,
                }

            if not current_path or len(current_path) < 2:
                break

            changed, changed_mask = self.reveal(known_map, current)
            visible_mask |= changed_mask
            if changed and current_path:
                path_rows = np.array([node[0] for node in current_path[1:]], dtype=np.int32)
                path_cols = np.array([node[1] for node in current_path[1:]], dtype=np.int32)
                path_affected = np.any(changed_mask[path_rows, path_cols])
                if path_affected:
                    replans += 1
                    current_path = self.planner.plan(known_map, current, self.goal, use_heuristic=True)
                    if not current_path or len(current_path) < 2:
                        break

            current = current_path[1]
            history.append(current)
            current_path = current_path[1:]

        return {
            "path": history if history[-1] == self.goal else None,
            "known_map": known_map,
            "replans": replans,
            "visible_mask": visible_mask,
        }


def get_interactive_points(cost_map, display_map=None):
    pygame.init()
    height, width = cost_map.shape
    scale = compute_scale(height, width)
    map_surface = pygame.surfarray.make_surface(normalize_to_surface(cost_map, display_map).swapaxes(0, 1))
    map_surface = pygame.transform.scale(map_surface, (width * scale, height * scale))
    screen = pygame.display.set_mode((width * scale, height * scale))
    pygame.display.set_caption("Select start (green) and goal (red)")

    start = None
    goal = None
    font = pygame.font.SysFont("arial", 18)

    while goal is None:
        screen.blit(map_surface, (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                point = (min(height - 1, y // scale), min(width - 1, x // scale))
                if start is None:
                    start = point
                elif point != start:
                    goal = point

        if start is not None:
            pygame.draw.circle(screen, (0, 255, 120), cell_to_screen(start, scale), max(4, scale + 1))
        instruction = "Click start and goal on the map"
        screen.blit(font.render(instruction, True, (255, 255, 255)), (12, 12))
        pygame.display.flip()

    return start, goal


def render_dashboard(
    screen,
    map_surface,
    scale,
    metrics_lines,
    start,
    goal,
    astar_path=None,
    dijkstra_path=None,
    rl_path=None,
    local_path=None,
    sensor_radius=0,
    robot_position=None,
):
    map_width = map_surface.get_width()
    map_height = map_surface.get_height()
    screen.fill((18, 18, 24))
    screen.blit(map_surface, (0, 0))

    draw_path(screen, dijkstra_path, (255, 255, 255), scale, 2)
    draw_path(screen, astar_path, (0, 230, 255), scale, 3)
    draw_path(screen, local_path, (255, 196, 0), scale, 3)
    draw_path(screen, rl_path, (255, 0, 180), scale, 3)

    pygame.draw.circle(screen, (0, 255, 120), cell_to_screen(start, scale), max(4, scale + 1))
    pygame.draw.circle(screen, (255, 60, 60), cell_to_screen(goal, scale), max(4, scale + 1))
    if robot_position is not None:
        pygame.draw.circle(screen, (255, 240, 40), cell_to_screen(robot_position, scale), max(4, scale + 1))
        if sensor_radius > 0:
            pygame.draw.circle(
                screen,
                (255, 230, 90),
                cell_to_screen(robot_position, scale),
                max(3, int(sensor_radius * scale)),
                1,
            )

    panel_rect = pygame.Rect(map_width, 0, PANEL_WIDTH, screen.get_height())
    pygame.draw.rect(screen, (28, 30, 36), panel_rect)
    pygame.draw.line(screen, (65, 70, 78), (map_width, 0), (map_width, screen.get_height()), 2)

    font = pygame.font.SysFont("consolas", 22)
    small_font = pygame.font.SysFont("consolas", 18)
    screen.blit(font.render("Terrain Navigation Demo", True, (245, 245, 245)), (map_width + 18, 18))

    legend = [
        ("Dijkstra", (255, 255, 255)),
        ("A*", (0, 230, 255)),
        ("Local A*", (255, 196, 0)),
        ("Q-learning", (255, 0, 180)),
    ]
    y = 62
    for label, color in legend:
        pygame.draw.line(screen, color, (map_width + 18, y + 9), (map_width + 58, y + 9), 4)
        screen.blit(small_font.render(label, True, (220, 220, 220)), (map_width + 70, y))
        y += 28

    y += 8
    for line in metrics_lines:
        rendered = small_font.render(line, True, (220, 220, 220))
        screen.blit(rendered, (map_width + 18, y))
        y += 24


def build_metrics_lines(dijkstra_path, astar_path, local_result, rl_result, cost_map, terrain_map, train_time):
    def describe(label, path):
        if not path:
            return f"{label}: no path"
        metrics = path_metrics(path, cost_map)
        return (
            f"{label}: steps={metrics['steps']} "
            f"cost={metrics['cost']:.1f} dist={metrics['distance_m']:.0f}m"
        )

    lines = [
        describe("Dijkstra", dijkstra_path),
        describe("A*", astar_path),
        describe("Local", local_result["path"]),
        describe("RL", rl_result["best_path"]),
        f"RL train: {train_time:.2f}s success={rl_result['success_rate'] * 100:.1f}%",
        f"Q weights: min={rl_result['q_min']:.2f} max={rl_result['q_max']:.2f} mean={rl_result['q_mean']:.2f}",
        f"Replans: {local_result['replans']}",
        f"RL segments: {rl_result['segment_successes']}/{rl_result['segments_total']}",
    ]

    if rl_result["best_path"]:
        hist = terrain_histogram(rl_result["best_path"], terrain_map)
        lines.append("RL terrain cells:")
        for name, count in hist.items():
            if count > 0:
                lines.append(f"  {name}: {count}")
        if all(c == 0 for c in hist.values()):
            lines.append("  (none)")

    return lines


def save_rl_weights(scene_name, start, goal, q_table):
    filename = f"{scene_name}_qtable_{start[0]}_{start[1]}_{goal[0]}_{goal[1]}.npy"
    path = os.path.join(OUTPUT_DIR, filename)
    np.save(path, q_table)
    return path


def sample_waypoints(path, segment_span=36):
    if not path:
        return []
    waypoints = [path[0]]
    index = segment_span
    while index < len(path) - 1:
        waypoints.append(path[index])
        index += segment_span
    if waypoints[-1] != path[-1]:
        waypoints.append(path[-1])
    return waypoints


def guidance_segment(full_path, start, goal):
    if not full_path:
        return None
    try:
        start_idx = full_path.index(start)
        goal_idx = full_path.index(goal)
    except ValueError:
        return None
    if start_idx <= goal_idx:
        return full_path[start_idx:goal_idx + 1]
    return list(reversed(full_path[goal_idx:start_idx + 1]))


def train_segmented_q_learning(cost_map, start, goal, astar_path):
    if not astar_path:
        return {
            "best_path": None,
            "success_rate": 0.0,
            "q_table": np.zeros((*cost_map.shape, len(ACTIONS)), dtype=np.float32),
            "q_min": 0.0,
            "q_max": 0.0,
            "q_mean": 0.0,
            "segment_successes": 0,
            "segments_total": 0,
        }

    global_q = np.zeros((*cost_map.shape, len(ACTIONS)), dtype=np.float32)
    waypoints = sample_waypoints(astar_path)
    stitched_path = [waypoints[0]]
    segment_successes = 0

    for seg_start, seg_goal in zip(waypoints[:-1], waypoints[1:]):
        segment_guide = guidance_segment(astar_path, seg_start, seg_goal)
        segment_steps = max(180, int(GridPlanner.heuristic(seg_start, seg_goal) * 5))
        agent = QLearningNavigator(cost_map, seg_start, seg_goal, guidance_path=segment_guide)
        result = agent.train(episodes=max(60, QL_EPISODES // 2), max_steps=segment_steps)
        segment_path = result["best_path"] or segment_guide
        if result["best_path"]:
            segment_successes += 1
        if not segment_path:
            return {
                "best_path": None,
                "success_rate": 0.0,
                "q_table": global_q,
                "q_min": 0.0,
                "q_max": 0.0,
                "q_mean": 0.0,
                "segment_successes": segment_successes,
                "segments_total": len(waypoints) - 1,
            }

        mask = np.any(agent.q_table != 0.0, axis=2)
        global_q[mask] = agent.q_table[mask]
        stitched_path.extend(segment_path[1:])

    return {
        "best_path": stitched_path,
        "success_rate": segment_successes / max(1, len(waypoints) - 1),
        "q_table": global_q,
        "q_min": float(global_q.min()),
        "q_max": float(global_q.max()),
        "q_mean": float(global_q.mean()),
        "segment_successes": segment_successes,
        "segments_total": len(waypoints) - 1,
    }


def run_navigation(scene_name):
    cost_map = load_cost_map(scene_name)
    terrain_map = load_terrain_map(scene_name)
    if terrain_map is None:
        terrain_map = infer_terrain_from_cost(cost_map)
    display_map = load_display_map(scene_name)
    if display_map is None:
        display_map = build_display_map(terrain_map, cost_map)
    height, width = cost_map.shape
    start, goal = get_interactive_points(cost_map, display_map)

    planner = GridPlanner(height, width)
    print("Computing baselines...")
    dijkstra_path = planner.plan(cost_map, start, goal, use_heuristic=False)
    astar_path = planner.plan(cost_map, start, goal, use_heuristic=True)

    print("Running local-sensing A*...")
    local_navigator = LocalSensingNavigator(cost_map, start, goal)
    local_result = local_navigator.run()

    print("Training Q-learning policy...")
    train_started = time.perf_counter()
    rl_result = train_segmented_q_learning(cost_map, start, goal, astar_path)
    train_time = time.perf_counter() - train_started
    weights_path = save_rl_weights(scene_name, start, goal, rl_result["q_table"])

    print(f"Saved Q-table weights to: {weights_path}")
    if dijkstra_path:
        print("Dijkstra:", path_metrics(dijkstra_path, cost_map))
    if astar_path:
        print("A*:", path_metrics(astar_path, cost_map))
    if local_result["path"]:
        print("Local sensing:", path_metrics(local_result["path"], cost_map), "replans=", local_result["replans"])
    if rl_result["best_path"]:
        print("Q-learning:", path_metrics(rl_result["best_path"], cost_map))
    else:
        print("Q-learning did not reach the goal. Try a longer training run or easier points.")

    pygame.init()
    scale = compute_scale(height, width)
    map_surface = pygame.surfarray.make_surface(normalize_to_surface(cost_map, display_map).swapaxes(0, 1))
    map_surface = pygame.transform.scale(map_surface, (width * scale, height * scale))
    # Window height must accommodate both map and tall panel content
    window_height = max(height * scale, 620)
    screen = pygame.display.set_mode((width * scale + PANEL_WIDTH, window_height))
    pygame.display.set_caption("Sentinel-2 path planning and RL")

    metrics_lines = build_metrics_lines(
        dijkstra_path,
        astar_path,
        local_result,
        rl_result,
        cost_map,
        terrain_map,
        train_time,
    )

    robot_path = rl_result["best_path"] or astar_path or dijkstra_path or local_result["path"] or [start]
    robot_index = 0
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        robot_position = robot_path[min(robot_index, len(robot_path) - 1)]
        render_dashboard(
            screen,
            map_surface,
            scale,
            metrics_lines,
            start,
            goal,
            astar_path=astar_path,
            dijkstra_path=dijkstra_path,
            rl_path=rl_result["best_path"],
            local_path=local_result["path"],
            sensor_radius=LOCAL_SENSOR_RADIUS if local_result["path"] else 0,
            robot_position=robot_position,
        )
        pygame.display.flip()

        if robot_index < len(robot_path) - 1:
            robot_index += 1
        clock.tick(20)

    pygame.quit()


# ─────────────────────────────────────────────
#  WEBOTS EXPORT FUNCTION
# ─────────────────────────────────────────────

def export_webots_path(scene_name, path_cells, start, goal, display_map, cost_map, terrain_map):
    """
    Export a planned path and display texture for use in Webots.

    This function can be called after run_navigation() to save the computed
    path in a format that the Webots path_follower controller can read.

    Args:
        scene_name: Scene name (e.g., "Douro_Vineyards_512")
        path_cells: List of (row, col) tuples forming the path
        start: (row, col) start
        goal: (row, col) goal
        display_map: (H, W, 3) uint8 RGB display map
        cost_map: (H, W) float32 cost map
        terrain_map: (H, W) uint8 terrain classification

    Returns:
        Path to the exported JSON waypoint file.
    """
    import json
    from PIL import Image

    height, width = cost_map.shape
    cell_size = 10.0  # m per pixel

    webots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webots")
    textures_dir = os.path.join(webots_dir, "worlds", "textures")
    paths_dir = os.path.join(webots_dir, "paths")
    os.makedirs(textures_dir, exist_ok=True)
    os.makedirs(paths_dir, exist_ok=True)

    # ── Export display texture ──
    img = Image.fromarray(display_map, mode="RGB")
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    tex_path = os.path.join(textures_dir, f"{scene_name}_display.png")
    img.save(tex_path)
    img.save(os.path.join(textures_dir, "current_display.png"))
    print(f"  ✓  Exported display texture: {tex_path}")

    # ── Build waypoints ──
    def grid_to_world(row, col):
        x = col * cell_size - (width * cell_size / 2.0)
        z = row * cell_size - (height * cell_size / 2.0)
        return round(x, 2), round(z, 2)

    waypoints = []
    for i, (row, col) in enumerate(path_cells):
        wx, wz = grid_to_world(row, col)
        waypoints.append({
            "index": i,
            "row": int(row),
            "col": int(col),
            "x_m": wx,
            "z_m": wz,
        })

    # Metrics
    total_dist = 0.0
    total_cost = 0.0
    for a, b in zip(path_cells[:-1], path_cells[1:]):
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        step = math.sqrt(dr * dr + dc * dc)
        total_dist += step
        total_cost += float(cost_map[b]) * step

    data = {
        "meta": {
            "scene": scene_name,
            "map_height_px": height,
            "map_width_px": width,
            "cell_size_m": cell_size,
            "map_center_x_m": 0.0,
            "map_center_z_m": 0.0,
            "map_extent_min_x_m": -(width * cell_size / 2.0),
            "map_extent_min_z_m": -(height * cell_size / 2.0),
            "map_extent_max_x_m": width * cell_size / 2.0,
            "map_extent_max_z_m": height * cell_size / 2.0,
        },
        "start": {
            "row": int(start[0]),
            "col": int(start[1]),
            "x_m": grid_to_world(start[0], start[1])[0],
            "z_m": grid_to_world(start[0], start[1])[1],
        },
        "goal": {
            "row": int(goal[0]),
            "col": int(goal[1]),
            "x_m": grid_to_world(goal[0], goal[1])[0],
            "z_m": grid_to_world(goal[0], goal[1])[1],
        },
        "path": waypoints,
        "metrics": {
            "num_waypoints": len(waypoints),
            "distance_px": round(total_dist, 1),
            "distance_m": round(total_dist * cell_size, 1),
            "cumulative_cost": round(total_cost, 1),
        },
    }

    filename = f"{scene_name}_path_{start[0]}_{start[1]}_{goal[0]}_{goal[1]}.json"
    path_file = os.path.join(paths_dir, filename)
    with open(path_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓  Exported waypoints: {path_file}")
    print(f"  ℹ  Open webots/worlds/terrain_navigation.wbt in Webots to visualize.")
    return path_file


if __name__ == "__main__":
    run_navigation("Douro_Vineyards_512")
    #run_navigation("Porto_City_512")
    #run_navigation("Serra_Estrela_Mountain_512")
    #run_navigation("Alentejo_Savanna_512")
