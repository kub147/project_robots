"""
Webots controller for planner path following and line-following experiments.

The planner still exports the route as waypoints, but the visual route is now
expected to be drawn into the ground texture. This controller keeps the old
waypoint follower as the default and adds camera-based diagnostics, a heuristic
line follower, and a small tabular Q-learning mode for the next project stage.
"""

import glob
import json
import math
import os
import random
import sys

try:
    from controller import Camera, Supervisor
except ImportError:
    print("=" * 60)
    print("ERROR: Cannot import the Webots Python controller library.")
    print("=" * 60)
    print()
    print("Fix in Webots: Preferences -> Python command -> set to:")
    print(sys.executable)
    sys.exit(1)


TIME_STEP = 16
ROBOT_Z = 90.0
SPEED_MPS = 80.0
LINE_FOLLOW_SPEED_MPS = 28.0
WAYPOINT_THRESHOLD_M = 2.0
PATH_MARKER_STEP = 12
PATH_SEGMENT_WIDTH_M = 34.0
CAMERA_NAME = "path camera"
LINE_MIN_PIXELS = 8
LINE_LOST_LIMIT = 45
LOG_INTERVAL_STEPS = 30

MODE_WAYPOINT = "waypoint"
MODE_DIAGNOSTIC = "diagnostic"
MODE_HEURISTIC = "heuristic"
MODE_RL = "rl-follow"
MODES = {MODE_WAYPOINT, MODE_DIAGNOSTIC, MODE_HEURISTIC, MODE_RL}

ACTIONS = [
    ("hard_left", 1.10, 0.45),
    ("left", 0.52, 0.78),
    ("straight", 0.0, 1.0),
    ("right", -0.52, 0.78),
    ("hard_right", -1.10, 0.45),
    ("recover", -1.55, 0.25),
]
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_NAME = "line_follow_qtable.json"

PATHS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../paths")
)


def find_path_file(scene, start=None, goal=None):
    if not os.path.exists(PATHS_DIR):
        return None

    if scene and start and goal:
        exact = os.path.join(
            PATHS_DIR,
            f"{scene}_path_{start[0]}_{start[1]}_{goal[0]}_{goal[1]}.json",
        )
        if os.path.exists(exact):
            return exact

    pattern = f"{scene}_path_*.json" if scene else "*_path_*.json"
    matches = glob.glob(os.path.join(PATHS_DIR, pattern))
    if not matches:
        return None

    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def parse_args(args):
    config = {
        "scene": None,
        "start": None,
        "goal": None,
        "path_file": None,
        "mode": MODE_WAYPOINT,
        "debug_path_3d": False,
        "markers": False,
        "train": False,
        "episodes": 80,
        "episode_steps": 900,
        "model": os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME),
    }

    i = 0
    while i < len(args):
        if args[i] == "--scene" and i + 1 < len(args):
            config["scene"] = args[i + 1]
            i += 2
        elif args[i] == "--start" and i + 2 < len(args):
            config["start"] = (int(args[i + 1]), int(args[i + 2]))
            i += 3
        elif args[i] == "--goal" and i + 2 < len(args):
            config["goal"] = (int(args[i + 1]), int(args[i + 2]))
            i += 3
        elif args[i] == "--path" and i + 1 < len(args):
            config["path_file"] = args[i + 1]
            i += 2
        elif args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            if mode not in MODES:
                print(f"Unknown mode '{mode}', falling back to {MODE_WAYPOINT}.")
                mode = MODE_WAYPOINT
            config["mode"] = mode
            i += 2
        elif args[i] == "--debug-path-3d":
            config["debug_path_3d"] = True
            i += 1
        elif args[i] == "--markers":
            config["markers"] = True
            i += 1
        elif args[i] == "--train":
            config["train"] = True
            i += 1
        elif args[i] == "--episodes" and i + 1 < len(args):
            config["episodes"] = max(1, int(args[i + 1]))
            i += 2
        elif args[i] == "--episode-steps" and i + 1 < len(args):
            config["episode_steps"] = max(50, int(args[i + 1]))
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            config["model"] = args[i + 1]
            i += 2
        else:
            i += 1

    return config


def load_waypoints(path_file):
    with open(path_file) as f:
        data = json.load(f)

    waypoints = [(float(point["x_m"]), float(point["z_m"])) for point in data["path"]]
    if not waypoints:
        raise ValueError(f"No waypoints in {path_file}")

    return data, waypoints


def set_pose(translation_field, rotation_field, x, y, heading):
    translation_field.setSFVec3f([x, y, ROBOT_Z])
    rotation_field.setSFRotation([0.0, 0.0, 1.0, heading])


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def heading_between(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return -math.atan2(dx, dy)


def initial_heading(waypoints):
    if len(waypoints) < 2:
        return 0.0
    return heading_between(waypoints[0], waypoints[1])


def add_path_line(supervisor, waypoints):
    root_children = supervisor.getRoot().getField("children")
    segment_template = """
Solid {{
  name "planned_path_segment"
  translation {cx:.2f} {cy:.2f} 76.00
  rotation 0 0 1 {rotation:.6f}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor 1 0.74 0
        emissiveColor 0.55 0.34 0
        roughness 0.65
        metalness 0
      }}
      geometry Box {{
        size {width:.2f} {length:.2f} 10.00
      }}
    }}
  ]
}}
"""

    for start, end in zip(waypoints, waypoints[1:]):
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < 1.0:
            continue

        root_children.importMFNodeFromString(
            -1,
            segment_template.format(
                cx=(x1 + x2) / 2.0,
                cy=(y1 + y2) / 2.0,
                rotation=-math.atan2(dx, dy),
                width=PATH_SEGMENT_WIDTH_M,
                length=length + PATH_SEGMENT_WIDTH_M,
            ),
        )


def add_visual_markers(supervisor, waypoints):
    root_children = supervisor.getRoot().getField("children")
    marker_template = """
Solid {{
  translation {x:.2f} {y:.2f} {z:.2f}
  children [
    Shape {{
      appearance PBRAppearance {{
        baseColor {r:.2f} {g:.2f} {b:.2f}
        emissiveColor {er:.2f} {eg:.2f} {eb:.2f}
        roughness 0.4
      }}
      geometry Sphere {{
        radius {radius:.2f}
      }}
    }}
  ]
}}
"""

    sx, sy = waypoints[0]
    gx, gy = waypoints[-1]
    root_children.importMFNodeFromString(
        -1,
        marker_template.format(
            x=sx,
            y=sy,
            z=70.0,
            radius=75.0,
            r=0.0,
            g=0.85,
            b=0.18,
            er=0.0,
            eg=0.25,
            eb=0.04,
        ),
    )
    root_children.importMFNodeFromString(
        -1,
        marker_template.format(
            x=gx,
            y=gy,
            z=70.0,
            radius=75.0,
            r=1.0,
            g=0.08,
            b=0.04,
            er=0.35,
            eg=0.02,
            eb=0.0,
        ),
    )


def enable_camera(supervisor, timestep):
    try:
        camera = supervisor.getDevice(CAMERA_NAME)
    except Exception:
        camera = None

    if camera is None:
        print(f"WARNING: camera device '{CAMERA_NAME}' not found.")
        return None

    camera.enable(timestep)
    print(f"Camera enabled: {CAMERA_NAME} ({camera.getWidth()}x{camera.getHeight()})")
    return camera


def detect_line_features(camera):
    if camera is None:
        return {
            "visible": False,
            "offset": 0.0,
            "angle": 0.0,
            "coverage": 0.0,
            "pixels": 0,
        }

    image = camera.getImage()
    if not image:
        return {
            "visible": False,
            "offset": 0.0,
            "angle": 0.0,
            "coverage": 0.0,
            "pixels": 0,
        }

    width = camera.getWidth()
    height = camera.getHeight()
    xs = []
    ys = []

    for y in range(height):
        for x in range(width):
            red = Camera.imageGetRed(image, width, x, y)
            green = Camera.imageGetGreen(image, width, x, y)
            blue = Camera.imageGetBlue(image, width, x, y)
            if red > 150 and blue > 140 and green < 105 and red + blue > 2 * green + 150:
                xs.append(x)
                ys.append(y)

    count = len(xs)
    if count < LINE_MIN_PIXELS:
        return {
            "visible": False,
            "offset": 0.0,
            "angle": 0.0,
            "coverage": count / float(width * height),
            "pixels": count,
        }

    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    offset = (mean_x - (width - 1) / 2.0) / max(1.0, (width - 1) / 2.0)

    var_x = sum((x - mean_x) ** 2 for x in xs) / count
    var_y = sum((y - mean_y) ** 2 for y in ys) / count
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / count
    axis_angle = 0.5 * math.atan2(2.0 * cov_xy, var_x - var_y)
    angle = normalize_angle(axis_angle - math.pi / 2.0)
    if angle > math.pi / 2.0:
        angle -= math.pi
    elif angle < -math.pi / 2.0:
        angle += math.pi

    return {
        "visible": True,
        "offset": max(-1.0, min(1.0, offset)),
        "angle": max(-1.0, min(1.0, angle / (math.pi / 2.0))),
        "coverage": count / float(width * height),
        "pixels": count,
    }


def nearest_path_progress(x, y, waypoints):
    best_index = 0
    best_dist = float("inf")
    for index, (px, py) in enumerate(waypoints):
        dist = (px - x) ** 2 + (py - y) ** 2
        if dist < best_dist:
            best_index = index
            best_dist = dist
    return best_index / max(1, len(waypoints) - 1), math.sqrt(best_dist)


def apply_motion(x, y, heading, features, action_idx, timestep, heuristic=False, last_offset=0.0):
    dt = timestep / 1000.0
    if heuristic:
        if features["visible"]:
            angular = -0.36 * features["offset"]
            speed_scale = 0.62 - 0.32 * min(1.0, abs(features["offset"]))
        else:
            search_direction = -1.0 if last_offset < 0.0 else 1.0
            angular = 0.58 * search_direction
            speed_scale = 0.0
    else:
        angular, speed_scale = ACTIONS[action_idx][1], ACTIONS[action_idx][2]
        if action_idx == len(ACTIONS) - 1 and features["visible"] and features["offset"] < 0:
            angular = -angular

    heading = normalize_angle(heading + angular * dt)
    speed = LINE_FOLLOW_SPEED_MPS * speed_scale
    x -= math.sin(heading) * speed * dt
    y += math.cos(heading) * speed * dt
    return x, y, heading


def state_from_features(features, progress_delta, previous_action):
    if not features["visible"]:
        return f"lost:{previous_action}"

    offset_bin = int(max(0, min(6, math.floor((features["offset"] + 1.0) * 3.5))))
    angle_bin = int(max(0, min(6, math.floor((features["angle"] + 1.0) * 3.5))))
    progress_bin = 1 if progress_delta > 0.0005 else 0
    return f"{offset_bin}:{angle_bin}:{progress_bin}:{previous_action}"


def q_values(q_table, state):
    if state not in q_table:
        q_table[state] = [0.0 for _ in ACTIONS]
    return q_table[state]


def choose_action(q_table, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    values = q_values(q_table, state)
    best = max(values)
    candidates = [index for index, value in enumerate(values) if value == best]
    return random.choice(candidates)


def load_q_table(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {key: [float(value) for value in values] for key, values in data.items()}


def save_q_table(path, q_table):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(q_table, f, indent=2, sort_keys=True)


def run_waypoint_loop(supervisor, timestep, translation_field, rotation_field, waypoints, camera, diagnostic=False):
    x, y = waypoints[0]
    target_index = 1 if len(waypoints) > 1 else 0
    heading = initial_heading(waypoints)
    step_count = 0
    set_pose(translation_field, rotation_field, x, y, heading)

    while supervisor.step(timestep) != -1:
        if diagnostic and step_count % LOG_INTERVAL_STEPS == 0:
            features = detect_line_features(camera)
            print(
                "camera "
                f"visible={features['visible']} "
                f"offset={features['offset']:.3f} "
                f"angle={features['angle']:.3f} "
                f"coverage={features['coverage']:.4f} "
                f"pixels={features['pixels']}"
            )
        step_count += 1

        if target_index >= len(waypoints):
            return

        tx, ty = waypoints[target_index]
        dx = tx - x
        dy = ty - y
        distance = math.hypot(dx, dy)

        if distance <= WAYPOINT_THRESHOLD_M:
            target_index += 1
            if target_index >= len(waypoints):
                x, y = waypoints[-1]
                set_pose(translation_field, rotation_field, x, y, heading)
                print("Goal reached.")
                return
            continue

        step_distance = SPEED_MPS * (timestep / 1000.0)
        travel = min(step_distance, distance)
        ux = dx / distance
        uy = dy / distance
        x += ux * travel
        y += uy * travel
        heading = -math.atan2(ux, uy)
        set_pose(translation_field, rotation_field, x, y, heading)


def run_heuristic_loop(supervisor, timestep, translation_field, rotation_field, waypoints, camera):
    x, y = waypoints[0]
    heading = initial_heading(waypoints)
    lost_steps = 0
    step_count = 0
    last_offset = 0.0
    set_pose(translation_field, rotation_field, x, y, heading)

    while supervisor.step(timestep) != -1:
        features = detect_line_features(camera)
        if features["visible"]:
            lost_steps = 0
            last_offset = features["offset"]
        else:
            lost_steps += 1

        x, y, heading = apply_motion(x, y, heading, features, 0, timestep, heuristic=True, last_offset=last_offset)
        set_pose(translation_field, rotation_field, x, y, heading)

        progress, distance = nearest_path_progress(x, y, waypoints)
        if step_count % LOG_INTERVAL_STEPS == 0:
            print(
                "heuristic "
                f"progress={progress:.3f} "
                f"dist={distance:.1f}m "
                f"visible={features['visible']} "
                f"offset={features['offset']:.3f} "
                f"angle={features['angle']:.3f}"
            )
        if progress >= 0.995 or lost_steps > LINE_LOST_LIMIT * 3:
            print(f"Heuristic finished: progress={progress:.3f}, lost_steps={lost_steps}")
            return
        step_count += 1


def reward_from_features(features, progress_delta, distance_to_path, lost_steps):
    if not features["visible"]:
        return -3.0 - 0.04 * lost_steps

    reward = 1.0
    reward += 2.4 * max(0.0, progress_delta * 100.0)
    reward += 1.1 * (1.0 - abs(features["offset"]))
    reward += 0.6 * (1.0 - abs(features["angle"]))
    reward -= min(3.0, distance_to_path / 140.0)
    return reward


def run_rl_training(supervisor, timestep, translation_field, rotation_field, waypoints, camera, config):
    q_table = load_q_table(config["model"])
    alpha = 0.22
    gamma = 0.92
    epsilon_start = 0.45
    epsilon_end = 0.06

    print(
        f"Training RL line follower: episodes={config['episodes']} "
        f"steps={config['episode_steps']} model={config['model']}"
    )

    for episode in range(config["episodes"]):
        x, y = waypoints[0]
        heading = initial_heading(waypoints)
        previous_action = 2
        previous_progress = 0.0
        total_reward = 0.0
        lost_steps = 0
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (episode / max(1, config["episodes"] - 1))
        set_pose(translation_field, rotation_field, x, y, heading)

        for step in range(config["episode_steps"]):
            if supervisor.step(timestep) == -1:
                save_q_table(config["model"], q_table)
                return

            features = detect_line_features(camera)
            state = state_from_features(features, 0.0, previous_action)
            action = choose_action(q_table, state, epsilon)
            x, y, heading = apply_motion(x, y, heading, features, action, timestep)
            set_pose(translation_field, rotation_field, x, y, heading)

            progress, distance = nearest_path_progress(x, y, waypoints)
            progress_delta = progress - previous_progress
            next_features = detect_line_features(camera)
            if next_features["visible"]:
                lost_steps = 0
            else:
                lost_steps += 1

            reward = reward_from_features(next_features, progress_delta, distance, lost_steps)
            next_state = state_from_features(next_features, progress_delta, action)
            values = q_values(q_table, state)
            next_values = q_values(q_table, next_state)
            values[action] += alpha * (reward + gamma * max(next_values) - values[action])

            total_reward += reward
            previous_action = action
            previous_progress = progress
            if progress >= 0.995 or lost_steps > LINE_LOST_LIMIT:
                break

        print(
            f"episode={episode + 1}/{config['episodes']} "
            f"reward={total_reward:.1f} progress={previous_progress:.3f} "
            f"states={len(q_table)} epsilon={epsilon:.2f}"
        )

    save_q_table(config["model"], q_table)
    print(f"Saved RL policy: {config['model']}")


def run_rl_follow_loop(supervisor, timestep, translation_field, rotation_field, waypoints, camera, config):
    q_table = load_q_table(config["model"])
    if not q_table:
        print("No RL policy found; using heuristic line follower instead.")
        run_heuristic_loop(supervisor, timestep, translation_field, rotation_field, waypoints, camera)
        return

    x, y = waypoints[0]
    heading = initial_heading(waypoints)
    previous_action = 2
    previous_progress = 0.0
    step_count = 0
    lost_steps = 0
    set_pose(translation_field, rotation_field, x, y, heading)

    while supervisor.step(timestep) != -1:
        features = detect_line_features(camera)
        state = state_from_features(features, 0.0, previous_action)
        action = choose_action(q_table, state, 0.0)
        x, y, heading = apply_motion(x, y, heading, features, action, timestep)
        set_pose(translation_field, rotation_field, x, y, heading)

        progress, distance = nearest_path_progress(x, y, waypoints)
        progress_delta = progress - previous_progress
        previous_action = action
        previous_progress = progress
        if features["visible"]:
            lost_steps = 0
        else:
            lost_steps += 1

        if step_count % LOG_INTERVAL_STEPS == 0:
            print(
                "rl "
                f"action={ACTIONS[action][0]} "
                f"progress={progress:.3f} "
                f"delta={progress_delta:.4f} "
                f"dist={distance:.1f}m "
                f"visible={features['visible']} "
                f"offset={features['offset']:.3f}"
            )

        if progress >= 0.995 or lost_steps > LINE_LOST_LIMIT:
            print(f"RL follow finished: progress={progress:.3f}, lost_steps={lost_steps}")
            return
        step_count += 1


def main():
    config = parse_args(sys.argv[1:])

    path_file = config["path_file"]
    if path_file is None:
        path_file = find_path_file(config["scene"], config["start"], config["goal"])
    if not path_file or not os.path.exists(path_file):
        print("ERROR: no path file found. Run webots/scripts/export_webots.py first.")
        return

    data, waypoints = load_waypoints(path_file)

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep()) or TIME_STEP
    robot_node = supervisor.getSelf()
    translation_field = robot_node.getField("translation")
    rotation_field = robot_node.getField("rotation")
    camera = enable_camera(supervisor, timestep)

    if config["debug_path_3d"]:
        add_path_line(supervisor, waypoints)
    if config["markers"]:
        add_visual_markers(supervisor, waypoints)

    print("=" * 60)
    print(f"Scene: {data['meta']['scene']}")
    print(f"Path file: {path_file}")
    print(f"Mode: {config['mode']}")
    print(f"Waypoints: {len(waypoints)}")
    print(f"Distance: {data['metrics']['distance_m']:.0f} m")
    print(f"Start: ({waypoints[0][0]:.0f}, {waypoints[0][1]:.0f})")
    print(f"Goal:  ({waypoints[-1][0]:.0f}, {waypoints[-1][1]:.0f})")
    print("=" * 60)

    if config["mode"] == MODE_DIAGNOSTIC:
        run_waypoint_loop(supervisor, timestep, translation_field, rotation_field, waypoints, camera, diagnostic=True)
    elif config["mode"] == MODE_HEURISTIC:
        run_heuristic_loop(supervisor, timestep, translation_field, rotation_field, waypoints, camera)
    elif config["mode"] == MODE_RL:
        if config["train"]:
            run_rl_training(supervisor, timestep, translation_field, rotation_field, waypoints, camera, config)
        run_rl_follow_loop(supervisor, timestep, translation_field, rotation_field, waypoints, camera, config)
    else:
        run_waypoint_loop(supervisor, timestep, translation_field, rotation_field, waypoints, camera, diagnostic=False)


if __name__ == "__main__":
    main()
