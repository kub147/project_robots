"""
Webots controller that follows exported planner waypoints.

The current world uses a simple visual Robot node on a top-down terrain map,
so this controller moves that node kinematically with the Supervisor API. This
keeps the Webots demo aligned with the Python planner without requiring a
physical wheel model yet.
"""

import glob
import json
import math
import os
import sys

try:
    from controller import Supervisor
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
WAYPOINT_THRESHOLD_M = 2.0
PATH_MARKER_STEP = 12
PATH_SEGMENT_WIDTH_M = 34.0

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
    scene = None
    start = None
    goal = None
    path_file = None

    i = 0
    while i < len(args):
        if args[i] == "--scene" and i + 1 < len(args):
            scene = args[i + 1]
            i += 2
        elif args[i] == "--start" and i + 2 < len(args):
            start = (int(args[i + 1]), int(args[i + 2]))
            i += 3
        elif args[i] == "--goal" and i + 2 < len(args):
            goal = (int(args[i + 1]), int(args[i + 2]))
            i += 3
        elif args[i] == "--path" and i + 1 < len(args):
            path_file = args[i + 1]
            i += 2
        else:
            i += 1

    return scene, start, goal, path_file


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

    for index, (x, y) in enumerate(waypoints):
        if index % PATH_MARKER_STEP != 0 and index != len(waypoints) - 1:
            continue
        root_children.importMFNodeFromString(
            -1,
            marker_template.format(
                x=x,
                y=y,
                z=55.0,
                radius=18.0,
                r=1.0,
                g=0.86,
                b=0.05,
                er=0.35,
                eg=0.25,
                eb=0.0,
            ),
        )

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


def main():
    scene, start, goal, path_file = parse_args(sys.argv[1:])

    if path_file is None:
        path_file = find_path_file(scene, start, goal)
    if not path_file or not os.path.exists(path_file):
        print("ERROR: no path file found. Run webots/scripts/export_webots.py first.")
        return

    data, waypoints = load_waypoints(path_file)

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep()) or TIME_STEP
    robot_node = supervisor.getSelf()
    translation_field = robot_node.getField("translation")
    rotation_field = robot_node.getField("rotation")

    x, y = waypoints[0]
    target_index = 1 if len(waypoints) > 1 else 0
    heading = 0.0
    set_pose(translation_field, rotation_field, x, y, heading)
    add_path_line(supervisor, waypoints)
    add_visual_markers(supervisor, waypoints)

    print("=" * 60)
    print(f"Scene: {data['meta']['scene']}")
    print(f"Path file: {path_file}")
    print(f"Waypoints: {len(waypoints)}")
    print(f"Distance: {data['metrics']['distance_m']:.0f} m")
    print(f"Start: ({waypoints[0][0]:.0f}, {waypoints[0][1]:.0f})")
    print(f"Goal:  ({waypoints[-1][0]:.0f}, {waypoints[-1][1]:.0f})")
    print("=" * 60)

    while supervisor.step(timestep) != -1:
        if target_index >= len(waypoints):
            continue

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
            continue

        step_distance = SPEED_MPS * (timestep / 1000.0)
        travel = min(step_distance, distance)
        ux = dx / distance
        uy = dy / distance
        x += ux * travel
        y += uy * travel
        heading = -math.atan2(ux, uy)
        set_pose(translation_field, rotation_field, x, y, heading)


if __name__ == "__main__":
    main()
