"""
path_follower.py — Webots robot controller for GPS + Compass waypoint following.
"""

import os
import math
import json
import glob

try:
    from controller import Robot, GPS, Compass, Gyro, Motor
except ImportError:
    import sys
    print("=" * 60)
    print("  ERROR: Cannot import Webots Python controller library.")
    print("=" * 60)
    print()
    print("  Fix: Webots → Preferences → Python command → set to:")
    import sys as _sys
    print(f"  {_sys.executable}")
    print()
    _sys.exit(1)

# ── Configuration ──
TIME_STEP = 16
WHEEL_RADIUS = 0.12
WHEEL_BASE = 0.40         # distance between wheels (x=-0.20 to x=0.20)
MAX_MOTOR_VEL = 20.0
ANGULAR_GAIN = 3.0
MIN_SPEED = 0.5
MAX_SPEED = 4.0
WP_THRESHOLD = 3.0        # waypoint reach threshold (m)
GOAL_THRESHOLD = 2.0
LOOKAHEAD = 15.0

PATHS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../paths")
)


def normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def heading_from_compass(values):
    """Return heading in radians. 0 = +Z (North), +pi/2 = +X (East)."""
    return math.atan2(values[0], values[2])


def find_path_file(scene, start=None, goal=None):
    cdir = os.path.dirname(os.path.abspath(__file__))
    pdir = os.path.normpath(os.path.join(cdir, "../../paths"))
    if not os.path.exists(pdir):
        return None
    pattern = f"{scene}_path_*.json"
    matches = glob.glob(os.path.join(pdir, pattern))
    if not matches:
        return None
    if start and goal:
        exact = os.path.join(pdir, f"{scene}_path_{start[0]}_{start[1]}_{goal[0]}_{goal[1]}.json")
        if os.path.exists(exact):
            return exact
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def compute_commands(target_heading, current_heading, distance):
    """
    Differential drive control law.
    Positive ω (right-left difference) = turn LEFT (CCW).
    When target is to the RIGHT of heading: angle_error > 0.
    To turn RIGHT: need ω < 0  →  right < left  →  left > right.
    """
    error = normalize_angle(target_heading - current_heading)
    
    # Speed ramp
    frac = min(1.0, distance / LOOKAHEAD)
    speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * frac
    
    # Angular correction (rad/s)
    angular = ANGULAR_GAIN * error
    
    # Linear wheel velocities in m/s
    left_lin = speed + angular * WHEEL_BASE / 2.0
    right_lin = speed - angular * WHEEL_BASE / 2.0
    
    # Convert to rad/s (motor velocity)
    left = left_lin / WHEEL_RADIUS
    right = right_lin / WHEEL_RADIUS
    
    # Clamp
    for v in [left, right]:
        if abs(v) > MAX_MOTOR_VEL:
            scale = MAX_MOTOR_VEL / abs(v)
            left *= scale
            right *= scale
    
    return left, right, math.degrees(error)


def main():
    import sys
    args = sys.argv[1:]
    scene = None
    start = None
    goal = None
    path_file = None
    
    i = 0
    while i < len(args):
        if args[i] == "--scene" and i + 1 < len(args):
            scene = args[i + 1]; i += 2
        elif args[i] == "--start" and i + 2 < len(args):
            start = (int(args[i+1]), int(args[i+2])); i += 3
        elif args[i] == "--goal" and i + 2 < len(args):
            goal = (int(args[i+1]), int(args[i+2])); i += 3
        elif args[i] == "--path" and i + 1 < len(args):
            path_file = args[i + 1]; i += 2
        else:
            i += 1

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    # Get devices
    gps = robot.getDevice("gps"); gps.enable(timestep)
    compass = robot.getDevice("compass"); compass.enable(timestep)
    gyro = robot.getDevice("gyro"); gyro.enable(timestep)
    left_motor = robot.getDevice("left_motor")
    right_motor = robot.getDevice("right_motor")
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)
    
    # Load path
    if path_file is None and scene:
        path_file = find_path_file(scene, start, goal)
    if not path_file or not os.path.exists(path_file):
        print(f"ERROR: no path file found. Run export_webots.py first.")
        return
    
    with open(path_file) as f:
        data = json.load(f)
    waypoints = data["path"]
    total_wp = len(waypoints)
    
    xs = [wp["x_m"] for wp in waypoints]
    zs = [wp["z_m"] for wp in waypoints]
    
    print(f"\n{'='*60}")
    print(f"  Scene: {data['meta']['scene']}")
    print(f"  Waypoints: {total_wp}")
    print(f"  Distance: {data['metrics']['distance_m']:.0f} m")
    print(f"  Start: ({xs[0]:.0f}, {zs[0]:.0f})")
    print(f"  Goal:  ({xs[-1]:.0f}, {zs[-1]:.0f})")
    print(f"{'='*60}\n")
    
    # Settle
    for _ in range(5):
        robot.step(timestep)
    
    # ── DIAGNOSTIC: print raw compass at start ──
    cv = compass.getValues()
    gv = gps.getValues()
    h = heading_from_compass(cv)
    print(f"  [DIAG] GPS: ({gv[0]:.1f}, {gv[1]:.1f}, {gv[2]:.1f})")
    print(f"  [DIAG] Raw compass: ({cv[0]:.4f}, {cv[1]:.4f}, {cv[2]:.4f})")
    print(f"  [DIAG] Heading: {math.degrees(h):.0f} deg (0=+Z, +90=+X)")
    print()
    
    # Navigation loop
    current_wp = 0
    step = 0
    heading = h
    
    while robot.step(timestep) != -1:
        step += 1
        
        gv = gps.getValues()
        cv = compass.getValues()
        heading = heading_from_compass(cv)
        
        rx, rz = gv[0], gv[2]
        
        # Check if we've reached all waypoints
        if current_wp >= total_wp:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
            if step % 50 == 0:
                print(f"  ✓ GOAL REACHED! Stopped.")
            continue
        
        # Current target waypoint
        tx, tz = xs[current_wp], zs[current_wp]
        dx = tx - rx
        dz = tz - rz
        dist = math.hypot(dx, dz)
        
        # Check arrival
        threshold = GOAL_THRESHOLD if current_wp == total_wp - 1 else WP_THRESHOLD
        if dist < threshold:
            if current_wp == total_wp - 1:
                print(f"  ✓ GOAL REACHED at ({rx:.0f}, {rz:.0f})")
                current_wp += 1
                continue
            else:
                old = current_wp
                current_wp += 1
                if step % 20 == 0:
                    print(f"  ✓ WP {old+1}/{total_wp} done → WP {current_wp+1}/{total_wp}")
                continue
        
        # Compute target heading
        target_h = math.atan2(dx, dz)
        
        # Get motor commands
        left_v, right_v, err_deg = compute_commands(target_h, heading, dist)
        
        left_motor.setVelocity(left_v)
        right_motor.setVelocity(right_v)
        
        # Status
        if step % 50 == 0:
            print(f"  [{step:5d}] WP {current_wp+1}/{total_wp} | "
                  f"pos=({rx:.0f},{rz:.0f}) target=({tx:.0f},{tz:.0f}) "
                  f"dist={dist:.0f}m heading={math.degrees(heading):.0f}° "
                  f"err={err_deg:.0f}° L={left_v:.1f} R={right_v:.1f}")
    
    print("Simulation ended.")


if __name__ == "__main__":
    main()
