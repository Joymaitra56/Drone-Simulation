import os
import sys
import time
import math
import json
import heapq
import csv
import io
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import mujoco  
# pyrefly: ignore [missing-import]
import mujoco.viewer
# pyrefly: ignore [missing-import]
from PIL import Image, ImageDraw
from sensor_simulation import SensorSimulationSuite

# Advanced Research Modules
from terrain_scanner import TerrainScanner
from dynamic_obstacles import DynamicObstacleManager
from subsumption import (
    BehaviorManager, KillSwitchBehavior, EmergencyLandingBehavior,
    ObstacleAvoidanceBehavior, NavigationBehavior, TerrainScanningBehavior,
    ReturnHomeBehavior,
)
from mission_manager import MissionManager
from mission_state import MissionState
from sensor_fusion import SensorFusion
from occupancy_grid_mapping import OccupancyGridMapping
from coverage_metrics import CoverageMetrics
from frontier_detection import FrontierDetection
from frontier_ranking import FrontierRanking
from imu_trajectory import TrajectoryRecorder, TrajectoryReplayer
from post_mission_analysis import PostMissionAnalysis


# ============================================================
#  Pre-allocated / shared constants (computed once at import)
# ============================================================

_PADS = [
    ("Home",  -4.0, -4.0, 0.6),
    ("Pad 1", -4.0,  4.0, 0.4),
    ("Pad 2",  4.0,  4.0, 0.4),
    ("Pad 3",  4.0, -4.0, 0.4),
    ("Pad 4",  0.0,  2.0, 0.4),
    ("Pad 5",  0.0, -2.0, 0.4),
]
_PADS_DICT = {name: (px, py) for name, px, py, _ in _PADS}

_ZERO3      = np.zeros(3, dtype=np.float64)
_EYE3_FLAT  = np.eye(3, dtype=np.float32).flatten()
_GEOM_COLOR = np.array([0.0, 1.0, 0.5, 0.8], dtype=np.float32)


# ============================================================
#  Terrain height
# ============================================================

def get_terrain_height(x, y):
    """Terrain height at world (x,y) — pure scalar math, no allocation."""
    h_val  = 120.0
    h_val += 60.0 * math.sin(0.4 * x)  * math.cos(0.4 * y)
    h_val += 30.0 * math.sin(1.1 * x + 0.5) * math.cos(0.9 * y - 0.2)
    h_val += 15.0 * math.sin(2.5 * x)  * math.sin(2.8 * y)
    h_val +=  5.0 * math.sin(6.0 * x)  * math.cos(5.5 * y)
    flatten_factor = 1.0
    for _, px, py, pr in _PADS:
        dx = x - px;  dy = y - py
        d = math.sqrt(dx*dx + dy*dy)
        if d < pr:
            flatten_factor = 0.0;  break
        elif d < pr + 0.4:
            t = (d - pr) / 0.4
            f = 3.0*t*t - 2.0*t*t*t
            if f < flatten_factor:
                flatten_factor = f
    return 0.4 * (h_val * flatten_factor / 255.0)


# ============================================================
#  A* Pathfinder  (replaces Dijkstra for user-commanded flights)
# ============================================================

def run_astar(grid, start, goal):
    """
    A* search on the occupancy grid.
    start / goal : (col, row) integer grid coordinates.
    Returns (path_list_of_(col,row), cost) or (None, inf).
    """
    height, width = grid.shape
    dirs = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
            (1,1,1.414),(1,-1,1.414),(-1,1,1.414),(-1,-1,1.414)]

    def heuristic(a, b):
        # Octile distance — admissible for 8-connected grid
        dx = abs(a[0]-b[0]);  dy = abs(a[1]-b[1])
        return max(dx, dy) + (1.414 - 1.0) * min(dx, dy)

    g_score = {start: 0.0}
    f_score = {start: heuristic(start, goal)}
    open_q  = [(f_score[start], start)]
    parent  = {}
    closed  = set()

    while open_q:
        _, cur = heapq.heappop(open_q)
        if cur == goal:
            path = [];  node = goal
            while node != start:
                path.append(node);  node = parent[node]
            path.append(start);  path.reverse()
            return path, g_score[goal]
        if cur in closed:
            continue
        closed.add(cur)
        cx, cy = cur
        for ddx, ddy, sc in dirs:
            nx, ny = cx+ddx, cy+ddy
            nb = (nx, ny)
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if grid[ny, nx] == 1 or nb in closed:
                continue
            # Obstacle inflation cost (same as Dijkstra version)
            near = any(
                0 <= nx+ox < width and 0 <= ny+oy < height and grid[ny+oy, nx+ox] == 1
                for ox in (-1,0,1) for oy in (-1,0,1)
            )
            tentative_g = g_score[cur] + sc + (3.0 if near else 0.0)
            if tentative_g < g_score.get(nb, 1e18):
                g_score[nb] = tentative_g
                f_score[nb] = tentative_g + heuristic(nb, goal)
                parent[nb]  = cur
                heapq.heappush(open_q, (f_score[nb], nb))

    return None, float('inf')


# ============================================================
#  Physics-based Drone Force Controller  (unchanged)
# ============================================================

class DronePhysicsController:
    """
    Purely MuJoCo-physics force controller.
    Writes to data.xfrc_applied[body_id] every step.
    No qpos manipulation whatsoever.
    """
    # Altitude PID
    KP_Z  = 28.0;  KI_Z = 4.5;  KD_Z = 12.0
    # Horizontal PD
    KP_XY = 18.0;  KD_XY = 9.0
    # Attitude
    KP_ATT = 6.0;  KD_ATT = 2.5
    # Yaw rate
    KP_YAW = 3.0
    # Drag
    K_DRAG_LIN = 1.8;  K_DRAG_ROT = 0.6
    # Limits
    MAX_LATERAL_FORCE = 30.0;  INT_CLAMP = 8.0

    def __init__(self, model, data, body_id):
        self.model   = model
        self.data    = data
        self.body_id = body_id
        total_mass   = 0.0
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == body_id:
                try:   total_mass += model.geom_mass[g]
                except: pass
        if total_mass < 0.01: total_mass = model.body_mass[body_id]
        if total_mass < 0.01: total_mass = 1.0
        self.mass      = float(total_mass)
        self.gravity   = float(abs(model.opt.gravity[2]))
        self._hover_ff = self.mass * self.gravity
        self._iz       = 0.0
        self._prev_ez  = 0.0
        print(f"[PHYSICS] mass={self.mass:.3f} kg  hover_thrust={self._hover_ff:.2f} N")

    def _read_state(self):
        cvel = self.data.cvel[self.body_id]
        return (self.data.xpos[self.body_id], cvel[3:], cvel[:3],
                self.data.xquat[self.body_id])

    @staticmethod
    def _quat_rpy(q):
        w,x,y,z = float(q[0]),float(q[1]),float(q[2]),float(q[3])
        roll  = math.atan2(2.0*(w*x+y*z), 1.0-2.0*(x*x+y*y))
        sinp  = 2.0*(w*y-z*x)
        sinp  = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)
        yaw   = math.atan2(2.0*(w*z+x*y), 1.0-2.0*(y*y+z*z))
        return roll, pitch, yaw

    def apply_forces(self, tx, ty, tz, target_yaw=None,
                     target_vx=0.0, target_vy=0.0, dt=0.005, mode="position"):
        pos, vel, omega, quat = self._read_state()
        # Altitude PID
        ez  = tz - float(pos[2])
        if self._prev_ez == 0.0:
            self._prev_ez = ez
        dez = (ez - self._prev_ez) / dt
        iz  = max(-self.INT_CLAMP, min(self.INT_CLAMP, self._iz + ez*dt))
        self._iz = iz;  self._prev_ez = ez
        f_z = self.KP_Z*ez + self.KI_Z*iz + self.KD_Z*dez + self._hover_ff
        # Horizontal PD
        vx = float(vel[0]);  vy = float(vel[1]);  vz = float(vel[2])
        if mode == "velocity":
            f_x = self.KP_XY*(target_vx - vx)*self.mass
            f_y = self.KP_XY*(target_vy - vy)*self.mass
        else:
            ex  = tx - float(pos[0]);  ey = ty - float(pos[1])
            f_x = (self.KP_XY*ex - self.KD_XY*vx)*self.mass
            f_y = (self.KP_XY*ey - self.KD_XY*vy)*self.mass
        f_x = max(-self.MAX_LATERAL_FORCE, min(self.MAX_LATERAL_FORCE, f_x))
        f_y = max(-self.MAX_LATERAL_FORCE, min(self.MAX_LATERAL_FORCE, f_y))
        # Drag
        kd  = self.K_DRAG_LIN
        f_x -= kd*vx;  f_y -= kd*vy;  f_z -= kd*vz
        f_z = max(0.0, min(50.0, f_z))
        # Attitude torques
        roll, pitch, cur_yaw = self._quat_rpy(quat)
        ox = float(omega[0]);  oy = float(omega[1]);  oz = float(omega[2])
        kr = self.K_DRAG_ROT
        t_roll  = -self.KP_ATT*roll  - (self.KD_ATT+kr)*ox
        t_pitch = -self.KP_ATT*pitch - (self.KD_ATT+kr)*oy
        if target_yaw is None:
            t_yaw = -(self.KD_ATT+kr)*oz
        else:
            ye = target_yaw - cur_yaw
            if ye >  math.pi: ye -= 2.0*math.pi
            if ye < -math.pi: ye += 2.0*math.pi
            t_yaw = self.KP_YAW*ye - (self.KD_ATT+kr)*oz
        xf = self.data.xfrc_applied
        b  = self.body_id
        xf[b,0]=f_x; xf[b,1]=f_y; xf[b,2]=f_z
        xf[b,3]=t_roll; xf[b,4]=t_pitch; xf[b,5]=t_yaw

    def apply_scan_spin(self, rate, dt=0.005):
        pos, vel, omega, quat = self._read_state()
        vx=float(vel[0]); vy=float(vel[1]); vz=float(vel[2])
        ox=float(omega[0]); oy=float(omega[1]); oz=float(omega[2])
        f_z = self._hover_ff - self.K_DRAG_LIN*vz
        f_x = -self.KP_XY*vx*self.mass - self.K_DRAG_LIN*vx
        f_y = -self.KP_XY*vy*self.mass - self.K_DRAG_LIN*vy
        roll, pitch, _ = self._quat_rpy(quat)
        kr = self.K_DRAG_ROT
        t_roll  = -self.KP_ATT*roll  - (self.KD_ATT+kr)*ox
        t_pitch = -self.KP_ATT*pitch - (self.KD_ATT+kr)*oy
        t_yaw   =  self.KP_YAW*(rate - oz) - kr*oz
        xf=self.data.xfrc_applied; b=self.body_id
        xf[b,0]=f_x; xf[b,1]=f_y; xf[b,2]=f_z
        xf[b,3]=t_roll; xf[b,4]=t_pitch; xf[b,5]=t_yaw

    def clear_forces(self):
        self.data.xfrc_applied[self.body_id, :] = 0.0

    def reset_integrators(self):
        self._iz = 0.0;  self._prev_ez = 0.0


# ============================================================
#  DroneAutopilot  –  User-Commanded + A* navigation
# ============================================================

class DroneAutopilot:
    """
    State machine that drives the drone purely via physics forces.
    The user commands a TARGET pad (Home / Pad 1 … Pad 5).
    The drone plans an A* path on the occupancy grid, takes off,
    follows that path using the physics PID controller, and lands
    at the requested pad.  No automatic visiting sequence.
    """

    def __init__(self, model, data):
        self.model = model
        self.data  = data

        try:
            self.joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drone_joint")
            self.qpos_adr = model.jnt_qposadr[self.joint_id]
            self.qvel_adr = model.jnt_dofadr[self.joint_id]
        except Exception as e:
            print(f"Error: drone_joint not found: {e}");  sys.exit(1)

        try:
            self.drone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
        except Exception as e:
            print(f"Error: drone body not found: {e}");  sys.exit(1)

        self.physics = DronePhysicsController(model, data, self.drone_body_id)

        # ── Flight parameters ──────────────────────────────────────────────
        self.hover_height   = 1.5
        self.resting_height = 0.12
        self.cruise_speed   = 1.4    # m/s  – physics-driven, not teleported
        self.home_x         = -4.0
        self.home_y         = -4.0

        # ── State machine ──────────────────────────────────────────────────
        # States: LANDED | TAKEOFF | HOVER | SCANNING | GOTO_TARGET |
        #         AUTOLAND | LANDING_HOME
        self.state          = "LANDED"
        self.current_pad    = "Home"   # pad where drone currently is / last landed

        # ── Scan state ────────────────────────────────────────────────────
        self.scan_accumulated_yaw = 0.0
        self.scan_speed           = 1.8   # rad/s

        # ── Occupancy grid & A* path ──────────────────────────────────────
        self.occ_grid             = None   # numpy int8 array (grid_size × grid_size)
        self.grid_size            = 160
        self.shortest_path_coords = None   # list of {x,y,grid_x,grid_y} dicts
        self.show_path            = False
        self.current_path_idx     = 0

        # Numpy path arrays for fast sensor math
        self._path_xy = None   # (N,2) float32
        self._path_z  = None   # (N,)  float32

        # ── User command ───────────────────────────────────────────────────
        self.target_pad    = None   # name of requested destination pad
        self.target_pos    = None   # (x, y) of target pad

        # ── Autoland state ────────────────────────────────────────────────
        self.land_state      = "DESCENDING"
        self.land_pad_name   = ""
        self.land_pad_pos    = (0.0, 0.0)
        self.land_start_time = 0.0

        # ── Terrain height cache ───────────────────────────────────────────
        self._terrain_home  = get_terrain_height(self.home_x, self.home_y)
        self._terrain_cache = {}

        # ── Buffered CSV logging ───────────────────────────────────────────
        self.trajectory_log_file = "flight_trajectory.csv"
        self.detection_log_file  = "detection_log.csv"
        self.log_step_counter    = 0
        self._traj_buf           = []
        self._TRAJ_FLUSH         = 100

        try:
            with open(self.trajectory_log_file, "w", newline='') as f:
                csv.writer(f).writerow(["t","state","x","y","z",
                                        "qw","qx","qy","qz","vx","vy","vz",
                                        "fx","fy","fz"])
        except Exception: pass
        try:
            with open(self.detection_log_file, "w", newline='') as f:
                csv.writer(f).writerow(["t","event","pad","x","y","z"])
        except Exception: pass

        # Load ground truth grid
        if os.path.exists("map_data.json"):
            try:
                with open("map_data.json") as f:
                    mdata = json.load(f)
                self.ground_truth_grid = np.array(mdata["grid"], dtype=np.int8)
                print("[AUTOPILOT] Loaded ground truth occupancy grid from map_data.json")
            except Exception as e:
                print(f"[AUTOPILOT] Warning loading map_data.json: {e}")
                _, self.ground_truth_grid = generate_map(self.model, self.data)
        else:
            _, self.ground_truth_grid = generate_map(self.model, self.data)

        # Initialize Terrain Scanner (Feature 1)
        self.scanner = TerrainScanner(self.ground_truth_grid, grid_size=self.grid_size, sensor_radius=2.5, sensor_type="directional", fov_deg=90.0)
        self.occ_grid = self.scanner.get_planning_grid()

        # Initialize Dynamic Obstacle Manager (Feature 2)
        self.obstacle_manager = DynamicObstacleManager(grid_size=self.grid_size)
        self.obstacle_manager.reset(self.model)

        # Initialize Subsumption Architecture Behavior Manager (Feature 3)
        self.behavior_manager = BehaviorManager()
        self.behavior_manager.add_behavior(KillSwitchBehavior())
        self.behavior_manager.add_behavior(EmergencyLandingBehavior())
        self.behavior_manager.add_behavior(ObstacleAvoidanceBehavior())
        self.behavior_manager.add_behavior(NavigationBehavior())
        self.behavior_manager.add_behavior(TerrainScanningBehavior())

        self.kill_switch_active = False
        self.path_blocked = False
        self.planner_algorithm = "A*"  # A* (default) or Dijkstra

        # Initialize Mission Manager (Phase 1)
        self.mission_manager = MissionManager(MissionState.IDLE)

        # Initialize Sensor Fusion (Phase 2)
        self.sensor_fusion = SensorFusion(self)

        # Initialize Occupancy Mapper (Phase 3)
        self.occupancy_mapper = OccupancyGridMapping(self)

        # Initialize Coverage Analyzer (Phase 4)
        self.coverage_analyzer = CoverageMetrics(self)
        self.coverage_analyzer.update(self.occupancy_mapper.get_grid(), self.data.time)

        # Initialize Frontier Detector (Phase 5)
        self.frontier_detector = FrontierDetection(self, connectivity=8)

        # Initialize Frontier Ranker (Phase 6)
        self.frontier_ranker = FrontierRanking(self)

        # Initialize Trajectory Recorder & Replayer (Step 2)
        self.trajectory_recorder = TrajectoryRecorder(record_hz=10.0, min_dist_m=0.05)
        self.trajectory_replayer = TrajectoryReplayer(
            arrival_radius_m=0.45,
            home_pos=(self.home_x, self.home_y),
            home_radius_m=0.40,
        )
        # Step 2/3 flags
        self.auto_return_active           = False
        self.explore_coverage_threshold   = 70.0   # % coverage to trigger auto-return
        self._awaiting_mission_completion = False  # set True when auto-return fires
        self._mission_complete_saved      = False  # prevent duplicate save calls

        # Register ReturnHomeBehavior in the subsumption stack (priority 2.5)
        self.behavior_manager.add_behavior(ReturnHomeBehavior())

        # Compute fallback Dijkstra tour path at startup
        self.dijkstra_tour_coords = self._compute_dijkstra_tour_path()

        # Save initial map
        self.save_discovered_map()

    # ── Helpers ───────────────────────────────────────────────────────────

    def _build_path_arrays(self):
        if not self.shortest_path_coords:
            self._path_xy = None;  self._path_z = None;  return
        n  = len(self.shortest_path_coords)
        xy = np.empty((n, 2), dtype=np.float32)
        z  = np.empty(n,      dtype=np.float32)
        for i, pt in enumerate(self.shortest_path_coords):
            x_, y_ = float(pt['x']), float(pt['y'])
            xy[i,0] = x_;  xy[i,1] = y_
            z[i]    = get_terrain_height(x_, y_)
        self._path_xy = xy;  self._path_z = z

    def _compute_dijkstra_tour_path(self):
        grid = self.ground_truth_grid
        grid_size = self.grid_size
        targets = []
        for name, px, py, _ in _PADS:
            gx = int((px + 8.0) / 16.0 * (grid_size - 1))
            gy = int((py + 8.0) / 16.0 * (grid_size - 1))
            targets.append((name, gx, gy))

        unvisited = list(targets[1:])
        cur = targets[0]
        global_path = []
        while unvisited:
            bd = 1e18
            bp = None
            bb = None
            for pad in unvisited:
                _, gx, gy = pad
                p, d = run_dijkstra(grid, (cur[1], cur[2]), (gx, gy))
                if p and d < bd:
                    bd = d
                    bp = p
                    bb = pad
            if bp:
                if global_path:
                    global_path.pop()
                global_path.extend(bp)
                unvisited.remove(bb)
                cur = bb
            else:
                break
        if not unvisited:
            p, _ = run_dijkstra(grid, (cur[1], cur[2]), (targets[0][1], targets[0][2]))
            if p:
                if global_path:
                    global_path.pop()
                global_path.extend(p)

        sp_points = [{"grid_x": c, "grid_y": r,
                      "x": round(-8.0 + 16.0 * c / (grid_size - 1), 3),
                      "y": round(-8.0 + 16.0 * r / (grid_size - 1), 3)}
                     for c, r in global_path]
        return sp_points

    def _get_pad_terrain(self, px, py):
        key = (round(px,3), round(py,3))
        if key not in self._terrain_cache:
            self._terrain_cache[key] = get_terrain_height(px, py)
        return self._terrain_cache[key]

    def _world_to_grid(self, wx, wy):
        gs = self.grid_size
        gx = int(round((wx + 8.0) / 16.0 * (gs - 1)))
        gy = int(round((wy + 8.0) / 16.0 * (gs - 1)))
        return (max(0, min(gs-1, gx)), max(0, min(gs-1, gy)))

    def _grid_to_world(self, gx, gy):
        gs = self.grid_size
        wx = -8.0 + 16.0 * gx / (gs - 1)
        wy = -8.0 + 16.0 * gy / (gs - 1)
        return wx, wy

    def plan_path(self, start_g, goal_g):
        """Plans path using selected algorithm (A* or Dijkstra) on discovered map."""
        if self.planner_algorithm == "Dijkstra":
            path_grid, cost = run_dijkstra(self.occ_grid, start_g, goal_g)
        else:
            path_grid, cost = run_astar(self.occ_grid, start_g, goal_g)
        return path_grid, cost

    def check_path_validity(self):
        """Checks if remaining waypoints on the path have become blocked by obstacles."""
        if not self.shortest_path_coords or not self.show_path:
            return False
        for i in range(self.current_path_idx, len(self.shortest_path_coords)):
            pt = self.shortest_path_coords[i]
            gx, gy = pt["grid_x"], pt["grid_y"]
            if self.occ_grid[gy, gx] == 1:
                return True
        return False

    def save_discovered_map(self, visual_size=640):
        """Generates ground_map.png and shortest_path_map.png based on explored region."""
        if not hasattr(self, "scanner") or self.scanner is None:
            return
        grid = self.scanner.discovered_grid
        explored = self.scanner.explored_mask
        grid_size = self.grid_size
        cell_w = visual_size / grid_size
        
        def to_px(wx, wy):
            return (int((wx + 8.0) / 16.0 * (visual_size - 1)),
                    int((8.0 - wy) / 16.0 * (visual_size - 1)))
                    
        for fname, draw_path in [("ground_map.png", False), ("shortest_path_map.png", True)]:
            img = Image.new("RGB", (visual_size, visual_size), (12, 16, 22))
            draw = ImageDraw.Draw(img)
            
            # Show full ground truth map to ensure visibility of all features
            for r in range(grid_size):
                for c in range(grid_size):
                    tx_ = c * cell_w
                    ty_ = (grid_size - 1 - r) * cell_w
                    if self.occ_grid[r, c] == 1:
                        # Distinguish fully discovered obstacles from undiscovered ones
                        if explored[r, c]:
                            draw.rectangle([tx_, ty_, tx_ + cell_w, ty_ + cell_w], fill=(230, 70, 70)) # Bright Red
                        else:
                            draw.rectangle([tx_, ty_, tx_ + cell_w, ty_ + cell_w], fill=(120, 30, 30)) # Dark Red
                    elif not explored[r, c]:
                        # Subtle fog background for unexplored free space
                        draw.rectangle([tx_, ty_, tx_ + cell_w, ty_ + cell_w], fill=(20, 25, 30))
                        
            # Draw cyber grid lines AFTER filling cells so they are visible throughout the entire map
            for g in np.arange(-8.0, 8.1, 1.0):
                draw.line([to_px(g, -8.0), to_px(g, 8.0)], fill=(0, 40, 60), width=1)
                draw.line([to_px(-8.0, g), to_px(8.0, g)], fill=(0, 40, 60), width=1)
                
            # Draw pads
            for name, px, py, pr in _PADS:
                cx, cy = to_px(px, py)
                rp = int(pr / 16.0 * visual_size)
                draw.ellipse([cx - rp, cy - rp, cx + rp, cy + rp], outline=(0, 255, 128), width=3)
                draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(0, 255, 128))
                draw.text((cx + rp + 5, cy - 6), f"{name}({px},{py})", fill=(0, 255, 128))
                
            # Draw shortest path (fall back to the calculated Dijkstra tour path
            # so the path map stays meaningful even when shortest_path_coords is None,
            # e.g. after landing, during scanning, or at initial startup)
            if draw_path:
                path_coords = self.shortest_path_coords
                if not path_coords:
                    path_coords = getattr(self, "dijkstra_tour_coords", None)
                if path_coords and len(path_coords) > 1:
                    pts = [to_px(pt['x'], pt['y']) for pt in path_coords]
                    draw.line(pts, fill=(0, 160, 80), width=6)
                    draw.line(pts, fill=(180, 255, 100), width=2)

            img.save(fname)
        
        # Save JSON data for metrics compatibility
        map_data = {
            "metadata": {"origin_x":-8.0,"origin_y":-8.0,"width_meters":16.0,
                         "height_meters":16.0,"grid_size":grid_size,
                         "resolution_meters_per_cell":16.0/grid_size},
            "landing_spots": [{"name":name,"x":px,"y":py,
                                "grid_x":int((px+8.0)/16.0*(grid_size-1)),
                                "grid_y":int((py+8.0)/16.0*(grid_size-1))}
                              for name,px,py,_ in _PADS],
            "visiting_sequence": ["Home"],
            "shortest_path": self.shortest_path_coords if self.shortest_path_coords else getattr(self, "dijkstra_tour_coords", []),
            "grid": grid.tolist()
        }
        with open("map_data.json","w") as f:
            json.dump(map_data, f, indent=2)
        save_map_to_csv(grid, map_data["metadata"], map_data["landing_spots"],
                        map_data["shortest_path"], [])

    # ── Logging ───────────────────────────────────────────────────────────

    def _flush_traj(self):
        if not self._traj_buf: return
        try:
            with open(self.trajectory_log_file, "a", newline='') as f:
                csv.writer(f).writerows(self._traj_buf)
            self._traj_buf.clear()
        except Exception: pass

    def log_telemetry(self):
        try:
            qa = self.qpos_adr;  va = self.qvel_adr
            qp = self.data.qpos;  qv = self.data.qvel
            xf = self.data.xfrc_applied[self.drone_body_id]
            self._traj_buf.append([
                round(self.data.time,3), self.state,
                round(float(qp[qa]),3), round(float(qp[qa+1]),3), round(float(qp[qa+2]),3),
                round(float(qp[qa+3]),4), round(float(qp[qa+4]),4),
                round(float(qp[qa+5]),4), round(float(qp[qa+6]),4),
                round(float(qv[va]),3), round(float(qv[va+1]),3), round(float(qv[va+2]),3),
                round(float(xf[0]),3), round(float(xf[1]),3), round(float(xf[2]),3),
            ])
            if len(self._traj_buf) >= self._TRAJ_FLUSH:
                self._flush_traj()
        except Exception: pass

    def log_event(self, event, pad="", px=None, py=None, pz=None):
        try:
            qa = self.qpos_adr;  qp = self.data.qpos
            if px is None: px = float(qp[qa])
            if py is None: py = float(qp[qa+1])
            if pz is None: pz = float(qp[qa+2])
            with open(self.detection_log_file, "a", newline='') as f:
                csv.writer(f).writerow([round(self.data.time,3), event, pad,
                                        round(px,3), round(py,3), round(pz,3)])
            print(f"[LOG] {event}  pad={pad}")
        except Exception as e:
            print(f"[LOG] Warning: {e}")

    # ── User-facing commands ───────────────────────────────────────────────

    def cmd_start_exploration(self) -> None:
        """
        E key — enters autonomous exploration mode.
        Transitions the mission manager to EXPLORE and activates the
        trajectory recorder.  Must be called when the drone is hovering
        (state == "HOVER") or taking off (state == "TAKEOFF").
        """
        if self.state not in ("HOVER", "TAKEOFF"):
            print("[EXPLORE] Drone must be hovering before entering exploration mode. "
                  "Press L to take off first.")
            return
        if self.auto_return_active:
            print("[EXPLORE] Already in auto-return mode.")
            return
        ok = self.mission_manager.transition_to(MissionState.EXPLORE)
        if not ok:
            # Force if already in a compatible state (e.g. IDLE → can't go direct)
            self.mission_manager.force_state(MissionState.EXPLORE)
        self.trajectory_recorder.clear()
        self.trajectory_recorder.start()
        self._awaiting_mission_completion = False
        self._mission_complete_saved      = False
        print("[EXPLORE] Exploration mode ACTIVE. Trajectory recording started. "
              f"Auto-return will trigger at {self.explore_coverage_threshold:.0f}% coverage "
              "or when no frontiers remain.")

    def _trigger_auto_return(self) -> None:
        """
        Internal method called when the coverage threshold is reached or
        no frontier regions remain.  Stops recording, reverses the trajectory,
        and activates the ReturnHomeBehavior.
        """
        log = self.trajectory_recorder.get_log()
        if len(log) < 2:
            print("[AUTO-RETURN] Trajectory log too short to reverse — cannot auto-return.")
            return

        self.trajectory_recorder.stop()
        self.mission_manager.transition_to(MissionState.RETURN_HOME)

        started = self.trajectory_replayer.start(log)
        if started:
            self.auto_return_active           = True
            self._awaiting_mission_completion = True
            dist = self.trajectory_recorder.get_total_distance_m()
            print(
                f"[AUTO-RETURN] Trajectory reversal activated: "
                f"{self.trajectory_replayer.get_waypoint_count()} waypoints, "
                f"{dist:.1f} m recorded path. "
                f"Coverage={self.coverage_analyzer.get_coverage_percentage():.1f}%"
            )
        else:
            print("[AUTO-RETURN] WARNING: Trajectory replayer failed to start.")
            self.mission_manager.force_state(MissionState.EXPLORE)   # revert

    def save_mission_data(self) -> None:
        """
        Step 3: Save complete mission artefacts after landing.
        Writes mission_log.json, then runs post-mission route analysis
        (Step 4) and saves mission_route_analysis.json + mission_route_map.png.
        """
        # ── 1. Final map (PNG + JSON + CSV) ──────────────────────────────
        self.save_discovered_map()

        # ── 2. Coverage statistics ────────────────────────────────────────
        stats: dict = {}
        if hasattr(self, "coverage_analyzer") and self.coverage_analyzer is not None:
            stats = self.coverage_analyzer.get_statistics()

        # ── 3. Trajectory log ─────────────────────────────────────────────
        traj_log: list = []
        if hasattr(self, "trajectory_recorder") and self.trajectory_recorder is not None:
            traj_log = [s.to_dict() for s in self.trajectory_recorder.get_log()]

        mission_data = {
            "mission_end_time_s": round(self.data.time, 3),
            "home_position":      {"x": self.home_x, "y": self.home_y},
            "coverage_statistics": stats,
            "trajectory_snapshot_count": len(traj_log),
            "trajectory_log": traj_log,
        }
        try:
            with open("mission_log.json", "w") as f:
                json.dump(mission_data, f, indent=2)
            print("[MISSION] mission_log.json saved.")
        except Exception as exc:
            print(f"[MISSION] Failed to save mission_log.json: {exc}")

        # ── 4. Post-mission route analysis (offline, drone does NOT fly) ──
        try:
            grid = self.occupancy_mapper.get_grid()
            analysis = PostMissionAnalysis(pads=_PADS, grid_size=self.grid_size)
            analysis.run(grid, run_astar, run_dijkstra)
            analysis.print_route_matrix()
            analysis.save_results("mission_route_analysis.json")
            analysis.save_map(grid, "mission_route_map.png")
        except Exception as exc:
            print(f"[MISSION] Post-mission analysis failed: {exc}")

        # ── 5. Advance mission manager to MISSION_COMPLETE ────────────────
        self.mission_manager.transition_to(MissionState.MISSION_COMPLETE)
        print("[MISSION] ✓ All mission artefacts saved. MissionState → MISSION_COMPLETE.")


    def cmd_takeoff(self):
        """L key – takeoff from current position."""
        if self.state == "LANDED":
            self.physics.reset_integrators()
            self.state = "TAKEOFF"
            self.log_event("TAKEOFF", self.current_pad)
            print("\n[CTRL] Takeoff! Climbing to hover height...")
        elif self.state in ("HOVER","SCANNING","GOTO_TARGET","AUTOLAND"):
            self.state = "LANDING_HOME"
            self.log_event("LAND_HOME_CMD")
            print("\n[CTRL] Returning home and landing...")

    def cmd_scan(self):
        """C key – start 360° scan to build occupancy grid."""
        if self.state == "HOVER":
            self.state = "SCANNING"
            self.scan_accumulated_yaw = 0.0
            self.log_event("SCAN_START")
            print("\n[SCAN] 360° scan started...")
        else:
            print(f"\n[SCAN] Must be HOVER. Current: {self.state}")

    def cmd_toggle_path(self):
        """S key – show/hide path overlay."""
        if not self.shortest_path_coords:
            print("\n[VIZ] No path yet. Fly to a target first or press C to scan.");  return
        self.show_path = not self.show_path
        print(f"\n[VIZ] Path overlay {'ON' if self.show_path else 'OFF'}.")

        # Initiate flight if path is shown and target pad is set
        if self.show_path and self.target_pad is not None:
            if self.state == "LANDED":
                self.physics.reset_integrators()
                self.state = "TAKEOFF"
                self.log_event("TAKEOFF_FOR_GOTO", self.target_pad)
                print("[NAV] Path shown. Taking off first...")
            elif self.state == "HOVER":
                self.state = "GOTO_TARGET"
                self.log_event("GOTO_TARGET_START", self.target_pad, self.target_pos[0], self.target_pos[1])
                print(f"[NAV] Path shown. Flying to {self.target_pad}...")
        # Hover/halt if path is hidden during takeoff/flight
        elif not self.show_path:
            if self.state in ("TAKEOFF", "GOTO_TARGET"):
                self.state = "HOVER"
                print("\n[NAV] Path line disappeared! Drone hovering in place.")

    def cmd_goto(self, pad_name):
        """
        Main user command: fly to a named pad.
        Plans A* from current position to target.
        """
        if self.occ_grid is None:
            print("\n[NAV] No map yet. Press C to scan first.");  return
        if pad_name not in _PADS_DICT:
            print(f"\n[NAV] Unknown pad '{pad_name}'. Options: {list(_PADS_DICT.keys())}");  return
        if self.state not in ("HOVER", "LANDED", "GOTO_TARGET"):
            print(f"\n[NAV] Must be hovering or landed. Current: {self.state}");  return

        tx, ty = _PADS_DICT[pad_name]
        if self.target_pad == pad_name and self.state == "GOTO_TARGET":
            print(f"\n[NAV] Already flying to {pad_name}.");  return

        # ── Plan path ──────────────────────────────────────────────────
        qa = self.qpos_adr;  qp = self.data.qpos
        drone_x = float(qp[qa]);  drone_y = float(qp[qa+1])
        start_g = self._world_to_grid(drone_x, drone_y)
        goal_g  = self._world_to_grid(tx, ty)

        print(f"\n[NAV] Planning {self.planner_algorithm} from {start_g} -> {goal_g} (target: {pad_name})...")
        path_grid, cost = self.plan_path(start_g, goal_g)

        if path_grid is None:
            print(f"[NAV] {self.planner_algorithm} failed - no path to {pad_name}!");  return

        # Convert grid path → world coordinates
        sp_points = []
        for (gx, gy) in path_grid:
            wx, wy = self._grid_to_world(gx, gy)
            sp_points.append({"grid_x": gx, "grid_y": gy, "x": round(wx,3), "y": round(wy,3)})

        self.shortest_path_coords = sp_points
        self._build_path_arrays()
        self.current_path_idx     = 0
        self.target_pad           = pad_name
        self.target_pos           = (tx, ty)
        self.show_path            = False   # Do NOT show the path yet (user must press S)
        self.save_discovered_map()

        print(f"[NAV] {self.planner_algorithm} found path: {len(sp_points)} waypoints, cost={cost:.1f}")
        print(f"[NAV] Planned path to {pad_name} at ({tx}, {ty}). Press S to show line and start flight.")

    # ── Line sensor (A*-path following) ───────────────────────────────────

    def _detect_line_sensor(self):
        """
        Finds the lookahead point on the A* path.
        Returns (x, y) world coords, or None if path is lost.
        """
        if not self.show_path or self._path_xy is None:
            return None
        xy  = self._path_xy
        n   = len(xy)
        idx = self.current_path_idx
        end = min(n - 1, idx + 20)   # scan 20 segments ahead

        qp = self.data.qpos;  qa = self.qpos_adr
        dx = float(qp[qa]);  dy = float(qp[qa+1])

        best_idx  = idx;  best_dist = 1e9
        best_cx   = dx;   best_cy   = dy

        for i in range(idx, end):
            p1x = float(xy[i,0]);    p1y = float(xy[i,1])
            p2x = float(xy[i+1,0]); p2y = float(xy[i+1,1])
            vx  = p2x-p1x;           vy  = p2y-p1y
            wx  = dx-p1x;            wy  = dy-p1y
            vl2 = vx*vx + vy*vy
            if vl2 < 1e-12: t = 0.0
            else:
                t = (wx*vx + wy*vy) / vl2
                t = max(0.0, min(1.0, t))
            cx = p1x + t*vx;  cy = p1y + t*vy
            d  = (dx-cx)**2 + (dy-cy)**2
            if d < best_dist:
                best_dist = d;  best_idx = i
                best_cx   = cx; best_cy   = cy

        self.current_path_idx = best_idx
        best_dist = math.sqrt(best_dist)

        if best_dist > 2.0:   # too far from path
            return None

        # Walk 0.6 m lookahead along path
        remaining = 0.6
        ti = best_idx
        px = best_cx;  py = best_cy
        while remaining > 0.0 and ti < n - 1:
            nx_ = float(xy[ti+1,0]);  ny_ = float(xy[ti+1,1])
            dvx = nx_-px;              dvy = ny_-py
            dl  = math.sqrt(dvx*dvx + dvy*dvy)
            if dl >= remaining:
                px += dvx/dl*remaining;  py += dvy/dl*remaining;  break
            remaining -= dl;  px = nx_;  py = ny_;  ti += 1
        return (px, py)

    # ── Main step ─────────────────────────────────────────────────────────

    def step(self):
        dt = self.model.opt.timestep
        self.log_step_counter += 1
        if self.log_step_counter % 20 == 0:
            self.log_telemetry()

        # Update scanner (Feature 1)
        if hasattr(self, "scanner") and self.scanner is not None:
            qp = self.data.qpos; qa = self.qpos_adr
            dx = float(qp[qa]); dy = float(qp[qa+1])
            _, _, cur_yaw = self.physics._quat_rpy(self.data.xquat[self.drone_body_id])
            self.scanner.update(dx, dy, cur_yaw)
            self.occ_grid = self.scanner.get_planning_grid()
            
            # Check path validity
            if self.state == "GOTO_TARGET" and self.check_path_validity():
                self.path_blocked = True

        # Update dynamic obstacles timers / periodic spawning (Feature 2)
        if hasattr(self, "obstacle_manager") and self.obstacle_manager is not None:
            self.obstacle_manager.step(self.model, self.data, self, self.ground_truth_grid, dt)

        # Step Mission Manager (Phase 1 Hook)
        if hasattr(self, "mission_manager") and self.mission_manager is not None:
            self.mission_manager.step(dt)

        # Step Sensor Fusion (Phase 2 Hook)
        if hasattr(self, "sensor_fusion") and self.sensor_fusion is not None:
            self.sensor_fusion.update()

        # Step heavy pipeline modules with throttled execution schedules to restore smooth flight
        is_first_step = (self.log_step_counter == 1)

        # Step Occupancy Mapping (Phase 3 Hook) - 10Hz (every 20 steps)
        if hasattr(self, "occupancy_mapper") and self.occupancy_mapper is not None:
            if is_first_step or (self.log_step_counter % 20 == 0):
                self.occupancy_mapper.update(self.sensor_fusion.get_state(), getattr(self, "sensor_suite", None))

        # Step Coverage Analysis (Phase 4 Hook) - 2Hz (every 100 steps)
        if hasattr(self, "coverage_analyzer") and self.coverage_analyzer is not None:
            if is_first_step or (self.log_step_counter % 100 == 0):
                self.coverage_analyzer.update(self.occupancy_mapper.get_grid(), self.data.time)

        # Step Frontier Detection (Phase 5 Hook) - 2Hz (every 100 steps)
        if hasattr(self, "frontier_detector") and self.frontier_detector is not None:
            if is_first_step or (self.log_step_counter % 100 == 0):
                self.frontier_detector.update(self.occupancy_mapper.get_grid())

        # Step Frontier Ranking (Phase 6 Hook) - 2Hz (every 100 steps)
        if hasattr(self, "frontier_ranker") and self.frontier_ranker is not None:
            if is_first_step or (self.log_step_counter % 100 == 0):
                self.frontier_ranker.update(
                    self.frontier_detector.get_frontier_regions(),
                    self.occupancy_mapper.get_grid(),
                    self.sensor_fusion.get_state()
                )

        # ── Step Trajectory Recorder (Step 2 Hook) ─────────────────────────
        # Runs every physics step; internal time/distance gates limit actual records.
        if (hasattr(self, "trajectory_recorder") and
                self.trajectory_recorder is not None and
                hasattr(self, "mission_manager") and
                self.mission_manager.is_in_state(MissionState.EXPLORE) and
                hasattr(self, "sensor_fusion")):
            self.trajectory_recorder.record(self.sensor_fusion.get_state())

        # ── Coverage-threshold check → auto-return trigger (Step 2/3 Hook) ──
        # Evaluated at 2 Hz (same cadence as coverage update) to avoid redundant checks.
        if (not self.auto_return_active and
                not self._awaiting_mission_completion and
                hasattr(self, "mission_manager") and
                self.mission_manager.is_in_state(MissionState.EXPLORE) and
                hasattr(self, "coverage_analyzer") and
                hasattr(self, "trajectory_recorder") and
                (is_first_step or self.log_step_counter % 100 == 0)):
            snap_count = self.trajectory_recorder.snapshot_count()
            if snap_count >= 3:
                cov = self.coverage_analyzer.get_coverage_percentage()
                no_frontiers = (
                    hasattr(self, "frontier_detector") and
                    self.frontier_detector.get_frontier_count() == 0
                )
                if cov >= self.explore_coverage_threshold or no_frontiers:
                    self._trigger_auto_return()

        # ── Mission completion detection (Step 3 Hook) ──────────────────────
        # Fires once after the drone lands following a RETURN_HOME sequence.
        if (not self._mission_complete_saved and
                self._awaiting_mission_completion and
                self.state == "LANDED"):
            self._mission_complete_saved = True
            print("[MISSION] Exploration mission complete — drone has landed. "
                  "Saving mission data…")
            self.save_mission_data()

        # Run subsumption decision step (Feature 3)
        self.behavior_manager.step(self, dt)


# ============================================================
#  Map / CSV utilities  (unchanged from previous version)
# ============================================================

def save_map_to_csv(grid, metadata, landing_spots, shortest_path, visiting_sequence):
    try:
        with open("map_metadata.csv","w",newline='') as f:
            w = csv.writer(f);  w.writerow(["Parameter","Value"])
            for k, v in metadata.items(): w.writerow([k, v])
        with open("landing_spots.csv","w",newline='') as f:
            w = csv.writer(f);  w.writerow(["Name","X","Y","Grid_X","Grid_Y"])
            for s in landing_spots: w.writerow([s["name"],s["x"],s["y"],s["grid_x"],s["grid_y"]])
        with open("shortest_path.csv","w",newline='') as f:
            w = csv.writer(f);  w.writerow(["Index","X","Y","Grid_X","Grid_Y"])
            for i, pt in enumerate(shortest_path):
                w.writerow([i, pt["x"], pt["y"], pt["grid_x"], pt["grid_y"]])
        with open("occupancy_grid.csv","w",newline='') as f:
            w = csv.writer(f)
            for row in grid: w.writerow(row)
        if visiting_sequence:
            with open("visiting_sequence.csv","w",newline='') as f:
                w = csv.writer(f);  w.writerow(["Order","Pad_Name"])
                for i, name in enumerate(visiting_sequence): w.writerow([i, name])
        print("[MAP] CSV files saved.")
    except Exception as e:
        print(f"[MAP] Error saving CSVs: {e}")


def run_dijkstra(grid, start_coords, end_coords):
    """Dijkstra – kept for generate_map's full-tour path (backward compat)."""
    height, width = grid.shape
    queue = [(0.0, start_coords[0], start_coords[1])]
    heapq.heapify(queue)
    dist = {start_coords: 0.0};  parent = {}
    dirs = [(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
            (1,1,1.414),(1,-1,1.414),(-1,1,1.414),(-1,-1,1.414)]
    found = False
    while queue:
        d, cx, cy = heapq.heappop(queue)
        if (cx,cy) == end_coords: found=True; break
        if d > dist.get((cx,cy), 1e18): continue
        for ddx,ddy,sc in dirs:
            nx,ny = cx+ddx, cy+ddy
            if 0<=nx<width and 0<=ny<height and grid[ny,nx]!=1:
                near = any(0<=nx+ox<width and 0<=ny+oy<height and grid[ny+oy,nx+ox]==1
                           for ox in (-1,0,1) for oy in (-1,0,1))
                nd = d + sc + (3.0 if near else 0.0)
                if nd < dist.get((nx,ny), 1e18):
                    dist[(nx,ny)] = nd;  parent[(nx,ny)] = (cx,cy)
                    heapq.heappush(queue, (nd,nx,ny))
    if not found: return None, float('inf')
    path=[]; curr=end_coords
    while curr!=start_coords: path.append(curr); curr=parent[curr]
    path.append(start_coords);  path.reverse()
    return path, dist[end_coords]


def generate_map(model, data, grid_size=160, visual_size=640):
    """
    Builds the occupancy grid from terrain + rocks.
    Also creates a Dijkstra full-tour path for the path_map image.
    Returns (sp_points_list, occ_grid_ndarray).
    """
    print("[MAP] Building occupancy grid...")

    grid  = np.zeros((grid_size, grid_size), dtype=np.int8)
    rocks = []
    for b in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
        if name and (name.startswith("rock_") or name.startswith("dyn_")):
            bx, by, _ = model.body_pos[b]
            g_adr = model.body_geomadr[b]
            br = max(model.geom_size[g_adr][0], model.geom_size[g_adr][1]) * 1.15
            rocks.append((float(bx), float(by), br))

    cols = np.linspace(-8.0, 8.0, grid_size, dtype=np.float32)
    rows = np.linspace(-8.0, 8.0, grid_size, dtype=np.float32)
    RX, RY = np.meshgrid(cols, rows)

    rock_mask = np.zeros((grid_size,grid_size), dtype=bool)
    for bx,by,br in rocks:
        rock_mask |= ((RX-bx)**2 + (RY-by)**2) < br**2

    pad_mask = np.zeros((grid_size,grid_size), dtype=bool)
    for _,px,py,pr in _PADS:
        pad_mask |= ((RX-px)**2 + (RY-py)**2) < pr**2

    H  = (120.0
          + 60.0*np.sin(0.4*RX)*np.cos(0.4*RY)
          + 30.0*np.sin(1.1*RX+0.5)*np.cos(0.9*RY-0.2)
          + 15.0*np.sin(2.5*RX)*np.sin(2.8*RY)
          +  5.0*np.sin(6.0*RX)*np.cos(5.5*RY))

    FF = np.ones((grid_size,grid_size), dtype=np.float32)
    for _,px,py,pr in _PADS:
        D = np.sqrt((RX-px)**2 + (RY-py)**2)
        FF[D<pr] = 0.0
        mask = (D>=pr) & (D<pr+0.4)
        t = (D[mask]-pr)/0.4
        FF[mask] = np.minimum(FF[mask], 3.0*t*t - 2.0*t*t*t)

    H_scaled = 0.4*(H*FF/255.0)
    grid[rock_mask & ~pad_mask] = 1
    grid[~pad_mask & (H_scaled > 0.22)] = 1

    # Build a Dijkstra full-tour path just for the visual map
    targets = []
    for name,px,py,_ in _PADS:
        gx = int((px+8.0)/16.0*(grid_size-1))
        gy = int((py+8.0)/16.0*(grid_size-1))
        targets.append((name,gx,gy))

    unvisited = list(targets[1:])
    cur = targets[0]
    global_path = [];  visit_order = ["Home"]
    while unvisited:
        bd=1e18; bp=None; bb=None
        for pad in unvisited:
            _,gx,gy = pad
            p,d = run_dijkstra(grid, (cur[1],cur[2]), (gx,gy))
            if p and d < bd: bd=d; bp=p; bb=pad
        if bp:
            if global_path: global_path.pop()
            global_path.extend(bp)
            unvisited.remove(bb);  cur=bb;  visit_order.append(bb[0])
        else: break
    if not unvisited:
        p,_ = run_dijkstra(grid, (cur[1],cur[2]), (targets[0][1],targets[0][2]))
        if p:
            if global_path: global_path.pop()
            global_path.extend(p);  visit_order.append("Home")

    # Visual images
    cell_w = visual_size / grid_size
    def to_px(wx,wy):
        return (int((wx+8.0)/16.0*(visual_size-1)),
                int((8.0-wy)/16.0*(visual_size-1)))

    for fname, draw_tour in [("ground_map.png", False), ("shortest_path_map.png", True)]:
        img  = Image.new("RGB", (visual_size,visual_size), (12, 16, 22))
        draw = ImageDraw.Draw(img)
        
        # Fill cells first (everything is unexplored initially)
        for r in range(grid_size):
            for c in range(grid_size):
                tx_ = c * cell_w
                ty_ = (grid_size - 1 - r) * cell_w
                if grid[r, c] == 1:
                    # Obstacles are dark red since they are unexplored initially
                    draw.rectangle([tx_, ty_, tx_ + cell_w, ty_ + cell_w], fill=(120, 30, 30))
                else:
                    # Subtle fog background for unexplored free space
                    draw.rectangle([tx_, ty_, tx_ + cell_w, ty_ + cell_w], fill=(20, 25, 30))
                    
        # Draw cyber grid lines AFTER filling cells so they are visible throughout
        for g in np.arange(-8.0,8.1,1.0):
            draw.line([to_px(g,-8.0), to_px(g,8.0)], fill=(0, 40, 60), width=1)
            draw.line([to_px(-8.0,g), to_px(8.0,g)], fill=(0, 40, 60), width=1)
            
        for name,px,py,pr in _PADS:
            cx,cy = to_px(px,py)
            rp = int(pr/16.0*visual_size)
            draw.ellipse([cx-rp,cy-rp,cx+rp,cy+rp], outline=(0,255,128), width=3)
            draw.ellipse([cx-4,cy-4,cx+4,cy+4], fill=(0,255,128))
            draw.text((cx+rp+5,cy-6), f"{name}({px},{py})", fill=(0,255,128))
        if draw_tour and global_path:
            pts = [(int(c*cell_w+cell_w/2), int((grid_size-1-r)*cell_w+cell_w/2))
                   for c,r in global_path]
            draw.line(pts, fill=(0,160,80),    width=6)
            draw.line(pts, fill=(180,255,100), width=2)
        img.save(fname)

    print(f"[MAP] ground_map.png and shortest_path_map.png saved.")

    sp_points = [{"grid_x":c,"grid_y":r,
                  "x":round(-8.0+16.0*c/(grid_size-1),3),
                  "y":round(-8.0+16.0*r/(grid_size-1),3)}
                 for c,r in global_path]

    map_data = {
        "metadata": {"origin_x":-8.0,"origin_y":-8.0,"width_meters":16.0,
                     "height_meters":16.0,"grid_size":grid_size,
                     "resolution_meters_per_cell":16.0/grid_size},
        "landing_spots": [{"name":name,"x":px,"y":py,
                            "grid_x":int((px+8.0)/16.0*(grid_size-1)),
                            "grid_y":int((py+8.0)/16.0*(grid_size-1))}
                          for name,px,py,_ in _PADS],
        "visiting_sequence": visit_order,
        "shortest_path": sp_points,
        "grid": grid.tolist()
    }
    with open("map_data.json","w") as f:
        json.dump(map_data, f, indent=2)
    save_map_to_csv(grid, map_data["metadata"], map_data["landing_spots"],
                    sp_points, visit_order)
    print("[MAP] map_data.json saved. Grid ready for A* navigation.\n")
    return sp_points, grid


# ============================================================
#  Simulation Entry Point
# ============================================================

def run_simulation():
    xml_path = "scene.xml"
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found!");  sys.exit(1)

    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    ap    = DroneAutopilot(model, data)
    sensor_suite = SensorSimulationSuite(dt=model.opt.timestep)
    ap.sensor_suite = sensor_suite

    # ── Keyboard → pad mapping ─────────────────────────────────────────────
    # Number keys 1-5 map to Pad 1-5; H maps to Home
    _KEY_PAD = {
        '1': "Pad 1", '2': "Pad 2", '3': "Pad 3",
        '4': "Pad 4", '5': "Pad 5", 'h': "Home",
        49: "Pad 1", 50: "Pad 2", 51: "Pad 3",   # ASCII codes
        52: "Pad 4", 53: "Pad 5",
        72: "Home",  104: "Home",
    }

    def key_callback(keycode):
        try:
            key_char = chr(keycode).lower() if 32 <= keycode <= 126 else None

            # L – takeoff / emergency land
            if key_char == 'l' or keycode in (76, 108):
                ap.cmd_takeoff()
            # C – scan
            elif key_char == 'c' or keycode in (67, 99):
                ap.cmd_scan()
            # S – toggle path overlay
            elif key_char == 's' or keycode in (83, 115):
                ap.cmd_toggle_path()
            # O – Spawn dynamic obstacle
            elif key_char == 'o' or keycode in (79, 111):
                if ap.state == "GOTO_TARGET":
                    ap.obstacle_manager.spawn_on_path(model, data, ap, ap.ground_truth_grid)
                else:
                    ap.obstacle_manager.spawn_random(model, data, ap.ground_truth_grid)
            # K – Kill Switch
            elif key_char == 'k' or keycode in (75, 107):
                ap.kill_switch_active = True
            # P – Toggle Pathfinder Algorithm
            elif key_char == 'p' or keycode in (80, 112):
                if ap.planner_algorithm == "A*":
                    ap.planner_algorithm = "Dijkstra"
                else:
                    ap.planner_algorithm = "A*"
                print(f"\n[PLANNER] Pathfinder algorithm toggled to: {ap.planner_algorithm}")
            # R – Toggle Random Spawning
            elif key_char == 'r' or keycode in (82, 114):
                ap.obstacle_manager.random_spawn_enabled = not ap.obstacle_manager.random_spawn_enabled
                status = "ENABLED (Interval: 12s)" if ap.obstacle_manager.random_spawn_enabled else "DISABLED"
                print(f"\n[DYNAMIC] Periodic random obstacle spawning: {status}")
            # E – Start autonomous exploration mode
            elif key_char == 'e' or keycode in (69, 101):
                ap.cmd_start_exploration()
            # A – Manually trigger auto-return home (early exit from exploration)
            elif key_char == 'a' or keycode in (65, 97):
                if ap.mission_manager.is_in_state(MissionState.EXPLORE):
                    print("\n[KEY] Manual auto-return triggered.")
                    ap._trigger_auto_return()
                else:
                    print("\n[KEY] Auto-return only available during EXPLORE mode (press E first).")
            # 1-5 / H – fly to pad
            elif key_char in _KEY_PAD:
                ap.cmd_goto(_KEY_PAD[key_char])
            elif keycode in _KEY_PAD:
                ap.cmd_goto(_KEY_PAD[keycode])

        except Exception as e:
            print(f"[KEY ERROR] {keycode}: {e}")

    print("Launching MuJoCo viewer (physics-only, no scripted animation)...")
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        print("\n+--------------------------------------------------------------+")
        print("|     Neon Grid Drone — Autonomous Exploration Sim            |")
        print("+--------------------------------------------------------------+")
        print("|  EXPLORATION PIPELINE (Steps 1-4)                           |")
        print("|   L         - Takeoff / Emergency land                       |")
        print("|   E         - Enter EXPLORE mode (records trajectory)        |")
        print("|   A         - Manual auto-return home (reverse trajectory)   |")
        print("+--------------------------------------------------------------+")
        print("|  MANUAL NAVIGATION                                           |")
        print("|   1-5       - Fly to Pad 1-5                                 |")
        print("|   H         - Fly back to Home (-4,-4)                       |")
        print("|   C         - Scan terrain (360° spin while hovering)        |")
        print("|   S         - Show / Hide path overlay                       |")
        print("+--------------------------------------------------------------+")
        print("|  SYSTEM                                                      |")
        print("|   O         - Spawn dynamic obstacle                         |")
        print("|   K         - Kill Switch (emergency stop)                   |")
        print("|   P         - Toggle pathfinder  (A* / Dijkstra)             |")
        print("|   R         - Toggle random obstacle spawning                |")
        print("+--------------------------------------------------------------+")
        print("|  AUTO-RETURN triggers automatically at:                      |")
        print(f"|   Coverage >= {ap.explore_coverage_threshold:.0f}%  OR  No frontier regions remain |")
        print("+--------------------------------------------------------------+")
        print("|  Click inside window first to capture keypresses!            |")
        print("+--------------------------------------------------------------+")
        print(f"\n  mass={ap.physics.mass:.3f} kg | "
              f"hover={ap.physics._hover_ff:.2f} N | "
              f"drag={ap.physics.K_DRAG_LIN:.2f} N*s/m")
        if ap.occ_grid is not None:
            print("  Occupancy grid loaded - ready for autonomous navigation.")
        else:
            print("  No map yet. Press L -> hover -> C to scan terrain first.")
        print()

        viewer.cam.lookat   = [-4.0, -4.0, 0.0]
        viewer.cam.distance = 4.5
        viewer.cam.elevation = -30.0
        viewer.cam.azimuth   = 135.0

        # Path overlay cache (rebuilt only when path changes)
        _ov1 = [];  _ov2 = []
        _ov_built    = [False]
        _prev_show   = [False]
        _prev_coords = [None]

        def _rebuild_overlay():
            _ov1.clear();  _ov2.clear()
            coords = ap.shortest_path_coords
            if not coords: return
            xy = ap._path_xy;  z = ap._path_z
            for i in range(len(coords)-1):
                if xy is not None:
                    x1,y1,z1 = float(xy[i,0]),   float(xy[i,1]),   float(z[i])
                    x2,y2,z2 = float(xy[i+1,0]), float(xy[i+1,1]), float(z[i+1])
                else:
                    x1,y1 = coords[i]['x'],   coords[i]['y']
                    x2,y2 = coords[i+1]['x'], coords[i+1]['y']
                    z1 = get_terrain_height(x1,y1)
                    z2 = get_terrain_height(x2,y2)
                _ov1.append(np.array([x1,y1,z1+0.12], dtype=np.float64))
                _ov2.append(np.array([x2,y2,z2+0.12], dtype=np.float64))
            _ov_built[0] = True

        step_count = 0
        try:
            while viewer.is_running():
                t0 = time.perf_counter()

                # ── 1. Physics autopilot step ──────────────────────────────
                ap.step()

                # ── 2. MuJoCo physics integration ─────────────────────────
                mujoco.mj_step(model, data)
                sensor_suite.step(model, data)

                step_count += 1

                # ── 3. Debug print (sparse) ────────────────────────────────
                if step_count % 1000 == 0 and ap.state != "LANDED":
                    qa = ap.qpos_adr
                    print(f"[DBG] {ap.state}  "
                          f"pos=({data.qpos[qa]:.2f},{data.qpos[qa+1]:.2f},{data.qpos[qa+2]:.2f})  "
                          f"target={ap.target_pad}  path_idx={ap.current_path_idx}")

                # ── 4. Camera follow ───────────────────────────────────────
                qa = ap.qpos_adr
                viewer.cam.lookat[0] = data.qpos[qa]
                viewer.cam.lookat[1] = data.qpos[qa+1]
                viewer.cam.lookat[2] = data.qpos[qa+2]

                # ── 5. Path overlay (rebuild only on change) ───────────────
                show   = ap.show_path
                coords = ap.shortest_path_coords
                if show != _prev_show[0] or coords is not _prev_coords[0]:
                    _prev_show[0]   = show
                    _prev_coords[0] = coords
                    _ov_built[0]    = False

                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    geom_idx = 0
                    max_g = viewer.user_scn.maxgeom
                    
                    # A. Draw path capsules
                    if show and coords:
                        if not _ov_built[0]:
                            _rebuild_overlay()
                        n_seg = min(len(_ov1), max_g - geom_idx - 10)
                        for i in range(n_seg):
                            g = viewer.user_scn.geoms[geom_idx]
                            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                                _ZERO3, _ZERO3, _EYE3_FLAT, _GEOM_COLOR)
                            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                                 0.025, _ov1[i], _ov2[i])
                            geom_idx += 1
                            
                    # B. Draw sensor range cylinder under the drone (translucent cyan)
                    if geom_idx < max_g and ap.state != "LANDED":
                        g = viewer.user_scn.geoms[geom_idx]
                        qp = data.qpos; qa = ap.qpos_adr
                        drone_x, drone_y = float(qp[qa]), float(qp[qa+1])
                        terrain_z = get_terrain_height(drone_x, drone_y)
                        
                        size = np.array([ap.scanner.sensor_radius, 0.005, 0.0], dtype=np.float64)
                        pos = np.array([drone_x, drone_y, terrain_z + 0.01], dtype=np.float64)
                        color = np.array([0.0, 0.8, 1.0, 0.15], dtype=np.float32)
                        
                        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER,
                                            size, pos, _EYE3_FLAT, color)
                        geom_idx += 1
                        
                    # C. Draw glowing rings around active dynamic obstacles
                    if hasattr(ap, "obstacle_manager") and ap.obstacle_manager is not None:
                        for name, (ox, oy, oz, orad) in ap.obstacle_manager.active_obstacles.items():
                            if geom_idx < max_g:
                                g = viewer.user_scn.geoms[geom_idx]
                                size = np.array([orad + 0.05, 0.01, 0.0], dtype=np.float64)
                                pos = np.array([ox, oy, oz - 0.2], dtype=np.float64)
                                color = np.array([1.0, 0.0, 0.8, 0.4], dtype=np.float32)
                                
                                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER,
                                                    size, pos, _EYE3_FLAT, color)
                                geom_idx += 1
                                
                    viewer.user_scn.ngeom = geom_idx

                # ── 6. Sync viewer ─────────────────────────────────────────
                viewer.sync()

                # ── 7. Real-time pacing ────────────────────────────────────
                elapsed = time.perf_counter() - t0
                sleep   = model.opt.timestep - elapsed
                if sleep > 1e-4:
                    time.sleep(sleep)

        except KeyboardInterrupt:
            print("\nSimulation terminated by user.")
        finally:
            ap._flush_traj()
            ap.save_discovered_map()
            sensor_suite.generate_plots_and_report()


if __name__ == "__main__":
    run_simulation()
