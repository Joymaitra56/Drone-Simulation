import math
import time
from sensor_simulation import get_terrain_height
from mission_state import MissionState

class Behavior:
    """Base class for all subsumption behaviors."""
    def __init__(self, name, priority):
        self.name = name
        self.priority = priority

    def check_trigger(self, drone):
        """Returns True if the behavior conditions are met and it wants to execute."""
        raise NotImplementedError

    def execute(self, drone, dt):
        """Executes the behavior logic, outputting forces/states to the drone."""
        raise NotImplementedError


class BehaviorManager:
    """Orchestrates behaviors, ensuring only the highest priority active behavior runs."""
    def __init__(self):
        self.behaviors = []
        self.last_active_name = None

    def add_behavior(self, behavior):
        self.behaviors.append(behavior)
        # Sort by priority descending (highest priority first)
        self.behaviors.sort(key=lambda b: b.priority, reverse=True)

    def step(self, drone, dt):
        """Evaluates and executes the highest-priority active behavior."""
        # Fast path: if landed, just clear forces
        if drone.state == "LANDED":
            drone.physics.clear_forces()
            if self.last_active_name != "Landed":
                print("[BEHAVIOR] Active behavior: Landed (Idle)")
                self.last_active_name = "Landed"
            return "Landed"

        for behavior in self.behaviors:
            if behavior.check_trigger(drone):
                if self.last_active_name != behavior.name:
                    print(f"[BEHAVIOR] Active behavior: {behavior.name} (Priority {behavior.priority})")
                    self.last_active_name = behavior.name
                behavior.execute(drone, dt)
                return behavior.name
        
        # Fallback: if no behavior triggered, clear forces
        drone.physics.clear_forces()
        return None


class KillSwitchBehavior(Behavior):
    """Priority 5: Emergency Stop / Kill Switch. Instantly kills motor thrust."""
    def __init__(self):
        super().__init__("Kill Switch", 5)

    def check_trigger(self, drone):
        return getattr(drone, "kill_switch_active", False)

    def execute(self, drone, dt):
        drone.physics.clear_forces()
        drone.state = "LANDED"
        drone.kill_switch_active = False  # Reset
        drone.target_pad = None
        drone.target_pos = None
        drone.shortest_path_coords = None
        drone._path_xy = None
        drone._path_z = None
        drone.show_path = False
        drone.log_event("KILL_SWITCH_TRIGGERED")
        print("\n=======================================================")
        print("!!! [KILL SWITCH ACTIVATED] MOTOR THRUST CUT IMMEDIATELY !!!")
        print("=======================================================")


class EmergencyLandingBehavior(Behavior):
    """Priority 4: Emergency Landing. Descent in-place to the terrain."""
    def __init__(self):
        super().__init__("Emergency Landing", 4)

    def check_trigger(self, drone):
        return drone.state == "LANDING_HOME"

    def execute(self, drone, dt):
        qp = drone.data.qpos; qa = drone.qpos_adr
        dx, dy, dz = float(qp[qa]), float(qp[qa+1]), float(qp[qa+2])
        
        tz = get_terrain_height(dx, dy) + drone.resting_height
        drone.physics.apply_forces(dx, dy, tz, target_yaw=0.0, dt=dt)
        
        if dz <= get_terrain_height(dx, dy) + drone.resting_height + 0.08:
            drone.state = "LANDED"
            drone.physics.clear_forces()
            drone._flush_traj()
            drone.log_event("EMERGENCY_LAND", "", dx, dy, dz)
            drone.shortest_path_coords = None
            drone._path_xy = None
            drone._path_z = None
            drone.show_path = False
            print("[CTRL] Safe landing complete.")


class ObstacleAvoidanceBehavior(Behavior):
    """Priority 3: Obstacle Avoidance & Path Replanning."""
    def __init__(self):
        super().__init__("Obstacle Avoidance / Replan", 3)

    def check_trigger(self, drone):
        # Trigger replanning if drone is actively navigating to a target and path gets blocked
        return drone.state == "GOTO_TARGET" and getattr(drone, "path_blocked", False)

    def execute(self, drone, dt):
        # 1. Pause normal navigation
        drone.physics.clear_forces()
        
        qp = drone.data.qpos; qa = drone.qpos_adr
        dx, dy = float(qp[qa]), float(qp[qa+1])
        
        # 2. Get current grid and goal grid positions
        start_g = drone._world_to_grid(dx, dy)
        tx, ty = drone.target_pos
        goal_g = drone._world_to_grid(tx, ty)
        
        print(f"\n[AVOIDANCE] Re-routing to {drone.target_pad} due to dynamic obstacle blockage...")
        
        # 3. Recalculate optimal path using A* or Dijkstra
        path_grid, cost = drone.plan_path(start_g, goal_g)
        
        if path_grid is not None:
            # Rebuild waypoints in world coords
            sp_points = []
            for (gx, gy) in path_grid:
                wx, wy = drone._grid_to_world(gx, gy)
                sp_points.append({"grid_x": gx, "grid_y": gy, "x": round(wx,3), "y": round(wy,3)})
                
            drone.shortest_path_coords = sp_points
            drone._build_path_arrays()
            drone.current_path_idx = 0
            # Reset the blocked flag
            drone.path_blocked = False
            drone.log_event("REPLAN_SUCCESS", drone.target_pad, tx, ty)
            drone.save_discovered_map()
            print(f"[AVOIDANCE] Path re-routing successful. Waypoints: {len(sp_points)}, cost: {cost:.1f}")
        else:
            # If trapped/no path, transition to hover
            drone.state = "HOVER"
            drone.path_blocked = False
            drone.shortest_path_coords = None
            drone._path_xy = None
            drone._path_z = None
            drone.log_event("REPLAN_FAILED", drone.target_pad)
            print("[AVOIDANCE ERROR] Re-routing failed: no path found! Hovering in place.")


class NavigationBehavior(Behavior):
    """Priority 2: Normal Navigation (Takeoff, Path Following, Hover, Autoland)."""
    def __init__(self):
        super().__init__("Navigation", 2)

    def check_trigger(self, drone):
        return drone.state in ("TAKEOFF", "HOVER", "GOTO_TARGET", "AUTOLAND")

    def execute(self, drone, dt):
        qa = drone.qpos_adr; va = drone.qvel_adr
        qp = drone.data.qpos; qv = drone.data.qvel
        dx, dy, dz = float(qp[qa]), float(qp[qa+1]), float(qp[qa+2])

        # ── TAKEOFF ───────────────────────────────────────────────────────
        if drone.state == "TAKEOFF":
            if not drone.show_path and drone.target_pad is not None:
                drone.state = "HOVER"
                print("\n[NAV] Path line disappeared! Drone hovering in place.")
                return

            cur_terrain = drone._get_pad_terrain(dx, dy)
            tz = cur_terrain + drone.hover_height
            drone.physics.apply_forces(dx, dy, tz, target_yaw=0.0, dt=dt)
            if abs(dz - tz) < 0.10 and abs(float(qv[va+2])) < 0.2:
                drone.log_event("HOVER_REACHED")
                if drone.target_pad is not None:
                    drone.state = "GOTO_TARGET"
                    drone.log_event("GOTO_TARGET_START", drone.target_pad,
                                   drone.target_pos[0], drone.target_pos[1])
                    print(f"[NAV] Hover reached. Flying to {drone.target_pad}...")
                else:
                    drone.state = "HOVER"
                    print("[CTRL] Hover reached. Use keys to fly.")

        # ── HOVER ─────────────────────────────────────────────────────────
        elif drone.state == "HOVER":
            tz = get_terrain_height(dx, dy) + drone.hover_height
            drone.physics.apply_forces(dx, dy, tz, target_yaw=0.0, dt=dt)

        # ── GOTO_TARGET (Path Following) ──────────────────────────────────
        elif drone.state == "GOTO_TARGET":
            if not drone.show_path:
                drone.state = "HOVER"
                print("\n[NAV] Path line disappeared! Drone hovering in place.")
                return

            target_pt = drone._detect_line_sensor()

            if target_pt is None:
                tz = get_terrain_height(dx, dy) + drone.hover_height
                drone.physics.apply_forces(dx, dy, tz, target_yaw=0.0, dt=dt)
            else:
                tx_pt, ty_pt = float(target_pt[0]), float(target_pt[1])
                vxd = tx_pt - dx;  vyd = ty_pt - dy
                dist = math.sqrt(vxd*vxd + vyd*vyd)
                if dist > 0.01:
                    inv = 1.0 / dist
                    des_vx = vxd * inv * drone.cruise_speed
                    des_vy = vyd * inv * drone.cruise_speed
                    des_yaw = math.atan2(vyd, vxd) if dist > 0.05 else None
                else:
                    des_vx = des_vy = 0.0;  des_yaw = None

                tz = get_terrain_height(dx, dy) + drone.hover_height
                drone.physics.apply_forces(tx_pt, ty_pt, tz,
                                          target_yaw=des_yaw,
                                          target_vx=des_vx, target_vy=des_vy,
                                          dt=dt, mode="velocity")

            # Check arrival at target pad
            if drone.target_pos is not None:
                tpx, tpy = drone.target_pos
                ex = dx - tpx;  ey = dy - tpy
                if ex*ex + ey*ey < 0.30**2:   # within 30 cm of pad centre
                    print(f"\n[NAV] Arrived at {drone.target_pad}. Initiating landing...")
                    drone.state = "AUTOLAND"
                    drone.land_state = "DESCENDING"
                    drone.land_pad_name = drone.target_pad
                    drone.land_pad_pos = drone.target_pos
                    drone.log_event("PAD_ARRIVED", drone.target_pad, tpx, tpy)
                    drone.target_pad = None
                    drone.target_pos = None

        # ── AUTOLAND ──────────────────────────────────────────────────────
        elif drone.state == "AUTOLAND":
            lx, ly = drone.land_pad_pos
            lt = drone._get_pad_terrain(lx, ly)

            if drone.land_state == "DESCENDING":
                tz = lt + drone.resting_height
                drone.physics.apply_forces(lx, ly, tz, target_yaw=0.0, dt=dt)
                if dz <= lt + drone.resting_height + 0.06:
                    drone.land_state = "WAITING"
                    drone.land_start_time = time.time()
                    drone.current_pad = drone.land_pad_name
                    drone.log_event("TOUCHDOWN", drone.land_pad_name, lx, ly, dz)
                    print(f"[LAND] Landed on {drone.land_pad_name}. Waiting 2 s...")

            elif drone.land_state == "WAITING":
                tz = lt + drone.resting_height
                drone.physics.apply_forces(lx, ly, tz, target_yaw=0.0, dt=dt)
                if time.time() - drone.land_start_time >= 2.0:
                    drone.state = "LANDED"
                    drone.physics.clear_forces()
                    drone._flush_traj()
                    drone.log_event("LANDED", drone.land_pad_name, lx, ly, dz)
                    drone.shortest_path_coords = None
                    drone._path_xy = None
                    drone._path_z = None
                    drone.show_path = False
                    drone.save_discovered_map()
                    print(f"[LAND] Drone landed at {drone.land_pad_name}. Use keys to fly next.")



class ReturnHomeBehavior(Behavior):
    """
    Priority 2.5: IMU-Based Return Home via trajectory reversal.

    Activates when ``drone.auto_return_active`` is True (set by
    DroneAutopilot._trigger_auto_return() after exploration ends).
    Pulls waypoints from ``drone.trajectory_replayer`` and commands
    the physics controller toward each one in turn.

    On completion, transitions the autopilot to LANDING_HOME so that the
    existing EmergencyLandingBehavior (priority 4) performs the descent.
    """

    def __init__(self) -> None:
        super().__init__("Return Home (Trajectory Reversal)", 2.5)

    def check_trigger(self, drone) -> bool:
        """Active whenever auto_return_active flag is set by the autopilot."""
        return getattr(drone, "auto_return_active", False)

    def execute(self, drone, dt) -> None:
        qa = drone.qpos_adr
        qp = drone.data.qpos
        dx, dy, dz = float(qp[qa]), float(qp[qa + 1]), float(qp[qa + 2])

        # Ask the replayer for the next waypoint
        wp = drone.trajectory_replayer.get_next_waypoint(dx, dy)

        if wp is None:
            # ── Replayer reports home reached → hand off to landing ─────
            drone.auto_return_active = False
            drone.mission_manager.transition_to(MissionState.LAND)
            drone.state = "LANDING_HOME"
            progress_pct = drone.trajectory_replayer.get_progress() * 100.0
            print(
                f"[RETURN] Home reached ({progress_pct:.0f}% trajectory consumed). "
                "Initiating landing sequence…"
            )
            return

        wx, wy, wz = wp

        # ── Target altitude: terrain + hover height ──────────────────────
        tz = get_terrain_height(dx, dy) + drone.hover_height

        # ── Velocity vector toward waypoint ─────────────────────────────
        vxd = wx - dx
        vyd = wy - dy
        dist = math.sqrt(vxd * vxd + vyd * vyd)

        if dist > 0.01:
            inv = 1.0 / dist
            des_vx  = vxd * inv * drone.cruise_speed
            des_vy  = vyd * inv * drone.cruise_speed
            des_yaw = math.atan2(vyd, vxd) if dist > 0.05 else None
        else:
            des_vx = des_vy = 0.0
            des_yaw = None

        # ── Apply forces via the existing physics controller ─────────────
        drone.physics.apply_forces(
            wx, wy, tz,
            target_yaw=des_yaw,
            target_vx=des_vx,
            target_vy=des_vy,
            dt=dt,
            mode="velocity",
        )


class TerrainScanningBehavior(Behavior):
    """Priority 1: Terrain Scanning (360° spin & onboard mapping)."""
    def __init__(self):
        super().__init__("Terrain Scanning", 1)

    def check_trigger(self, drone):
        return drone.state == "SCANNING"

    def execute(self, drone, dt):
        qa = drone.qpos_adr
        qp = drone.data.qpos
        dx, dy = float(qp[qa]), float(qp[qa+1])
        
        tz = get_terrain_height(dx, dy) + drone.hover_height
        drone.physics.apply_scan_spin(drone.scan_speed, dt=dt)
        
        # Altitude correction on top of spin
        pos_z = float(drone.data.xpos[drone.drone_body_id][2])
        drone.data.xfrc_applied[drone.drone_body_id, 2] += drone.physics.KP_Z * (tz - pos_z)
        
        # Accumulate rotation
        drone.scan_accumulated_yaw += abs(float(drone.data.cvel[drone.drone_body_id][2])) * dt
        
        if drone.scan_accumulated_yaw >= 2.0 * math.pi:
            drone.state = "HOVER"
            drone.scan_accumulated_yaw = 0.0
            drone.log_event("SCAN_COMPLETE")
            print("[SCAN] 360° scan complete! Terrain scanner mapping finished.")
            drone.shortest_path_coords = None   # clear old path
            drone._path_xy = None
            drone._path_z = None
            drone.show_path = False
            drone.save_discovered_map()
            print("[SCAN] Map ready. Use keys to fly.")
