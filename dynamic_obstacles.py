import math
import random
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import mujoco
from sensor_simulation import get_terrain_height

# Radii for collision/grid marking for each dynamic obstacle
_OBSTACLE_SIZES = {
    "dynamic_obstacle_1": 0.45,
    "dynamic_obstacle_2": 0.50,
    "dynamic_obstacle_3": 0.45,
    "dynamic_obstacle_4": 0.38,
    "dynamic_obstacle_5": 0.35,
}

class DynamicObstacleManager:
    """
    Manages runtime spawning and positioning of dynamic obstacles.
    Repositions static bodies that are pre-defined in scene.xml but kept underground initially.
    """
    def __init__(self, grid_size=160):
        self.grid_size = grid_size
        self.obstacle_names = list(_OBSTACLE_SIZES.keys())
        self.active_obstacles = {}  # maps body_name -> (x, y, z, radius)
        self.inactive_obstacles = list(self.obstacle_names)
        
        # Periodic spawning state
        self.spawn_timer = 0.0
        self.spawn_interval = 12.0  # Spawn a dynamic obstacle every 12 seconds
        self.random_spawn_enabled = False

    def reset(self, model):
        """Reset obstacle positions to hidden underground."""
        self.active_obstacles.clear()
        self.inactive_obstacles = list(self.obstacle_names)
        for name in self.obstacle_names:
            try:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    model.body_pos[body_id] = [0.0, 0.0, -10.0]
            except Exception:
                pass

    def spawn_obstacle(self, model, data, x, y, ground_truth_grid):
        """
        Spawns a dynamic obstacle at coordinates (x, y).
        Moves the body in MuJoCo and updates the ground truth grid.
        """
        if not self.inactive_obstacles:
            # Re-use the oldest active obstacle (FIFO)
            oldest_name = list(self.active_obstacles.keys())[0]
            # Remove from active grid
            ox, oy, _, orad = self.active_obstacles[oldest_name]
            self._update_grid_obstacle(ground_truth_grid, ox, oy, orad, remove=True)
            # Make inactive
            self.inactive_obstacles.append(oldest_name)
            del self.active_obstacles[oldest_name]

        name = self.inactive_obstacles.pop(0)
        radius = _OBSTACLE_SIZES[name]
        
        # Calculate terrain height to place it flat on the floor
        # Obstacles are box or ellipsoid. Box size z is ~0.5 to 0.7.
        # Height offset = half of its z-height (0.5 for large, 0.4 for small)
        z_offset = 0.5 if "1" in name or "2" in name or "3" in name else 0.4
        z = get_terrain_height(x, y) + z_offset

        # Reposition in MuJoCo
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                model.body_pos[body_id] = [x, y, z]
                # Force updates kinematic state
                mujoco.mj_forward(model, data)
                
                self.active_obstacles[name] = (x, y, z, radius)
                # Update ground truth grid so drone scanner can detect it
                self._update_grid_obstacle(ground_truth_grid, x, y, radius, remove=False)
                print(f"[DYNAMIC] Spawned {name} at ({x:.2f}, {y:.2f}, {z:.2f}) with radius {radius:.2f}m")
                return name
        except Exception as e:
            print(f"[DYNAMIC ERROR] Failed to spawn {name}: {e}")
            self.inactive_obstacles.append(name)
        return None

    def spawn_on_path(self, model, data, drone, ground_truth_grid):
        """
        Spawns an obstacle directly in front of the drone along its current path.
        """
        if not drone.shortest_path_coords or not drone.show_path:
            return self.spawn_random(model, data, ground_truth_grid)
        
        # Look ahead 1.2 to 2.5 meters on path
        xy = drone._path_xy
        idx = drone.current_path_idx
        qp = drone.data.qpos; qa = drone.qpos_adr
        dx, dy = float(qp[qa]), float(qp[qa+1])
        
        target_pt = None
        for i in range(idx, len(xy)):
            px, py = float(xy[i, 0]), float(xy[i, 1])
            dist = math.sqrt((px - dx)**2 + (py - dy)**2)
            if 1.2 <= dist <= 2.5:
                target_pt = (px, py)
                break
        
        if not target_pt and idx < len(xy):
            target_pt = (float(xy[-1, 0]), float(xy[-1, 1]))
            
        if target_pt:
            # Spawn exactly on the waypoint
            return self.spawn_obstacle(model, data, target_pt[0], target_pt[1], ground_truth_grid)
        
        return self.spawn_random(model, data, ground_truth_grid)

    def spawn_random(self, model, data, ground_truth_grid):
        """
        Spawns an obstacle at a random valid coordinate away from landing pads.
        """
        pads = [
            (-4.0, -4.0, 1.2),  # Home
            (-4.0,  4.0, 1.0),  # Pad 1
            ( 4.0,  4.0, 1.0),  # Pad 2
            ( 4.0, -4.0, 1.0),  # Pad 3
            ( 0.0,  2.0, 1.0),  # Pad 4
            ( 0.0, -2.0, 1.0)   # Pad 5
        ]
        
        valid = False
        rx, ry = 0.0, 0.0
        max_attempts = 100
        attempts = 0
        
        while not valid and attempts < max_attempts:
            attempts += 1
            rx = random.uniform(-6.0, 6.0)
            ry = random.uniform(-6.0, 6.0)
            
            # Avoid pads
            valid = True
            for px, py, r in pads:
                if math.sqrt((rx - px)**2 + (ry - py)**2) < r:
                    valid = False
                    break
            
            # Avoid active obstacles
            for ox, oy, _, orad in self.active_obstacles.values():
                if math.sqrt((rx - ox)**2 + (ry - oy)**2) < orad + 0.5:
                    valid = False
                    break
                    
        if valid:
            return self.spawn_obstacle(model, data, rx, ry, ground_truth_grid)
        return None

    def step(self, model, data, drone, ground_truth_grid, dt):
        """Updates timers and executes periodic random spawning if enabled."""
        if not self.random_spawn_enabled:
            return None
            
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0.0
            # 80% chance of spawning
            if random.random() < 0.8:
                # If drone is currently flying, spawn in front of it!
                if drone.state == "GOTO_TARGET":
                    return self.spawn_on_path(model, data, drone, ground_truth_grid)
                else:
                    return self.spawn_random(model, data, ground_truth_grid)
        return None

    def _update_grid_obstacle(self, grid, ox, oy, radius, remove=False):
        """Marks or unmarks the obstacle coordinates on the occupancy grid."""
        cols = np.linspace(-8.0, 8.0, self.grid_size, dtype=np.float32)
        rows = np.linspace(-8.0, 8.0, self.grid_size, dtype=np.float32)
        RX, RY = np.meshgrid(cols, rows)
        
        # Obstacle mask
        mask = ((RX - ox)**2 + (RY - oy)**2) < radius**2
        
        # Don't overwrite landing pads
        pads = [
            (-4.0, -4.0, 0.6),  # Home
            (-4.0,  4.0, 0.4),
            ( 4.0,  4.0, 0.4),
            ( 4.0, -4.0, 0.4),
            ( 0.0,  2.0, 0.4),
            ( 0.0, -2.0, 0.4)
        ]
        pad_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        for px, py, pr in pads:
            pad_mask |= ((RX - px)**2 + (RY - py)**2) < pr**2
            
        final_mask = mask & ~pad_mask
        grid[final_mask] = 0 if remove else 1
