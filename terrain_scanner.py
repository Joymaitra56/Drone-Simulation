import math
# pyrefly: ignore [missing-import]
import numpy as np

# Landing pads coordinates matching simulate.py for pre-revealing
_PADS = [
    ("Home",  -4.0, -4.0, 0.6),
    ("Pad 1", -4.0,  4.0, 0.4),
    ("Pad 2",  4.0,  4.0, 0.4),
    ("Pad 3",  4.0, -4.0, 0.4),
    ("Pad 4",  0.0,  2.0, 0.4),
    ("Pad 5",  0.0, -2.0, 0.4),
]

class TerrainScanner:
    """
    Onboard Terrain Scanning Layer.
    Maintains an internal discovered occupancy grid and simulates range-based sensing.
    Modular design allows it to work as an omnidirectional sensor or directional LiDAR-like camera.
    """
    def __init__(self, ground_truth_grid, grid_size=160, sensor_radius=2.5, sensor_type="directional", fov_deg=90.0):
        self.grid_size = grid_size
        self.sensor_radius = sensor_radius
        self.sensor_type = sensor_type
        self.fov_rad = math.radians(fov_deg)
        self.ground_truth_grid = ground_truth_grid
        
        # -1 = Unknown/Unexplored, 0 = Free, 1 = Occupied
        self.discovered_grid = np.full((grid_size, grid_size), -1, dtype=np.int8)
        self.explored_mask = np.zeros((grid_size, grid_size), dtype=bool)
        
        # Pre-reveal landing spots so the drone always starts with clear local landing/takeoff zones
        self._reveal_landing_pads()

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

    def _reveal_landing_pads(self):
        """Pre-explores areas around landing pads."""
        for name, px, py, pr in _PADS:
            # Reveal slightly larger than physical pad radius (adding 0.4m)
            self.reveal_circle(px, py, pr + 0.4)

    def reveal_circle(self, cx, cy, radius):
        """Force-reveals all cells within a circular area as ground truth (usually free space)."""
        g_min_x, g_min_y = self._world_to_grid(cx - radius, cy - radius)
        g_max_x, g_max_y = self._world_to_grid(cx + radius, cy + radius)
        
        for gy in range(min(g_min_y, g_max_y), max(g_min_y, g_max_y) + 1):
            for gx in range(min(g_min_x, g_max_x), max(g_min_x, g_max_x) + 1):
                wx, wy = self._grid_to_world(gx, gy)
                dist = math.sqrt((wx - cx)**2 + (wy - cy)**2)
                if dist <= radius:
                    self.explored_mask[gy, gx] = True
                    self.discovered_grid[gy, gx] = self.ground_truth_grid[gy, gx]

    def update(self, drone_x, drone_y, drone_yaw):
        """
        Simulates range scanning around the drone.
        Checks cells within sensor radius and FOV, revealing them in discovered_grid.
        """
        g_min_x, g_min_y = self._world_to_grid(drone_x - self.sensor_radius, drone_y - self.sensor_radius)
        g_max_x, g_max_y = self._world_to_grid(drone_x + self.sensor_radius, drone_y + self.sensor_radius)
        
        for gy in range(min(g_min_y, g_max_y), max(g_min_y, g_max_y) + 1):
            for gx in range(min(g_min_x, g_max_x), max(g_min_x, g_max_x) + 1):
                wx, wy = self._grid_to_world(gx, gy)
                dist = math.sqrt((wx - drone_x)**2 + (wy - drone_y)**2)
                
                if dist <= self.sensor_radius:
                    if self.sensor_type == "directional":
                        # Ray-angle alignment check matching front-facing sensor
                        angle = math.atan2(wy - drone_y, wx - drone_x)
                        diff = angle - drone_yaw
                        # Normalize to [-pi, pi]
                        diff = (diff + math.pi) % (2.0 * math.pi) - math.pi
                        if abs(diff) <= self.fov_rad / 2.0:
                            self.explored_mask[gy, gx] = True
                            self.discovered_grid[gy, gx] = self.ground_truth_grid[gy, gx]
                    else:
                        # Omnidirectional scanner
                        self.explored_mask[gy, gx] = True
                        self.discovered_grid[gy, gx] = self.ground_truth_grid[gy, gx]

    def get_planning_grid(self):
        """
        Returns the occupancy grid mapped for pathfinders (A* / Dijkstra).
        Treats unexplored regions (-1) as free space (0) so the pathfinder 
        plans optimistically through unexplored space.
        """
        grid = np.copy(self.discovered_grid)
        grid[grid == -1] = 0
        return grid
