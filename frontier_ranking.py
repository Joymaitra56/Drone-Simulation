import math
# pyrefly: ignore [missing-import]
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sensor_fusion import FusedState

# Cell state definitions matching mapping modules
CELL_UNKNOWN = -1
CELL_FREE = 0
CELL_OCCUPIED = 1

class FrontierRanking:
    """
    Frontier Ranking module for autonomous exploration.
    Evaluates frontier regions based on travel cost (via existing A* or Dijkstra),
    information gain (unexplored neighbors), safety (distance to obstacles), and reachability.
    Selects the best target candidate. Implements a caching optimization layer.
    """
    def __init__(self, autopilot: Any, weights: Optional[Dict[str, float]] = None) -> None:
        """
        Initializes the FrontierRanking class.
        
        Args:
            autopilot (Any): Reference to the DroneAutopilot instance.
            weights (Optional[Dict[str, float]]): Configurable weights for scoring components.
                Expected keys: 'info_gain', 'travel_cost', 'safety', 'reachability'.
        """
        self.autopilot: Any = autopilot
        self.grid_size: int = autopilot.grid_size
        
        # Default weights matching example specification
        self.weights: Dict[str, float] = {
            "info_gain": 0.40,
            "travel_cost": 0.30,
            "safety": 0.20,
            "reachability": 0.10
        }
        if weights is not None:
            self.weights.update(weights)

        # Output states
        self.ranked_frontiers: List[Dict[str, Any]] = []
        self._best_frontier: Optional[Dict[str, Any]] = None
        
        # Caching state variables
        self._cached_pos_key: Tuple[float, float] = (0.0, 0.0)
        self._cached_grid_sum: int = -1
        self._cached_regions_len: int = -1

    def update(self, frontier_regions: List[Dict[str, Any]], occupancy_grid: np.ndarray,
               fused_state: FusedState) -> None:
        """
        Evaluates, scores, and ranks all frontier regions.
        Caches results if position, map, and regions lists remain identical.
        
        Args:
            frontier_regions (List[Dict[str, Any]]): List of region structures from detector.
            occupancy_grid (np.ndarray): The mapping grid of shape (grid_size, grid_size).
            fused_state (FusedState): The active fused pose estimate.
        """
        if not fused_state.valid:
            self.ranked_frontiers.clear()
            self._best_frontier = None
            return

        # 1. Check cache keys for optimization
        grid_sum = int(np.sum(occupancy_grid == CELL_FREE) + np.sum(occupancy_grid == CELL_OCCUPIED) * 2)
        pos_key = (round(fused_state.x, 2), round(fused_state.y, 2))
        regions_len = len(frontier_regions)

        if (pos_key == self._cached_pos_key and 
            grid_sum == self._cached_grid_sum and 
            regions_len == self._cached_regions_len and 
            self.ranked_frontiers):
            # Cache hit: no need to re-evaluate or replan paths
            return

        # Cache miss: update keys and compute rankings
        self._cached_pos_key = pos_key
        self._cached_grid_sum = grid_sum
        self._cached_regions_len = regions_len

        self.ranked_frontiers.clear()
        self._best_frontier = None

        if not frontier_regions:
            return

        # Locate drone grid position
        start_g = self.autopilot._world_to_grid(fused_state.x, fused_state.y)
        
        candidate_rankings: List[Dict[str, Any]] = []

        # 2. Evaluate every frontier region independently
        for region in frontier_regions:
            region_id = region["region_id"]
            centroid_g = region["centroid_grid"]
            goal_g = (int(round(centroid_g[0])), int(round(centroid_g[1])))
            size = region["size"]

            # Component A: Travel Cost using existing pathfinder
            path, cost = self.autopilot.plan_path(start_g, goal_g)
            
            reachability = 0.0
            travel_score = 0.0
            if path is not None and not math.isinf(cost):
                reachability = 1.0
                # Exponential decay travel score rewards closer targets (clamped [0.0, 1.0])
                travel_score = math.exp(-0.05 * cost)
            else:
                # Skip isolated or unreachable frontiers
                continue

            # Component B: Information Gain
            # Scan 8-neighbors of all cells in region to count surrounding unknown cells
            unique_unknowns = set()
            offsets = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for gx, gy in region["cells"]:
                for dx, dy in offsets:
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        if occupancy_grid[ny, nx] == CELL_UNKNOWN:
                            unique_unknowns.add((nx, ny))
            
            info_gain = len(unique_unknowns)
            # Normalize information gain score (assumes 50 unknown cells is max utility)
            info_gain_score = min(1.0, info_gain / 50.0)

            # Component C: Safety Score
            # Evaluate distance to nearest occupied cells in a window around the centroid
            min_obs_dist = self._compute_obstacle_clearance(goal_g, occupancy_grid)
            # Normalize safety score (search radius limit is 6.0 grid units)
            safety_score = min(1.0, min_obs_dist / 6.0)

            # 3. Calculate normalized weighted score
            final_score = (
                self.weights["info_gain"] * info_gain_score +
                self.weights["travel_cost"] * travel_score +
                self.weights["safety"] * safety_score +
                self.weights["reachability"] * reachability
            )

            candidate_rankings.append({
                "region_id": region_id,
                "centroid": region["centroid_world"],
                "centroid_grid": centroid_g,
                "size": size,
                "travel_cost": cost if reachability > 0.0 else float('inf'),
                "info_gain": info_gain,
                "safety_score": safety_score,
                "reachability": reachability,
                "score": final_score
            })

        # 4. Sort candidates by final score descending (highest score first)
        # Apply secondary sort key (region_id) to ensure deterministic tie-breaking
        candidate_rankings.sort(key=lambda x: (x["score"], -x["region_id"]), reverse=True)

        self.ranked_frontiers = candidate_rankings
        if self.ranked_frontiers:
            self._best_frontier = self.ranked_frontiers[0]

    def _compute_obstacle_clearance(self, center_g: Tuple[int, int], grid: np.ndarray) -> float:
        """
        Computes the grid distance from center_g to the nearest occupied cell.
        Limits search window to 6 cells to maintain low computational cost.
        
        Args:
            center_g (Tuple[int, int]): Bounding centroid grid coordinate.
            grid (np.ndarray): Occupancy grid.
            
        Returns:
            float: Grid-space distance to nearest obstacle, clamped to 6.0.
        """
        cx, cy = center_g
        search_radius = 6
        min_dist = float(search_radius)

        y_min = max(0, cy - search_radius)
        y_max = min(self.grid_size, cy + search_radius + 1)
        x_min = max(0, cx - search_radius)
        x_max = min(self.grid_size, cx + search_radius + 1)

        # Quick check for occupied cells in the window
        occupied_ys, occupied_xs = np.where(grid[y_min:y_max, x_min:x_max] == CELL_OCCUPIED)
        
        if len(occupied_ys) > 0:
            # Shift back to absolute coordinates
            abs_ys = occupied_ys + y_min
            abs_xs = occupied_xs + x_min
            
            dists = np.sqrt((abs_xs - cx) ** 2 + (abs_ys - cy) ** 2)
            min_dist = float(np.min(dists))

        return min(float(search_radius), min_dist)

    def get_best_frontier(self) -> Optional[Dict[str, Any]]:
        """
        Gets the highest ranked frontier candidate.
        
        Returns:
            Optional[Dict[str, Any]]: The selected frontier, or None if empty.
        """
        return self._best_frontier

    def get_ranked_frontiers(self) -> List[Dict[str, Any]]:
        """
        Gets all evaluated frontiers sorted by score.
        
        Returns:
            List[Dict[str, Any]]: List of ranked frontiers.
        """
        return self.ranked_frontiers

    def get_scores(self) -> List[float]:
        """
        Gets scores of all evaluated frontiers in ranked order.
        
        Returns:
            List[float]: List of scores.
        """
        return [f["score"] for f in self.ranked_frontiers]

    def reset(self) -> None:
        """
        Resets and clears the rankings and cache buffers.
        """
        self.ranked_frontiers.clear()
        self._best_frontier = None
        self._cached_pos_key = (0.0, 0.0)
        self._cached_grid_sum = -1
        self._cached_regions_len = -1
