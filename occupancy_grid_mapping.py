import math
# pyrefly: ignore [missing-import]
import numpy as np
from typing import Dict, Any, Tuple
from sensor_fusion import FusedState

# Define cell state constants matching simulate.py and terrain_scanner.py
CELL_UNKNOWN = -1
CELL_FREE = 0
CELL_OCCUPIED = 1

class OccupancyGridMapping:
    """
    Incremental Occupancy Grid Mapping module.
    Maintains a probabilistic grid using log-odds accumulation based on
    fused state estimates and sensor FOV bounds. Highly optimized using
    vectorized NumPy computations.
    """
    def __init__(self, autopilot: Any, min_log_odds: float = -3.0, max_log_odds: float = 3.0,
                 threshold_free: float = -1.0, threshold_occupied: float = 1.0) -> None:
        """
        Initializes the OccupancyGridMapping class.
        
        Args:
            autopilot (Any): Reference to the DroneAutopilot instance.
            min_log_odds (float): Clamped minimum value of log-odds.
            max_log_odds (float): Clamped maximum value of log-odds.
            threshold_free (float): Threshold below which a cell is marked FREE.
            threshold_occupied (float): Threshold above which a cell is marked OCCUPIED.
        """
        self.autopilot: Any = autopilot
        self.grid_size: int = autopilot.grid_size
        self.min_log_odds: float = min_log_odds
        self.max_log_odds: float = max_log_odds
        self.threshold_free: float = threshold_free
        self.threshold_occupied: float = threshold_occupied

        # Log-odds representation initialized to 0.0 (representing P=0.5 probability)
        self.log_odds: np.ndarray = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Pre-reveal landing spots so they are marked free initially
        self._sync_and_pre_reveal()

    def _sync_and_pre_reveal(self) -> None:
        """Synchronizes and initializes pre-revealed spots with log-odds representation."""
        scanner = getattr(self.autopilot, "scanner", None)
        if scanner is not None:
            # Set pre-revealed landing pad cells to strong free log-odds (-3.0)
            free_mask = (scanner.discovered_grid == CELL_FREE)
            self.log_odds[free_mask] = self.min_log_odds
            
            # Set pre-revealed rocks to strong occupied log-odds (3.0)
            occupied_mask = (scanner.discovered_grid == CELL_OCCUPIED)
            self.log_odds[occupied_mask] = self.max_log_odds

    def update(self, fused_state: FusedState, sensor_data: Any) -> None:
        """
        Incrementally updates the occupancy grid using fused state estimates.
        Uses the scanner FOV to update log-odds values of local cells.
        Uses highly efficient vectorized meshgrid matrix operations.
        
        Args:
            fused_state (FusedState): Fused pose estimate.
            sensor_data (Any): Reference to the SensorSimulationSuite.
        """
        if not fused_state.valid:
            return

        scanner = getattr(self.autopilot, "scanner", None)
        if scanner is None:
            return

        drone_x: float = fused_state.x
        drone_y: float = fused_state.y
        drone_yaw: float = fused_state.yaw
        sensor_radius: float = scanner.sensor_radius
        fov_rad: float = scanner.fov_rad
        sensor_type: str = scanner.sensor_type
        
        # Determine grid indices corresponding to the sensor's boundary area
        g_min_x, g_min_y = scanner._world_to_grid(drone_x - sensor_radius, drone_y - sensor_radius)
        g_max_x, g_max_y = scanner._world_to_grid(drone_x + sensor_radius, drone_y + sensor_radius)

        x_start: int = min(g_min_x, g_max_x)
        x_end: int = max(g_min_x, g_max_x) + 1
        y_start: int = min(g_min_y, g_max_y)
        y_end: int = max(g_min_y, g_max_y) + 1

        # Extract coordinates of the Region of Interest (ROI)
        cols = np.linspace(-8.0, 8.0, self.grid_size, dtype=np.float32)[x_start:x_end]
        rows = np.linspace(-8.0, 8.0, self.grid_size, dtype=np.float32)[y_start:y_end]
        
        if len(cols) == 0 or len(rows) == 0:
            return

        RX, RY = np.meshgrid(cols, rows)
        
        # Distances from drone to all cells in ROI
        DX = RX - drone_x
        DY = RY - drone_y
        dists_sq = DX * DX + DY * DY
        radius_mask = dists_sq <= (sensor_radius * sensor_radius)
        
        if sensor_type == "directional":
            angles = np.arctan2(DY, DX)
            diffs = angles - drone_yaw
            # Normalize to [-pi, pi]
            diffs = (diffs + np.pi) % (2.0 * np.pi) - np.pi
            fov_mask = np.abs(diffs) <= (fov_rad / 2.0)
            mask = radius_mask & fov_mask
        else:
            mask = radius_mask

        # For cells in view, perform log-odds update
        if np.any(mask):
            # Crop ground truth and log-odds to the ROI
            gt_roi = scanner.ground_truth_grid[y_start:y_end, x_start:x_end]
            lo_roi = self.log_odds[y_start:y_end, x_start:x_end]
            disc_roi = scanner.discovered_grid[y_start:y_end, x_start:x_end]
            expl_roi = scanner.explored_mask[y_start:y_end, x_start:x_end]
            
            # Mask applied to the ROI slice
            mask_roi = mask
            
            # Where ground truth is occupied (1) and in view:
            occ_mask = mask_roi & (gt_roi == CELL_OCCUPIED)
            # Where ground truth is free (0) and in view:
            free_mask = mask_roi & (gt_roi == CELL_FREE)
            
            # Update log-odds
            lo_roi[occ_mask] = np.minimum(self.max_log_odds, lo_roi[occ_mask] + 0.8)
            lo_roi[free_mask] = np.maximum(self.min_log_odds, lo_roi[free_mask] - 0.4)
            
            # Apply thresholds to update states
            t_occ = lo_roi >= self.threshold_occupied
            disc_roi[mask_roi & t_occ] = CELL_OCCUPIED
            expl_roi[mask_roi & t_occ] = True
            
            t_free = lo_roi <= self.threshold_free
            
            # For cells that were previously occupied (1), require log_odds to drop below -2.0 to clear (hysteresis)
            prev_occ_mask = (disc_roi == CELL_OCCUPIED)
            clear_occ_mask = mask_roi & t_free & prev_occ_mask & (lo_roi <= -2.0)
            clear_other_mask = mask_roi & t_free & ~prev_occ_mask
            
            disc_roi[clear_occ_mask] = CELL_FREE
            expl_roi[clear_occ_mask] = True
            
            disc_roi[clear_other_mask] = CELL_FREE
            expl_roi[clear_other_mask] = True
            
        # Synchronize autopilot's main planning grid
        self.autopilot.occ_grid = scanner.get_planning_grid()

    def get_grid(self) -> np.ndarray:
        """
        Gets the current occupancy grid state (-1 = Unknown, 0 = Free, 1 = Occupied).
        
        Returns:
            np.ndarray: The occupancy grid array.
        """
        scanner = getattr(self.autopilot, "scanner", None)
        if scanner is not None:
            return scanner.discovered_grid
        return np.full((self.grid_size, self.grid_size), CELL_UNKNOWN, dtype=np.int8)

    def reset(self) -> None:
        """
        Resets the mapping log-odds representation and synchronizes with TerrainScanner.
        """
        self.log_odds.fill(0.0)
        scanner = getattr(self.autopilot, "scanner", None)
        if scanner is not None:
            scanner.discovered_grid.fill(CELL_UNKNOWN)
            scanner.explored_mask.fill(False)
            scanner._reveal_landing_pads()
        self._sync_and_pre_reveal()
        
        # Synchronize planning grid
        if scanner is not None:
            self.autopilot.occ_grid = scanner.get_planning_grid()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Computes statistics regarding mapping metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing metric values.
        """
        grid = self.get_grid()
        total_cells = int(self.grid_size * self.grid_size)
        
        unknown_cells = int(np.sum(grid == CELL_UNKNOWN))
        free_cells = int(np.sum(grid == CELL_FREE))
        occupied_cells = int(np.sum(grid == CELL_OCCUPIED))
        explored_cells = free_cells + occupied_cells
        
        coverage = (explored_cells / total_cells) * 100.0 if total_cells > 0 else 0.0

        return {
            "total_cells": total_cells,
            "explored_cells": explored_cells,
            "unknown_cells": unknown_cells,
            "free_cells": free_cells,
            "occupied_cells": occupied_cells,
            "coverage_percentage": coverage
        }
