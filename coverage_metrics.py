# pyrefly: ignore [missing-import]
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# Constants matching cell states
CELL_UNKNOWN = -1
CELL_FREE = 0
CELL_OCCUPIED = 1

class CoverageMetrics:
    """
    Coverage Analysis module for evaluation of autonomous exploration progress.
    Extracts metrics from the occupancy grid mapping layer and computes coverage percentages,
    growth rates, exploration efficiency, and mission elapsed time. Keeps a history
    of coverage over time.
    """
    def __init__(self, autopilot: Any) -> None:
        """
        Initializes the CoverageMetrics class.
        
        Args:
            autopilot (Any): Reference to the DroneAutopilot instance.
        """
        self.autopilot: Any = autopilot
        self.grid_size: int = autopilot.grid_size
        
        # Dimensions & resolution
        self.resolution: float = 16.0 / self.grid_size
        self.cell_area: float = self.resolution * self.resolution
        
        # Coverage statistics
        self.total_cells: int = self.grid_size * self.grid_size
        self.explored_cells: int = 0
        self.unknown_cells: int = self.total_cells
        self.free_cells: int = 0
        self.occupied_cells: int = 0
        self.coverage_percentage: float = 0.0
        self.coverage_growth_rate: float = 0.0  # in % per second
        self.total_mapped_area: float = 0.0     # in m^2
        self.exploration_efficiency: float = 0.0 # in m^2/sec
        
        # Timing trackers
        self.start_time: Optional[float] = None
        self.elapsed_time: float = 0.0
        
        # Historical tracking data
        self.history: List[Tuple[float, float]] = []  # List of (timestamp, coverage_pct)

    def update(self, grid: np.ndarray, timestamp: float) -> None:
        """
        Processes the current occupancy grid to update all coverage statistics.
        Keeps track of historical coverage progress and filters growth rate estimates.
        
        Args:
            grid (np.ndarray): Sensed occupancy grid of shape (grid_size, grid_size).
            timestamp (float): Current simulation timestamp.
        """
        if self.start_time is None:
            self.start_time = timestamp
            
        self.elapsed_time = timestamp - self.start_time
        
        # Calculate cell counts using fast NumPy array operations
        self.unknown_cells = int(np.sum(grid == CELL_UNKNOWN))
        self.free_cells = int(np.sum(grid == CELL_FREE))
        self.occupied_cells = int(np.sum(grid == CELL_OCCUPIED))
        self.explored_cells = self.free_cells + self.occupied_cells
        
        # Calculate percentages and areas
        self.coverage_percentage = (self.explored_cells / self.total_cells) * 100.0 if self.total_cells > 0 else 0.0
        self.total_mapped_area = self.explored_cells * self.cell_area
        
        # Calculate current growth rate over a moving window of ~1.0 second (to avoid differential noise)
        growth_rate = 0.0
        if len(self.history) > 0:
            target_t = timestamp - 1.0
            found_prev = False
            
            # Search backward to find history close to 1 second ago
            for t_hist, cov_hist in reversed(self.history):
                if t_hist <= target_t:
                    dt = timestamp - t_hist
                    if dt > 1e-3:
                        growth_rate = (self.coverage_percentage - cov_hist) / dt
                        found_prev = True
                        break
            if not found_prev and len(self.history) > 1:
                # Fallback to the oldest history record
                t_hist, cov_hist = self.history[0]
                dt = timestamp - t_hist
                if dt > 1e-3:
                    growth_rate = (self.coverage_percentage - cov_hist) / dt
                    
        self.coverage_growth_rate = growth_rate
        
        # Exploration efficiency: total mapped area in m^2 divided by elapsed time
        self.exploration_efficiency = (
            self.total_mapped_area / self.elapsed_time 
            if self.elapsed_time > 0.1 else 0.0
        )
        
        # Append to historical tracking record
        # Limit history append rate to avoid massive memory accumulation (e.g. max once per 0.1s in sim time)
        if not self.history or (timestamp - self.history[-1][0]) >= 0.1:
            self.history.append((timestamp, self.coverage_percentage))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Compiles all computed exploration metrics into a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary containing active metric telemetry values.
        """
        return {
            "total_cells": self.total_cells,
            "explored_cells": self.explored_cells,
            "unknown_cells": self.unknown_cells,
            "free_cells": self.free_cells,
            "occupied_cells": self.occupied_cells,
            "coverage_percentage": self.coverage_percentage,
            "coverage_growth_rate": self.coverage_growth_rate,
            "total_mapped_area_m2": self.total_mapped_area,
            "exploration_efficiency_m2_per_sec": self.exploration_efficiency,
            "elapsed_time_seconds": self.elapsed_time
        }

    def get_coverage_percentage(self) -> float:
        """Gets current coverage percentage."""
        return self.coverage_percentage

    def get_explored_cells(self) -> int:
        """Gets count of explored cells."""
        return self.explored_cells

    def get_unknown_cells(self) -> int:
        """Gets count of unknown cells."""
        return self.unknown_cells

    def get_occupied_cells(self) -> int:
        """Gets count of occupied cells."""
        return self.occupied_cells

    def get_free_cells(self) -> int:
        """Gets count of free cells."""
        return self.free_cells

    def reset(self) -> None:
        """Resets the history and statistics."""
        self.explored_cells = 0
        self.unknown_cells = self.total_cells
        self.free_cells = 0
        self.occupied_cells = 0
        self.coverage_percentage = 0.0
        self.coverage_growth_rate = 0.0
        self.total_mapped_area = 0.0
        self.exploration_efficiency = 0.0
        self.start_time = None
        self.elapsed_time = 0.0
        self.history.clear()
