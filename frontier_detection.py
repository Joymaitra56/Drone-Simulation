# pyrefly: ignore [missing-import]
import numpy as np
from typing import Dict, Any, List, Tuple, Set

# Cell state definitions matching mapping modules
CELL_UNKNOWN = -1
CELL_FREE = 0
CELL_OCCUPIED = 1

class FrontierDetection:
    """
    Frontier Detection module for autonomous exploration.
    Identifies frontier cells (free space adjacent to unexplored/unknown space)
    and groups them into connected regions (components), calculating geometric statistics.
    Optimized for execution efficiency.
    """
    def __init__(self, autopilot: Any, connectivity: int = 8) -> None:
        """
        Initializes the FrontierDetection class.
        
        Args:
            autopilot (Any): Reference to the DroneAutopilot instance.
            connectivity (int): Neighborhood connectivity for frontiers (4 or 8). Defaults to 8.
        """
        self.autopilot: Any = autopilot
        self.grid_size: int = autopilot.grid_size
        self.connectivity: int = connectivity
        
        # State stores
        self.frontier_cells: List[Tuple[int, int]] = []  # List of (gx, gy) grid coords
        self.frontier_regions: List[Dict[str, Any]] = [] # Detailed region dictionaries
        self._update_counter: int = 0

    def update(self, grid: np.ndarray) -> None:
        """
        Extracts frontier cells from the occupancy grid and clusters them into connected regions.
        Uses fast NumPy array shifts to locate candidate boundary cells.
        
        Args:
            grid (np.ndarray): Occupancy grid of shape (grid_size, grid_size).
        """
        self.frontier_cells.clear()
        self.frontier_regions.clear()
        self._update_counter += 1

        height, width = grid.shape
        free_mask = (grid == CELL_FREE)
        unknown_mask = (grid == CELL_UNKNOWN)

        # Offsets based on 4-neighbor or 8-neighbor connectivity
        offsets_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        offsets_8 = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        offsets = offsets_8 if self.connectivity == 8 else offsets_4

        # Union of shifted unknown masks to find free cells adjacent to unknown cells
        has_unknown_neighbor = np.zeros_like(unknown_mask)
        for dy, dx in offsets:
            shifted = np.zeros_like(unknown_mask)
            
            y_src_start, y_src_end = max(0, -dy), height - max(0, dy)
            x_src_start, x_src_end = max(0, -dx), width - max(0, dx)
            
            y_dst_start, y_dst_end = max(0, dy), height - max(0, -dy)
            x_dst_start, x_dst_end = max(0, dx), width - max(0, -dx)
            
            shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = unknown_mask[y_src_start:y_src_end, x_src_start:x_src_end]
            has_unknown_neighbor |= shifted

        # Frontier cells are FREE cells that touch at least one UNKNOWN cell
        frontier_mask = free_mask & has_unknown_neighbor
        
        # Convert mask to list of (gx, gy) coordinate tuples
        ys, xs = np.where(frontier_mask)
        self.frontier_cells = [(int(x), int(y)) for x, y in zip(xs, ys)]
        
        if not self.frontier_cells:
            return

        # Cluster frontier cells into connected components using a queue-based BFS
        self._cluster_frontier_regions(offsets)

    def _cluster_frontier_regions(self, offsets: List[Tuple[int, int]]) -> None:
        """
        Groups adjacent frontier cells into connected components using BFS.
        Uses an index-pointer queue structure to achieve O(V + E) linear execution time.
        
        Args:
            offsets (List[Tuple[int, int]]): Grid neighborhood direction vectors.
        """
        # Convert to set of (gx, gy) tuples for fast O(1) lookups and removals
        unvisited: Set[Tuple[int, int]] = set(self.frontier_cells)
        region_id = 0

        scanner = getattr(self.autopilot, "scanner", None)

        while unvisited:
            # Pick a starting seed node
            start = next(iter(unvisited))
            unvisited.remove(start)
            
            # Connected component collection
            region_cells: List[Tuple[int, int]] = [start]
            queue: List[Tuple[int, int]] = [start]
            
            head = 0
            # BFS queue pointer loop
            while head < len(queue):
                cx, cy = queue[head]
                head += 1
                for dx, dy in offsets:
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        region_cells.append(neighbor)
                        queue.append(neighbor)

            # Compute geometric statistics for this region component
            region_id += 1
            xs = [c[0] for c in region_cells]
            ys = [c[1] for c in region_cells]
            size = len(region_cells)

            # Grid Centroid
            centroid_gx = float(sum(xs)) / size
            centroid_gy = float(sum(ys)) / size

            # Calculate World Centroid using coordinate converters if available
            centroid_wx = 0.0
            centroid_wy = 0.0
            if scanner is not None:
                centroid_wx, centroid_wy = scanner._grid_to_world(centroid_gx, centroid_gy)
            else:
                # Default world mapping conversion fallback
                centroid_wx = -8.0 + 16.0 * centroid_gx / (self.grid_size - 1)
                centroid_wy = -8.0 + 16.0 * centroid_gy / (self.grid_size - 1)

            # Bounding box [min_gx, min_gy, max_gx, max_gy]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

            # Store region
            self.frontier_regions.append({
                "region_id": region_id,
                "cells": region_cells,
                "size": size,
                "centroid_grid": (centroid_gx, centroid_gy),
                "centroid_world": (centroid_wx, centroid_wy),
                "bbox": bbox
            })

    def get_frontiers(self) -> List[Tuple[int, int]]:
        """
        Gets the raw coordinates of all detected frontier cells.
        
        Returns:
            List[Tuple[int, int]]: List of (gx, gy) coordinate tuples.
        """
        return self.frontier_cells

    def get_frontier_regions(self) -> List[Dict[str, Any]]:
        """
        Gets the list of structured frontier regions.
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing region metrics.
        """
        return self.frontier_regions

    def get_frontier_count(self) -> int:
        """
        Gets the count of detected frontier cells.
        
        Returns:
            int: The total cell count.
        """
        return len(self.frontier_cells)

    def reset(self) -> None:
        """
        Resets the detected frontiers list and regions back to defaults.
        """
        self.frontier_cells.clear()
        self.frontier_regions.clear()
        self._update_counter = 0
