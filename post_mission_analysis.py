"""
post_mission_analysis.py — Post-Mission Optimal Route Analysis.

After the mission has ended and the drone has landed, this module uses
the fully completed occupancy grid to compute optimal collision-free
paths between every pair of predefined landing pads.

Design
------
- Offline only: the drone does NOT physically fly these routes.
- Reuses the existing run_astar and run_dijkstra planners from simulate.py
  (passed in as callable references to avoid circular imports).
- Produces three artefacts:
    mission_route_analysis.json  — full results matrix + statistics
    mission_route_map.png        — visual overlay of all computed routes

Grid conventions (matching simulate.py):
    grid_size  = 160 cells
    world span = –8.0 m … +8.0 m  (16 m total)
    cell 0     corresponds to –8.0 m
"""

import time
import math
import json
from typing import List, Tuple, Dict, Any, Optional, Callable

# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Internal coordinate helpers (mirrors simulate.py)
# ---------------------------------------------------------------------------

_WORLD_MIN = -8.0
_WORLD_MAX  =  8.0
_WORLD_SPAN = _WORLD_MAX - _WORLD_MIN


def _world_to_grid(wx: float, wy: float, gs: int) -> Tuple[int, int]:
    """Converts world (x, y) to occupancy grid (col, row)."""
    gx = int(round((wx - _WORLD_MIN) / _WORLD_SPAN * (gs - 1)))
    gy = int(round((wy - _WORLD_MIN) / _WORLD_SPAN * (gs - 1)))
    return (max(0, min(gs - 1, gx)), max(0, min(gs - 1, gy)))


# ---------------------------------------------------------------------------
# Route data container
# ---------------------------------------------------------------------------

class PadRoute:
    """
    Holds the planning result for one (pad_a → pad_b, algorithm) combination.

    Attributes:
        pad_a, pad_b    : Names of the source and destination landing pads.
        algorithm       : "A*" or "Dijkstra".
        path_length     : Number of grid cells in the computed path.
        travel_cost     : Accumulated path cost (same units as the planner).
        computation_ms  : Wall-clock planning time in milliseconds.
        reachable       : False if no path was found.
        path_grid       : List of (col, row) grid coordinates along the path.
    """

    __slots__ = (
        'pad_a', 'pad_b', 'algorithm',
        'path_length', 'travel_cost', 'computation_ms',
        'reachable', 'path_grid',
    )

    def __init__(
        self,
        pad_a: str,
        pad_b: str,
        algorithm: str,
        path_length: int,
        travel_cost: float,
        computation_ms: float,
        reachable: bool,
        path_grid: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        self.pad_a = pad_a
        self.pad_b = pad_b
        self.algorithm = algorithm
        self.path_length = path_length
        self.travel_cost = travel_cost
        self.computation_ms = computation_ms
        self.reachable = reachable
        self.path_grid: List[Tuple[int, int]] = path_grid or []

    def to_dict(self) -> Dict[str, Any]:
        """Serialises the route to a JSON-friendly dictionary."""
        return {
            'pad_a': self.pad_a,
            'pad_b': self.pad_b,
            'algorithm': self.algorithm,
            'path_length_waypoints': self.path_length,
            'travel_cost': round(self.travel_cost, 4) if self.reachable else None,
            'computation_time_ms': round(self.computation_ms, 4),
            'reachable': self.reachable,
        }


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------

class PostMissionAnalysis:
    """
    Offline post-mission route planner and analyser.

    After the autonomous exploration mission completes and the drone has
    landed, this class:

      1. Iterates over every unique pair of landing pads (n*(n-1)/2 pairs).
      2. Plans the optimal route using **both** A* and Dijkstra.
      3. Records path length, travel cost, and wall-clock computation time.
      4. Produces a JSON results file and a PNG route visualisation.

    Usage::

        from post_mission_analysis import PostMissionAnalysis
        analysis = PostMissionAnalysis(pads=_PADS, grid_size=160)
        analysis.run(grid, run_astar, run_dijkstra)
        analysis.print_route_matrix()
        analysis.save_results("mission_route_analysis.json")
        analysis.save_map(grid, "mission_route_map.png")
    """

    def __init__(
        self,
        pads: List[Tuple[str, float, float, float]],
        grid_size: int = 160,
    ) -> None:
        """
        Args:
            pads:      List of (name, x_world, y_world, pad_radius) tuples
                       matching the ``_PADS`` constant in simulate.py.
            grid_size: Occupancy grid side length in cells.
        """
        self.pads: List[Tuple[str, float, float, float]] = pads
        self.grid_size: int = grid_size
        self.routes: List[PadRoute] = []
        self._completed: bool = False

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        grid: np.ndarray,
        astar_fn: Callable,
        dijkstra_fn: Callable,
    ) -> List[Dict[str, Any]]:
        """
        Plans all pairwise routes between every landing pad using both
        A* and Dijkstra algorithms.

        Args:
            grid:        Completed occupancy grid (grid_size × grid_size, int8).
                         Cell values: 0 = free, 1 = occupied, –1 = unknown.
            astar_fn:    Callable: ``run_astar(grid, start, goal) → (path, cost)``.
            dijkstra_fn: Callable: ``run_dijkstra(grid, start, goal) → (path, cost)``.

        Returns:
            List of serialised route dictionaries (one per algorithm per pair).
        """
        self.routes.clear()

        # Resolve grid coordinates for each pad
        pad_grids: List[Tuple[str, Tuple[int, int]]] = []
        for name, wx, wy, _ in self.pads:
            gx, gy = _world_to_grid(wx, wy, self.grid_size)
            pad_grids.append((name, (gx, gy)))

        algorithms: List[Tuple[str, Callable]] = [
            ("A*",       astar_fn),
            ("Dijkstra", dijkstra_fn),
        ]

        n = len(pad_grids)
        total_pairs = n * (n - 1) // 2
        print(f"[POST-MISSION] Planning routes for {n} pads "
              f"({total_pairs} pairs × 2 algorithms)…")

        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                name_a, g_a = pad_grids[i]
                name_b, g_b = pad_grids[j]
                pair_idx += 1

                for alg_name, alg_fn in algorithms:
                    t0 = time.perf_counter()
                    try:
                        path, cost = alg_fn(grid, g_a, g_b)
                    except Exception as exc:
                        print(f"[POST-MISSION] WARNING: {alg_name} "
                              f"{name_a}→{name_b} raised {exc}")
                        path, cost = None, float('inf')
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    reachable = (path is not None and
                                 not math.isinf(cost) and
                                 len(path) > 0)

                    route = PadRoute(
                        pad_a=name_a,
                        pad_b=name_b,
                        algorithm=alg_name,
                        path_length=len(path) if reachable else 0,
                        travel_cost=cost if reachable else float('inf'),
                        computation_ms=elapsed_ms,
                        reachable=reachable,
                        path_grid=list(path) if reachable else [],
                    )
                    self.routes.append(route)

        self._completed = True
        reachable_count = sum(1 for r in self.routes if r.reachable)
        print(f"[POST-MISSION] Analysis complete: "
              f"{reachable_count}/{len(self.routes)} routes reachable.")
        return [r.to_dict() for r in self.routes]

    # ------------------------------------------------------------------
    # Output: JSON results
    # ------------------------------------------------------------------

    def save_results(
        self,
        json_path: str = "mission_route_analysis.json",
    ) -> None:
        """
        Saves the complete analysis results, including summary statistics
        and a comparison table, to a JSON file.

        Args:
            json_path: Output file path.
        """
        if not self._completed:
            print("[POST-MISSION] Analysis not yet run — nothing to save.")
            return

        def _stats(routes: List[PadRoute]) -> Dict[str, Any]:
            reachable = [r for r in routes if r.reachable]
            if not reachable:
                return {"reachable_pairs": 0}
            costs = [r.travel_cost for r in reachable]
            times = [r.computation_ms for r in reachable]
            lengths = [r.path_length for r in reachable]
            return {
                "reachable_pairs":    len(reachable),
                "mean_cost":          round(sum(costs) / len(costs), 4),
                "min_cost":           round(min(costs), 4),
                "max_cost":           round(max(costs), 4),
                "mean_path_length":   round(sum(lengths) / len(lengths), 2),
                "mean_computation_ms": round(sum(times) / len(times), 4),
                "total_computation_ms": round(sum(times), 4),
            }

        astar_routes    = [r for r in self.routes if r.algorithm == "A*"]
        dijkstra_routes = [r for r in self.routes if r.algorithm == "Dijkstra"]

        # Build A* vs Dijkstra comparison for each pair
        comparisons: List[Dict[str, Any]] = []
        seen: set = set()
        for r in self.routes:
            key = (r.pad_a, r.pad_b)
            if key in seen:
                continue
            seen.add(key)
            astar_r    = next((x for x in astar_routes
                               if x.pad_a == r.pad_a and x.pad_b == r.pad_b), None)
            dijkstra_r = next((x for x in dijkstra_routes
                               if x.pad_a == r.pad_a and x.pad_b == r.pad_b), None)

            entry: Dict[str, Any] = {"pair": f"{r.pad_a} ↔ {r.pad_b}"}
            if astar_r and dijkstra_r and astar_r.reachable and dijkstra_r.reachable:
                cost_diff = dijkstra_r.travel_cost - astar_r.travel_cost
                time_diff = dijkstra_r.computation_ms - astar_r.computation_ms
                entry["astar_cost"]      = round(astar_r.travel_cost, 4)
                entry["dijkstra_cost"]   = round(dijkstra_r.travel_cost, 4)
                entry["cost_diff"]       = round(cost_diff, 4)
                entry["astar_ms"]        = round(astar_r.computation_ms, 4)
                entry["dijkstra_ms"]     = round(dijkstra_r.computation_ms, 4)
                entry["time_diff_ms"]    = round(time_diff, 4)
                entry["faster_algorithm"] = (
                    "A*" if astar_r.computation_ms < dijkstra_r.computation_ms
                    else "Dijkstra"
                )
            comparisons.append(entry)

        data: Dict[str, Any] = {
            "summary": {
                "total_pads":  len(self.pads),
                "total_pairs": len(self.pads) * (len(self.pads) - 1) // 2,
                "A*":          _stats(astar_routes),
                "Dijkstra":    _stats(dijkstra_routes),
            },
            "comparison": comparisons,
            "routes":     [r.to_dict() for r in self.routes],
        }

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[POST-MISSION] Results saved -> {json_path}")
        except Exception as exc:
            print(f"[POST-MISSION] Failed to save results: {exc}")

    # ------------------------------------------------------------------
    # Output: route map PNG
    # ------------------------------------------------------------------

    def save_map(
        self,
        grid: np.ndarray,
        png_path: str = "mission_route_map.png",
        visual_size: int = 640,
    ) -> None:
        """
        Renders a visualisation of the completed occupancy grid overlaid
        with every computed route.

        Route colours:
          - A*       → bright green  (30, 220, 100)
          - Dijkstra → cyan          (0,  190, 230)

        Args:
            grid:        Completed occupancy grid array.
            png_path:    Output PNG file path.
            visual_size: Square image dimension in pixels.
        """
        if not self._completed:
            print("[POST-MISSION] Analysis not yet run — no map to save.")
            return

        gs = self.grid_size
        cell_w = visual_size / gs

        def to_px(wx: float, wy: float) -> Tuple[int, int]:
            return (
                int((wx - _WORLD_MIN) / _WORLD_SPAN * (visual_size - 1)),
                int((_WORLD_MAX - wy)  / _WORLD_SPAN * (visual_size - 1)),
            )

        def grid_to_px(gx: int, gy: int) -> Tuple[int, int]:
            wx = _WORLD_MIN + _WORLD_SPAN * gx / (gs - 1)
            wy = _WORLD_MIN + _WORLD_SPAN * gy / (gs - 1)
            return to_px(wx, wy)

        # ── Canvas ─────────────────────────────────────────────────────────
        img  = Image.new("RGB", (visual_size, visual_size), (10, 14, 22))
        draw = ImageDraw.Draw(img)

        # ── Grid lines ─────────────────────────────────────────────────────
        for g in np.arange(_WORLD_MIN, _WORLD_MAX + 0.1, 1.0):
            draw.line([to_px(g, _WORLD_MIN), to_px(g, _WORLD_MAX)],
                      fill=(0, 35, 55), width=1)
            draw.line([to_px(_WORLD_MIN, g), to_px(_WORLD_MAX, g)],
                      fill=(0, 35, 55), width=1)

        # ── Occupied cells ──────────────────────────────────────────────────
        ys_occ, xs_occ = np.where(grid == 1)
        for r_c, c_c in zip(ys_occ, xs_occ):
            tx = c_c * cell_w
            ty = (gs - 1 - r_c) * cell_w
            draw.rectangle([tx, ty, tx + cell_w, ty + cell_w],
                           fill=(200, 55, 55))

        # ── Free cells (subtle fill to show mapped area) ───────────────────
        ys_free, xs_free = np.where(grid == 0)
        for r_c, c_c in zip(ys_free, xs_free):
            tx = c_c * cell_w
            ty = (gs - 1 - r_c) * cell_w
            draw.rectangle([tx, ty, tx + cell_w, ty + cell_w],
                           fill=(18, 38, 52))

        # ── Route paths (draw actual grid paths, A* first then Dijkstra) ───
        route_colours = {"A*": (30, 220, 100), "Dijkstra": (0, 190, 230)}
        route_widths  = {"A*": 2, "Dijkstra": 2}

        for route in self.routes:
            if not route.reachable or len(route.path_grid) < 2:
                continue
            colour = route_colours.get(route.algorithm, (200, 200, 200))
            width  = route_widths.get(route.algorithm, 1)
            pts = [grid_to_px(gx, gy) for gx, gy in route.path_grid]
            draw.line(pts, fill=colour, width=width)

        # ── Landing pads ───────────────────────────────────────────────────
        pad_world = {name: (wx, wy) for name, wx, wy, _ in self.pads}
        for name, wx, wy, pr in self.pads:
            cx, cy = to_px(wx, wy)
            rp = max(5, int(pr / _WORLD_SPAN * visual_size))
            draw.ellipse([cx - rp, cy - rp, cx + rp, cy + rp],
                         outline=(0, 255, 128), width=3)
            draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4],
                         fill=(0, 255, 128))
            draw.text((cx + rp + 4, cy - 6), name, fill=(0, 255, 128))

        # ── Legend ─────────────────────────────────────────────────────────
        draw.rectangle([4, 4, 160, 52], fill=(0, 0, 0, 180))
        draw.text(( 8,  8), "A* route",       fill=(30, 220, 100))
        draw.text(( 8, 22), "Dijkstra route", fill=(0,  190, 230))
        draw.text(( 8, 36), "Landing Pad",    fill=(0,  255, 128))

        try:
            img.save(png_path)
            print(f"[POST-MISSION] Route map saved → {png_path}")
        except Exception as exc:
            print(f"[POST-MISSION] Failed to save map: {exc}")

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def print_route_matrix(self) -> None:
        """Prints a formatted route matrix table to stdout."""
        if not self._completed:
            print("[POST-MISSION] Analysis not yet run.")
            return

        print("\n[POST-MISSION] ── Route Analysis Matrix ────────────────────────")
        header = f"  {'PAD A':<12} {'PAD B':<12} {'ALGO':<10} {'COST':>9} {'LEN':>5} {'ms':>8}  OK"
        print(header)
        print("  " + "─" * (len(header) - 2))
        for r in self.routes:
            cost_s = f"{r.travel_cost:9.2f}" if r.reachable else "      N/A"
            ok_s   = "✓" if r.reachable else "✗"
            print(f"  {r.pad_a:<12} {r.pad_b:<12} {r.algorithm:<10} "
                  f"{cost_s} {r.path_length:>5} {r.computation_ms:>8.3f}  {ok_s}")
        print("  " + "─" * (len(header) - 2))

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def is_completed(self) -> bool:
        """True if ``run()`` has been called at least once."""
        return self._completed

    def get_best_route(
        self,
        pad_a: str,
        pad_b: str,
        algorithm: str = "A*",
    ) -> Optional[PadRoute]:
        """
        Looks up the route record for a specific pad pair and algorithm.

        Args:
            pad_a:     Source pad name.
            pad_b:     Destination pad name.
            algorithm: "A*" or "Dijkstra".

        Returns:
            PadRoute if found, None otherwise.
        """
        for r in self.routes:
            if r.algorithm != algorithm:
                continue
            if (r.pad_a == pad_a and r.pad_b == pad_b) or \
               (r.pad_a == pad_b and r.pad_b == pad_a):
                return r
        return None
