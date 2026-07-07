"""
imu_trajectory.py — IMU-Based Trajectory Recording and Reverse Replay.

Provides two classes that together implement dead-reckoning–based
return-home capability for the autonomous exploration pipeline:

  TrajectoryRecorder  — records pose snapshots from SensorFusion during
                        exploration at a configurable frequency.
  TrajectoryReplayer  — reverses and smooths the recorded trajectory,
                        exposing a waypoint-by-waypoint interface used by
                        ReturnHomeBehavior in subsumption.py.

Design principles
-----------------
- Zero dependency on A* / Dijkstra: uses only the fused pose history.
- Non-intrusive: recorder just appends to an in-memory list; nothing is
  written to disk until save_mission_data() requests it.
- Thread-safe for single-threaded step loops (no locking needed).
"""

import math
from typing import List, Optional, Tuple, Dict, Any

from sensor_fusion import FusedState


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

class TrajectorySnapshot:
    """
    A lightweight, slot-optimised container for one recorded pose sample.

    Attributes:
        timestamp : Simulation time at recording (seconds).
        x, y, z   : Fused world position in metres.
        yaw        : Heading angle in radians.
        vx, vy, vz : Fused velocity in m/s.
    """

    __slots__ = ('timestamp', 'x', 'y', 'z', 'yaw', 'vx', 'vy', 'vz')

    def __init__(
        self,
        timestamp: float,
        x: float, y: float, z: float,
        yaw: float,
        vx: float, vy: float, vz: float,
    ) -> None:
        self.timestamp: float = timestamp
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.yaw: float = yaw
        self.vx: float = vx
        self.vy: float = vy
        self.vz: float = vz

    def to_dict(self) -> Dict[str, float]:
        """Serialises the snapshot to a plain dictionary for JSON export."""
        return {
            'timestamp': round(self.timestamp, 4),
            'x': round(self.x, 4),
            'y': round(self.y, 4),
            'z': round(self.z, 4),
            'yaw': round(self.yaw, 4),
            'vx': round(self.vx, 4),
            'vy': round(self.vy, 4),
            'vz': round(self.vz, 4),
        }

    def __repr__(self) -> str:
        return (f"TrajectorySnapshot(t={self.timestamp:.2f}, "
                f"x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, "
                f"yaw={math.degrees(self.yaw):.1f}°)")


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class TrajectoryRecorder:
    """
    Continuously records FusedState snapshots during the EXPLORE phase.

    Two complementary gates prevent log bloat:
      - Time gate   : enforces a maximum recording frequency (``record_hz``).
      - Distance gate: skips duplicate entries when the drone is hovering
                       still (``min_dist_m`` minimum displacement in XY).

    Usage::

        recorder = TrajectoryRecorder(record_hz=10.0, min_dist_m=0.05)
        recorder.start()                          # enable recording
        recorder.record(sensor_fusion.get_state())  # call every step
        log = recorder.get_log()                  # forward-ordered list
    """

    def __init__(
        self,
        record_hz: float = 10.0,
        min_dist_m: float = 0.05,
    ) -> None:
        """
        Args:
            record_hz:   Maximum snapshot frequency in Hz.
            min_dist_m:  Minimum XY displacement (metres) between consecutive
                         snapshots. Set to 0.0 to disable the distance gate.
        """
        self.record_hz: float = record_hz
        self.min_dist_m: float = min_dist_m
        self._min_interval: float = 1.0 / max(record_hz, 1e-6)

        self._log: List[TrajectorySnapshot] = []
        self._last_record_time: float = -1.0
        self._last_x: float = float('nan')
        self._last_y: float = float('nan')
        self.is_recording: bool = False

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Activates snapshot recording. Does NOT clear previous log data."""
        self.is_recording = True

    def stop(self) -> None:
        """Pauses recording without clearing the existing log."""
        self.is_recording = False

    def clear(self) -> None:
        """Resets the trajectory log and all internal gate state."""
        self._log.clear()
        self._last_record_time = -1.0
        self._last_x = float('nan')
        self._last_y = float('nan')

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def record(self, fused_state: FusedState) -> bool:
        """
        Attempts to append a snapshot from ``fused_state``.

        Args:
            fused_state: Current unified state from SensorFusion.update().

        Returns:
            True  — if a new snapshot was appended to the log.
            False — if recording is off, state is invalid, or the gate blocked.
        """
        if not self.is_recording or not fused_state.valid:
            return False

        t = fused_state.timestamp

        # ── Time gate ──────────────────────────────────────────────────────
        if t - self._last_record_time < self._min_interval:
            return False

        # ── Distance gate ──────────────────────────────────────────────────
        if not math.isnan(self._last_x):
            dx = fused_state.x - self._last_x
            dy = fused_state.y - self._last_y
            if math.sqrt(dx * dx + dy * dy) < self.min_dist_m:
                # Advance time gate even if position gate blocks, so we
                # re-evaluate at the next interval instead of every step.
                self._last_record_time = t
                return False

        # ── Record snapshot ────────────────────────────────────────────────
        snap = TrajectorySnapshot(
            timestamp=t,
            x=fused_state.x, y=fused_state.y, z=fused_state.z,
            yaw=fused_state.yaw,
            vx=fused_state.vx, vy=fused_state.vy, vz=fused_state.vz,
        )
        self._log.append(snap)
        self._last_record_time = t
        self._last_x = fused_state.x
        self._last_y = fused_state.y
        return True

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_log(self) -> List[TrajectorySnapshot]:
        """Returns the snapshot list ordered forward in time (oldest first)."""
        return self._log

    def snapshot_count(self) -> int:
        """Returns the number of snapshots currently in the log."""
        return len(self._log)

    def get_total_distance_m(self) -> float:
        """
        Computes total XY path length of the recorded trajectory in metres.
        Returns 0.0 if fewer than two snapshots are present.
        """
        total = 0.0
        for i in range(1, len(self._log)):
            dx = self._log[i].x - self._log[i - 1].x
            dy = self._log[i].y - self._log[i - 1].y
            total += math.sqrt(dx * dx + dy * dy)
        return total


# ---------------------------------------------------------------------------
# Replayer
# ---------------------------------------------------------------------------

class TrajectoryReplayer:
    """
    Reverses and smooths a recorded exploration trajectory, then exposes
    waypoints one at a time for use by ReturnHomeBehavior.

    Algorithm
    ---------
    1. Reverse the snapshot list (last recorded position becomes first
       waypoint — nearest to the drone at return time).
    2. Apply a 3-point moving-average to XY positions to reduce sensor
       jitter without introducing significant path deviation.
    3. Ensure the final waypoint is exactly the Home Pad position so the
       drone reliably arrives within ``home_radius_m``.
    4. Advance the waypoint index whenever the drone enters
       ``arrival_radius_m`` of the current target.
    5. Declare completion when within ``home_radius_m`` of Home Pad.

    Usage::

        replayer = TrajectoryReplayer(home_pos=(-4.0, -4.0))
        replayer.start(recorder.get_log())
        # inside ReturnHomeBehavior.execute():
        wp = replayer.get_next_waypoint(drone_x, drone_y)
        if wp is None and replayer.is_complete:
            # initiate landing
    """

    def __init__(
        self,
        arrival_radius_m: float = 0.45,
        home_pos: Tuple[float, float] = (-4.0, -4.0),
        home_radius_m: float = 0.40,
    ) -> None:
        """
        Args:
            arrival_radius_m: XY distance (m) at which the replayer advances
                              to the next waypoint.
            home_pos:         World (x, y) coordinates of the Home Pad.
            home_radius_m:    XY distance (m) from home_pos at which
                              the replayer declares the return complete.
        """
        self.arrival_radius_m: float = arrival_radius_m
        self.home_pos: Tuple[float, float] = home_pos
        self.home_radius_m: float = home_radius_m

        # Internal waypoint list: (x, y, z) in reversed order
        self._waypoints: List[Tuple[float, float, float]] = []
        self._wp_idx: int = 0
        self.is_active: bool = False
        self.is_complete: bool = False

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def start(self, log: List[TrajectorySnapshot]) -> bool:
        """
        Prepares the reversed, smoothed waypoint sequence from a recorder log.

        Args:
            log: Forward-ordered list of ``TrajectorySnapshot`` objects.

        Returns:
            True  — replayer initialised with at least one waypoint.
            False — log was empty; replayer not activated.
        """
        self.reset()

        if not log:
            return False

        # ── 1. Reverse ─────────────────────────────────────────────────────
        reversed_snaps = list(reversed(log))

        # ── 2. Extract raw XY + Z ──────────────────────────────────────────
        raw: List[Tuple[float, float, float]] = [
            (s.x, s.y, s.z) for s in reversed_snaps
        ]

        # ── 3. 3-point moving-average smoothing ────────────────────────────
        n = len(raw)
        smoothed: List[Tuple[float, float, float]] = []
        for i in range(n):
            lo = max(0, i - 1)
            hi = min(n - 1, i + 1)
            count = hi - lo + 1
            sx = sum(raw[k][0] for k in range(lo, hi + 1)) / count
            sy = sum(raw[k][1] for k in range(lo, hi + 1)) / count
            sz = raw[i][2]   # keep per-sample altitude for terrain following
            smoothed.append((sx, sy, sz))

        # ── 4. Ensure final waypoint is exactly Home Pad ───────────────────
        hx, hy = self.home_pos
        if smoothed:
            lx, ly, lz = smoothed[-1]
            dist_to_home = math.sqrt((lx - hx) ** 2 + (ly - hy) ** 2)
            if dist_to_home > self.home_radius_m:
                smoothed.append((hx, hy, lz))
        else:
            smoothed.append((hx, hy, 0.15))

        self._waypoints = smoothed
        self._wp_idx = 0
        self.is_active = True
        self.is_complete = False
        return True

    def reset(self) -> None:
        """Resets the replayer to a clean, inactive state."""
        self._waypoints.clear()
        self._wp_idx = 0
        self.is_active = False
        self.is_complete = False

    # ------------------------------------------------------------------
    # Per-step interface
    # ------------------------------------------------------------------

    def get_next_waypoint(
        self,
        drone_x: float,
        drone_y: float,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Returns the (x, y, z) of the current target waypoint, advancing the
        internal pointer whenever the drone is within ``arrival_radius_m``.

        Completion is declared when the drone is within ``home_radius_m``
        of the Home Pad (regardless of remaining waypoints).

        Args:
            drone_x: Current drone world X position.
            drone_y: Current drone world Y position.

        Returns:
            (x, y, z) tuple — next waypoint to navigate toward, or
            None            — if the return journey is complete.
        """
        if not self.is_active or self.is_complete:
            return None

        # ── Check home arrival (short-circuit all remaining waypoints) ─────
        hx, hy = self.home_pos
        dist_home = math.sqrt((drone_x - hx) ** 2 + (drone_y - hy) ** 2)
        if dist_home <= self.home_radius_m:
            self.is_complete = True
            return None

        # ── Advance past already-reached waypoints ─────────────────────────
        while self._wp_idx < len(self._waypoints):
            wx, wy, wz = self._waypoints[self._wp_idx]
            dist = math.sqrt((drone_x - wx) ** 2 + (drone_y - wy) ** 2)
            if dist <= self.arrival_radius_m:
                self._wp_idx += 1
            else:
                return (wx, wy, wz)

        # All waypoints consumed without triggering home check — declare done
        self.is_complete = True
        return None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_progress(self) -> float:
        """
        Returns fraction [0.0, 1.0] of waypoints that have been consumed.
        Returns 1.0 when complete or if no waypoints were loaded.
        """
        if not self._waypoints:
            return 1.0
        return min(1.0, self._wp_idx / len(self._waypoints))

    def get_waypoint_count(self) -> int:
        """Total number of waypoints in the reversed trajectory."""
        return len(self._waypoints)

    def get_remaining_count(self) -> int:
        """Number of waypoints not yet reached."""
        return max(0, len(self._waypoints) - self._wp_idx)

    def get_current_target(self) -> Optional[Tuple[float, float, float]]:
        """Returns the current target waypoint without modifying the index."""
        if self._wp_idx < len(self._waypoints):
            return self._waypoints[self._wp_idx]
        return None
