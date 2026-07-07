import math
import time
from typing import Dict, Any, Optional

class FusedState:
    """
    Data class container representing the unified state estimate of the drone.
    Contains position, velocity, orientation (roll, pitch, yaw), EMA rangefinder
    measurements, timestamp, and verification flag.
    """
    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        ema_alt: float = 0.0,
        ema_front: float = 0.0,
        timestamp: float = 0.0,
        valid: bool = False
    ):
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.vx: float = vx
        self.vy: float = vy
        self.vz: float = vz
        self.roll: float = roll
        self.pitch: float = pitch
        self.yaw: float = yaw
        self.ema_alt: float = ema_alt
        self.ema_front: float = ema_front
        self.timestamp: float = timestamp
        self.valid: bool = valid

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the fused state into a standardized dictionary representation.
        
        Returns:
            Dict[str, Any]: Nested dictionary containing state information.
        """
        return {
            "position": {"x": self.x, "y": self.y, "z": self.z},
            "velocity": {"vx": self.vx, "vy": self.vy, "vz": self.vz},
            "orientation": {"roll": self.roll, "pitch": self.pitch, "yaw": self.yaw},
            "ema": {"alt": self.ema_alt, "front": self.ema_front},
            "timestamp": self.timestamp,
            "valid": self.valid
        }

    def __str__(self) -> str:
        return (f"FusedState(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f}, "
                f"roll={self.roll:.3f}, pitch={self.pitch:.3f}, yaw={self.yaw:.3f}, "
                f"valid={self.valid})")


class SensorFusion:
    """
    Sensor Fusion module for the autonomous exploration pipeline.
    Accesses variables in SensorSimulationSuite to compile a unified,
    real-time state estimate without modifying or duplicating core filters.
    """
    def __init__(self, autopilot: Any) -> None:
        """
        Initializes the SensorFusion module.
        
        Args:
            autopilot (Any): Reference to the DroneAutopilot instance.
        """
        self.autopilot: Any = autopilot
        self._fused_state: FusedState = FusedState()
        self._last_update_time: float = 0.0
        self._update_counter: int = 0

    def update(self) -> None:
        """
        Reads filter state data from the autopilot's SensorSimulationSuite
        and updates the internal FusedState. Handles missing references gracefully.
        """
        sensor_suite = getattr(self.autopilot, "sensor_suite", None)
        if sensor_suite is None:
            self._fused_state.valid = False
            return

        # Ensure Kalman Filters are initialized within the suite
        if getattr(sensor_suite, "kf_x", None) is None:
            self._fused_state.valid = False
            return

        try:
            # 1. Extract position from the existing 1D Kalman Filters
            x: float = float(sensor_suite.kf_x.x[0])
            y: float = float(sensor_suite.kf_y.x[0])
            z: float = float(sensor_suite.kf_z.x[0])

            # 2. Extract velocity from the existing 1D Kalman Filters
            vx: float = float(sensor_suite.kf_x.x[1])
            vy: float = float(sensor_suite.kf_y.x[1])
            vz: float = float(sensor_suite.kf_z.x[1])

            # 3. Extract attitude (roll/pitch) from the existing Complementary Filter
            roll: float = float(sensor_suite.comp_roll)
            pitch: float = float(sensor_suite.comp_pitch)

            # 4. Extract yaw from physics data via RPY conversion helper
            quat = self.autopilot.data.xquat[self.autopilot.drone_body_id]
            _, _, yaw = self.autopilot.physics._quat_rpy(quat)

            # 5. Extract rangefinder values from existing EMA Filters
            ema_alt: float = float(sensor_suite.ema_alt)
            ema_front: float = float(sensor_suite.ema_front)

            # 6. Set timestamp and update local variables
            timestamp: float = float(self.autopilot.data.time)
            self._last_update_time = time.time()
            self._update_counter += 1

            # Compile into unified state
            self._fused_state = FusedState(
                x=x, y=y, z=z,
                vx=vx, vy=vy, vz=vz,
                roll=roll, pitch=pitch, yaw=yaw,
                ema_alt=ema_alt, ema_front=ema_front,
                timestamp=timestamp,
                valid=True
            )
        except Exception as e:
            # Prevent crashes if state vectors are partially updated or modified
            self._fused_state.valid = False

    def get_state(self) -> FusedState:
        """
        Retrieves the current fused state estimate.
        
        Returns:
            FusedState: The active state estimate.
        """
        return self._fused_state

    def reset(self) -> None:
        """
        Resets the internal fused state to default parameters.
        """
        self._fused_state = FusedState()
        self._update_counter = 0
        self._last_update_time = 0.0

    def get_health_status(self) -> Dict[str, Any]:
        """
        Evaluates the health and status of the Sensor Fusion module and underlying filters.
        
        Returns:
            Dict[str, Any]: Diagnostic information containing operational flags.
        """
        sensor_suite = getattr(self.autopilot, "sensor_suite", None)
        suite_connected: bool = sensor_suite is not None
        filters_initialized: bool = (
            suite_connected and 
            getattr(sensor_suite, "kf_x", None) is not None
        )

        # Validate that estimates are not NaN or infinite
        state_sane: bool = False
        if self._fused_state.valid:
            state_sane = (
                not math.isnan(self._fused_state.x) and
                not math.isnan(self._fused_state.y) and
                not math.isnan(self._fused_state.z) and
                not math.isinf(self._fused_state.x)
            )

        return {
            "sensor_suite_connected": suite_connected,
            "filters_initialized": filters_initialized,
            "state_estimate_valid": self._fused_state.valid,
            "state_estimate_sane": state_sane,
            "update_count": self._update_counter,
            "last_update_age_seconds": (
                time.time() - self._last_update_time 
                if self._last_update_time > 0.0 else -1.0
            )
        }
