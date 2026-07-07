import logging
from mission_state import MissionState

# Setup basic logging for mission state changes
logging.basicConfig(level=logging.INFO, format="[MISSION MANAGER] %(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MissionManager")

class MissionManager:
    """
    Modular Mission Manager for coordinates, timing, and state transitions of the drone.
    Implements a strict Finite State Machine (FSM) ensuring valid state transitions.
    """
    
    # Valid state transition mapping representing legal pathways in the mission flight plan
    _VALID_TRANSITIONS = {
        MissionState.IDLE: {
            MissionState.TAKEOFF,
            MissionState.EMERGENCY
        },
        MissionState.TAKEOFF: {
            MissionState.EXPLORE,
            MissionState.RETURN_HOME,
            MissionState.LAND,
            MissionState.EMERGENCY
        },
        MissionState.EXPLORE: {
            MissionState.RETURN_HOME,
            MissionState.MISSION_COMPLETE,
            MissionState.LAND,
            MissionState.EMERGENCY
        },
        MissionState.RETURN_HOME: {
            MissionState.LAND,
            MissionState.EMERGENCY
        },
        MissionState.LAND: {
            MissionState.IDLE,
            MissionState.MISSION_COMPLETE,
            MissionState.EMERGENCY
        },
        MissionState.MISSION_COMPLETE: {
            MissionState.IDLE,
            MissionState.EMERGENCY
        },
        MissionState.EMERGENCY: {
            MissionState.LAND,
            MissionState.IDLE
        }
    }

    def __init__(self, initial_state=MissionState.IDLE):
        """
        Initializes the MissionManager FSM with an initial state.
        
        Args:
            initial_state (MissionState): The starting state of the mission manager. Defaults to IDLE.
        """
        self._current_state = initial_state
        logger.info(f"Initialized with state: {self._current_state}")

    def get_state(self):
        """
        Gets the current state of the mission.
        
        Returns:
            MissionState: The active mission state.
        """
        return self._current_state

    def is_in_state(self, state):
        """
        Checks if the mission manager is currently in the specified state.
        
        Args:
            state (MissionState): The state to check.
            
        Returns:
            bool: True if matches, False otherwise.
        """
        return self._current_state == state

    def transition_to(self, target_state):
        """
        Attempts to transition the FSM from the current state to target_state.
        Logs transitions and validation errors.
        
        Args:
            target_state (MissionState): The state to transition into.
            
        Returns:
            bool: True if the transition was successful, False otherwise.
        """
        if not isinstance(target_state, MissionState):
            logger.error(f"Cannot transition to invalid state type: {target_state}")
            return False

        if target_state == self._current_state:
            # Already in target state, ignore as a success
            return True

        valid_next_states = self._VALID_TRANSITIONS.get(self._current_state, set())
        if target_state in valid_next_states:
            old_state = self._current_state
            self._current_state = target_state
            logger.info(f"Transition SUCCESS: {old_state} -> {self._current_state}")
            return True
        else:
            logger.warning(
                f"Transition REJECTED: Illegal path from {self._current_state} -> {target_state}"
            )
            return False

    def force_state(self, target_state):
        """
        Forcefully overwrites the state machine. Primarily used for manual overrides.
        
        Args:
            target_state (MissionState): The state to force the FSM to assume.
        """
        if not isinstance(target_state, MissionState):
            logger.error(f"Cannot force invalid state type: {target_state}")
            return
        
        old_state = self._current_state
        self._current_state = target_state
        logger.warning(f"State FORCED: {old_state} -> {self._current_state}")

    def reset(self):
        """
        Resets the state machine back to IDLE.
        """
        logger.info("Resetting mission manager state back to IDLE")
        self._current_state = MissionState.IDLE

    def step(self, dt):
        """
        Periodic tick update of the mission state.
        Currently handles state transition telemetry logging or timers.
        
        Args:
            dt (float): Time increment since last tick.
        """
        pass
