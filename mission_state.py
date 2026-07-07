import enum

class MissionState(enum.Enum):
    """
    Enumeration representing the high-level operational states of an autonomous drone exploration mission.
    """
    IDLE = "IDLE"
    TAKEOFF = "TAKEOFF"
    EXPLORE = "EXPLORE"
    RETURN_HOME = "RETURN_HOME"
    LAND = "LAND"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    EMERGENCY = "EMERGENCY"

    def __str__(self):
        return self.name
