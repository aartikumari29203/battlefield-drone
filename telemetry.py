from dataclasses import dataclass


@dataclass
class DroneTelemetry:
    """
    Drone position and orientation data.

    These are currently simulated because the project
    does not have a real drone telemetry source.
    """

    latitude: float
    longitude: float
    altitude: float

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


def get_simulated_telemetry():
    """
    Return simulated drone telemetry for development/testing.
    """

    return DroneTelemetry(
        latitude=12.9716,
        longitude=77.5946,
        altitude=100.0,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
    )