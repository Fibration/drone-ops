from typing import TypedDict

from src.models import DroneState


class DroneInitialState(TypedDict):
    id: str
    position: tuple[int, int]
    battery: float


def initialize_drone_states(
    drones: list[tuple[int, int]], default_battery: float
) -> dict[str, DroneState]:
    """Create simulation states keyed by drone ID from initial telemetry."""
    return {
        str(i): DroneState(
            id=str(i),
            position=position,
            battery=default_battery,
            status="green",
            comms=True,
            assignment=None,
        )
        for i, position in enumerate(drones)
    }
