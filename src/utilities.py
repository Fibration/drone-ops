from typing import Literal, TypedDict, overload
from collections.abc import Iterable

from src.models import DroneState, Point


@overload
def points_in_bounds(
    points: Iterable[Point], bounds: tuple[Point, Point], *, return_mask: Literal[False] = False
) -> bool: ...


@overload
def points_in_bounds(
    points: Iterable[Point], bounds: tuple[Point, Point], *, return_mask: Literal[True]
) -> list[bool]: ...


@overload
def points_in_bounds(
    points: Iterable[Point], bounds: tuple[Point, Point], *, return_mask: bool
) -> bool | list[bool]: ...


def points_in_bounds(
    points: Iterable[Point], bounds: tuple[Point, Point], *, return_mask: bool = False
) -> bool | list[bool]:
    """Check inclusive bounds, returning all-in-bounds or a mask in input order.

    Empty input returns True, or [] when return_mask=True. Boolean mode
    stops at the first out-of-bounds point; mask mode consumes every point.
    """
    (min_x, min_y), (max_x, max_y) = bounds
    checks = (min_x <= x <= max_x and min_y <= y <= max_y for x, y in points)
    return list(checks) if return_mask else all(checks)


class DroneInitialState(TypedDict):
    id: str
    position: tuple[int, int]
    battery: float


def initialize_drone_states(
    drones: list[tuple[int, int]], default_battery: float
) -> dict[str, DroneState]:
    """Create simulation states keyed by drone ID from initial positions."""
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
