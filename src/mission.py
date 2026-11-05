from src.simulator import Simulator
from src.map import Terrain, extract_from_raw
from src.models import (
    DroneState,
    MissionPlan,
    MissionState,
    MissionStatus,
    Objective,
    Region,
)
from src.utility import initialize_drone_states
from src.drone import Objective as DroneObjective, ObjectiveType
from math import isqrt
from typing import Callable
from src.models import MissionEvent


RAW_MAP = [
    "A . . . X X . . .",
    ". . . . X X . T .",
    ". . . . . . . . .",
    ". N N N . . . . .",
    ". N N N . . . . .",
    ". . . . . . . . B",
]


def divide_search(
    map: list[list[Terrain]], num_drones: int, drone_positions: list[tuple[int, int]]
) -> dict[str, DroneObjective]:
    """Assign one rectangular search sector to each drone, keyed by its index."""
    if num_drones < 0 or num_drones != len(drone_positions):
        raise ValueError("num_drones must match the number of drone positions")
    plans = {}
    if not map or not map[0] or not num_drones:
        return plans
    rows, cols = len(map), len(map[0])
    if any(len(row) != cols for row in map):
        raise ValueError("The terrain map must be rectangular")
    for row, col in drone_positions:
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError("Drone positions must be within the terrain map")

    if num_drones > rows * cols:
        raise ValueError("There must be at least one map cell per drone")
    available = set(range(num_drones))
    sector_count = num_drones
    band_count = max(
        (sector_count + cols - 1) // cols,
        min(rows, sector_count, max(1, isqrt(sector_count * rows // cols))),
    )
    sectors = []
    for band in range(band_count):
        top = band * rows // band_count
        bottom = (band + 1) * rows // band_count - 1
        column_count = sector_count // band_count + (band < sector_count % band_count)
        for column in range(column_count):
            left = column * cols // column_count
            right = (column + 1) * cols // column_count - 1
            sectors.append(((top, left), (bottom, right)))

    for (top, left), (bottom, right) in sectors:
        drone = min(
            available,
            key=lambda i: (
                max(top - drone_positions[i][0], 0, drone_positions[i][0] - bottom)
                + max(left - drone_positions[i][1], 0, drone_positions[i][1] - right),
                i,
            ),
        )
        available.remove(drone)
        plans[str(drone)] = DroneObjective(
            ObjectiveType.SEARCH_SECTOR,
            sector=len(plans),
            bounds=((top, left), (bottom, right)),
        )
    return plans


def mission(
    simulator: Simulator,
    map: list[list[Terrain]],
    drones: dict[str, DroneState],
    mission_commander: Callable[[MissionState], MissionState],
    monitoring_checks,
) -> MissionState:
    drone_ids = list(drones)
    search_sectors = divide_search(
        map, len(drone_ids), [drones[drone_id].position for drone_id in drone_ids]
    )
    initial_plan = MissionPlan(
        plan={
            drone_id: search_sectors[str(index)]
            for index, drone_id in enumerate(drone_ids)
        }
    )
    for drone_id in initial_plan.plan:
        drones[drone_id].assignment = "search for targets"
        drones[drone_id].route = []

    mission_state = MissionState(
        drones=drones,
        objectives=[Objective(description="find and reach all targets")],
        no_fly_zones=[Region(coordinates=[
            (row, col)
            for row, cells in enumerate(map)
            for col, cell in enumerate(cells)
            if cell == Terrain.NO_FLY_ZONE
        ])],
        discovered_targets=[],
        events=[],
        explored_cells=set(),
        current_plan=initial_plan,
        proposed_action=None,
        next_agent=None,
        safety_status=None,
        human_approval=None,
        step=0,
        mission_status=MissionStatus.ONGOING,
    )

    while mission_state.mission_status == MissionStatus.ONGOING:
        plan = mission_state.current_plan
        if plan is None or not any(plan.plan.values()):
            mission_state.mission_status = MissionStatus.FAILED
            break

        simulator.drones = mission_state.drones
        for drone_id, drone in mission_state.drones.items():
            objective = plan.plan.get(drone_id)
            drone.route = objective.copy() if isinstance(objective, list) else []

        event, updated_drones, explored_cells, timestep = simulator.execute_plan(
            plan.plan, monitoring_checks, mission_state.explored_cells
        )
        mission_state.explored_cells = explored_cells
        mission_state.drones = updated_drones
        mission_state.step = timestep
        mission_state.events.append(
            MissionEvent(
                description=event,
                drones={
                    drone_id: drone.model_copy(deep=True)
                    for drone_id, drone in updated_drones.items()
                },
            )
        )
        mission_state = mission_commander(mission_state)

    return mission_state


if __name__ == "__main__":
    from src.mission_commander import naive_mission_commander

    truth_terrain, terrain, drone_positions, _ = extract_from_raw(RAW_MAP)
    sorted_drone_positions = sorted(drone_positions, key=lambda x: x[0])
    drones = initialize_drone_states(sorted_drone_positions, default_battery=25.0)
    simulator = Simulator(truth_terrain, drones, targets=[(1, 7)])

    mission(
        simulator,
        terrain,
        drones,
        mission_commander=naive_mission_commander,
        monitoring_checks={"battery_reserve_threshold": 10.0},
    )
