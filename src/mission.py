from src.simulator import Simulator
from src.map import Terrain, extract_from_raw
from src.models import (
    MissionPlan,
    MissionState,
    MissionStatus,
    Objective,
    ObjectiveType,
    Region,
    Sector,
)
from src.utility import initialize_drone_states
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


def create_sectors(map: list[list[Terrain]], sector_count: int) -> dict[str, Sector]:
    """Partition the map into rectangular sectors with inclusive bounds."""
    if sector_count < 0:
        raise ValueError("sector_count must be non-negative")
    if not map or not map[0] or not sector_count:
        return {}
    rows, cols = len(map), len(map[0])
    if any(len(row) != cols for row in map):
        raise ValueError("The terrain map must be rectangular")
    if sector_count > rows * cols:
        raise ValueError("There must be at least one map cell per sector")
    band_count = max(
        (sector_count + cols - 1) // cols,
        min(rows, sector_count, max(1, isqrt(sector_count * rows // cols))),
    )
    sectors: dict[str, Sector] = {}
    for band in range(band_count):
        top = band * rows // band_count
        bottom = (band + 1) * rows // band_count - 1
        column_count = sector_count // band_count + (band < sector_count % band_count)
        for column in range(column_count):
            left = column * cols // column_count
            right = (column + 1) * cols // column_count - 1
            sector_id = str(len(sectors))
            sectors[sector_id] = Sector(
                sector_id=sector_id,
                bounds=((top, left), (bottom, right)),
            )
    return sectors


def divide_search(
    sectors: dict[str, Sector], drone_positions: list[tuple[int, int]]
) -> dict[str, Objective]:
    """Assign each prepared sector to the nearest available drone by index."""
    if len(sectors) != len(drone_positions):
        raise ValueError("There must be one sector per drone")
    available = set(range(len(drone_positions)))
    plans = {}
    for sector_id, sector in sectors.items():
        (top, left), (bottom, right) = sector.bounds
        drone = min(
            available,
            key=lambda i: (
                max(
                    top - drone_positions[int(i)][0],
                    0,
                    drone_positions[int(i)][0] - bottom,
                )
                + max(
                    left - drone_positions[int(i)][1],
                    0,
                    drone_positions[int(i)][1] - right,
                ),
                i,
            ),
        )
        available.remove(drone)
        plans[str(drone)] = Objective(
            drone_id=str(drone),
            objective_type=ObjectiveType.SEARCH_SECTOR,
            sector_id=sector_id,
        )
    return plans


def mission(
    simulator: Simulator,
    map: list[list[Terrain]],
    mission_team: Callable[[MissionState], MissionState],
    monitoring_checks,
    sectors: dict[str, Sector] | None = None,
) -> MissionState:
    queries = simulator.telemetry_queries
    drone_ids = queries.drone_ids()
    readings = queries.position(drone_ids)
    positions: list[tuple[int, int]] = []
    for drone_id in drone_ids:
        samples = readings[drone_id]
        if not samples or not isinstance(samples[-1].value, tuple):
            raise ValueError(f"No reported initial position for drone {drone_id}")
        positions.append(samples[-1].value)
    if sectors is None:
        sectors = create_sectors(map, len(drone_ids))
    for row, col in positions:
        if not (0 <= row < len(map) and 0 <= col < len(map[row])):
            raise ValueError("Drone positions must be within the terrain map")
    search_sectors = divide_search(
        sectors, positions
    )
    initial_plan = MissionPlan(
        plan={
            drone_id: search_sectors[str(index)].model_copy(update={"drone_id": drone_id})
            for index, drone_id in enumerate(drone_ids)
        }
    )
    mission_state = MissionState(
        drone_ids=drone_ids,
        objectives=[],
        no_fly_zones=[
            Region(
                coordinates=[
                    (row, col)
                    for row, cells in enumerate(map)
                    for col, cell in enumerate(cells)
                    if cell == Terrain.NO_FLY_ZONE
                ]
            )
        ],
        discovered_targets={},
        events=[],
        explored_cells=set(),
        completed_sectors=set(),
        current_plan=initial_plan,
        telemetry=[],
        proposed_plans=[],
        under_review=None,
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

        events, explored_cells, timestep = simulator.execute_plan(
            plan.plan,
            monitoring_checks,
            sectors,
            mission_state.discovered_targets,
            mission_state.explored_cells,
        )
        mission_state.explored_cells = explored_cells
        mission_state.step = timestep
        mission_state.latest_events = [
            MissionEvent(
                description=event,
                step=timestep,
            )
            for event in events
        ]
        mission_state.events.extend(mission_state.latest_events)
        mission_state = mission_team(mission_state)

    return mission_state


if __name__ == "__main__":
    from src.mission_graph import build_mission_graph

    def control_cycle(state: MissionState) -> MissionState:
        return MissionState.model_validate(graph.invoke(state))

    truth_terrain, terrain, drone_positions, _ = extract_from_raw(RAW_MAP)
    sorted_drone_positions = sorted(drone_positions, key=lambda x: x[0])
    drones = initialize_drone_states(sorted_drone_positions, default_battery=25.0)
    simulator = Simulator(truth_terrain, drones, targets=[(1, 7)])
    graph = build_mission_graph(telemetry_queries=simulator.telemetry_queries).compile()
    sectors = create_sectors(terrain, len(drones))

    mission(
        simulator,
        terrain,
        mission_team=control_cycle,
        sectors=sectors,
        monitoring_checks={"battery_reserve_threshold": 10.0},
    )
