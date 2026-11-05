from simulator import Simulator
from map import Terrain, extract_from_raw
from models import DroneState, MissionState, MissionStatus, Objective, Region
from utility import initialize_drone_states

raw_map = [
    "A . . . X X . . .",
    ". . . . X X . T .",
    ". . . . . . . . .",
    ". N N N . . . . .",
    ". N N N . . . . .",
    ". . . . . . . . B",
]

truth_terrain, terrain, drone_positions, no_fly_zones = extract_from_raw(raw_map)
drones = initialize_drone_states(drone_positions, default_battery=25.0)
simulator = Simulator(truth_terrain, drones, targets=[(1, 7)])


def mission(
    simulator: Simulator, map: list[list[Terrain]], drones: dict[str, DroneState]
):
    mission_state = MissionState(
        drones=drones,
        objectives=[Objective(description="find and reach all targets")],
        no_fly_zones=[Region(coordinates=no_fly_zones)],
        discovered_targets=[],
        events=[],
        current_plan=None,
        proposed_action=None,
        next_agent=None,
        safety_status=None,
        human_approval=None,
        step=0,
        mission_status=MissionStatus.ONGOING,
    )

    while mission_state.mission_status == MissionStatus.ONGOING:
        pass
