import re

from src.drone import Objective as DroneObjective, ObjectiveType
from src.map import no_fly_zone_from_event
from src.models import MissionState, MissionStatus, Region, Target


def naive_mission_commander(mission_state: MissionState) -> MissionState:
    if not mission_state.events:
        return mission_state

    latest = mission_state.events[-1]
    zone = no_fly_zone_from_event(latest.description)
    if zone is not None:
        known_cells = {
            cell for region in mission_state.no_fly_zones for cell in region.coordinates
        }
        new_cells = [cell for cell in zone if cell not in known_cells]
        if new_cells:
            mission_state.no_fly_zones.append(Region(coordinates=new_cells))
        return mission_state

    if latest.description == "Mission plan exhausted":
        reached_targets = {drone.position for drone in mission_state.drones.values()}
        for event in mission_state.events:
            reached = re.fullmatch(
                r"Drone (.+) reached the target at \((-?\d+),\s*(-?\d+)\)",
                event.description,
            )
            if reached is not None:
                _, row, col = reached.groups()
                reached_targets.add((int(row), int(col)))
        mission_state.mission_status = (
            MissionStatus.SUCCESS
            if all(
                target.position in reached_targets
                for target in mission_state.discovered_targets
            )
            else MissionStatus.FAILED
        )
        mission_state.next_agent = "finish"
        return mission_state

    detected = re.fullmatch(
        r"Drone (.+) detected a target at \((-?\d+),\s*(-?\d+)\)",
        latest.description,
    )
    if detected is not None:
        drone_id, row, col = detected.groups()
        target = (int(row), int(col))
        if mission_state.current_plan is not None and drone_id in mission_state.drones:
            mission_state.current_plan.plan[drone_id] = DroneObjective(
                ObjectiveType.GOTO, target=target
            )
            mission_state.drones[drone_id].assignment = f"go to {target}"
            mission_state.drones[drone_id].route = []
            if not any(item.position == target for item in mission_state.discovered_targets):
                mission_state.discovered_targets.append(Target(position=target))

    return mission_state
