import re

from src.client import client
import json
from src.drone import Objective, ObjectiveType
from src.map import Terrain, no_fly_zone_from_event
from src.models import (
    CommanderDecision,
    MissionState,
    MissionStatus,
    Objective,
    Region,
    Sector,
    Target,
)


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
                for target in mission_state.discovered_targets.values()
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
            for target_id, item in mission_state.discovered_targets.items():
                if item.position == target:
                    break
            else:
                target_id = str(len(mission_state.discovered_targets))
                while target_id in mission_state.discovered_targets:
                    target_id = str(int(target_id) + 1)
                mission_state.discovered_targets[target_id] = Target(
                    target_id=target_id, position=target
                )

            mission_state.current_plan.plan[drone_id] = Objective(
                drone_id=drone_id,
                objective_type=ObjectiveType.GOTO,
                target_id=target_id,
            )
            mission_state.drones[drone_id].assignment = f"go to {target}"
            mission_state.drones[drone_id].route = []

    return mission_state


class MissionCommander:
    def __init__(self, map: list[list[Terrain]], sectors: dict[str, Sector]):
        self.instructions = "You are a mission commander for a UAV fleet.\
              The mission objective is to find all the targets and \
                  move at least one drone to each target without running out of battery.\
                      Reason through the best course of action after the latest event.\
                        Then for each drone, choose from four options:\
                            hold position;\
                                  return to base;\
                                  search a sector (the sector choice is autoallocated);\
                                  go to a specific target.\
                                include your reasoning."
        self.map_bounds = {
            "rows": len(map),
            "columns": len(map[0]),
            "coordinate_convention": "zero-based (row, column), bounds inclusive",
        }
        self.sectors = {
            key: sector.model_dump_json() for key, sector in sectors.items()
        }

    def run(self, mission_state: MissionState) -> CommanderDecision:
        decision = client.responses.parse(
            model="gpt-4.1",
            input=[
                {"role": "developer", "content": self.instructions},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "mission_state": mission_state.model_dump_json(),
                            "map_bounds": self.map_bounds,
                            "sectors": self.sectors,
                        }
                    ),
                },
            ],
            text_format=CommanderDecision,
        )

        parsed = decision.output_parsed
        if parsed is None:
            raise RuntimeError("Mission commander returned no parsed decision")
        return parsed
