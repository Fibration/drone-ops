import re

from src.map import no_fly_zone_from_event
from src.models import MissionState, Region, Target


def update_no_fly_zones(mission_state: MissionState) -> bool:
    """Record the latest zone event; return whether it was a zone event."""
    if not mission_state.events:
        return False
    zone = no_fly_zone_from_event(mission_state.events[-1].description)
    if zone is None:
        return False
    known_cells = {
        cell for region in mission_state.no_fly_zones for cell in region.coordinates
    }
    new_cells = [cell for cell in zone if cell not in known_cells]
    if new_cells:
        mission_state.no_fly_zones.append(Region(coordinates=new_cells))
    return True


def update_discovered_targets(mission_state: MissionState) -> tuple[str, Target] | None:
    """Record a detection and return its drone ID and stored target."""
    if not mission_state.events:
        return None
    detected = re.fullmatch(
        r"Drone (.+) detected a target at \((-?\d+),\s*(-?\d+)\)",
        mission_state.events[-1].description,
    )
    if detected is None:
        return None
    drone_id, row, col = detected.groups()
    position = (int(row), int(col))
    for target in mission_state.discovered_targets.values():
        if target.position == position:
            return drone_id, target
    target_id = str(len(mission_state.discovered_targets))
    while target_id in mission_state.discovered_targets:
        target_id = str(int(target_id) + 1)
    target = Target(target_id=target_id, position=position)
    mission_state.discovered_targets[target_id] = target
    return drone_id, target
