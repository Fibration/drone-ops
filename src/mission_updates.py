import re

from src.map import no_fly_zone_from_event
from src.models import MissionState, Region, Target


def update_no_fly_zones(mission_state: MissionState) -> bool:
    """Record every zone update in the latest event batch."""
    known_cells = {
        cell for region in mission_state.no_fly_zones for cell in region.coordinates
    }
    found = False
    for event in mission_state.latest_events:
        zone = no_fly_zone_from_event(event.description)
        if zone is None:
            continue
        found = True
        new_cells = [cell for cell in zone if cell not in known_cells]
        if new_cells:
            mission_state.no_fly_zones.append(Region(coordinates=new_cells))
            known_cells.update(new_cells)
    return found


def update_discovered_targets(mission_state: MissionState) -> list[tuple[str, Target]]:
    """Record every detection in the batch and return its drone and target."""
    detections = []
    by_position = {
        target.position: target for target in mission_state.discovered_targets.values()
    }
    for event in mission_state.latest_events:
        detected = re.fullmatch(
            r"Drone (.+) detected a target at \((-?\d+),\s*(-?\d+)\)",
            event.description,
        )
        if detected is None:
            continue
        drone_id, row, col = detected.groups()
        position = (int(row), int(col))
        target = by_position.get(position)
        if target is None:
            target_id = str(len(mission_state.discovered_targets))
            while target_id in mission_state.discovered_targets:
                target_id = str(int(target_id) + 1)
            target = Target(target_id=target_id, position=position)
            mission_state.discovered_targets[target_id] = target
            by_position[position] = target
        detections.append((drone_id, target))
    return detections


def update_completed_sectors(mission_state: MissionState) -> set[str]:
    """Accumulate sector completions from the latest simulator event batch."""
    for event in mission_state.latest_events:
        completed = re.fullmatch(r"Sector (.+) fully explored", event.description)
        if completed is not None:
            mission_state.completed_sectors.add(completed.group(1))
    return mission_state.completed_sectors
