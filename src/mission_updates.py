from src.models import FreeWatchType, MissionState, MissionStatus, Target, WatchAlert, WatchType


def parse_target_detection(alert: WatchAlert) -> tuple[str, tuple[int, int]] | None:
    if (alert.type is FreeWatchType.NEW_TARGET_DETECTED
            and alert.drone_id is not None and alert.position is not None):
        return alert.drone_id, alert.position
    return None


def evaluate_mission_status(state: MissionState) -> MissionStatus:
    for alert in reversed(state.latest_events):
        if alert.type is FreeWatchType.MISSION_COMPLETE:
            if alert.mission_status is None:
                raise ValueError('MISSION_COMPLETE requires a mission status')
            return alert.mission_status
    return state.mission_status


def update_discovered_targets(state: MissionState) -> list[tuple[str, Target]]:
    detections = []
    for alert in state.latest_events:
        detected = parse_target_detection(alert)
        if detected is None:
            continue
        if alert.target_id is None:
            raise ValueError('NEW_TARGET_DETECTED requires target_id')
        drone_id, position = detected
        target = Target(target_id=alert.target_id, position=position)
        state.discovered_targets[alert.target_id] = target
        detections.append((drone_id, target))
    return detections


def update_completed_sectors(state: MissionState) -> set[str]:
    for alert in state.latest_events:
        if alert.type is WatchType.SECTOR_COMPLETE and alert.sector_id is not None:
            state.completed_sectors.add(alert.sector_id)
    return state.completed_sectors
