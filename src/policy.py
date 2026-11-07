from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from src.models import (
    AgentType, MissionState, MissionStatus, Objective, ObjectiveType, Plan, Region, Target,
)
from src.mission_updates import (
    evaluate_mission_status, parse_target_detection, update_completed_sectors,
    update_discovered_targets, update_no_fly_zones,
)

Result = TypeVar("Result", covariant=True)
Telemetry = dict[str, str | int | float]
SafetyOutcome = Literal["approve", "revise", "human_reviewer"]


class Policy(Protocol[Result]):
    def evaluate(self, state: MissionState) -> Result: ...


@dataclass(frozen=True)
class CommanderRoutingDecision:
    next_agent: AgentType | None
    under_review: Plan | None = None
    mission_status: MissionStatus = MissionStatus.ONGOING


@dataclass(frozen=True)
class FleetMonitorResult:
    telemetry: Telemetry
    no_fly_zones: list[Region]
    discovered_targets: dict[str, Target]
    completed_sectors: set[str]
    needs_replan: bool


class NaiveFleetMonitorPolicy:
    def evaluate(self, state: MissionState) -> FleetMonitorResult:
        observed = state.model_copy(update={
            "no_fly_zones": list(state.no_fly_zones),
            "discovered_targets": dict(state.discovered_targets),
            "completed_sectors": set(state.completed_sectors),
        })
        target_count = len(observed.discovered_targets)
        update_no_fly_zones(observed)
        update_completed_sectors(observed)
        detections = update_discovered_targets(observed)
        redundant = (
            bool(state.latest_events)
            and len(detections) == len(state.latest_events)
            and len(observed.discovered_targets) == target_count
            and state.current_plan is not None
        )
        telemetry: Telemetry = {"step": state.step}
        for drone_id, drone in state.drones.items():
            telemetry[f"{drone_id}.battery"] = drone.battery
            telemetry[f"{drone_id}.comms"] = drone.comms
        return FleetMonitorResult(
            telemetry=telemetry, no_fly_zones=observed.no_fly_zones,
            discovered_targets=observed.discovered_targets,
            completed_sectors=observed.completed_sectors, needs_replan=not redundant,
        )


class NaiveMissionCommanderPolicy:
    def evaluate(self, state: MissionState) -> CommanderRoutingDecision:
        status = evaluate_mission_status(state)
        if status != MissionStatus.ONGOING:
            return CommanderRoutingDecision(next_agent=None, mission_status=status)
        if not state.needs_replan and state.current_plan is not None:
            return CommanderRoutingDecision(next_agent=None)
        if not state.proposed_plans:
            return CommanderRoutingDecision(next_agent=AgentType.MISSION_PLANNER)
        return CommanderRoutingDecision(
            next_agent=AgentType.SAFETY_REVIEWER, under_review=state.proposed_plans[0]
        )


class NaiveMissionPlannerPolicy:
    def evaluate(self, state: MissionState) -> list[Plan]:
        current = state.current_plan.plan if state.current_plan is not None else {}
        plan = {
            drone_id: current[drone_id].model_copy(deep=True)
            if drone_id in current else Objective(
                drone_id=drone_id, objective_type=ObjectiveType.HOLD
            )
            for drone_id in state.drones
        }
        targets = {target.position: target for target in state.discovered_targets.values()}
        for event in state.latest_events:
            detected = parse_target_detection(event.description)
            if detected is None:
                continue
            drone_id, position = detected
            target = targets.get(position)
            if drone_id in plan and target is not None:
                plan[drone_id] = Objective(
                    drone_id=drone_id, objective_type=ObjectiveType.GOTO,
                    target_id=target.target_id,
                )
        return [Plan(
            plan=plan,
            reason="Keep assignments; send detecting drones to their discovered targets.",
            estimated_completion_times=0,
        )]


class NaiveSafetyReviewerPolicy:
    def evaluate(self, state: MissionState) -> SafetyOutcome:
        return "approve"
