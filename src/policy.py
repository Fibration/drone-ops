from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar

from src.models import (
    AgentType, MissionState, MissionStatus, Objective, ObjectiveType, Plan,
    TelemetryField, TelemetryRequest, TelemetryResult,
)
from src.mission_updates import (
    evaluate_mission_status, parse_target_detection,
)

from src.telemetry import TelemetryQueries

Result = TypeVar("Result", covariant=True)
SafetyOutcome = Literal["approve", "revise", "human_reviewer"]


class Policy(Protocol[Result]):
    def evaluate(self, state: MissionState) -> Result: ...


@dataclass(frozen=True)
class CommanderRoutingDecision:
    next_agent: AgentType | None
    under_review: Plan | None = None
    mission_status: MissionStatus = MissionStatus.ONGOING
    needs_replan: bool = True
    telemetry_requests: list[TelemetryRequest] = field(default_factory=list)


@dataclass(frozen=True)
class FleetMonitorObservation:
    telemetry: TelemetryQueries
    requests: tuple[TelemetryRequest, ...]
    step: int


class FleetMonitorPolicy(Protocol):
    def evaluate(self, observation: FleetMonitorObservation) -> list[TelemetryResult]: ...


class NaiveFleetMonitorPolicy:
    def evaluate(self, observation: FleetMonitorObservation) -> list[TelemetryResult]:
        requests = observation.requests or tuple(TelemetryRequest(field=field) for field in (
            TelemetryField.BATTERY_LEVEL, TelemetryField.POSITION, TelemetryField.COMMS,
        ))
        readers = {
            TelemetryField.BATTERY_LEVEL: observation.telemetry.battery_level,
            TelemetryField.POSITION: observation.telemetry.position,
            TelemetryField.COMMS: observation.telemetry.comms,
            TelemetryField.STATUS: observation.telemetry.status,
            TelemetryField.ASSIGNMENT: observation.telemetry.assignment,
        }
        return [TelemetryResult(request=request, readings=readers[request.field](
            request.drone_ids, at_step=request.at_step, history=request.history,
        )) for request in requests]


class NaiveMissionCommanderPolicy:
    def evaluate(self, state: MissionState) -> CommanderRoutingDecision:
        status = evaluate_mission_status(state)
        if status != MissionStatus.ONGOING:
            return CommanderRoutingDecision(next_agent=None, mission_status=status)
        if state.telemetry_requests:
            return CommanderRoutingDecision(
                next_agent=AgentType.FLEET_MONITOR,
                telemetry_requests=list(state.telemetry_requests),
            )
        redundant = (
            bool(state.latest_events) and not state.new_target_ids
            and all(parse_target_detection(event.description) is not None
                    for event in state.latest_events)
            and state.current_plan is not None
        )
        if redundant:
            return CommanderRoutingDecision(next_agent=None, needs_replan=False)
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
            for drone_id in state.drone_ids
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
