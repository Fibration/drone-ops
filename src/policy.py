from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from src.models import (
    AgentType, CommanderReviewOutcome, MissionState, MissionStatus, Objective, ObjectiveType, Plan,
    MissionPlan, MonitoringPlan, ReviewOutcome, ReviewResult,
)
from src.mission_updates import (
    evaluate_mission_status, parse_target_detection,
)


Result = TypeVar("Result", covariant=True)


class Policy(Protocol[Result]):
    def evaluate(self, state: MissionState) -> Result: ...


@dataclass(frozen=True)
class CommanderRoutingDecision:
    next_agent: AgentType | Literal["human_reviewer"] | None
    under_review: Plan | None = None
    mission_status: MissionStatus = MissionStatus.ONGOING
    needs_replan: bool = True
    commander_rethink: CommanderReviewOutcome | None = None


@dataclass(frozen=True)
class FleetMonitorObservation:
    accepted_plan: MissionPlan
    step: int
    safety_review: ReviewResult | None = None


class FleetMonitorPolicy(Protocol):
    def evaluate(self, observation: FleetMonitorObservation) -> MonitoringPlan: ...


class NaiveFleetMonitorPolicy:
    def evaluate(self, observation: FleetMonitorObservation) -> MonitoringPlan:
        raise NotImplementedError('Inject a fleet-monitor policy returning a MonitoringPlan; '
                                  'the watch-selection heuristic is not implemented yet.')


class NaiveMissionCommanderPolicy:
    def evaluate(self, state: MissionState) -> CommanderRoutingDecision:
        status = evaluate_mission_status(state)
        if status != MissionStatus.ONGOING:
            return CommanderRoutingDecision(next_agent=None, mission_status=status)
        if state.human_approval is False:
            return CommanderRoutingDecision(
                next_agent=AgentType.MISSION_PLANNER,
                commander_rethink=CommanderReviewOutcome.REPLAN,
            )
        if (state.safety_status is not None
                and state.safety_status.outcome in (ReviewOutcome.REPLAN, ReviewOutcome.RECONSIDER_PRIORITY)):
            return CommanderRoutingDecision(
                next_agent=AgentType.MISSION_PLANNER,
                commander_rethink=CommanderReviewOutcome.REPLAN,
            )
        if state.new_target_ids:
            return CommanderRoutingDecision(next_agent=AgentType.MISSION_PLANNER)
        redundant = (
            bool(state.latest_events) and not state.new_target_ids
            and all(parse_target_detection(event) is not None
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
            detected = parse_target_detection(event)
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
    def evaluate(self, state: MissionState) -> ReviewResult:
        return ReviewResult(outcome=ReviewOutcome.APPROVE,
                            critique="Approved by the default safety policy.",
                            concerns=[], mandatory_watches=[])
