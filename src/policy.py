from src.models import (
    ToolRegistry, AgentType, CommanderReviewOutcome, CommanderRoutingDecision,
    FleetMonitorObservation, MissionState, MissionStatus, MonitoringPlan,
    Objective, ObjectiveType, Plan, ReviewOutcome, ReviewResult,
)
from src.mission_updates import evaluate_mission_status, parse_target_detection


class NaiveFleetMonitorPolicy:
    def evaluate(self, observation: FleetMonitorObservation, *, tools: ToolRegistry | None = None) -> MonitoringPlan:
        raise NotImplementedError('Inject a fleet-monitor policy returning a MonitoringPlan; '
                                  'the watch-selection heuristic is not implemented yet.')


class NaiveMissionCommanderPolicy:
    def evaluate(self, state: MissionState, *, tools: ToolRegistry | None = None) -> CommanderRoutingDecision:
        status = evaluate_mission_status(state)
        if status != MissionStatus.ONGOING:
            return CommanderRoutingDecision(next_agent=None, mission_status=status)
        if state.human_approval is False:
            return CommanderRoutingDecision(
                next_agent=AgentType.MISSION_PLANNER,
                commander_rethink=CommanderReviewOutcome.REPLAN,
            )
        if (state.review_status is not None
                and state.review_status.outcome in (ReviewOutcome.REPLAN, ReviewOutcome.RECONSIDER_PRIORITY)):
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
            and not state.needs_replan
        )
        if redundant:
            assert state.current_plan is not None
            return CommanderRoutingDecision(
                next_agent=AgentType.REVIEWER,
                under_review=Plan(
                    plan={key: objective.model_copy(deep=True)
                          for key, objective in state.current_plan.plan.items()},
                    reason="Retain the accepted plan: these detections add no new targets.",
                    estimated_completion_times=0,
                ),
                needs_replan=False,
            )
        if not state.proposed_plans:
            return CommanderRoutingDecision(next_agent=AgentType.MISSION_PLANNER)
        return CommanderRoutingDecision(
            next_agent=AgentType.REVIEWER, under_review=state.proposed_plans[0]
        )


class NaiveMissionPlannerPolicy:
    def evaluate(self, state: MissionState, *, tools: ToolRegistry | None = None) -> list[Plan]:
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


class NaiveReviewerPolicy:
    def evaluate(self, state: MissionState, *, tools: ToolRegistry | None = None) -> ReviewResult:
        return ReviewResult(outcome=ReviewOutcome.APPROVE,
                            critique="Approved by the default review policy.",
                            concerns=[], mandatory_watches=[])
