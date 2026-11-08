from src.models import Agent, AgentType, MissionPlan, MissionState, Plan
from src.mission_updates import update_completed_sectors, update_discovered_targets, update_no_fly_zones
from src.telemetry import RecordedTelemetry, TelemetryQueries, merge_telemetry
from src.policy import (
    CommanderRoutingDecision,
    FleetMonitorObservation,
    FleetMonitorPolicy,
    NaiveFleetMonitorPolicy,
    NaiveMissionCommanderPolicy,
    NaiveMissionPlannerPolicy,
    NaiveSafetyReviewerPolicy,
    Policy,
    SafetyOutcome,
)


class FleetMonitor(Agent):
    def __init__(self, policy: FleetMonitorPolicy | None = None,
                 telemetry_queries: TelemetryQueries | None = None):
        self.policy = policy if policy is not None else NaiveFleetMonitorPolicy()
        self.telemetry_queries = telemetry_queries

    def run(self, state: MissionState) -> dict:
        queries = self.telemetry_queries
        if queries is None:
            if state.drone_ids:
                raise ValueError("FleetMonitor requires an injected telemetry provider")
            queries = RecordedTelemetry(())
        answers = self.policy.evaluate(FleetMonitorObservation(
            telemetry=queries, requests=tuple(state.telemetry_requests), step=state.step,
        ))
        for request in state.telemetry_requests:
            if not any(answer.request == request for answer in answers):
                raise ValueError(f"Fleet monitor did not fulfill telemetry request: {request}")
        observed = state.model_copy(update={
            "no_fly_zones": list(state.no_fly_zones),
            "discovered_targets": dict(state.discovered_targets),
            "completed_sectors": set(state.completed_sectors),
        })
        previous_targets = set(state.discovered_targets)
        update_no_fly_zones(observed)
        update_completed_sectors(observed)
        update_discovered_targets(observed)
        followup = state.next_agent == AgentType.FLEET_MONITOR and bool(state.telemetry_requests)
        update = {
            "telemetry": merge_telemetry(state.telemetry, answers),
            "available_telemetry": queries.available_telemetry(),
            "telemetry_requests": [],
            "no_fly_zones": observed.no_fly_zones,
            "discovered_targets": observed.discovered_targets,
            "completed_sectors": observed.completed_sectors,
            "new_target_ids": (state.new_target_ids if followup else set())
                              | (set(observed.discovered_targets) - previous_targets),
        }
        if not followup:
            update.update({"proposed_plans": [], "under_review": None,
                           "safety_status": None, "human_approval": None})
        return update


class MissionCommander(Agent):
    def __init__(self, policy: Policy[CommanderRoutingDecision] | None = None):
        self.policy = policy if policy is not None else NaiveMissionCommanderPolicy()

    def run(self, state: MissionState) -> dict:
        decision = self.policy.evaluate(state)
        return {
            "next_agent": decision.next_agent, "under_review": decision.under_review,
            "mission_status": decision.mission_status,
            "needs_replan": decision.needs_replan,
            "telemetry_requests": decision.telemetry_requests,
        }


class MissionPlanner(Agent):
    def __init__(self, policy: Policy[list[Plan]] | None = None):
        self.policy = policy if policy is not None else NaiveMissionPlannerPolicy()

    def run(self, state: MissionState) -> dict:
        return {"proposed_plans": self.policy.evaluate(state)}


class SafetyReviewer(Agent):
    def __init__(self, policy: Policy[SafetyOutcome] | None = None):
        self.policy = policy if policy is not None else NaiveSafetyReviewerPolicy()

    def run(self, state: MissionState) -> dict:
        outcome = self.policy.evaluate(state)
        update: dict = {"safety_status": outcome}
        if outcome == "approve":
            if state.under_review is None:
                raise ValueError("No proposed plan selected for approval")
            update["current_plan"] = MissionPlan(plan=state.under_review.plan)
            update["needs_replan"] = False
        return update
