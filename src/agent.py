from src.models import Agent, MissionPlan, MissionState, Plan
from src.policy import (
    CommanderRoutingDecision,
    FleetMonitorResult,
    NaiveFleetMonitorPolicy,
    NaiveMissionCommanderPolicy,
    NaiveMissionPlannerPolicy,
    NaiveSafetyReviewerPolicy,
    Policy,
    SafetyOutcome,
)


class FleetMonitor(Agent):
    def __init__(self, policy: Policy[FleetMonitorResult] | None = None):
        self.policy = policy if policy is not None else NaiveFleetMonitorPolicy()

    def run(self, state: MissionState) -> dict:
        observation = self.policy.evaluate(state)
        return {
            "telemetry": observation.telemetry,
            "no_fly_zones": observation.no_fly_zones,
            "discovered_targets": observation.discovered_targets,
            "completed_sectors": observation.completed_sectors,
            "needs_replan": observation.needs_replan,
            "proposed_plans": [], "under_review": None,
            "safety_status": None, "human_approval": None,
        }


class MissionCommander(Agent):
    def __init__(self, policy: Policy[CommanderRoutingDecision] | None = None):
        self.policy = policy if policy is not None else NaiveMissionCommanderPolicy()

    def run(self, state: MissionState) -> dict:
        decision = self.policy.evaluate(state)
        return {
            "next_agent": decision.next_agent, "under_review": decision.under_review,
            "mission_status": decision.mission_status,
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
