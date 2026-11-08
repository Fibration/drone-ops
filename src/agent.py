from src.models import Agent, AgentType, MissionPlan, MissionState, MonitoringPlan, Plan, ReviewOutcome, ReviewResult
from src.mission_updates import update_completed_sectors, update_discovered_targets
from src.policy import (
    CommanderRoutingDecision,
    FleetMonitorObservation,
    FleetMonitorPolicy,
    NaiveFleetMonitorPolicy,
    NaiveMissionCommanderPolicy,
    NaiveMissionPlannerPolicy,
    NaiveSafetyReviewerPolicy,
    Policy,
)


class FleetMonitor(Agent):
    def __init__(self, policy: FleetMonitorPolicy | None = None):
        self.policy = policy if policy is not None else NaiveFleetMonitorPolicy()

    def run(self, state: MissionState) -> dict:
        if state.current_plan is None or state.needs_replan:
            raise ValueError("FleetMonitor requires an accepted plan")
        result = self.policy.evaluate(FleetMonitorObservation(
            accepted_plan=state.current_plan.model_copy(deep=True),
            step=state.step,
            safety_review=state.safety_status.model_copy(deep=True) if state.safety_status else None,
        ))
        if not isinstance(result, MonitoringPlan):
            raise TypeError("Fleet monitor policy must return a MonitoringPlan")
        monitoring_plan = MonitoringPlan.model_validate(result.model_dump())
        return {"monitoring_plan": monitoring_plan}



class MissionCommander(Agent):
    def __init__(self, policy: Policy[CommanderRoutingDecision] | None = None):
        self.policy = policy if policy is not None else NaiveMissionCommanderPolicy()

    def run(self, state: MissionState) -> dict:
        observed = state.model_copy(update={
            "discovered_targets": dict(state.discovered_targets),
            "completed_sectors": set(state.completed_sectors),
        })
        previous_targets = set(state.discovered_targets)
        update_discovered_targets(observed)
        update_completed_sectors(observed)
        observed.new_target_ids = set(observed.discovered_targets) - previous_targets
        decision = self.policy.evaluate(observed)
        update = {
            "next_agent": decision.next_agent, "under_review": decision.under_review,
            "mission_status": decision.mission_status,
            "needs_replan": decision.needs_replan,
            "commander_rethink": decision.commander_rethink,
        }
        for field in ("discovered_targets", "completed_sectors", "new_target_ids"):
            if getattr(observed, field) != getattr(state, field):
                update[field] = getattr(observed, field)
        if decision.next_agent == AgentType.MISSION_PLANNER:
            update["proposed_plans"] = []
        elif decision.next_agent == AgentType.SAFETY_REVIEWER:
            update.update({"safety_status": None, "human_approval": None})
        elif decision.next_agent == "human_reviewer":
            update["under_review"] = decision.under_review or state.under_review
            update["human_approval"] = None
        return update


class MissionPlanner(Agent):
    def __init__(self, policy: Policy[list[Plan]] | None = None):
        self.policy = policy if policy is not None else NaiveMissionPlannerPolicy()

    def run(self, state: MissionState) -> dict:
        proposals = self.policy.evaluate(state)
        return {"proposed_plans": proposals, "safety_status": None, "human_approval": None}


class SafetyReviewer(Agent):
    def __init__(self, policy: Policy[ReviewResult] | None = None):
        self.policy = policy if policy is not None else NaiveSafetyReviewerPolicy()

    def run(self, state: MissionState) -> dict:
        outcome = self.policy.evaluate(state)
        update: dict = {"safety_status": outcome}
        if outcome.outcome == ReviewOutcome.APPROVE:
            if state.under_review is None:
                raise ValueError("No proposed plan selected for approval")
            update["current_plan"] = MissionPlan(plan=state.under_review.plan)
            update["monitoring_plan"] = None
            update["needs_replan"] = False
        else:
            update["needs_replan"] = True
        return update
