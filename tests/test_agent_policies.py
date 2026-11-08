import unittest
from unittest.mock import Mock

from src.agent import FleetMonitor, MissionCommander, MissionPlanner, SafetyReviewer
from src.mission_graph import build_mission_graph
from src.models import (
    AgentType, Evaluation, MissionEvent, MissionPlan, MissionState, MissionStatus,
    Objective, ObjectiveType, Plan, Target,
)
from src.policy import (
    CommanderRoutingDecision, NaiveFleetMonitorPolicy,
    NaiveMissionCommanderPolicy, NaiveMissionPlannerPolicy,
)
from src.telemetry import RecordedTelemetry
from src.models import TelemetrySnapshot
from src.utility import initialize_drone_states


class AgentPolicyTest(unittest.TestCase):
    def state(self):
        return MissionState(
            drone_ids=list({}), objectives=[], no_fly_zones=[], discovered_targets={}, events=[],
            explored_cells=set(), telemetry=[], current_plan=None, proposed_plans=[],
            under_review=None, next_agent=None, safety_status=None, human_approval=None,
            step=0, mission_status=MissionStatus.ONGOING,
        )

    def test_agents_wrap_injected_policy_results_in_correct_fields(self):
        candidate = Plan(plan={}, reason="Candidate", estimated_completion_times=2)
        cases = [
            (MissionCommander, CommanderRoutingDecision(AgentType.SAFETY_REVIEWER, candidate),
             {"next_agent": AgentType.SAFETY_REVIEWER, "under_review": candidate,
              "mission_status": MissionStatus.ONGOING, "needs_replan": True,
              "telemetry_requests": [], "commander_rethink": None,
              "safety_status": None, "human_approval": None}),
            (MissionPlanner, [candidate], {"proposed_plans": [candidate]}),
            (SafetyReviewer, Evaluation(decision="revise", reason="Review"),
             {"safety_status": Evaluation(decision="revise", reason="Review"), "needs_replan": True}),
        ]
        for agent_type, policy_result, expected in cases:
            with self.subTest(agent=agent_type.__name__):
                state = self.state()
                before = state.model_dump()
                policy = Mock(spec=["evaluate"])
                policy.evaluate.return_value = policy_result
                self.assertEqual(agent_type(policy=policy).run(state), expected)
                policy.evaluate.assert_called_once_with(state)
                self.assertEqual(state.model_dump(), before)

    def test_naive_commander_requests_planning_without_proposals(self):
        state = self.state()
        state.under_review = Plan(plan={}, reason="Stale", estimated_completion_times=0)
        decision = NaiveMissionCommanderPolicy().evaluate(state)
        self.assertEqual(decision.next_agent, AgentType.MISSION_PLANNER)
        self.assertIsNone(decision.under_review)
        self.assertIsNotNone(state.under_review)
        self.assertEqual(MissionCommander().run(state), {
            "next_agent": AgentType.MISSION_PLANNER, "under_review": None,
            "mission_status": MissionStatus.ONGOING, "needs_replan": True,
            "telemetry_requests": [], "commander_rethink": None,
            "proposed_plans": [], "safety_status": None, "human_approval": None,
        })

    def test_naive_commander_selects_first_proposed_plan(self):
        state = self.state()
        state.proposed_plans = [
            Plan(plan={}, reason="First", estimated_completion_times=5),
            Plan(plan={}, reason="Second", estimated_completion_times=1),
        ]
        decision = NaiveMissionCommanderPolicy().evaluate(state)
        self.assertEqual(decision.next_agent, AgentType.SAFETY_REVIEWER)
        self.assertEqual(decision.under_review, state.proposed_plans[0])
        self.assertIsNone(state.under_review)

    def test_fleet_policy_updates_all_knowledge_without_mutating_state(self):
        state = self.state()
        drones = initialize_drone_states([(0, 0)], 25.0)
        state.drone_ids = list(drones)
        state.latest_events = [MissionEvent(description=text) for text in (
            "Drone 0 detected a target at (1, 7)",
            "New no-fly zone added: rows 3-4, columns 4-8",
            "Sector west fully explored",
        )]
        before = state.model_dump()
        result = FleetMonitor(telemetry_queries=RecordedTelemetry([TelemetrySnapshot(step=0, drones=drones)])).run(state)
        self.assertEqual(result["discovered_targets"]["0"].position, (1, 7))
        self.assertEqual(result["completed_sectors"], {"west"})
        self.assertEqual(len(result["no_fly_zones"][0].coordinates), 10)
        self.assertEqual(result["telemetry"][0].readings["0"][0].value, 25.0)
        self.assertEqual(result["new_target_ids"], {"0"})
        self.assertNotIn("needs_replan", result)
        self.assertEqual(state.model_dump(), before)

    def test_planner_redirects_detecting_drone_and_preserves_other_assignments(self):
        state = self.state()
        state.drone_ids = ["0", "1"]
        state.current_plan = MissionPlan(plan={
            "0": Objective(drone_id="0", objective_type=ObjectiveType.HOLD),
            "1": Objective(drone_id="1", objective_type=ObjectiveType.RETURN_TO_BASE),
        })
        state.discovered_targets = {"target": Target(target_id="target", position=(1, 7))}
        state.latest_events = [MissionEvent(
            description="Drone 0 detected a target at (1, 7)", 
        )]
        before = state.model_dump()
        proposal = NaiveMissionPlannerPolicy().evaluate(state)[0]
        self.assertEqual(proposal.plan["0"].objective_type, ObjectiveType.GOTO)
        self.assertEqual(proposal.plan["0"].target_id, "target")
        self.assertEqual(proposal.plan["1"], state.current_plan.plan["1"])
        self.assertEqual(state.model_dump(), before)

    def test_commander_classifies_exhaustion_from_known_arrivals(self):
        for reached in (False, True):
            with self.subTest(reached=reached):
                state = self.state()
                state.discovered_targets = {"target": Target(target_id="target", position=(1, 7))}
                state.latest_events = [MissionEvent(description="Mission plan exhausted")]
                state.events = ([MissionEvent(description="Drone 0 reached the target at (1, 7)")] if reached else []) + state.latest_events
                decision = NaiveMissionCommanderPolicy().evaluate(state)
                self.assertIsNone(decision.next_agent)
                self.assertEqual(decision.mission_status,
                                 MissionStatus.SUCCESS if reached else MissionStatus.FAILED)
                self.assertEqual(state.mission_status, MissionStatus.ONGOING)

    def test_graph_reuses_current_plan_for_redundant_detections(self):
        state = self.state()
        drones = initialize_drone_states([(0, 0)], 25.0)
        state.drone_ids = list(drones)
        state.current_plan = MissionPlan(plan={
            "0": Objective(drone_id="0", objective_type=ObjectiveType.HOLD),
        })
        state.discovered_targets = {"target": Target(target_id="target", position=(1, 7))}
        state.latest_events = [MissionEvent(
            description="Drone 0 detected a target at (1, 7)", 
        )]
        state.proposed_plans = [Plan(plan={}, reason="Previous cycle", estimated_completion_times=0)]
        policy = Mock(spec=["evaluate"])
        workflow = build_mission_graph({
            AgentType.MISSION_PLANNER: MissionPlanner(policy=policy).run,
        }, telemetry_queries=RecordedTelemetry([TelemetrySnapshot(step=0, drones=drones)])).compile()
        result = MissionState.model_validate(workflow.invoke(state))
        policy.evaluate.assert_not_called()
        self.assertEqual(result.current_plan, state.current_plan)
        self.assertFalse(result.needs_replan)
        self.assertEqual(result.proposed_plans, [])
        self.assertIsNone(result.under_review)
        self.assertEqual(result.mission_status, MissionStatus.ONGOING)


if __name__ == "__main__":
    unittest.main()
