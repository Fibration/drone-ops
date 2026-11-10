import unittest
from unittest.mock import Mock

from src.agent import FleetMonitor, MissionCommander, MissionPlanner, Reviewer
from src.mission_graph import build_mission_graph
from src.models import (
    CommanderRoutingDecision, AgentType, ReviewResult, ReviewOutcome, CommanderReviewOutcome, WatchAlert, FreeWatchType, MissionPlan, MissionState, MissionStatus,
    Objective, ObjectiveType, Plan, Target,
    WatchPredicate, WatchType, MonitoringPlan,
)
from src.policy import (
    NaiveMissionCommanderPolicy, NaiveMissionPlannerPolicy, NaiveReviewerPolicy,
)
from src.utilities import initialize_drone_states


class AgentPolicyTest(unittest.TestCase):
    def build_graph(self, nodes=None):
        monitor = Mock()
        monitor.evaluate.return_value = MonitoringPlan(
            watches=[WatchPredicate(type=WatchType.SECTOR_COMPLETE, sector_id=str(i)) for i in range(3)],
            reason='Explicit test selection',
        )
        return build_mission_graph(nodes, agents={
            AgentType.MISSION_COMMANDER: MissionCommander(policy=NaiveMissionCommanderPolicy()),
            AgentType.MISSION_PLANNER: MissionPlanner(policy=NaiveMissionPlannerPolicy()),
            AgentType.REVIEWER: Reviewer(policy=NaiveReviewerPolicy()),
            AgentType.FLEET_MONITOR: FleetMonitor(policy=monitor),
        })

    def test_reviewer_can_challenge_a_decision_to_keep_the_current_plan(self):
        state = self.state()
        state.drone_ids = ['0']
        state.current_plan = MissionPlan(plan={
            '0': Objective(drone_id='0', objective_type=ObjectiveType.HOLD),
        })
        state.needs_replan = False
        state.discovered_targets = {'target': Target(target_id='target', position=(1, 7))}
        state.latest_events = [WatchAlert(type=FreeWatchType.NEW_TARGET_DETECTED, step=1,
                                         drone_id='0', target_id='target', position=(1, 7))]
        planner = Mock()
        planner.evaluate.return_value = [Plan(
            plan={'0': Objective(drone_id='0', objective_type=ObjectiveType.RETURN_TO_BASE)},
            reason='Change priorities: return to base', estimated_completion_times=0,
        )]
        reviewer = Mock()
        reviewer.evaluate.side_effect = [
            ReviewResult(outcome=ReviewOutcome.RECONSIDER_PRIORITY, critique='Reconsider priorities',
                         concerns=[], mandatory_watches=[]),
            ReviewResult(outcome=ReviewOutcome.APPROVE, critique='Updated priorities approved',
                         concerns=[], mandatory_watches=[]),
        ]
        graph = self.build_graph({
            AgentType.MISSION_PLANNER: MissionPlanner(policy=planner).run,
            AgentType.REVIEWER: Reviewer(policy=reviewer).run,
            AgentType.FLEET_MONITOR: FleetMonitor(policy=self.monitoring_policy()).run,
        }).compile()
        result = MissionState.model_validate(graph.invoke(state))
        planner.evaluate.assert_called_once()
        self.assertEqual(reviewer.evaluate.call_count, 2)
        retained = reviewer.evaluate.call_args_list[0].args[0]
        self.assertFalse(retained.needs_replan)
        self.assertEqual(retained.under_review.plan, state.current_plan.plan)
        self.assertIn('Retain', retained.under_review.reason)
        self.assertEqual(result.current_plan.plan['0'].objective_type, ObjectiveType.RETURN_TO_BASE)

    def test_commander_ingests_typed_detections_before_deciding(self):
        state = self.state()
        state.current_plan = MissionPlan(plan={})
        state.proposed_plans = [Plan(plan={}, reason='Old proposal', estimated_completion_times=0)]
        state.latest_events = [WatchAlert(type=FreeWatchType.NEW_TARGET_DETECTED, step=1,
                                         drone_id='0', target_id='target_0', position=(1, 2))]
        update = MissionCommander(policy=NaiveMissionCommanderPolicy()).run(state)
        self.assertEqual(update['next_agent'], AgentType.MISSION_PLANNER)
        self.assertEqual(update['discovered_targets']['target_0'].position, (1, 2))
        self.assertEqual(update['new_target_ids'], {'target_0'})
        self.assertFalse(state.discovered_targets)

    def test_priority_feedback_and_mandatory_watches_reach_planner(self):
        state = self.state()
        state.review_status = ReviewResult(
            outcome=ReviewOutcome.RECONSIDER_PRIORITY, critique='Reserve battery for return',
            concerns=['Low return reserve'], mandatory_watches=[WatchPredicate(
                type=WatchType.RETURN_RESERVE_BELOW, drone_id='0', threshold=10,
            )],
        )
        state.proposed_plans = [Plan(plan={}, reason='Rejected', estimated_completion_times=0)]
        update = MissionCommander(policy=NaiveMissionCommanderPolicy()).run(state)
        replanning = MissionState.model_validate(state.model_dump() | update)
        self.assertEqual(replanning.next_agent, AgentType.MISSION_PLANNER)
        self.assertEqual(replanning.commander_rethink, CommanderReviewOutcome.REPLAN)
        self.assertEqual(replanning.review_status, state.review_status)
        policy = Mock()
        policy.evaluate.return_value = []
        result = MissionPlanner(policy=policy).run(replanning)
        self.assertEqual(policy.evaluate.call_args.args[0].review_status.mandatory_watches,
                         state.review_status.mandatory_watches)
        self.assertIsNone(result['review_status'])

    def state(self):
        return MissionState(
            drone_ids=list({}), objectives=[], no_fly_zones=[], discovered_targets={}, events=[],
            explored_cells=set(), current_plan=None, proposed_plans=[],
            under_review=None, next_agent=None, review_status=None, human_approval=None,
            step=0, mission_status=MissionStatus.ONGOING,
        )

    def test_agents_wrap_injected_policy_results_in_correct_fields(self):
        candidate = Plan(plan={}, reason="Candidate", estimated_completion_times=2)
        cases = [
            (MissionCommander, CommanderRoutingDecision(AgentType.REVIEWER, candidate),
             {"next_agent": AgentType.REVIEWER, "under_review": candidate,
              "mission_status": MissionStatus.ONGOING, "needs_replan": True,
              "commander_rethink": None,
              "review_status": None, "human_approval": None}),
            (MissionPlanner, [candidate], {"proposed_plans": [candidate],
                                        "review_status": None, "human_approval": None}),
            (Reviewer, ReviewResult(outcome="replan", critique="Review", concerns=[], mandatory_watches=[]),
             {"review_status": ReviewResult(outcome="replan", critique="Review", concerns=[], mandatory_watches=[]), "needs_replan": True}),
        ]
        for agent_type, policy_result, expected in cases:
            with self.subTest(agent=agent_type.__name__):
                state = self.state()
                before = state.model_dump()
                policy = Mock(spec=["evaluate"])
                policy.evaluate.return_value = policy_result
                self.assertEqual(agent_type(policy=policy).run(state), expected)
                policy.evaluate.assert_called_once_with(state, tools=None)
                self.assertEqual(state.model_dump(), before)

    def test_naive_commander_requests_planning_without_proposals(self):
        state = self.state()
        state.under_review = Plan(plan={}, reason="Stale", estimated_completion_times=0)
        decision = NaiveMissionCommanderPolicy().evaluate(state)
        self.assertEqual(decision.next_agent, AgentType.MISSION_PLANNER)
        self.assertIsNone(decision.under_review)
        self.assertIsNotNone(state.under_review)
        self.assertEqual(MissionCommander(policy=NaiveMissionCommanderPolicy()).run(state), {
            "next_agent": AgentType.MISSION_PLANNER, "under_review": None,
            "mission_status": MissionStatus.ONGOING, "needs_replan": True,
            "commander_rethink": None,
            "proposed_plans": [],
        })

    def test_naive_commander_selects_first_proposed_plan(self):
        state = self.state()
        state.proposed_plans = [
            Plan(plan={}, reason="First", estimated_completion_times=5),
            Plan(plan={}, reason="Second", estimated_completion_times=1),
        ]
        decision = NaiveMissionCommanderPolicy().evaluate(state)
        self.assertEqual(decision.next_agent, AgentType.REVIEWER)
        self.assertEqual(decision.under_review, state.proposed_plans[0])
        self.assertIsNone(state.under_review)

    def monitoring_policy(self):
        policy = Mock(spec=['evaluate'])
        policy.evaluate.return_value = MonitoringPlan(
            watches=[WatchPredicate(type=WatchType.SECTOR_COMPLETE, sector_id=str(i)) for i in range(3)],
            reason='Explicit test selection',
        )
        return policy

    def test_fleet_monitor_returns_monitoring_plan_based_on_accepted_plan(self):
        state = self.state()
        state.current_plan = MissionPlan(plan={})
        state.needs_replan = False
        before = state.model_dump()
        policy = self.monitoring_policy()
        result = FleetMonitor(policy=policy).run(state)
        self.assertEqual(result, {'monitoring_plan': policy.evaluate.return_value})
        observation = policy.evaluate.call_args.args[0]
        self.assertEqual(observation.accepted_plan, state.current_plan)
        self.assertIsNot(observation.accepted_plan, state.current_plan)
        self.assertEqual(state.model_dump(), before)
        restored = MissionState.model_validate(state.model_dump() | result)
        self.assertEqual(len(restored.monitoring_plan.watches), 3)

    def test_fleet_monitor_rejects_missing_or_unaccepted_plan(self):
        state = self.state()
        monitor = FleetMonitor(policy=self.monitoring_policy())
        with self.assertRaises(ValueError):
            monitor.run(state)
        state.current_plan = MissionPlan(plan={})
        with self.assertRaises(ValueError):
            monitor.run(state)

    def test_planner_redirects_detecting_drone_and_preserves_other_assignments(self):
        state = self.state()
        state.drone_ids = ["0", "1"]
        state.current_plan = MissionPlan(plan={
            "0": Objective(drone_id="0", objective_type=ObjectiveType.HOLD),
            "1": Objective(drone_id="1", objective_type=ObjectiveType.RETURN_TO_BASE),
        })
        state.discovered_targets = {"target": Target(target_id="target", position=(1, 7))}
        state.latest_events = [WatchAlert(type=FreeWatchType.NEW_TARGET_DETECTED, step=0,
                    drone_id='0', target_id='target', position=(1, 7))]
        before = state.model_dump()
        proposal = NaiveMissionPlannerPolicy().evaluate(state)[0]
        self.assertEqual(proposal.plan["0"].objective_type, ObjectiveType.GOTO)
        self.assertEqual(proposal.plan["0"].target_id, "target")
        self.assertEqual(proposal.plan["1"], state.current_plan.plan["1"])
        self.assertEqual(state.model_dump(), before)

    def test_commander_classifies_completion_from_typed_alert(self):
        for status in (MissionStatus.SUCCESS, MissionStatus.FAILED):
            state = self.state()
            state.latest_events = [WatchAlert(type=FreeWatchType.MISSION_COMPLETE,
                                             step=1, mission_status=status)]
            decision = NaiveMissionCommanderPolicy().evaluate(state)
            self.assertIsNone(decision.next_agent)
            self.assertEqual(decision.mission_status, status)
            self.assertEqual(state.mission_status, MissionStatus.ONGOING)

    def test_graph_reuses_current_plan_for_redundant_detections(self):
        state = self.state()
        drones = initialize_drone_states([(0, 0)], 25.0)
        state.drone_ids = list(drones)
        state.current_plan = MissionPlan(plan={
            "0": Objective(drone_id="0", objective_type=ObjectiveType.HOLD),
        })
        state.needs_replan = False
        state.discovered_targets = {"target": Target(target_id="target", position=(1, 7))}
        state.latest_events = [WatchAlert(type=FreeWatchType.NEW_TARGET_DETECTED, step=0,
                    drone_id='0', target_id='target', position=(1, 7))]
        state.proposed_plans = [Plan(plan={}, reason="Previous cycle", estimated_completion_times=0)]
        policy = Mock(spec=["evaluate"])
        workflow = self.build_graph({
            AgentType.MISSION_PLANNER: MissionPlanner(policy=policy).run,
            AgentType.FLEET_MONITOR: FleetMonitor(policy=self.monitoring_policy()).run,
        }).compile()
        result = MissionState.model_validate(workflow.invoke(state))
        policy.evaluate.assert_not_called()
        self.assertEqual(result.current_plan, state.current_plan)
        self.assertFalse(result.needs_replan)
        self.assertIsNotNone(result.monitoring_plan)
        self.assertEqual(result.under_review.plan, state.current_plan.plan)
        self.assertIn('Retain', result.under_review.reason)
        self.assertEqual(result.mission_status, MissionStatus.ONGOING)


if __name__ == "__main__":
    unittest.main()
