import unittest
from unittest.mock import Mock, patch

from langgraph.graph import END

from src.agent import FleetMonitor, MissionCommander, MissionPlanner, Reviewer
from src.mission_graph import (
    HUMAN_REVIEW_NODE, build_mission_graph, route_commander,
    route_human_review, route_review,
)
from src.models import AgentType, ReviewResult, ReviewOutcome, CommanderReviewOutcome, MissionPlan, MissionState, MissionStatus, Plan, MonitoringPlan, WatchPredicate, WatchType
from src.models import CommanderRoutingDecision
from src.policy import NaiveMissionCommanderPolicy, NaiveMissionPlannerPolicy, NaiveReviewerPolicy


class MissionGraphTest(unittest.TestCase):
    def test_graph_requires_explicit_configuration_for_every_agent_role(self):
        with self.assertRaises(ValueError):
            build_mission_graph()
        with self.assertRaises(ValueError):
            build_mission_graph(agents={
                AgentType.MISSION_COMMANDER: MissionCommander(policy=NaiveMissionCommanderPolicy()),
            })

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

    def state(self):
        return MissionState(
            drone_ids=[], objectives=[], no_fly_zones=[], discovered_targets={}, events=[],
            explored_cells=set(), current_plan=None, proposed_plans=[], under_review=None,
            next_agent=None, review_status=None, human_approval=None, step=0,
            mission_status=MissionStatus.ONGOING,
        )

    def run_path(self, destinations, outcomes=(), human_outcomes=()):
        trace = []
        destinations, outcomes, human_outcomes = map(iter, (destinations, outcomes, human_outcomes))

        def handler(name):
            def run(state):
                trace.append(name)
                if name == AgentType.MISSION_COMMANDER:
                    destination = next(destinations)
                    if destination == END:
                        return {'mission_status': MissionStatus.SUCCESS, 'next_agent': None}
                    return {'next_agent': destination}
                if name == AgentType.REVIEWER:
                    return {'review_status': ReviewResult(outcome=next(outcomes), critique='Review', concerns=[], mandatory_watches=[])}
                if name == HUMAN_REVIEW_NODE:
                    return {'human_approval': next(human_outcomes)}
                return {}
            return run

        names = (*AgentType, HUMAN_REVIEW_NODE)
        self.build_graph({name: handler(name) for name in names}).compile().invoke(self.state())
        return trace

    def test_start_enters_commander_and_terminal_missions_end_directly(self):
        self.assertEqual(self.run_path([END]), [AgentType.MISSION_COMMANDER])
        for status in (MissionStatus.SUCCESS, MissionStatus.FAILED):
            state = self.state()
            state.mission_status = status
            self.assertEqual(route_commander(state), END)

    def test_approved_plan_finishes_through_fleet_monitor(self):
        self.assertEqual(
            self.run_path([AgentType.MISSION_PLANNER, AgentType.REVIEWER], ['approve']),
            [AgentType.MISSION_COMMANDER, AgentType.MISSION_PLANNER,
             AgentType.MISSION_COMMANDER, AgentType.REVIEWER, AgentType.FLEET_MONITOR],
        )
        initial = self.state()
        monitor_policy = Mock()
        monitor_policy.evaluate.return_value = MonitoringPlan(
            watches=[WatchPredicate(type=WatchType.SECTOR_COMPLETE, sector_id=str(i)) for i in range(3)],
            reason='Explicit test selection',
        )
        updates = list(self.build_graph({AgentType.FLEET_MONITOR: FleetMonitor(policy=monitor_policy).run}).compile().stream(initial))
        self.assertEqual([name for update in updates for name in update],
                         ['MISSION_COMMANDER', 'MISSION_PLANNER', 'MISSION_COMMANDER',
                          'REVIEWER', 'FLEET_MONITOR'])
        review = updates[-2]['REVIEWER']
        self.assertEqual(review['review_status'].outcome, 'approve')
        self.assertFalse(review['needs_replan'])
        self.assertIsNotNone(review['current_plan'])

    def test_revision_returns_to_commander_who_can_resubmit_or_replan(self):
        for destination in (AgentType.REVIEWER, AgentType.MISSION_PLANNER):
            destinations = [AgentType.REVIEWER, destination]
            expected = [AgentType.MISSION_COMMANDER, AgentType.REVIEWER,
                        AgentType.MISSION_COMMANDER, destination]
            if destination == AgentType.MISSION_PLANNER:
                destinations.append(AgentType.REVIEWER)
                expected.extend([AgentType.MISSION_COMMANDER, AgentType.REVIEWER])
            self.assertEqual(self.run_path(destinations, ['replan', 'approve']),
                             expected + [AgentType.FLEET_MONITOR])

    def test_commander_can_escalate_revision_and_human_rejection_returns_to_commander(self):
        self.assertEqual(
            self.run_path([AgentType.REVIEWER, HUMAN_REVIEW_NODE,
                           HUMAN_REVIEW_NODE], ['replan'], [False, True]),
            [AgentType.MISSION_COMMANDER, AgentType.REVIEWER,
             AgentType.MISSION_COMMANDER, HUMAN_REVIEW_NODE,
             AgentType.MISSION_COMMANDER, HUMAN_REVIEW_NODE, AgentType.FLEET_MONITOR],
        )

    def test_real_human_review_can_reject_then_approve(self):
        initial = self.state()
        candidate = Plan(plan={}, reason='Candidate', estimated_completion_times=0)
        initial.under_review = candidate
        initial.review_status = ReviewResult(outcome='replan', critique='Priority concern', concerns=[], mandatory_watches=[])
        policy = Mock()
        policy.evaluate.return_value = CommanderRoutingDecision(
            next_agent=HUMAN_REVIEW_NODE,
            commander_rethink=CommanderReviewOutcome.HUMAN_REQUIRED,
        )
        workflow = self.build_graph({
            AgentType.MISSION_COMMANDER: MissionCommander(policy=policy).run,
            AgentType.FLEET_MONITOR: lambda state: {},
        }).compile()
        with patch('builtins.input', side_effect=['n', 'y']) as prompt:
            final = MissionState.model_validate(workflow.invoke(initial))
        self.assertEqual(prompt.call_count, 2)
        self.assertIn('Priority concern', prompt.call_args_list[0].args[0])
        self.assertIn('HUMAN_REQUIRED', prompt.call_args_list[0].args[0])
        self.assertTrue(final.human_approval)
        self.assertFalse(final.needs_replan)
        self.assertEqual(final.current_plan.plan, candidate.plan)
        self.assertIs(policy.evaluate.call_args_list[1].args[0].human_approval, False)

    def test_default_commander_replans_after_revision_then_can_accept_new_proposal(self):
        initial = self.state()
        policy = Mock()
        policy.evaluate.side_effect = [
            ReviewResult(outcome='replan', critique='Change priorities', concerns=[], mandatory_watches=[]),
            ReviewResult(outcome='approve', critique='Revised plan accepted', concerns=[], mandatory_watches=[]),
        ]
        planner_states = []
        def planner(state, *, tools=None):
            planner_states.append(state)
            return [Plan(plan={}, reason='New proposal', estimated_completion_times=0)]
        planner_policy = Mock()
        planner_policy.evaluate.side_effect = planner
        workflow = self.build_graph({
            AgentType.REVIEWER: Reviewer(policy=policy).run,
            AgentType.MISSION_PLANNER: MissionPlanner(policy=planner_policy).run,
            AgentType.FLEET_MONITOR: lambda state: {},
        }).compile()
        final = MissionState.model_validate(workflow.invoke(initial))
        self.assertEqual(len(planner_states), 2)
        self.assertEqual(planner_states[1].commander_rethink, CommanderReviewOutcome.REPLAN)
        self.assertEqual(planner_states[1].review_status.critique, 'Change priorities')
        self.assertEqual(planner_states[1].proposed_plans, [])
        self.assertEqual(final.review_status.outcome, 'approve')
        self.assertFalse(final.needs_replan)

    def test_routes_and_state_round_trip_support_human_destination(self):
        state = self.state()
        state.next_agent = HUMAN_REVIEW_NODE
        restored = MissionState.model_validate_json(state.model_dump_json())
        self.assertEqual(route_commander(restored), HUMAN_REVIEW_NODE)
        self.assertIsNone(restored.commander_rethink)
        for outcome, destination in (('approve', AgentType.FLEET_MONITOR),
                                     ('replan', AgentType.MISSION_COMMANDER),
                                     ('reconsider_priority', AgentType.MISSION_COMMANDER)):
            state.review_status = ReviewResult(outcome=outcome, critique='Review', concerns=[], mandatory_watches=[])
            self.assertEqual(route_review(state), destination)
        for approval, destination in ((True, AgentType.FLEET_MONITOR),
                                      (False, AgentType.MISSION_COMMANDER)):
            state.human_approval = approval
            self.assertEqual(route_human_review(state), destination)
        state.current_plan = MissionPlan(plan={})
        state.needs_replan = False
        state.next_agent = AgentType.REVIEWER
        self.assertEqual(route_commander(state), AgentType.REVIEWER)
        state.next_agent = AgentType.FLEET_MONITOR
        with self.assertRaises(ValueError):
            route_commander(state)

    def test_missing_or_invalid_review_decisions_are_rejected(self):
        for router in (route_commander, route_review, route_human_review):
            with self.assertRaises(ValueError):
                router(self.state())
        state = self.state()
        with self.assertRaises(ValueError):
            ReviewResult(outcome='human_reviewer', critique='Unsupported review outcome',
                         concerns=[], mandatory_watches=[])


if __name__ == '__main__':
    unittest.main()
