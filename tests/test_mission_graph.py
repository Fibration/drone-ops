import unittest
from unittest.mock import Mock, patch

from langgraph.graph import END

from src.agent import MissionCommander, SafetyReviewer
from src.mission_graph import (
    HUMAN_REVIEW_NODE, build_mission_graph, route_commander,
    route_human_review, route_safety_review,
)
from src.models import AgentType, Evaluation, MissionPlan, MissionState, MissionStatus, Plan
from src.policy import CommanderRoutingDecision


class MissionGraphTest(unittest.TestCase):
    def state(self):
        return MissionState(
            drone_ids=[], objectives=[], no_fly_zones=[], discovered_targets={}, events=[],
            explored_cells=set(), current_plan=None, proposed_plans=[], under_review=None,
            next_agent=None, safety_status=None, human_approval=None, step=0,
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
                if name == AgentType.SAFETY_REVIEWER:
                    return {'safety_status': Evaluation(decision=next(outcomes), reason='Review')}
                if name == HUMAN_REVIEW_NODE:
                    return {'human_approval': next(human_outcomes)}
                return {}
            return run

        names = (*AgentType, HUMAN_REVIEW_NODE)
        build_mission_graph({name: handler(name) for name in names}).compile().invoke(self.state())
        return trace

    def test_start_enters_commander_and_terminal_missions_end_directly(self):
        self.assertEqual(self.run_path([END]), [AgentType.MISSION_COMMANDER])
        for status in (MissionStatus.SUCCESS, MissionStatus.FAILED):
            state = self.state()
            state.mission_status = status
            self.assertEqual(route_commander(state), END)

    def test_approved_plan_finishes_through_fleet_monitor(self):
        self.assertEqual(
            self.run_path([AgentType.MISSION_PLANNER, AgentType.SAFETY_REVIEWER], ['approve']),
            [AgentType.MISSION_COMMANDER, AgentType.MISSION_PLANNER,
             AgentType.MISSION_COMMANDER, AgentType.SAFETY_REVIEWER, AgentType.FLEET_MONITOR],
        )
        initial = self.state()
        updates = list(build_mission_graph().compile().stream(initial))
        self.assertEqual([name for update in updates for name in update],
                         ['MISSION_COMMANDER', 'MISSION_PLANNER', 'MISSION_COMMANDER',
                          'SAFETY_REVIEWER', 'FLEET_MONITOR'])
        safety = updates[-2]['SAFETY_REVIEWER']
        self.assertEqual(safety['safety_status'].decision, 'approve')
        self.assertFalse(safety['needs_replan'])
        self.assertIsNotNone(safety['current_plan'])

    def test_revision_returns_to_commander_who_can_resubmit_or_replan(self):
        for destination in (AgentType.SAFETY_REVIEWER, AgentType.MISSION_PLANNER):
            destinations = [AgentType.SAFETY_REVIEWER, destination]
            expected = [AgentType.MISSION_COMMANDER, AgentType.SAFETY_REVIEWER,
                        AgentType.MISSION_COMMANDER, destination]
            if destination == AgentType.MISSION_PLANNER:
                destinations.append(AgentType.SAFETY_REVIEWER)
                expected.extend([AgentType.MISSION_COMMANDER, AgentType.SAFETY_REVIEWER])
            self.assertEqual(self.run_path(destinations, ['revise', 'approve']),
                             expected + [AgentType.FLEET_MONITOR])

    def test_commander_can_escalate_revision_and_human_rejection_returns_to_commander(self):
        self.assertEqual(
            self.run_path([AgentType.SAFETY_REVIEWER, HUMAN_REVIEW_NODE,
                           HUMAN_REVIEW_NODE], ['revise'], [False, True]),
            [AgentType.MISSION_COMMANDER, AgentType.SAFETY_REVIEWER,
             AgentType.MISSION_COMMANDER, HUMAN_REVIEW_NODE,
             AgentType.MISSION_COMMANDER, HUMAN_REVIEW_NODE, AgentType.FLEET_MONITOR],
        )

    def test_real_human_review_can_reject_then_approve(self):
        initial = self.state()
        candidate = Plan(plan={}, reason='Candidate', estimated_completion_times=0)
        initial.under_review = candidate
        initial.safety_status = Evaluation(decision='revise', reason='Safety concern')
        policy = Mock()
        policy.evaluate.return_value = CommanderRoutingDecision(
            next_agent=HUMAN_REVIEW_NODE,
            commander_rethink=Evaluation(decision='arbitrate', reason='Need human judgement'),
        )
        workflow = build_mission_graph({
            AgentType.MISSION_COMMANDER: MissionCommander(policy=policy).run,
            AgentType.FLEET_MONITOR: lambda state: {},
        }).compile()
        with patch('builtins.input', side_effect=['n', 'y']) as prompt:
            final = MissionState.model_validate(workflow.invoke(initial))
        self.assertEqual(prompt.call_count, 2)
        self.assertIn('Safety concern', prompt.call_args_list[0].args[0])
        self.assertIn('Need human judgement', prompt.call_args_list[0].args[0])
        self.assertTrue(final.human_approval)
        self.assertFalse(final.needs_replan)
        self.assertEqual(final.current_plan.plan, candidate.plan)
        self.assertIs(policy.evaluate.call_args_list[1].args[0].human_approval, False)

    def test_default_commander_replans_after_revision_then_can_accept_new_proposal(self):
        initial = self.state()
        policy = Mock()
        policy.evaluate.side_effect = [
            Evaluation(decision='revise', reason='Change priorities'),
            Evaluation(decision='approve', reason='Revised plan accepted'),
        ]
        planner_states = []
        def planner(state):
            planner_states.append(state)
            return {'proposed_plans': [Plan(plan={}, reason='New proposal', estimated_completion_times=0)]}
        workflow = build_mission_graph({
            AgentType.SAFETY_REVIEWER: SafetyReviewer(policy=policy).run,
            AgentType.MISSION_PLANNER: planner,
            AgentType.FLEET_MONITOR: lambda state: {},
        }).compile()
        final = MissionState.model_validate(workflow.invoke(initial))
        self.assertEqual(len(planner_states), 2)
        self.assertEqual(planner_states[1].commander_rethink.reason, 'Change priorities')
        self.assertEqual(planner_states[1].proposed_plans, [])
        self.assertEqual(final.safety_status.decision, 'approve')
        self.assertFalse(final.needs_replan)

    def test_routes_and_state_round_trip_support_human_destination(self):
        state = self.state()
        state.next_agent = HUMAN_REVIEW_NODE
        restored = MissionState.model_validate_json(state.model_dump_json())
        self.assertEqual(route_commander(restored), HUMAN_REVIEW_NODE)
        self.assertIsNone(restored.commander_rethink)
        for outcome, destination in (('approve', AgentType.FLEET_MONITOR),
                                     ('revise', AgentType.MISSION_COMMANDER)):
            state.safety_status = Evaluation(decision=outcome, reason='Review')
            self.assertEqual(route_safety_review(state), destination)
        for approval, destination in ((True, AgentType.FLEET_MONITOR),
                                      (False, AgentType.MISSION_COMMANDER)):
            state.human_approval = approval
            self.assertEqual(route_human_review(state), destination)
        state.current_plan = MissionPlan(plan={})
        state.needs_replan = False
        self.assertEqual(route_commander(state), AgentType.FLEET_MONITOR)

    def test_missing_or_invalid_review_decisions_are_rejected(self):
        for router in (route_commander, route_safety_review, route_human_review):
            with self.assertRaises(ValueError):
                router(self.state())
        state = self.state()
        state.safety_status = Evaluation(decision='human_reviewer', reason='Unsupported safety outcome')
        with self.assertRaises(ValueError):
            route_safety_review(state)


if __name__ == '__main__':
    unittest.main()
