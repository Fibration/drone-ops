import unittest
from langgraph.graph import END
from unittest.mock import patch

from src.mission_graph import (
    HUMAN_REVIEW_NODE, build_mission_graph, route_commander, route_safety_review, route_human_review,
)
from src.models import AgentType, MissionState, MissionStatus


class MissionGraphTest(unittest.TestCase):
    def state(self):
        return MissionState(
            drone_ids=list({}), objectives=[], no_fly_zones=[], discovered_targets={}, events=[],
            explored_cells=set(), current_plan=None, telemetry=[], proposed_plans=[],
            under_review=None, next_agent=None,
            safety_status=None, human_approval=None, step=0, mission_status=MissionStatus.ONGOING,
        )

    def run_path(self, destinations, outcomes=(), human_outcomes=()):
        trace = []
        destinations = iter(destinations)
        outcomes = iter(outcomes)
        human_outcomes = iter(human_outcomes)

        def handler(name):
            def run(state):
                trace.append(name)
                self.assertIsInstance(state, MissionState)
                if name == AgentType.MISSION_COMMANDER:
                    destination = next(destinations)
                    if destination == END:
                        return {"mission_status": MissionStatus.SUCCESS, "next_agent": None}
                    return {"next_agent": destination}
                if name == AgentType.SAFETY_REVIEWER:
                    return {"safety_status": next(outcomes)}
                if name == HUMAN_REVIEW_NODE:
                    return {"human_approval": next(human_outcomes)}
                return {}
            return run

        names = (AgentType.FLEET_MONITOR, AgentType.MISSION_COMMANDER, AgentType.MISSION_PLANNER, AgentType.SAFETY_REVIEWER,
                 HUMAN_REVIEW_NODE)
        workflow = build_mission_graph({name: handler(name) for name in names}).compile()
        workflow.invoke(self.state())
        return trace

    def test_commander_can_finish(self):
        self.assertEqual(self.run_path([END]), [AgentType.FLEET_MONITOR, AgentType.MISSION_COMMANDER])

    def test_real_first_pass_agents_finish_planning_cycle(self):
        workflow = build_mission_graph().compile()
        initial = self.state()
        updates = list(workflow.stream(initial, config={"recursion_limit": 10}))
        self.assertEqual(
            [name for update in updates for name in update],
            [AgentType.FLEET_MONITOR, AgentType.MISSION_COMMANDER, AgentType.MISSION_PLANNER,
             AgentType.MISSION_COMMANDER, AgentType.SAFETY_REVIEWER],
        )
        result = initial.model_dump()
        for update in updates:
            for fields in update.values():
                result.update(fields)
        final = MissionState.model_validate(result)
        self.assertEqual(final.safety_status, "approve")
        self.assertEqual(len(final.proposed_plans), 1)
        self.assertEqual(final.under_review, final.proposed_plans[0])
        self.assertIsNotNone(final.current_plan)
        assert final.current_plan is not None
        self.assertEqual(final.current_plan.plan, final.under_review.plan)
        self.assertEqual(final.mission_status, MissionStatus.ONGOING)

    def test_partial_updates_preserve_state_and_output_converts_to_model(self):
        initial = self.state()
        initial.explored_cells = {(1, 2)}
        initial.completed_sectors = {"west"}
        workflow = build_mission_graph({
            AgentType.FLEET_MONITOR: lambda state: {"step": state.step + 1},
            AgentType.MISSION_COMMANDER: lambda state: {
                "mission_status": MissionStatus.SUCCESS, "next_agent": None,
            },
        }).compile()
        result = MissionState.model_validate(workflow.invoke(initial))
        self.assertEqual(result.step, 1)
        self.assertIsNone(result.next_agent)
        self.assertEqual(result.mission_status, MissionStatus.SUCCESS)
        self.assertEqual(result.explored_cells, {(1, 2)})
        self.assertEqual(result.completed_sectors, {"west"})
        self.assertEqual(initial.step, 0)
        self.assertIsNone(initial.next_agent)

    def test_telemetry_and_planning_loops_then_approval_ends_cycle(self):
        self.assertEqual(
            self.run_path([AgentType.FLEET_MONITOR, AgentType.MISSION_PLANNER, AgentType.SAFETY_REVIEWER], ["approve"]),
            [AgentType.FLEET_MONITOR, AgentType.MISSION_COMMANDER, AgentType.FLEET_MONITOR, AgentType.MISSION_COMMANDER,
             AgentType.MISSION_PLANNER, AgentType.MISSION_COMMANDER, AgentType.SAFETY_REVIEWER],
        )

    def test_revision_returns_to_planner_then_commander(self):
        self.assertEqual(
            self.run_path([AgentType.SAFETY_REVIEWER, AgentType.SAFETY_REVIEWER], ["revise", "approve"]),
            [AgentType.FLEET_MONITOR, AgentType.MISSION_COMMANDER, AgentType.SAFETY_REVIEWER,
             AgentType.MISSION_PLANNER, AgentType.MISSION_COMMANDER, AgentType.SAFETY_REVIEWER],
        )

    def test_human_approval_ends_cycle(self):
        self.assertEqual(
            self.run_path([AgentType.SAFETY_REVIEWER], ["human_reviewer"], [True]),
            [AgentType.FLEET_MONITOR, AgentType.MISSION_COMMANDER, AgentType.SAFETY_REVIEWER, HUMAN_REVIEW_NODE],
        )

    def test_human_rejection_returns_to_planner(self):
        self.assertEqual(
            self.run_path([AgentType.SAFETY_REVIEWER, AgentType.SAFETY_REVIEWER],
                          ["human_reviewer", "approve"], [False]),
            [AgentType.FLEET_MONITOR, AgentType.MISSION_COMMANDER, AgentType.SAFETY_REVIEWER, HUMAN_REVIEW_NODE,
             AgentType.MISSION_PLANNER, AgentType.MISSION_COMMANDER, AgentType.SAFETY_REVIEWER],
        )

    def test_human_rejection_can_be_followed_by_human_approval(self):
        self.assertEqual(
            self.run_path([AgentType.SAFETY_REVIEWER, AgentType.SAFETY_REVIEWER],
                          ["human_reviewer", "human_reviewer"], [False, True]),
            [AgentType.FLEET_MONITOR, AgentType.MISSION_COMMANDER, AgentType.SAFETY_REVIEWER, HUMAN_REVIEW_NODE,
             AgentType.MISSION_PLANNER, AgentType.MISSION_COMMANDER, AgentType.SAFETY_REVIEWER, HUMAN_REVIEW_NODE],
        )

    def test_terminal_mission_status_routes_to_end(self):
        for status in (MissionStatus.SUCCESS, MissionStatus.FAILED):
            with self.subTest(status=status):
                state = self.state()
                state.mission_status = status
                self.assertEqual(route_commander(state), END)

    def test_routers_distinguish_agents_from_end_and_human_review(self):
        state = self.state()
        state.next_agent = AgentType.MISSION_PLANNER
        self.assertIs(route_commander(state), AgentType.MISSION_PLANNER)
        for outcome, destination in (
            ("approve", END),
            ("revise", AgentType.MISSION_PLANNER),
            ("human_reviewer", HUMAN_REVIEW_NODE),
        ):
            state.safety_status = outcome
            self.assertIs(route_safety_review(state), destination)
        state.human_approval = True
        self.assertIs(route_human_review(state), END)
        state.human_approval = False
        self.assertIs(route_human_review(state), AgentType.MISSION_PLANNER)

    def test_next_agent_survives_json_round_trip_as_enum(self):
        state = self.state()
        state.next_agent = AgentType.SAFETY_REVIEWER
        restored = MissionState.model_validate_json(state.model_dump_json())
        self.assertIs(restored.next_agent, AgentType.SAFETY_REVIEWER)

    def test_agent_types_only_identify_software_agents(self):
        self.assertEqual(set(AgentType), {
            AgentType.FLEET_MONITOR, AgentType.MISSION_COMMANDER,
            AgentType.MISSION_PLANNER, AgentType.SAFETY_REVIEWER,
        })

    def test_real_input_handler_can_reject_then_approve(self):
        workflow = build_mission_graph({
            AgentType.SAFETY_REVIEWER: lambda state: {"safety_status": "human_reviewer"},
        }).compile()
        with patch("builtins.input", side_effect=["n", "y"]) as prompt:
            result = MissionState.model_validate(workflow.invoke(
                self.state(), config={"recursion_limit": 15},
            ))
        self.assertEqual(prompt.call_count, 2)
        self.assertIs(result.human_approval, True)
        self.assertIsNotNone(result.under_review)
        self.assertIsNotNone(result.current_plan)

    def test_missing_routing_decisions_are_rejected(self):
        for router in (route_commander, route_safety_review, route_human_review):
            with self.subTest(router=router.__name__):
                with self.assertRaises(ValueError):
                    router(self.state())


if __name__ == "__main__":
    unittest.main()
