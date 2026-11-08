import unittest
from unittest.mock import Mock

from pydantic import ValidationError

from src.agent import FleetMonitor, MissionCommander, MissionPlanner
from src.map import Terrain
from src.mission_graph import build_mission_graph
from src.models import (
    AgentType, MissionEvent, MissionPlan, MissionState, MissionStatus, Objective,
    ObjectiveType, TelemetryField, TelemetryRequest, TelemetrySnapshot,
)
from src.policy import (
    CommanderRoutingDecision, FleetMonitorObservation, NaiveFleetMonitorPolicy,
    NaiveMissionCommanderPolicy, NaiveMissionPlannerPolicy,
)
from src.simulator import Simulator
from src.telemetry import RecordedTelemetry, has_telemetry
from src.utility import initialize_drone_states


class TelemetryTest(unittest.TestCase):
    def snapshots(self):
        readings = []
        for step in range(3):
            drones = initialize_drone_states([(0, step), (1, step)], 25.0 - step)
            drones["1"].status = "amber"
            readings.append(TelemetrySnapshot(step=step, drones=drones))
        return readings

    def state(self):
        history = self.snapshots()
        drones = history[-1].drones
        return MissionState(
            drone_ids=list(drones), objectives=[], no_fly_zones=[], discovered_targets={},
            events=[], latest_events=[], explored_cells=set(), telemetry=[],
             proposed_plans=[], under_review=None,
            current_plan=MissionPlan(plan={key: Objective(drone_id=key, objective_type=ObjectiveType.HOLD)
                                          for key in drones}),
            next_agent=None, safety_status=None, human_approval=None,
            step=2, mission_status=MissionStatus.ONGOING,
        )

    def test_mission_state_and_events_exclude_raw_drone_state(self):
        state = self.state()
        state.events = [MissionEvent(description="Mission plan exhausted", step=2)]
        payload = state.model_dump()
        self.assertEqual(payload["drone_ids"], ["0", "1"])
        self.assertNotIn("drones", payload)
        self.assertNotIn("telemetry_history", payload)
        self.assertEqual(payload["events"], [{"description": "Mission plan exhausted", "step": 2}])
        with self.assertRaises(ValidationError):
            MissionState.model_validate({**payload, "drones": self.snapshots()[0].drones})
        with self.assertRaises(ValidationError):
            MissionEvent.model_validate({"description": "event", "drones": {}})

    def test_injected_graph_reads_live_provider_without_changing_received_telemetry(self):
        drones = initialize_drone_states([(0, 3), (0, 0)], 25.0)
        simulator = Simulator([[Terrain.EMPTY] * 4], drones, [])
        provider = simulator.telemetry_queries
        workflow = build_mission_graph(telemetry_queries=provider).compile()
        state = self.state()
        before = MissionState.model_validate(workflow.invoke(state))
        position_request = TelemetryRequest(field=TelemetryField.POSITION)
        def received_position(current):
            result = next(item for item in current.telemetry if item.request == position_request)
            return result.readings["0"][0]
        self.assertEqual(received_position(before).value, (0, 3))
        simulator.execute_plan(
            {"0": Objective(drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE)},
            {"battery_reserve_threshold": 0.0}, {}, {},
        )
        self.assertIs(provider, simulator.telemetry_queries)
        self.assertEqual(provider.position("0")["0"][0].value, (0, 0))
        self.assertEqual(received_position(before).value, (0, 3))
        before.step = simulator.timestep
        after = MissionState.model_validate(workflow.invoke(before))
        self.assertEqual((received_position(after).step, received_position(after).value), (3, (0, 0)))
        self.assertNotIn("telemetry_history", after.model_dump())

    def test_monitor_requires_provider_when_drones_are_known(self):
        with self.assertRaisesRegex(ValueError, "injected telemetry provider"):
            FleetMonitor().run(self.state())

    def test_latest_historical_single_multiple_and_all_drone_queries(self):
        queries = RecordedTelemetry(self.snapshots())
        self.assertEqual(queries.drone_ids(), ["0", "1"])
        self.assertEqual(queries.battery_level("0")["0"][0].value, 23.0)
        self.assertEqual(set(queries.position()), {"0", "1"})
        self.assertEqual(queries.position(["0", "1"])["1"][0].value, (1, 2))
        historical = queries.battery_level(["0"], at_step=1, history=True)["0"]
        self.assertEqual([(item.step, item.value) for item in historical], [(0, 25.0), (1, 24.0)])
        self.assertEqual(queries.position("0", at_step=1)["0"][0].value, (0, 1))
        self.assertEqual(queries.comms("0")["0"][0].value, True)
        self.assertEqual(queries.status("1")["1"][0].value, "amber")
        self.assertIsNone(queries.assignment("0")["0"][0].value)
        self.assertEqual({item.field for item in queries.available_telemetry()}, set(TelemetryField))

    def test_missing_readings_unknown_ids_and_results_are_independent(self):
        queries = RecordedTelemetry(self.snapshots()[2:])
        self.assertEqual(queries.position("0", at_step=1), {"0": []})
        self.assertEqual(queries.position([]), {})
        with self.assertRaisesRegex(ValueError, "Unknown drone IDs"):
            queries.battery_level(["missing"])
        with self.assertRaises(ValueError):
            queries.battery_level(at_step=-1)
        result = queries.battery_level("0")
        result["0"][0].value = -1
        self.assertEqual(queries.battery_level("0")["0"][0].value, 23.0)

    def test_disconnected_drone_keeps_last_received_reading_and_timestamp(self):
        history = self.snapshots()
        history[-1].drones["0"].comms = False
        queries = RecordedTelemetry(history)
        position = queries.position("0")["0"][0]
        self.assertEqual((position.step, position.value), (1, (0, 1)))
        self.assertEqual(queries.comms("0")["0"][0].value, False)
        self.assertEqual([item.step for item in queries.battery_level("0", history=True)["0"]], [0, 1])

    def test_fleet_policy_receives_query_tools_not_full_state(self):
        state = self.state()
        before = state.model_dump()
        policy = Mock(spec=["evaluate"])
        policy.evaluate.side_effect = NaiveFleetMonitorPolicy().evaluate
        result = FleetMonitor(policy=policy, telemetry_queries=RecordedTelemetry(self.snapshots())).run(state)
        observation = policy.evaluate.call_args.args[0]
        self.assertIsInstance(observation, FleetMonitorObservation)
        self.assertFalse(hasattr(observation, "drones"))
        self.assertFalse(hasattr(observation, "mission_state"))
        self.assertFalse(hasattr(observation, "events"))
        self.assertEqual(observation.telemetry.position("0")["0"][0].value, (0, 2))
        self.assertEqual(len(result["telemetry"]), 3)
        self.assertNotIn("needs_replan", result)
        self.assertEqual(state.model_dump(), before)

    def test_simulator_records_each_step_without_mutable_drone_aliases(self):
        drones = initialize_drone_states([(0, 3)], 25.0)
        simulator = Simulator([[Terrain.EMPTY] * 4], drones, [])
        simulator.execute_plan(
            {"0": Objective(drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE)},
            {"battery_reserve_threshold": 0.0}, {}, {},
        )
        self.assertEqual([item.step for item in simulator.telemetry_history], [0, 1, 2, 3])
        self.assertEqual([item.value for item in RecordedTelemetry(simulator.telemetry_history)
                          .battery_level("0", history=True)["0"]], [25.0, 24.0, 23.0, 22.0])
        drones["0"].battery = 99.0
        self.assertEqual(simulator.telemetry_history[-1].drones["0"].battery, 22.0)

    def test_commander_requests_additional_telemetry_and_resumes_planning(self):
        state = self.state()
        event = MissionEvent(description="Drone 0 detected a target at (0, 4)")
        state.latest_events = [event]
        state.events = [event]
        requests = [
            TelemetryRequest(field=TelemetryField.BATTERY_LEVEL, drone_ids=["0"], history=True, at_step=1),
            TelemetryRequest(field=TelemetryField.STATUS, drone_ids=["1"]),
        ]
        observations = []

        class RequestingCommander:
            def evaluate(policy_self, current):
                observations.append(current.model_copy(deep=True))
                available = {item.field for item in current.available_telemetry}
                if not all(has_telemetry(current.telemetry, request) for request in requests):
                    self.assertTrue({request.field for request in requests} <= available)
                    return CommanderRoutingDecision(
                        next_agent=AgentType.FLEET_MONITOR, telemetry_requests=requests,
                    )
                return NaiveMissionCommanderPolicy().evaluate(current)

        workflow = build_mission_graph({
            AgentType.MISSION_COMMANDER: MissionCommander(policy=RequestingCommander()).run,
        }, telemetry_queries=RecordedTelemetry(self.snapshots())).compile()
        result = MissionState.model_validate(workflow.invoke(state, config={"recursion_limit": 12}))
        self.assertEqual(len(observations), 3)  # Request -> plan -> review.
        self.assertEqual(len(observations[0].telemetry), 3)
        self.assertEqual(len(observations[1].telemetry), 5)
        self.assertEqual(observations[1].new_target_ids, {"0"})
        self.assertEqual(result.telemetry_requests, [])
        self.assertEqual(result.safety_status, "approve")
        assert result.current_plan is not None
        self.assertEqual(result.current_plan.plan["0"].objective_type, ObjectiveType.GOTO)
        self.assertTrue(all(has_telemetry(result.telemetry, request) for request in requests))
        self.assertTrue(all(item.drone_id == "0" for result_item in result.telemetry
                            if result_item.request == requests[0]
                            for readings in result_item.readings.values() for item in readings))

    def test_telemetry_followup_preserves_proposals_already_being_considered(self):
        request = TelemetryRequest(field=TelemetryField.STATUS, drone_ids=["0"])

        class ReviewingCommander:
            def evaluate(self, state):
                if state.proposed_plans and not has_telemetry(state.telemetry, request):
                    return CommanderRoutingDecision(
                        next_agent=AgentType.FLEET_MONITOR, telemetry_requests=[request],
                    )
                return NaiveMissionCommanderPolicy().evaluate(state)

        planner = Mock(spec=["evaluate"])
        planner.evaluate.side_effect = NaiveMissionPlannerPolicy().evaluate
        workflow = build_mission_graph({
            AgentType.MISSION_COMMANDER: MissionCommander(policy=ReviewingCommander()).run,
            AgentType.MISSION_PLANNER: MissionPlanner(policy=planner).run,
        }, telemetry_queries=RecordedTelemetry(self.snapshots())).compile()
        result = MissionState.model_validate(workflow.invoke(self.state(), config={"recursion_limit": 12}))
        planner.evaluate.assert_called_once()
        self.assertEqual(result.safety_status, "approve")
        self.assertEqual(len(result.proposed_plans), 1)
        self.assertTrue(has_telemetry(result.telemetry, request))


if __name__ == "__main__":
    unittest.main()
