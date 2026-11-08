import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.client import client
from src.map import Terrain, extract_from_raw
from src.mission import RAW_MAP, create_sectors, mission
from src.mission_commander import mission_team
from src.models import (
    IncidentType,
    CommanderDecision, Objective, ObjectiveType, MissionStatus,
)
from src.simulator import Simulator
from src.utility import initialize_drone_states


class MissionTeamTest(unittest.TestCase):
    def run_scenario(self, parse_response):
        truth, terrain, positions, _ = extract_from_raw(RAW_MAP)
        drones = initialize_drone_states(sorted(positions), default_battery=25.0)
        sectors = create_sectors(terrain, len(drones))
        zone = {(row, col) for row in (3, 4) for col in range(4, 9)}
        zone_event = "New no-fly zone added: rows 3-4, columns 4-8"
        simulator = Simulator(truth, drones, [(1, 7)], scenario={5: (IncidentType.NEW_NO_FLY_ZONE, zone_event)})
        calls = 0
        zone_seen = False

        def bounded_parse(**kwargs):
            nonlocal calls
            if calls >= 6:
                self.fail("Stopped at six API requests; mission did not complete")
            calls += 1
            payload = json.loads(kwargs["input"][1]["content"])
            self.assertIsInstance(payload["mission_state"], dict)
            self.assertIsInstance(payload["sectors"]["0"], dict)
            self.assertIs(kwargs["text_format"], CommanderDecision)
            return parse_response(**kwargs)

        def team(state):
            nonlocal zone_seen
            self.assertLess(len(state.events), 30, "Mission did not terminate")
            self.assertTrue(state.latest_events)
            self.assertEqual(state.events[-len(state.latest_events):], state.latest_events)
            for event in state.latest_events:
                self.assertEqual(event.step, state.step)
            latest = [event.description for event in state.latest_events]
            if zone_event in latest:
                zone_seen = True
                self.assertEqual(state.step, 5)
                for row, col in zone:
                    self.assertEqual(simulator.map[row][col], Terrain.NO_FLY_ZONE)
            if zone_seen:
                self.assertFalse(any("entered a no-fly zone" in event for event in latest))
                if zone_event not in latest:
                    for drone in simulator.drones.values():
                        self.assertNotIn(drone.position, zone)
            before = calls
            target_count = len(state.discovered_targets)
            previous_plan = state.current_plan
            result = mission_team(state, terrain, sectors)
            if "Mission plan exhausted" in latest:
                self.assertEqual(calls, before, "Exhaustion should not call the API")
            elif (
                all("detected a target" in event for event in latest)
                and len(result.discovered_targets) == target_count
            ):
                self.assertEqual(calls, before, "Known target should not call the API")
                self.assertIs(result.current_plan, previous_plan)
            else:
                self.assertEqual(
                    calls, before + 1, "New events must reach the commander"
                )
            if zone_seen:
                known = {
                    cell
                    for region in result.no_fly_zones
                    for cell in region.coordinates
                }
                self.assertTrue(zone <= known)
            return result

        with patch(
            "src.mission_commander.client.responses.parse", side_effect=bounded_parse
        ):
            result = mission(
                simulator,
                terrain,
                
                team,
                {"battery_reserve_threshold": 10.0},
                sectors=sectors,
            )
        self.assertTrue(zone_seen)
        self.assertEqual(result.mission_status, MissionStatus.SUCCESS)
        self.assertEqual(result.events[-1].description, "Mission plan exhausted")
        self.assertEqual(len(result.discovered_targets), 1)
        self.assertTrue(
            any("reached the target at (1, 7)" in e.description for e in result.events)
        )
        return result, calls

    @staticmethod
    def replay_response(**kwargs):
        state = json.loads(kwargs["input"][1]["content"])["mission_state"]
        objectives = dict(state["current_plan"]["plan"])
        for event in state["latest_events"]:
            description = event["description"]
            if "detected a target" in description:
                drone_id = description.split()[1]
                objectives[drone_id] = {
                    "drone_id": drone_id,
                    "objective_type": "goto",
                    "target_id": next(iter(state["discovered_targets"])),
                }
            elif "reached the target" in description:
                drone_id = description.split()[1]
                objectives[drone_id] = {"drone_id": drone_id, "objective_type": "hold"}
        decision = CommanderDecision(
            objectives=list(objectives.values()), reason="Replay"
        )
        return SimpleNamespace(output_parsed=decision)

    def test_scenario_with_mocked_structured_responses(self):
        result, calls = self.run_scenario(self.replay_response)
        self.assertEqual(calls, 5)
        self.assertEqual(result.step, 22)
        self.assertEqual(
            {e.description for e in result.events if e.description.startswith("Sector ")},
            {"Sector 0 fully explored", "Sector 1 fully explored"},
        )

    def test_unknown_sector_is_rejected_before_execution(self):
        def invalid_response(**kwargs):
            decision = CommanderDecision(
                objectives=[
                    Objective(
                        drone_id="0",
                        objective_type=ObjectiveType.SEARCH_SECTOR,
                        sector_id="unknown",
                    ),
                    Objective(drone_id="1", objective_type=ObjectiveType.HOLD),
                ],
                reason="Invalid sector",
            )
            return SimpleNamespace(output_parsed=decision)

        with self.assertRaisesRegex(ValueError, "Unknown sector ID"):
            self.run_scenario(invalid_response)

    @unittest.skipUnless(
        os.getenv("RUN_OPENAI_MISSION_TEST") == "1", "Opt-in live API test"
    )
    def test_live_scenario(self):
        # Disable SDK retries so six requests is a hard endpoint-call limit.
        live_client = client.with_options(max_retries=0, timeout=45.0)
        usage = {"requests": 0, "responses": 0, "input_tokens": 0, "output_tokens": 0}

        def live_response(**kwargs):
            usage["requests"] += 1
            response = live_client.responses.parse(**kwargs, max_output_tokens=1000)
            usage["responses"] += 1
            if response.usage:
                usage["input_tokens"] += response.usage.input_tokens
                usage["output_tokens"] += response.usage.output_tokens
            if response.output_parsed:
                print(
                    "Commander:", response.output_parsed.model_dump_json(), flush=True
                )
            return response

        try:
            result, _ = self.run_scenario(live_response)
            print(
                "Mission:",
                result.mission_status.value,
                "step:",
                result.step,
                flush=True,
            )
        finally:
            print("Live API usage:", usage, flush=True)


if __name__ == "__main__":
    unittest.main()
