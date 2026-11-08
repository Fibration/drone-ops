import unittest

from src.map import Terrain, extract_from_raw
from src.mission import RAW_MAP, create_sectors, mission
from src.mission_commander import NaiveCommanderPolicy, mission_team
from src.mission_graph import build_mission_graph
from src.models import IncidentType, MissionState, MissionStatus
from src.simulator import Simulator
from src.utility import initialize_drone_states


class MissionIntegrationTest(unittest.TestCase):
    def test_graph_policies_complete_example_with_no_fly_zone(self):
        truth, terrain, positions, _ = extract_from_raw(RAW_MAP)
        drones = initialize_drone_states(sorted(positions), default_battery=25.0)
        simulator = Simulator(truth, drones, targets=[(1, 7)], scenario={
            5: (IncidentType.NEW_NO_FLY_ZONE,
                "New no-fly zone added: rows 3-4, columns 4-8"),
        })

        graph = build_mission_graph(telemetry_queries=simulator.telemetry_queries).compile()

        def control_cycle(state):
            self.assertLess(len(state.events), 100, "Mission did not terminate")
            return MissionState.model_validate(graph.invoke(state, config={"recursion_limit": 10}))

        result = mission(simulator, terrain,  control_cycle,
                         {"battery_reserve_threshold": 10.0})
        self.assertEqual(result.mission_status, MissionStatus.SUCCESS)
        self.assertEqual(result.completed_sectors, {"0", "1"})
        self.assertEqual(len(result.discovered_targets), 1)
        self.assertTrue(any("reached the target at (1, 7)" in event.description
                            for event in result.events))
        known_zones = {cell for region in result.no_fly_zones for cell in region.coordinates}
        self.assertTrue({(row, col) for row in (3, 4) for col in range(4, 9)} <= known_zones)
        assert result.current_plan is not None
        self.assertEqual(set(result.current_plan.plan), set(drones))

    def test_example_search_completes_successfully(self):
        truth, terrain, positions, _ = extract_from_raw(RAW_MAP)
        drones = initialize_drone_states(
            sorted(positions, key=lambda position: position[0]),
            default_battery=25.0,
        )
        sectors = create_sectors(terrain, len(drones))
        simulator = Simulator(truth, drones, targets=[(1, 7)])

        def commander(state):
            self.assertLess(len(state.events), 200, "Mission did not terminate")
            return mission_team(state, terrain, sectors, policy=NaiveCommanderPolicy())

        result = mission(
            simulator,
            terrain,
            
            mission_team=commander,
            monitoring_checks={"battery_reserve_threshold": 10.0},
        )

        self.assertEqual(result.mission_status, MissionStatus.SUCCESS)
        self.assertEqual(result.events[-1].description, "Mission plan exhausted")
        self.assertIn(
            (1, 7), [target.position for target in result.discovered_targets.values()]
        )
        self.assertEqual(len(result.discovered_targets), 1)
        self.assertTrue(
            any(
                "reached the target at (1, 7)" in event.description
                for event in result.events
            )
        )
        self.assertIn((1, 7), result.explored_cells)
        self.assertGreater(result.step, 0)

    def test_example_search_with_new_no_fly_zone_at_step_five(self):
        truth, terrain, positions, _ = extract_from_raw(RAW_MAP)
        drones = initialize_drone_states(
            sorted(positions, key=lambda position: position[0]),
            default_battery=25.0,
        )
        zone = {(row, col) for row in (3, 4) for col in range(4, 9)}
        zone_event = "New no-fly zone added: rows 3-4, columns 4-8"
        sectors = create_sectors(terrain, len(drones))
        simulator = Simulator(truth, drones, targets=[(1, 7)], scenario={5: (IncidentType.NEW_NO_FLY_ZONE, zone_event)})
        zone_added = False

        def commander(state):
            nonlocal zone_added
            self.assertLess(len(state.events), 200, "Mission did not terminate")
            latest = [event.description for event in state.latest_events]
            if zone_event in latest:
                self.assertEqual(state.step, 5)
                self.assertFalse(zone_added, "Scenario event repeated")
                for row, col in zone:
                    self.assertEqual(simulator.map[row][col], Terrain.NO_FLY_ZONE)
                zone_added = True
            if zone_added:
                self.assertFalse(any("entered a no-fly zone" in event for event in latest))
                if zone_event not in latest:
                    for drone in simulator.drones.values():
                        self.assertNotIn(drone.position, zone)
            result = mission_team(state, terrain, sectors, policy=NaiveCommanderPolicy())
            if zone_added:
                known_zones = {
                    cell
                    for region in result.no_fly_zones
                    for cell in region.coordinates
                }
                self.assertTrue(
                    zone <= known_zones, "Commander did not record the new zone"
                )
            return result

        result = mission(
            simulator,
            terrain,
            
            mission_team=commander,
            monitoring_checks={"battery_reserve_threshold": 10.0},
        )

        self.assertTrue(zone_added, "Mission never reached the scenario event")
        self.assertEqual(result.mission_status, MissionStatus.SUCCESS)
        self.assertEqual(result.events[-1].description, "Mission plan exhausted")
        self.assertIn(
            (1, 7), [target.position for target in result.discovered_targets.values()]
        )
        self.assertEqual(len(result.discovered_targets), 1)
        self.assertTrue(
            any(
                "reached the target at (1, 7)" in event.description
                for event in result.events
            )
        )
        self.assertGreater(result.step, 5)


if __name__ == "__main__":
    unittest.main()
