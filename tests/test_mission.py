import unittest

from src.map import Terrain, extract_from_raw
from src.mission import RAW_MAP, mission
from src.mission_commander import naive_mission_commander
from src.models import IncidentType, MissionStatus
from src.simulator import Simulator
from src.utility import initialize_drone_states


class MissionIntegrationTest(unittest.TestCase):
    def test_example_search_completes_successfully(self):
        truth, terrain, positions, _ = extract_from_raw(RAW_MAP)
        drones = initialize_drone_states(
            sorted(positions, key=lambda position: position[0]),
            default_battery=25.0,
        )
        simulator = Simulator(truth, drones, targets=[(1, 7)])

        def commander(state):
            self.assertLess(len(state.events), 200, "Mission did not terminate")
            return naive_mission_commander(state)

        result = mission(
            simulator,
            terrain,
            drones,
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
                    for drone in state.drones.values():
                        self.assertNotIn(drone.position, zone)
            result = naive_mission_commander(state)
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
            drones,
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
