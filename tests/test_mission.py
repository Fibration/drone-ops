import unittest

from src.map import extract_from_raw
from src.mission import RAW_MAP, mission
from src.mission_commander import naive_mission_commander
from src.models import MissionStatus
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
            mission_commander=commander,
            monitoring_checks={"battery_reserve_threshold": 10.0},
        )

        self.assertEqual(result.mission_status, MissionStatus.SUCCESS)
        self.assertEqual(result.events[-1].description, "Mission plan exhausted")
        self.assertIn((1, 7), [target.position for target in result.discovered_targets])
        self.assertTrue(
            any(
                "reached the target at (1, 7)" in event.description
                for event in result.events
            )
        )
        self.assertIn((1, 7), result.explored_cells)
        self.assertGreater(result.step, 0)


if __name__ == "__main__":
    unittest.main()
