import unittest

from src.map import Terrain
from src.models import Objective, ObjectiveType
from src.simulator import Simulator
from src.utility import initialize_drone_states


class SensorObservationTest(unittest.TestCase):
    def test_initial_scan_covers_full_circle_once_without_movement(self):
        drones = initialize_drone_states([(4, 4)], 25.0)
        simulator = Simulator([[Terrain.EMPTY] * 9 for _ in range(9)], drones, [])
        plan = {"0": Objective(drone_id="0", objective_type=ObjectiveType.HOLD)}
        explored = set()
        _,  observed, step = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, {}, {}, explored,
        )
        self.assertIs(observed, explored)
        self.assertEqual(len(explored), 29)
        self.assertIn((1, 4), explored)  # Radius boundary.
        self.assertIn((2, 2), explored)  # Diagonal inside the circle.
        self.assertNotIn((1, 3), explored)  # Outside the circle.
        self.assertEqual(step, 0)
        self.assertEqual(drones["0"].battery, 25.0)
        initial = explored.copy()
        drones["0"].position = (0, 0)
        simulator.execute_plan(plan, {"battery_reserve_threshold": 0.0}, {}, {}, explored)
        self.assertEqual(explored, initial, "Resuming must not repeat the initial scan")

    def test_controllers_see_the_same_shared_observations(self):
        drones = initialize_drone_states([(0, 0), (8, 8)], 25.0)
        simulator = Simulator([[Terrain.EMPTY] * 9 for _ in range(9)], drones, [])
        explored = set()
        plan = {key: Objective(drone_id=key, objective_type=ObjectiveType.HOLD)
                for key in drones}
        simulator.execute_plan(plan, {"battery_reserve_threshold": 0.0}, {}, {}, explored)
        for controller in simulator.controllers.values():
            self.assertIs(controller.explored_cells, explored)
            self.assertIn((0, 3), controller.explored_cells)
            self.assertIn((8, 5), controller.explored_cells)
        explored.add((4, 4))
        for controller in simulator.controllers.values():
            self.assertIn((4, 4), controller.explored_cells)


if __name__ == "__main__":
    unittest.main()
