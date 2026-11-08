import unittest

from src.map import Terrain
from src.models import IncidentType, Objective, ObjectiveType
from src.simulator import Simulator
from src.utility import initialize_drone_states


class BatteryDrainTest(unittest.TestCase):
    def test_silent_drain_is_discovered_at_monitoring_and_applied_once(self):
        for initial_battery in (25.0, 5.0):
            with self.subTest(initial_battery=initial_battery):
                drones = initialize_drone_states([(0, 10), (1, 10)], 25.0)
                drones["0"].battery = initial_battery
                simulator = Simulator(
                    [[Terrain.EMPTY] * 12 for _ in range(2)], drones, [],
                    scenario={1: (IncidentType.BATTERY_DRAIN, "0")},
                )
                plan = {
                    key: Objective(drone_id=key, objective_type=ObjectiveType.RETURN_TO_BASE)
                    for key in drones
                }
                event,  _, step = simulator.execute_plan(
                    plan, {"battery_reserve_threshold": 14.0}, {}, {},
                )
                self.assertEqual(step, 5)
                self.assertEqual(drones["1"].battery, 20.0)
                if initial_battery <= 10:
                    self.assertEqual(event, ["Drone 0 ran out of battery at (0, 10)"])
                    self.assertEqual(drones["0"].position, (0, 10))
                    self.assertEqual(drones["0"].battery, 0.0)
                    self.assertEqual(simulator.controllers["0"].objective.objective_type,
                                     ObjectiveType.HOLD)
                else:
                    self.assertEqual(event, ["Drone 0 battery low 10.0 at (0, 5)"])
                    self.assertEqual(drones["0"].battery, 10.0)
                    self.assertEqual(drones["0"].position, (0, 5))
                self.assertEqual(simulator.controllers["1"].objective.objective_type,
                                 ObjectiveType.RETURN_TO_BASE)
                # Resume the same plan: the incident must not drain the battery again.
                resumed_event,  _, _ = simulator.execute_plan(
                    plan, {"battery_reserve_threshold": 0.0}, {}, {},
                )
                self.assertTrue(any("returned to base" in event for event in resumed_event))
                self.assertEqual(drones["0"].battery,
                                 0.0 if initial_battery <= 10 else 5.0)
                if initial_battery <= 10:
                    self.assertEqual(drones["0"].position, (0, 10))


if __name__ == "__main__":
    unittest.main()
