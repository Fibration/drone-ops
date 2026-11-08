import unittest

from src.map import Terrain
from src.models import IncidentType, Objective, ObjectiveType, Target
from src.simulator import Simulator
from src.utility import initialize_drone_states


class CommsFailureTest(unittest.TestCase):
    def test_silent_failure_is_monitored_then_drone_finishes_accepted_route(self):
        drones = initialize_drone_states([(0, 0), (1, 0)], 25.0)
        simulator = Simulator(
            [[Terrain.EMPTY] * 10 for _ in range(2)], drones, [],
            scenario={2: (IncidentType.COMMS_FAILURE, "0")},
        )
        targets = {"waypoint": Target(target_id="waypoint", position=(0, 8))}
        accepted = Objective(drone_id="0", objective_type=ObjectiveType.GOTO,
                             target_id="waypoint")
        plan = {
            "0": accepted,
            "1": Objective(drone_id="1", objective_type=ObjectiveType.HOLD),
        }
        event,  explored, step = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, {}, targets,
        )

        # Failure at step 2 is silent; the first notification is monitoring at 5.
        self.assertEqual(step, 5)
        self.assertEqual(event, ["Drone 0 lost communication at (0, 5)"])
        self.assertFalse(drones["0"].comms)
        self.assertTrue(drones["1"].comms)
        self.assertEqual(drones["0"].battery, 20.0)
        controller = simulator.controllers["0"]

        # Mission control's replacement cannot be delivered during the outage.
        plan["0"] = Objective(drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE)
        event,  explored, step = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, {}, targets, explored,
        )
        self.assertEqual(event, ["Mission plan exhausted"])
        self.assertEqual(step, 8)
        self.assertIs(simulator.controllers["0"], controller)
        self.assertEqual(controller.objective, accepted)
        self.assertEqual(drones["0"].position, (0, 8))
        self.assertEqual(drones["0"].battery, 17.0)
        self.assertEqual(drones["1"].position, (1, 0))
        self.assertEqual(drones["1"].battery, 25.0)

        # With no remaining accepted steps, it holds even when omitted from a plan.
        event,  _, final_step = simulator.execute_plan(
            {"1": plan["1"]}, {"battery_reserve_threshold": 0.0}, {}, targets, explored,
        )
        self.assertEqual(event, ["Mission plan exhausted"])
        self.assertEqual(final_step, step)
        self.assertEqual(drones["0"].position, (0, 8))
        self.assertEqual(drones["0"].battery, 17.0)


if __name__ == "__main__":
    unittest.main()
