import unittest

from src.drone import DroneController
from src.map import Terrain
from src.models import Objective, ObjectiveType, Sector


class ControllerStepTest(unittest.TestCase):
    def returning_controller(self):
        controller = DroneController("0", (0, 3), [[Terrain.EMPTY] * 4], {}, {})
        controller.update_objective(Objective(
            drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE
        ))
        return controller

    def test_repeated_planning_does_not_consume_a_step(self):
        controller = self.returning_controller()
        self.assertEqual(controller.plan_step(), (0, -1))
        self.assertEqual(controller.plan_step(), (0, -1))
        self.assertEqual(controller.position, (0, 3))
        self.assertEqual(controller.route_index, 0)
        controller.apply_step((0, 2))
        self.assertEqual(controller.position, (0, 2))
        self.assertEqual(controller.route_index, 1)
        self.assertEqual(controller.plan_step(), (0, -1))

    def test_applying_an_unplanned_or_wrong_move_does_not_advance(self):
        controller = self.returning_controller()
        with self.assertRaisesRegex(ValueError, "No step"):
            controller.apply_step((0, 2))
        controller.plan_step()
        with self.assertRaisesRegex(ValueError, "does not match"):
            controller.apply_step((0, 1))
        self.assertEqual(controller.position, (0, 3))
        self.assertEqual(controller.route_index, 0)
        controller.apply_step((0, 2))
        with self.assertRaisesRegex(ValueError, "No step"):
            controller.apply_step((0, 1))

    def test_replacing_objective_discards_unapplied_step(self):
        controller = self.returning_controller()
        controller.plan_step()
        controller.update_objective(Objective(drone_id="0", objective_type=ObjectiveType.HOLD))
        self.assertIsNone(controller.plan_step())
        with self.assertRaisesRegex(ValueError, "No step"):
            controller.apply_step((0, 2))

    def test_lost_comms_allows_accepted_route_but_no_further_search(self):
        sectors = {"search": Sector(sector_id="search", bounds=((0, 4), (0, 5)))}
        controller = DroneController("0", (0, 0), [[Terrain.EMPTY] * 6], sectors, {})
        controller.update_objective(Objective(
            drone_id="0", objective_type=ObjectiveType.SEARCH_SECTOR, sector_id="search"
        ))
        controller.update_comms(False)
        for col in range(1, 5):
            self.assertEqual(controller.plan_step(), (0, 1))
            controller.apply_step((0, col))
        self.assertIsNone(controller.plan_step())
        self.assertEqual(controller.position, (0, 4))


if __name__ == "__main__":
    unittest.main()
