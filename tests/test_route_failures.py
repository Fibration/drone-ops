import unittest

from src.drone import DroneController
from src.map import Terrain
from src.models import Objective, ObjectiveType, Sector, Target
from src.pathfinder import NoRouteFound, astar
from src.simulator import Simulator
from src.utility import initialize_drone_states


class RouteFailureTest(unittest.TestCase):
    def test_astar_distinguishes_no_route_from_arrival_and_invalid_input(self):
        terrain = [[Terrain.EMPTY, Terrain.OBSTACLE, Terrain.EMPTY]]
        self.assertEqual(astar(terrain, (0, 0), (0, 2)), NoRouteFound((0, 0), (0, 2)))
        self.assertEqual(astar(terrain, (0, 0), (0, 0)), [])
        with self.assertRaises(ValueError):
            astar(terrain, (0, 0), (0, 3))

    def test_unreachable_target_keeps_list_route_and_surfaces_event(self):
        terrain = [[Terrain.EMPTY, Terrain.NO_FLY_ZONE, Terrain.EMPTY]]
        targets = {"target": Target(target_id="target", position=(0, 2))}
        objective = Objective(drone_id="0", objective_type=ObjectiveType.GOTO,
                              target_id="target")
        controller = DroneController("0", terrain, {}, targets)
        controller.update_objective(objective)
        self.assertEqual(controller.route, [])
        self.assertEqual(controller.failure_event, "Drone 0 cannot reach target target at (0, 2)")
        self.assertIsNone(controller.next_action())
        drones = initialize_drone_states([(0, 0)], 25.0)
        simulator = Simulator(terrain, drones, [])
        event, _, _, step = simulator.execute_plan(
            {"0": objective}, {"battery_reserve_threshold": 0.0}, {}, targets
        )
        self.assertEqual(event, "Drone 0 cannot reach target target at (0, 2)")
        self.assertEqual(step, 0)
        self.assertEqual(drones["0"].position, (0, 0))
        controller.update_map([[Terrain.EMPTY] * 3])
        controller.update_objective(objective)
        self.assertIsNone(controller.failure_event)
        self.assertEqual(controller.route, [(0, 1), (0, 1)])
        self.assertEqual(controller.next_action(), (0, 1))

    def test_search_tries_other_cells_when_nearest_is_unreachable(self):
        empty, blocked = Terrain.EMPTY, Terrain.OBSTACLE
        terrain = [[empty, blocked, empty, blocked],
                   [empty, blocked, blocked, blocked],
                   [empty, empty, empty, empty]]
        sector = Sector(sector_id="sector", bounds=((0, 0), (2, 3)))
        controller = DroneController("0", terrain, {"sector": sector}, {})
        controller.update_explored_cells({(0, 0), (1, 0), (2, 0)})
        controller.update_objective(Objective(
            drone_id="0", objective_type=ObjectiveType.SEARCH_SECTOR, sector_id="sector"
        ))
        position = (0, 0)
        for dr, dc in controller.route:
            position = position[0] + dr, position[1] + dc
        self.assertEqual(position, (2, 1))

    def test_unreachable_search_is_not_reported_as_exhausted(self):
        terrain = [[Terrain.EMPTY, Terrain.OBSTACLE, Terrain.EMPTY]]
        sectors = {"sector": Sector(sector_id="sector", bounds=((0, 0), (0, 2)))}
        drones = initialize_drone_states([(0, 0)], 25.0)
        simulator = Simulator(terrain, drones, [])
        event, _, _, step = simulator.execute_plan(
            {"0": Objective(drone_id="0", objective_type=ObjectiveType.SEARCH_SECTOR,
                            sector_id="sector")},
            {"battery_reserve_threshold": 0.0}, sectors, {}, {(0, 0)},
        )
        self.assertEqual(event, "Drone 0 cannot search sector sector: remaining cells are unreachable")
        self.assertEqual(step, 0)

    def test_completed_and_hold_routes_are_empty_without_failure(self):
        terrain = [[Terrain.EMPTY]]
        targets = {"target": Target(target_id="target", position=(0, 0))}
        for objective in (
            Objective(drone_id="0", objective_type=ObjectiveType.HOLD),
            Objective(drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE),
            Objective(drone_id="0", objective_type=ObjectiveType.GOTO, target_id="target"),
        ):
            with self.subTest(objective=objective.objective_type):
                controller = DroneController("0", terrain, {}, targets)
                controller.update_objective(objective)
                self.assertEqual(controller.route, [])
                self.assertIsNone(controller.failure_event)
                self.assertIsNone(controller.next_action())


if __name__ == "__main__":
    unittest.main()
