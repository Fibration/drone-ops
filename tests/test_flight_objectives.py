import unittest

from src.drone import DroneController
from src.map import Terrain, extract_from_raw
from src.mission import RAW_MAP, mission
from src.models import MissionStatus, Objective, ObjectiveType
from src.pathfinder import astar
from src.simulator import Simulator
from src.utility import initialize_drone_states


class FlightObjectiveTest(unittest.TestCase):
    def test_return_route_avoids_obstacles_and_accepts_base_marker(self):
        terrain = [
            [Terrain.DRONE, Terrain.EMPTY, Terrain.EMPTY],
            [Terrain.EMPTY, Terrain.NO_FLY_ZONE, Terrain.EMPTY],
        ]
        self.assertIsNone(astar(terrain, (1, 2), (0, 0)))
        controller = DroneController(
            Objective(drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE), {}, {}
        )
        controller.calculate_route(terrain, (1, 2), set())
        self.assertIsNotNone(controller.route)
        position = (1, 2)
        assert controller.route
        for dr, dc in controller.route:
            position = (position[0] + dr, position[1] + dc)
            self.assertNotEqual(terrain[position[0]][position[1]], Terrain.NO_FLY_ZONE)
        self.assertEqual(position, (0, 0))

    def test_hold_stays_stationary_while_other_drone_returns(self):
        terrain = [
            [Terrain.DRONE, Terrain.EMPTY, Terrain.EMPTY],
            [Terrain.EMPTY, Terrain.EMPTY, Terrain.EMPTY],
        ]
        drones = initialize_drone_states([(1, 2), (0, 2)], 25.0)
        simulator = Simulator(terrain, drones, targets=[])
        plan = {
            "0": Objective(drone_id="0", objective_type=ObjectiveType.HOLD),
            "1": Objective(drone_id="1", objective_type=ObjectiveType.RETURN_TO_BASE),
        }
        event, _, explored, timestep = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 10.0}, {}, {}
        )
        self.assertEqual(event, "Drone 1 returned to base at (0, 0)")
        self.assertEqual(drones["0"].position, (1, 2))
        self.assertEqual(drones["0"].battery, 25.0)
        self.assertEqual(drones["1"].position, (0, 0))
        self.assertEqual(drones["1"].battery, 23.0)
        event, _, _, final_step = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 10.0}, {}, {}, explored
        )
        self.assertEqual(event, "Mission plan exhausted")
        self.assertEqual(final_step, timestep)

    def test_blocked_base_reports_unreachable(self):
        for blocked in (Terrain.OBSTACLE, Terrain.NO_FLY_ZONE):
            with self.subTest(blocked=blocked):
                drones = initialize_drone_states([(0, 1)], 25.0)
                simulator = Simulator([[blocked, Terrain.EMPTY]], drones, [])
                event, _, _, step = simulator.execute_plan(
                    {
                        "0": Objective(
                            drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE
                        )
                    },
                    {"battery_reserve_threshold": 10.0},
                    {},
                    {},
                )
                self.assertEqual(event, "Drone 0 cannot return to base at (0, 0)")
                self.assertEqual(step, 0)
                self.assertEqual(drones["0"].position, (0, 1))

    def test_mission_applies_hold_and_return_after_scenario(self):
        truth, terrain, positions, _ = extract_from_raw(RAW_MAP)
        drones = initialize_drone_states(sorted(positions), 25.0)
        simulator = Simulator(truth, drones, [], scenario={2: "Return fleet"})
        held_position = None

        def commander(state):
            nonlocal held_position
            self.assertLess(len(state.events), 20, "Mission did not terminate")
            event = state.events[-1].description
            if event == "Return fleet":
                held_position = state.drones["1"].position
                state.current_plan.plan = {
                    "0": Objective(
                        drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE
                    ),
                    "1": Objective(drone_id="1", objective_type=ObjectiveType.HOLD),
                }
            elif event == "Mission plan exhausted":
                state.mission_status = MissionStatus.SUCCESS
            return state

        result = mission(
            simulator, terrain, drones, commander, {"battery_reserve_threshold": 10.0}
        )
        self.assertEqual(result.mission_status, MissionStatus.SUCCESS)
        self.assertEqual(result.drones["0"].position, (0, 0))
        self.assertEqual(result.drones["0"].assignment, "return to base at (0, 0)")
        self.assertEqual(result.drones["1"].position, held_position)
        self.assertEqual(result.drones["1"].assignment, "hold position")
        self.assertTrue(
            any("returned to base" in event.description for event in result.events)
        )


if __name__ == "__main__":
    unittest.main()
