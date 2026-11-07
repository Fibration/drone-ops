import unittest

from src.drone import DroneController
from src.map import Terrain, extract_from_raw
from src.mission import RAW_MAP, mission
from src.models import MissionStatus, Objective, ObjectiveType, Sector, Target
from src.pathfinder import NoRouteFound, astar
from src.simulator import Simulator
from src.utility import initialize_drone_states


class FlightObjectiveTest(unittest.TestCase):
    def test_simulator_initializes_controllers_on_hold_without_drone_routes(self):
        drones = initialize_drone_states([(0, 1)], 25.0)
        drones["0"].comms = False
        simulator = Simulator([[Terrain.EMPTY] * 2], drones, [])
        controller = simulator.controllers["0"]
        self.assertEqual(controller.objective.objective_type, ObjectiveType.HOLD)
        self.assertEqual(controller.position, (0, 1))
        self.assertEqual(controller.route, [])
        self.assertNotIn("route", drones["0"].model_dump())
        simulator.execute_plan(
            {"0": Objective(drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE)},
            {"battery_reserve_threshold": 0.0}, {}, {},
        )
        self.assertIs(simulator.controllers["0"], controller)
        self.assertEqual(controller.objective.objective_type, ObjectiveType.HOLD)
        self.assertEqual(drones["0"].position, (0, 1))

    def test_empty_battery_drone_never_moves(self):
        drones = initialize_drone_states([(0, 2)], 0.0)
        simulator = Simulator([[Terrain.EMPTY] * 3], drones, [])
        event, _, _, step = simulator.execute_plan(
            {"0": Objective(drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE)},
            {"battery_reserve_threshold": 0.0}, {}, {},
        )
        self.assertEqual(event, "Mission plan exhausted")
        self.assertEqual(step, 0)
        self.assertEqual(drones["0"].position, (0, 2))
        self.assertEqual(drones["0"].battery, 0.0)
        self.assertEqual(simulator.controllers["0"].route[simulator.controllers["0"].route_index:], [])

    def test_drone_stops_when_battery_runs_out(self):
        drones = initialize_drone_states([(0, 3)], 1.0)
        simulator = Simulator([[Terrain.EMPTY] * 4], drones, [])
        plan = {"0": Objective(drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE)}
        first_event, _, _, _ = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, {}, {}
        )
        self.assertEqual(first_event, "Drone 0 ran out of battery at (0, 2)")
        self.assertEqual(drones["0"].position, (0, 2))
        for _ in range(3):
            event, _, _, step = simulator.execute_plan(
                plan, {"battery_reserve_threshold": 0.0}, {}, {},
            )
            self.assertEqual(event, "Mission plan exhausted")
            self.assertEqual(step, 1)
            self.assertEqual(drones["0"].battery, 0.0)
            self.assertEqual(drones["0"].position, (0, 2))

    def test_battery_event_does_not_repeat_while_other_drone_continues(self):
        drones = initialize_drone_states([(0, 3), (1, 4)], 25.0)
        drones["0"].battery = 1.0
        simulator = Simulator([[Terrain.EMPTY] * 5 for _ in range(2)], drones, [])
        plan = {
            drone_id: Objective(drone_id=drone_id, objective_type=ObjectiveType.RETURN_TO_BASE)
            for drone_id in drones
        }
        first_event, _, _, _ = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, {}, {}
        )
        self.assertEqual(first_event, "Drone 0 ran out of battery at (0, 2)")
        next_event, _, _, _ = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, {}, {}
        )
        self.assertEqual(next_event, "Drone 1 returned to base at (0, 0)")
        self.assertEqual(drones["0"].position, (0, 2))
        self.assertEqual(drones["0"].battery, 0.0)
        self.assertEqual(drones["1"].position, (0, 0))

    def test_disconnected_drone_finishes_delivered_route_then_holds(self):
        drones = initialize_drone_states([(0, 0)], 25.0)
        simulator = Simulator([[Terrain.EMPTY] * 4], drones, [], scenario={1: "Paused"})
        target = Target(target_id="0", position=(0, 2))
        simulator.execute_plan(
            {"0": Objective(drone_id="0", objective_type=ObjectiveType.GOTO, target_id="0")},
            {"battery_reserve_threshold": 0.0}, {}, {"0": target},
        )
        drones["0"].comms = False
        # An undeliverable return command must not replace the saved route.
        plan = {"0": Objective(drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE)}
        event, _, _, step = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, {}, {},
        )
        self.assertEqual(event, "Mission plan exhausted")
        self.assertEqual(drones["0"].position, (0, 2))
        self.assertEqual(drones["0"].battery, 23.0)
        self.assertEqual(simulator.controllers["0"].route[simulator.controllers["0"].route_index:], [])
        sectors = {"0": Sector(sector_id="0", bounds=((0, 0), (0, 3)))}
        plan["0"] = Objective(drone_id="0", objective_type=ObjectiveType.SEARCH_SECTOR,
                              sector_id="0")
        event, _, _, final_step = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, sectors, {},
        )
        self.assertEqual(event, "Mission plan exhausted")
        self.assertEqual(final_step, step)
        self.assertEqual(drones["0"].position, (0, 2))

    def test_disconnected_route_survives_pause_and_omitted_plan(self):
        drones = initialize_drone_states([(0, 3)], 25.0)
        simulator = Simulator([[Terrain.EMPTY] * 4], drones, [], scenario={2: "Paused"})
        plan = {"0": Objective(drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE)}
        event, _, _, _ = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, {}, {},
        )
        self.assertEqual(event, "Paused")
        controller = simulator.controllers["0"]
        self.assertEqual(controller.route[controller.route_index:], [(0, -1), (0, -1)])
        drones["0"].comms = False
        event, _, _, _ = simulator.execute_plan(
            {}, {"battery_reserve_threshold": 0.0}, {}, {},
        )
        self.assertEqual(event, "Drone 0 returned to base at (0, 0)")
        self.assertIs(simulator.controllers["0"], controller)
        self.assertEqual(drones["0"].position, (0, 0))
        self.assertEqual(simulator.controllers["0"].route[simulator.controllers["0"].route_index:], [])

    def test_return_route_avoids_obstacles_and_accepts_base_marker(self):
        terrain = [
            [Terrain.DRONE, Terrain.EMPTY, Terrain.EMPTY],
            [Terrain.EMPTY, Terrain.NO_FLY_ZONE, Terrain.EMPTY],
        ]
        self.assertEqual(astar(terrain, (1, 2), (0, 0)), NoRouteFound((1, 2), (0, 0)))
        controller = DroneController("0", terrain, {}, {})
        controller.update_position((1, 2))
        controller.update_objective(
            Objective(drone_id="0", objective_type=ObjectiveType.RETURN_TO_BASE)
        )
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
