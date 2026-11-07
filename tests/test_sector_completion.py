import unittest

from src.map import Terrain
from src.models import IncidentType, Objective, ObjectiveType, Sector, Target
from src.simulator import Simulator
from src.utility import initialize_drone_states


class SectorCompletionTest(unittest.TestCase):
    def test_initial_completions_are_each_reported_once_without_movement(self):
        drones = initialize_drone_states([(0, 0)], 25.0)
        simulator = Simulator([[Terrain.EMPTY, Terrain.OBSTACLE, Terrain.EMPTY]], drones, [])
        sectors = {
            "west": Sector(sector_id="west", bounds=((0, 0), (0, 1))),
            "east": Sector(sector_id="east", bounds=((0, 2), (0, 2))),
        }
        explored = set()
        events = []
        for _ in range(4):
            event, _, explored, step = simulator.execute_plan(
                {}, {"battery_reserve_threshold": 0.0}, sectors, {}, explored,
            )
            events.append(event)
            self.assertEqual(step, 0)
        self.assertEqual(events, [["Sector west fully explored", "Sector east fully explored"],
                                  ["Mission plan exhausted"], ["Mission plan exhausted"],
                                  ["Mission plan exhausted"]])
        self.assertEqual(drones["0"].battery, 25.0)

    def test_other_drone_observations_complete_sector(self):
        drones = initialize_drone_states([(0, 0), (1, 0)], 25.0)
        simulator = Simulator([[Terrain.EMPTY] * 9 for _ in range(2)], drones, [])
        sectors = {"observed": Sector(sector_id="observed", bounds=((0, 4), (0, 4)))}
        targets = {"waypoint": Target(target_id="waypoint", position=(0, 8))}
        plan = {
            "0": Objective(drone_id="0", objective_type=ObjectiveType.GOTO, target_id="waypoint"),
            "1": Objective(drone_id="1", objective_type=ObjectiveType.HOLD),
        }
        event, _, explored, step = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, sectors, targets,
        )
        self.assertEqual(event, ["Sector observed fully explored"])
        self.assertEqual(step, 1)
        self.assertIn((0, 4), explored)
        self.assertEqual(drones["1"].position, (1, 0))
        plan["0"] = Objective(drone_id="0", objective_type=ObjectiveType.HOLD)
        event, _, _, final_step = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, sectors, targets, explored,
        )
        self.assertEqual(event, ["Mission plan exhausted"])
        self.assertEqual(final_step, step)

    def test_new_no_fly_zone_completes_sector_without_losing_either_event(self):
        drones = initialize_drone_states([(0, 0)], 25.0)
        zone_event = "New no-fly zone added: rows 0-0, columns 4-5"
        simulator = Simulator([[Terrain.EMPTY] * 9], drones, [],
                              scenario={1: (IncidentType.NEW_NO_FLY_ZONE, zone_event)})
        sectors = {"zone": Sector(sector_id="zone", bounds=((0, 4), (0, 5)))}
        targets = {"waypoint": Target(target_id="waypoint", position=(0, 8))}
        plan = {"0": Objective(drone_id="0", objective_type=ObjectiveType.GOTO,
                               target_id="waypoint")}
        first, _, explored, step = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, sectors, targets,
        )
        self.assertEqual(first, [zone_event, "Sector zone fully explored"])
        second, _, _, second_step = simulator.execute_plan(
            {}, {"battery_reserve_threshold": 0.0}, sectors, targets, explored,
        )
        self.assertEqual(second, ["Mission plan exhausted"])
        self.assertEqual(second_step, step)
        third, _, _, _ = simulator.execute_plan(
            {}, {"battery_reserve_threshold": 0.0}, sectors, targets, explored,
        )
        self.assertEqual(third, ["Mission plan exhausted"])

    def test_disconnected_controller_keeps_route_after_sector_completion(self):
        drones = initialize_drone_states([(0, 0)], 25.0)
        simulator = Simulator([[Terrain.EMPTY] * 6], drones, [],
                              scenario={1: (IncidentType.COMMS_FAILURE, "0")})
        sectors = {"search": Sector(sector_id="search", bounds=((0, 4), (0, 4)))}
        accepted = Objective(drone_id="0", objective_type=ObjectiveType.SEARCH_SECTOR,
                             sector_id="search")
        event, _, explored, step = simulator.execute_plan(
            {"0": accepted}, {"battery_reserve_threshold": 0.0}, sectors, {},
        )
        self.assertEqual(event, ["Sector search fully explored"])
        self.assertEqual(step, 1)
        controller = simulator.controllers["0"]
        event, _, _, step = simulator.execute_plan(
            {"0": Objective(drone_id="0", objective_type=ObjectiveType.HOLD)},
            {"battery_reserve_threshold": 0.0}, sectors, {}, explored,
        )
        self.assertEqual(event, ["Mission plan exhausted"])
        self.assertEqual(step, 4)
        self.assertEqual(controller.objective, accepted)
        self.assertEqual(drones["0"].position, (0, 4))


if __name__ == "__main__":
    unittest.main()
