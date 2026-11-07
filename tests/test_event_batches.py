import unittest
from unittest.mock import patch

from src.map import Terrain
from src.mission_commander import mission_team, naive_mission_commander
from src.mission_updates import update_completed_sectors
from src.models import (
    CommanderDecision, IncidentType, MissionEvent, MissionPlan, MissionState,
    MissionStatus, Objective, ObjectiveType, Sector, Target,
)
from src.simulator import Simulator
from src.utility import initialize_drone_states


class EventBatchTest(unittest.TestCase):
    def test_all_unreachable_objectives_are_reported_before_any_movement(self):
        terrain = [[Terrain.EMPTY, Terrain.OBSTACLE, Terrain.EMPTY, Terrain.EMPTY]
                   for _ in range(2)]
        drones = initialize_drone_states([(0, 0), (1, 0)], 25.0)
        simulator = Simulator(terrain, drones, [])
        targets = {"waypoint": Target(target_id="waypoint", position=(0, 3))}
        plan = {key: Objective(drone_id=key, objective_type=ObjectiveType.GOTO,
                               target_id="waypoint") for key in drones}
        events, _, _, step = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, {}, targets,
        )
        self.assertEqual(events, ["Drone 0 cannot reach target waypoint at (0, 3)",
                                  "Drone 1 cannot reach target waypoint at (0, 3)"])
        self.assertEqual(step, 0)
        self.assertTrue(all(drone.battery == 25.0 for drone in drones.values()))

    def test_two_batteries_exhausted_in_same_step_are_both_reported_once(self):
        drones = initialize_drone_states([(0, 3), (1, 3)], 1.0)
        simulator = Simulator([[Terrain.EMPTY] * 4 for _ in range(2)], drones, [])
        plan = {key: Objective(drone_id=key, objective_type=ObjectiveType.RETURN_TO_BASE)
                for key in drones}
        events, _, explored, step = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, {}, {},
        )
        self.assertEqual(step, 1)
        self.assertEqual(events, ["Drone 0 ran out of battery at (0, 2)",
                                  "Drone 1 ran out of battery at (0, 3)"])
        events, _, _, step = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 0.0}, {}, {}, explored,
        )
        self.assertEqual(events, ["Mission plan exhausted"])
        self.assertEqual(step, 1)

    def test_monitoring_collects_low_battery_and_comms_for_all_drones(self):
        drones = initialize_drone_states([(0, 10), (1, 10)], 25.0)
        simulator = Simulator([[Terrain.EMPTY] * 12 for _ in range(2)], drones, [],
                              scenario={2: (IncidentType.COMMS_FAILURE, "0")})
        plan = {key: Objective(drone_id=key, objective_type=ObjectiveType.RETURN_TO_BASE)
                for key in drones}
        events, _, _, step = simulator.execute_plan(
            plan, {"battery_reserve_threshold": 22.0}, {}, {},
        )
        self.assertEqual(step, 5)
        self.assertEqual(events, ["Drone 0 battery low 20.0 at (0, 5)",
                                  "Drone 0 lost communication at (0, 5)",
                                  "Drone 1 battery low 20.0 at (0, 6)"])

    def test_multiple_targets_and_sector_completion_share_one_batch(self):
        drones = initialize_drone_states([(0, 0)], 25.0)
        simulator = Simulator([[Terrain.EMPTY] * 9 for _ in range(2)], drones,
                              targets=[(0, 4), (1, 3)])
        sectors = {"survey": Sector(sector_id="survey", bounds=((0, 4), (0, 4)))}
        targets = {"waypoint": Target(target_id="waypoint", position=(0, 8))}
        events, _, _, step = simulator.execute_plan(
            {"0": Objective(drone_id="0", objective_type=ObjectiveType.GOTO,
                            target_id="waypoint")},
            {"battery_reserve_threshold": 0.0}, sectors, targets,
        )
        self.assertEqual(step, 1)
        self.assertEqual(events, ["Drone 0 detected a target at (0, 4)",
                                  "Drone 0 detected a target at (1, 3)",
                                  "Sector survey fully explored"])

    def make_state(self, descriptions):
        drones = initialize_drone_states([(0, 0), (1, 0)], 25.0)
        events = [MissionEvent(description=text, drones=drones) for text in descriptions]
        objectives = [Objective(drone_id=key, objective_type=ObjectiveType.HOLD)
                      for key in drones]
        state = MissionState(
            drones=drones, objectives=[], no_fly_zones=[], discovered_targets={},
            events=events.copy(), latest_events=events, explored_cells=set(),
            current_plan=MissionPlan(plan={obj.drone_id: obj for obj in objectives}),
            proposed_action=None, next_agent=None, safety_status=None,
            human_approval=None, step=5, mission_status=MissionStatus.ONGOING,
        )
        return state, CommanderDecision(objectives=objectives, reason="Batch test")

    def test_all_zone_and_target_updates_precede_one_commander_decision(self):
        state, decision = self.make_state([
            "New no-fly zone added: rows 3-3, columns 0-1",
            "New no-fly zone added: rows 3-3, columns 1-2",
            "Drone 0 detected a target at (0, 4)",
            "Drone 1 detected a target at (0, 4)",
            "Drone 1 detected a target at (1, 3)",
            "Sector survey fully explored",
        ])

        def decide(observation):
            updated = observation.mission_state
            self.assertEqual({target.position for target in updated.discovered_targets.values()},
                             {(0, 4), (1, 3)})
            cells = [cell for region in updated.no_fly_zones for cell in region.coordinates]
            self.assertCountEqual(cells, [(3, 0), (3, 1), (3, 2)])
            self.assertEqual(len(updated.latest_events), 6)
            self.assertEqual(updated.completed_sectors, {"survey"})
            return decision

        with patch("src.mission_commander.OpenAICommanderPolicy.decide", side_effect=decide) as run:
            mission_team(state, [[Terrain.EMPTY] * 6 for _ in range(6)], {})
        run.assert_called_once()

    def test_completed_sectors_accumulate_and_ignore_repeated_events(self):
        state, _ = self.make_state([
            "Sector west fully explored", "Sector east fully explored",
            "Sector west fully explored", "Drone 0 battery low 5.0 at (0, 0)",
        ])
        state.completed_sectors.add("previous")
        self.assertEqual(update_completed_sectors(state), {"previous", "west", "east"})
        self.assertEqual(update_completed_sectors(state), {"previous", "west", "east"})
        state.latest_events = []
        self.assertEqual(update_completed_sectors(state), {"previous", "west", "east"})

    def test_only_entirely_redundant_detection_batches_skip_commander(self):
        for other_events in ([], ["Sector survey fully explored"]):
            with self.subTest(other_events=other_events):
                state, decision = self.make_state([
                    "Drone 0 detected a target at (0, 4)",
                    "Drone 1 detected a target at (0, 4)",
                    *other_events,
                ])
                state.discovered_targets["0"] = Target(target_id="0", position=(0, 4))
                with patch("src.mission_commander.MissionCommander.run", return_value=decision) as run:
                    mission_team(state, [[Terrain.EMPTY] * 6 for _ in range(6)], {})
                self.assertEqual(run.call_count, 1 if other_events else 0)

    def test_exhaustion_is_handled_locally_even_when_not_last_in_batch(self):
        state, _ = self.make_state(["Mission plan exhausted", "Sector survey fully explored"])
        with patch("src.mission_commander.MissionCommander.run") as run:
            result = mission_team(state, [[Terrain.EMPTY] * 6 for _ in range(6)], {})
        self.assertEqual(result.mission_status, MissionStatus.SUCCESS)
        run.assert_not_called()

    def test_naive_commander_applies_zone_and_detection_from_same_batch(self):
        state, _ = self.make_state([
            "Drone 0 detected a target at (0, 4)",
            "New no-fly zone added: rows 3-3, columns 0-1",
            "Sector survey fully explored",
        ])
        result = naive_mission_commander(state)
        assert result.current_plan is not None
        self.assertEqual(result.current_plan.plan["0"].objective_type, ObjectiveType.GOTO)
        self.assertEqual(len(result.discovered_targets), 1)
        self.assertEqual(result.no_fly_zones[0].coordinates, [(3, 0), (3, 1)])
        self.assertEqual(result.completed_sectors, {"survey"})


if __name__ == "__main__":
    unittest.main()
