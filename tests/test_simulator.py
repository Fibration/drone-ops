import random
import unittest
from unittest.mock import patch

from src.drone import DroneController
from src.map import Terrain
from src.models import DroneState, IncidentType, Objective, ObjectiveType, MonitoringPlan, WatchPredicate, WatchType, FreeWatchType
from src.scenario import Difficulty, Scenario, Sector, Weather, World, generate_world
from src.simulator import Simulator


def drone(drone_id='0', position=(0, 0), battery=20):
    return DroneState(id=drone_id, position=position, battery=battery,
                      status='idle', comms=True, assignment=None)


def world(weather=Weather.CLOUDY, visibility=2, movement_cost=1, drones=None):
    sector = Sector('0', ((0, 0), (4, 6)), visibility, movement_cost,
                    0, weather, population_rate=0)
    return World([sector], 0, [], [], drones or {'0': drone()})


def order(kind, drone_id='0', **kwargs):
    return Objective(drone_id=drone_id, objective_type=kind, **kwargs)


def add_target(w, position):
    w.targets['target_0'] = position
    w.active_targets['0'].append('target_0')


class WorldSimulationTest(unittest.TestCase):
    def test_wind_cost_matches_route_estimate_and_simulated_move(self):
        from src.parameters import wind_penalty
        from src.watch import estimate_route_cost

        for wind, expected in (
            ((0, 1), 1 - wind_penalty),
            ((0, -1), 1 + wind_penalty),
            ((1, 0), 1.0),
        ):
            with self.subTest(wind=wind):
                w = world()
                w.sectors[0].wind_direction = wind
                self.assertAlmostEqual(estimate_route_cost(w, '0', (0, 1)), expected)
                sim = self.simulator(w)
                with patch('src.simulator.random.random', return_value=0.99):
                    sim.step({'0': order(ObjectiveType.SEARCH_SECTOR, sector_id='0')})
                self.assertEqual(w.drones['0'].position, (0, 1))
                self.assertAlmostEqual(w.drones['0'].battery, 20 - expected)

    def test_cached_neighbours_handle_staggered_three_two_three_layout(self):
        bounds = [((0, 0), (1, 1)), ((0, 2), (1, 3)), ((0, 4), (1, 5)),
                  ((2, 0), (3, 2)), ((2, 3), (3, 5)),
                  ((4, 0), (5, 1)), ((4, 2), (5, 3)), ((4, 4), (5, 5))]
        sectors = [Sector(str(i), bounds, 2, 1, 0, Weather.CLOUDY, 0)
                   for i, bounds in enumerate(bounds)]
        w = World(sectors, 0, [], [], {})
        self.assertEqual({key: set(ids) for key, ids in w.sector_neighbours.items()}, {
            '0': {'1', '3'}, '1': {'0', '2', '3', '4'}, '2': {'1', '4'},
            '3': {'0', '1', '4', '5', '6'}, '4': {'1', '2', '3', '6', '7'},
            '5': {'3', '6'}, '6': {'3', '4', '5', '7'}, '7': {'4', '6'},
        })
        with patch.object(w, '_build_sector_neighbours', side_effect=AssertionError('rebuilt')):
            w.update_sector_weather(Weather.CLOUDY)

    def test_weather_neighbour_averages_use_snapshot_and_ignore_sector_order(self):
        for reverse in (False, True):
            sectors = [Sector(str(i), ((0, i), (0, i)), 2, 1, 0, weather,
                              population_rate=0)
                       for i, weather in enumerate((Weather.CLOUDY, Weather.CLOUDY,
                                                    Weather.BLIZZARD))]
            if reverse:
                sectors.reverse()
            w = World(sectors, 0, [], [], {})
            rolls = {'0': 0.025, '1': 0.045, '2': 0.9}
            with patch('src.scenario.random.random',
                       side_effect=[rolls[s.sector_id] for s in sectors]):
                w.update_sector_weather(Weather.CLEAR)
            self.assertEqual({s.sector_id: s.weather for s in sectors},
                             {'0': Weather.CLEAR, '1': Weather.FOGGY,
                              '2': Weather.BLIZZARD})

    def test_weather_scenario_and_neighbours_have_equal_weight(self):
        for scenario_weather, neighbour_weather, current in (
            (Weather.CLEAR, Weather.FOGGY, Weather.CLOUDY),
            (Weather.BLIZZARD, Weather.CLEAR, Weather.FOGGY),
        ):
            sectors = [Sector('0', ((0, 0), (0, 0)), 2, 1, 0, current, 0),
                       Sector('1', ((0, 1), (0, 1)), 2, 1, 0, neighbour_weather, 0),
                       # A diagonal sector must not influence sector 0.
                       Sector('2', ((1, 1), (1, 1)), 2, 1, 0, Weather.BLIZZARD, 0)]
            w = World(sectors, 0, [], [], {})
            with patch('src.scenario.random.random', return_value=0.045):
                w.update_sector_weather(scenario_weather)
            self.assertEqual(sectors[0].weather, current)

    def test_isolated_sector_weather_falls_back_to_scenario(self):
        w = world(weather=Weather.CLOUDY)
        with patch('src.scenario.random.random', return_value=0.045):
            w.update_sector_weather(Weather.BLIZZARD)
        self.assertEqual(w.sectors[0].weather, Weather.FOGGY)

    def test_weather_comms_failure_probabilities_for_moving_drones(self):
        for weather, roll, expected in (
            (Weather.STORMY, 0.009, False),
            (Weather.STORMY, 0.01, True),
            (Weather.BLIZZARD, 0.019, False),
            (Weather.BLIZZARD, 0.02, True),
            (Weather.CLEAR, 0.0, True),
            (Weather.CLOUDY, 0.0, True),
            (Weather.FOGGY, 0.0, True),
        ):
            with self.subTest(weather=weather, roll=roll):
                w = world(weather=weather, drones={'0': drone(position=(0, 4))})
                sim = self.simulator(w)
                with patch('src.simulator.random.random', return_value=roll):
                    sim.step({'0': order(ObjectiveType.RETURN_TO_BASE)})
                self.assertEqual(w.drones['0'].comms, expected)
                self.assertEqual(w.drones['0'].position, (0, 3))
                self.assertEqual(sim._watch_history[-1].comms['0'], expected)

        w = world(weather=Weather.BLIZZARD)
        sim = self.simulator(w)
        with patch('src.simulator.random.random', return_value=0.0) as roll:
            sim.step({})
        self.assertTrue(w.drones['0'].comms)
        roll.assert_not_called()

    def test_holding_drones_face_weather_risk_each_turn(self):
        for weather, probability in ((Weather.STORMY, 0.01), (Weather.BLIZZARD, 0.02)):
            with self.subTest(weather=weather):
                w = world(weather=weather, drones={'0': drone(position=(0, 4))})
                sim = self.simulator(w)
                with patch('src.simulator.random.random',
                           side_effect=[probability, 0.0]) as roll:
                    sim.step({})
                    self.assertTrue(w.drones['0'].comms)
                    sim.step({})
                self.assertFalse(w.drones['0'].comms)
                self.assertEqual(w.drones['0'].position, (0, 4))
                self.assertEqual(roll.call_count, 2)
                with patch('src.simulator.random.random') as roll:
                    sim.step({})
                roll.assert_not_called()

        w = world(weather=Weather.BLIZZARD,
                  drones={'0': drone(position=(0, 4), battery=0)})
        sim = self.simulator(w)
        with patch('src.simulator.random.random') as roll:
            sim.step({})
        roll.assert_not_called()

    def test_base_recharges_empty_batteries_and_caps_at_100(self):
        w = world(drones={
            '0': drone(battery=0), '1': drone('1', battery=98),
            '2': drone('2', position=(4, 6), battery=20),
        })
        w.base_position = (1, 1)
        w.drones['0'].position = w.drones['1'].position = w.base_position
        sim = self.simulator(w, recharge_rate=5.0)
        sim.step({})
        self.assertEqual([d.battery for d in w.drones.values()], [5.0, 100.0, 20])
        sim.step({})
        self.assertEqual([d.battery for d in w.drones.values()], [10.0, 100.0, 20])
        self.assertEqual(sim._watch_history[-1].battery['0'], 10.0)

    def test_arriving_at_base_recharges_but_departing_does_not(self):
        w = world(drones={'0': drone(position=(0, 1))})
        sim = self.simulator(w, recharge_rate=5.0)
        sim.step({'0': order(ObjectiveType.RETURN_TO_BASE)})
        self.assertEqual(w.drones['0'].battery, 24.0)
        sim.step({'0': order(ObjectiveType.SEARCH_SECTOR, sector_id='0')})
        self.assertNotEqual(w.drones['0'].position, w.base_position)
        self.assertEqual(w.drones['0'].battery, 23.0)

    def test_target_archive_survives_rescue_and_expiry(self):
        w = world()
        w.target_count = 3
        w._place_targets()
        original = w.targets.copy()
        archive = w.targets
        rescued = next(iter(original))
        w.update_target_success(rescued, 1)
        with patch.object(w.sectors[0], 'update_targets', return_value=[]):
            w.update_active_targets()
        self.assertIs(w.targets, archive)
        self.assertEqual(w.targets, original)
        self.assertEqual(w.successful_targets, [rescued])
        self.assertEqual(set(w.failed_targets), set(original) - {rescued})
        self.assertFalse(list(w.iter_active_targets()))
        for position in original.values():
            self.assertEqual(w.check_for_target(position), Terrain.EMPTY)

    def simulator(self, w, recharge_rate=0.0):
        scenario = Scenario(
            (max(s.bounds[1][0] for s in w.sectors) + 1,
             max(s.bounds[1][1] for s in w.sectors) + 1),
            len(w.sectors), w.target_count, Difficulty.MEDIUM, len(w.drones),
            w.base_position, {}, w.sectors[0].weather, recharge_rate,
        )
        weather = patch.object(w, 'update_sector_weather')
        weather.start()
        self.addCleanup(weather.stop)
        return Simulator(w, scenario)

    def test_fractional_weather_visibility_and_blizzard_current_cell(self):
        for visibility, radii in ((2, (2.5, 2, 1.5, 1, 0)),
                                  (3, (3.5, 3, 2.5, 2, 0))):
            for weather, radius in zip(Weather, radii):
                with self.subTest(visibility=visibility, weather=weather):
                    w = world(weather=weather, visibility=visibility)
                    for position in ((2, 3), (0, 0)):
                        sector = w.sectors[0]
                        expected = {(x, y) for x in range(5) for y in range(7)
                                    if (x-position[0])**2 + (y-position[1])**2 <= radius**2}
                        self.assertEqual(sector.visible_cells(position), expected)

    def test_generated_world_runs_with_shared_nondefault_base(self):
        random.seed(17)
        scenario = Scenario((25, 20), 4, 12, Difficulty.MEDIUM, 3,
                            (12, 10), {}, Weather.CLOUDY, recharge_rate=5.0)
        w = generate_world(scenario)
        sim = Simulator(w, scenario)
        sim.step({drone_id: order(ObjectiveType.SEARCH_SECTOR, drone_id, sector_id=str(i))
                  for i, drone_id in enumerate(w.drones)})
        self.assertIs(sim.drones, w.drones)
        self.assertEqual(w.base_position, (12, 10))
        self.assertEqual(sim.timestep, 1)
        self.assertTrue(sim.explored_cells)
        self.assertTrue(all(w.sector_at(d.position) for d in w.drones.values()))
        self.assertTrue(all(d.battery < 80 for d in w.drones.values()))

    def test_destination_sector_weather_controls_battery_and_sensing(self):
        start = Sector('start', ((0, 0), (0, 0)), 2, 0.5, 0,
                       Weather.CLEAR, population_rate=0)
        destination = Sector('end', ((0, 1), (0, 4)), 2, 1.5, 0,
                             Weather.BLIZZARD, population_rate=0)
        w = World([start, destination], 0, [], [], {'0': drone()})
        sim = self.simulator(w)
        with patch('src.simulator.random.random', return_value=0.99):
            sim.step({'0': order(ObjectiveType.SEARCH_SECTOR, sector_id='end')})
        self.assertEqual(w.drones['0'].position, (0, 1))
        self.assertEqual(w.drones['0'].battery, 17.9)
        self.assertEqual(sim.explored_cells, {(0, 0), (0, 1)})

    def test_rescue_on_last_battery_move_is_recorded_before_decay(self):
        w = world(drones={'0': drone(battery=1)})
        add_target(w, (0, 1))
        sim = self.simulator(w)
        with patch.object(w, 'update_active_targets', wraps=w.update_active_targets) as decay:
            events = sim.step({'0': order(ObjectiveType.GOTO, target_id='target_0')})
        self.assertEqual(w.successful_targets, ['target_0'])
        self.assertEqual(w.success_times, {'target_0': 1})
        self.assertEqual(w.targets, {'target_0': (0, 1)})
        self.assertFalse(w.active_targets['0'])
        with self.assertRaises(KeyError):
            w.update_target_success('target_0', 2)
        self.assertFalse(sim.discovered_targets)
        self.assertEqual(w.drones['0'].battery, 0)
        self.assertIn((0, 3), sim.explored_cells)
        self.assertFalse(any(event.type is FreeWatchType.MISSION_COMPLETE for event in events))
        self.assertTrue(any(event.type is FreeWatchType.DRONE_HARD_FAILURE for event in events))
        decay.assert_called_once()
        self.assertFalse(any(event.type is FreeWatchType.DRONE_HARD_FAILURE for event in sim.step()))

    def test_expiration_removes_discovered_target_and_updates_world(self):
        w = world()
        add_target(w, (0, 2))
        sim = self.simulator(w)
        with patch.object(w.sectors[0], 'update_targets', return_value=[]):
            events = sim.step({})
        self.assertEqual(w.failed_targets, ['target_0'])
        self.assertEqual(w.targets, {'target_0': (0, 2)})
        self.assertFalse(w.active_targets['0'])
        with self.assertRaises(KeyError):
            w.update_target_success('target_0', 2)
        self.assertFalse(sim.discovered_targets)
        self.assertFalse(any(event.type is FreeWatchType.MISSION_COMPLETE for event in events))
        self.assertFalse(sim.step({}))

    def test_incident_changes_world_before_route_execution(self):
        w = world(drones={'0': drone(position=(0, 3))})
        scenario = Scenario((5, 7), 1, 0, Difficulty.MEDIUM, 1, (0, 0),
                            {1: (IncidentType.NEW_NO_FLY_ZONE,
                                 'New no-fly zone added: rows 0-0, columns 2-2')},
                            Weather.CLOUDY, recharge_rate=5.0)
        sim = Simulator(w, scenario)
        with patch.object(w, 'update_sector_weather') as weather:
            sim.step({'0': order(ObjectiveType.RETURN_TO_BASE)})
        self.assertIn((0, 2), w.no_fly)
        self.assertNotEqual(w.drones['0'].position, (0, 2))
        weather.assert_called_once_with(Weather.CLOUDY)

    def test_comms_failure_preserves_route_but_stops_reporting(self):
        w = world(drones={'0': drone(position=(0, 4))})
        sim = self.simulator(w)
        sim.step({'0': order(ObjectiveType.RETURN_TO_BASE)})
        before = sim.explored_cells.copy()
        w.drones['0'].comms = False
        sim.step({'0': order(ObjectiveType.HOLD)})
        self.assertEqual(w.drones['0'].position, (0, 2))
        self.assertEqual(sim.explored_cells, before)
        self.assertEqual(sim._watch_history[0].battery['0'], 20)

    def test_disconnected_route_cannot_enter_new_restriction(self):
        w = world(drones={'0': drone(position=(0, 4))})
        sim = self.simulator(w)
        sim.step({'0': order(ObjectiveType.RETURN_TO_BASE)})
        w.drones['0'].comms = False
        w.no_fly.add((0, 2))
        events = sim.step()
        self.assertEqual(w.drones['0'].position, (0, 3))
        self.assertFalse(events)

    def test_shared_base_allows_returns_but_other_cells_prevent_collisions(self):
        w = world(drones={'0': drone(position=(0, 1)), '1': drone('1', (1, 0))})
        sim = self.simulator(w)
        sim.step({key: order(ObjectiveType.RETURN_TO_BASE, key) for key in w.drones})
        self.assertTrue(all(d.position == w.base_position for d in w.drones.values()))
        w = world(drones={'0': drone(), '1': drone('1')}, visibility=0,
                  weather=Weather.BLIZZARD)
        sim = self.simulator(w)
        sim.step({key: order(ObjectiveType.SEARCH_SECTOR, key, sector_id='0') for key in w.drones})
        self.assertNotEqual(w.drones['0'].position, w.drones['1'].position)

    def test_execute_plan_stops_and_reports_unreachable_target(self):
        w = world()
        add_target(w, (0, 2))
        w.obstacles.update({(0, 1), (1, 2), (0, 3)})
        sim = self.simulator(w)
        sim.set_monitoring_plan(MonitoringPlan(
            watches=[WatchPredicate(type=WatchType.ROUTE_UNREACHABLE, drone_id='0'),
                     WatchPredicate(type=WatchType.CHARGE_ABOVE, drone_id='0', threshold=100),
                     WatchPredicate(type=WatchType.WEATHER_AT_LEAST, sector_id='0', threshold='blizzard')],
            reason='Watch the accepted route',
        ))
        events, explored, timestep = sim.execute_plan(
            {'0': order(ObjectiveType.GOTO, target_id='target_0')})
        self.assertTrue(any(event.type is WatchType.ROUTE_UNREACHABLE for event in events))
        self.assertEqual(timestep, 1)
        self.assertEqual(w.drones['0'].position, (0, 0))
        explored.clear()
        self.assertTrue(sim.explored_cells)

    def test_sector_completion_is_reported_once_and_hold_still_advances_decay(self):
        w = world(visibility=20)
        sim = self.simulator(w)
        with patch.object(w, 'update_active_targets', wraps=w.update_active_targets) as decay:
            events = sim.step({})
            later = sim.step({})
        self.assertIn('0', sim.completed_sectors)
        self.assertFalse(later)
        self.assertEqual(decay.call_count, 2)
        self.assertEqual(w.drones['0'].battery, 20)

    def test_planning_is_repeatable_and_commits_world_state(self):
        w = world(drones={'0': drone(position=(0, 3))})
        controller = DroneController('0', w)
        controller.update_objective(order(ObjectiveType.RETURN_TO_BASE))
        self.assertEqual(controller.plan_step(), (0, -1))
        self.assertEqual(controller.plan_step(), (0, -1))
        self.assertEqual(w.drones['0'].position, (0, 3))
        with self.assertRaises(ValueError):
            controller.apply_step((0, 1))
        controller.apply_step((0, 2))
        self.assertEqual(w.drones['0'].position, (0, 2))
        controller.update_objective(order(ObjectiveType.HOLD))
        self.assertIsNone(controller.plan_step())

    def test_controller_does_not_route_to_undiscovered_world_targets(self):
        w = world()
        add_target(w, (4, 6))
        controller = DroneController('0', w)
        controller.update_objective(order(ObjectiveType.GOTO, target_id='target_0'))
        self.assertIn('unavailable', controller.failure_event)
        self.assertIsNone(controller.plan_step())
        with self.assertRaises(AssertionError):
            controller.update_objective(order(ObjectiveType.GOTO))


if __name__ == '__main__':
    unittest.main()
