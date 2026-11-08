from dataclasses import replace
from math import log
import unittest
from unittest.mock import Mock, patch

from src.models import DroneState, Objective, ObjectiveType, MonitoringPlan, WatchPredicate, WatchType, FreeWatchType, WatchAlert, MissionStatus, IncidentType
from src.scenario import Difficulty, Scenario, Sector, Weather, World
from src.simulator import Simulator
from src.watch import FREE_WATCH_HOOKS, WATCH_HOOKS, WatchSample, WatchContext, check_free_watch, check_watch, forecast_weather_risk, target_survival_probability, reachable_cells


def make_world():
    drone = DroneState(id='d', position=(0, 2), battery=10,
                       status='idle', comms=False, assignment=None)
    sector = Sector('s', ((0, 0), (4, 4)), 2, 1, 0, Weather.STORMY, 0)
    world = World([sector], 0, [], [], {'d': drone})
    world.targets['t'] = (2, 2)
    world.active_targets['s'] = ['t']
    return world


class WatchTest(unittest.TestCase):
    def test_free_hooks_cover_all_types_and_entity_conditions(self):
        self.assertEqual(set(FREE_WATCH_HOOKS), set(FreeWatchType))
        context = replace(self.context, new_target_ids={'t'})
        self.assertTrue(check_free_watch(FreeWatchType.NEW_TARGET_DETECTED, context, target_id='t'))
        self.assertFalse(check_free_watch(FreeWatchType.NEW_TARGET_DETECTED, self.context, target_id='t'))
        self.assertFalse(check_free_watch(FreeWatchType.DRONE_HARD_FAILURE, context, drone_id='d'))
        self.world.drones['d'].battery = 0
        self.assertTrue(check_free_watch(FreeWatchType.DRONE_HARD_FAILURE, context, drone_id='d'))
        self.world.drones['d'].position = self.world.base_position
        self.assertFalse(check_free_watch(FreeWatchType.DRONE_HARD_FAILURE, context, drone_id='d'))
        self.assertFalse(check_free_watch(FreeWatchType.MISSION_COMPLETE, context))
        self.world.active_targets['s'].clear()
        self.world.failed_targets.append('t')
        self.assertFalse(check_free_watch(FreeWatchType.MISSION_COMPLETE, context))
        self.assertTrue(check_free_watch(FreeWatchType.MISSION_COMPLETE,
                                        replace(context, explored_cells=reachable_cells(self.world))))

    def test_charge_watch_only_applies_at_base(self):
        predicate = WatchPredicate(type=WatchType.CHARGE_ABOVE, drone_id='d', threshold=9)
        self.assertFalse(check_watch(predicate, self.context))
        self.world.drones['d'].position = self.world.base_position
        self.assertTrue(check_watch(predicate, self.context))
        self.assertFalse(check_watch(predicate.model_copy(update={'threshold': 10}), self.context))

    def test_forecast_is_deterministic_and_ignores_neighbours_without_mutation(self):
        self.world.sectors[0].weather = Weather.FOGGY
        with patch('random.random', side_effect=AssertionError('Simulator RNG consumed')):
            risk = forecast_weather_risk(self.world, Weather.BLIZZARD, 's', 1)
            repeated = forecast_weather_risk(self.world, Weather.BLIZZARD, 's', 1)
        # One step can worsen from foggy to stormy with probability 0.2.
        self.assertAlmostEqual(risk, 0.2)
        self.assertEqual(risk, repeated)
        self.assertEqual(self.world.sectors[0].weather, Weather.FOGGY)
        neighbour = Sector('neighbour', ((5, 0), (5, 4)), 2, 1, 0, Weather.BLIZZARD, 0)
        self.world.sectors.append(neighbour)
        self.world.sector_neighbours = self.world._build_sector_neighbours()
        self.assertAlmostEqual(forecast_weather_risk(self.world, Weather.BLIZZARD, 's', 1), risk)
        neighbour.weather = Weather.CLEAR
        self.assertAlmostEqual(forecast_weather_risk(self.world, Weather.BLIZZARD, 's', 1), risk)
        self.world.sectors[0].weather = Weather.CLEAR
        self.assertEqual(forecast_weather_risk(self.world, Weather.BLIZZARD, 's', 2), 0)
        self.world.sectors[0].weather = Weather.STORMY
        self.assertEqual(forecast_weather_risk(self.world, Weather.CLEAR, 's', 1), 1)

    def test_survival_is_specific_to_target_and_uses_supplied_prior(self):
        self.assertAlmostEqual(target_survival_probability('t', self.world, log(2), 2), 0.25)
        self.assertEqual(target_survival_probability('t', self.world, 0, 10), 1.0)
        self.world.targets['other'] = (3, 3)
        self.world.active_targets['s'].append('other')
        self.assertAlmostEqual(target_survival_probability('t', self.world, log(2), 2), 0.25)
        self.world.update_target_success('other', 1)
        self.assertAlmostEqual(target_survival_probability('t', self.world, log(2), 2), 0.25)
        predicate = WatchPredicate(type=WatchType.TARGET_SURVIVAL_PROBABILITY_BELOW,
                                   target_id='t', horizon=2, threshold=0.3, decay_rate_prior=log(2))
        for actual_rate in (0, 0.001, 100):
            self.world.sectors[0].decay_rate = actual_rate
            self.assertTrue(check_watch(predicate, self.context))
        with self.assertRaises(ValueError):
            check_watch(predicate.model_copy(update={'decay_rate_prior': None}), self.context)
        self.world.update_target_success('t', 1)
        self.assertEqual(target_survival_probability('t', self.world, log(2), 2), 1.0)
        self.assertFalse(check_watch(predicate, self.context))
        self.world.targets['failed'] = (4, 4)
        self.world.failed_targets.append('failed')
        self.assertEqual(target_survival_probability('failed', self.world, log(2), 2), 0.0)
        with self.assertRaises(KeyError):
            target_survival_probability('unknown', self.world, log(2), 2)

    def setUp(self):
        self.world = make_world()
        snapshots = []
        for step, battery, comms in ((0, 20, True), (1, 15, False), (2, 12, False), (3, 10, False)):
            snapshots.append(WatchSample(step=step, battery={'d': battery}, comms={'d': comms}))
        self.context = WatchContext(
            world=self.world, step=3, history=snapshots, completed_sectors={'s'},
            route_costs={'d': 12}, baseline_route_costs={'d': 10},
            scenario_weather=Weather.CLOUDY,
        )

    def test_every_watch_type_has_hook_and_comparison_boundaries_work(self):
        self.assertEqual(set(WATCH_HOOKS), set(WatchType))
        # Stormy return cost is two moves at 1.5 each: reserve = 10 - 3 = 7.
        cases = [
            (WatchType.RETURN_RESERVE_BELOW, dict(drone_id='d'), 8, 7),
            (WatchType.BATTERY_DRAIN_ABOVE, dict(drone_id='d', horizon=3), 9, 10),
            (WatchType.WEATHER_FORECAST_RISK_ABOVE, dict(sector_id='s', horizon=2), 0.9, 1.0),
            (WatchType.ROUTE_COST_INCREASED_BY, dict(drone_id='d'), 1, 2),
            (WatchType.TARGET_SURVIVAL_PROBABILITY_BELOW, dict(target_id='t', horizon=2, decay_rate_prior=log(2)), 0.3, 0.25),
        ]
        for watch_type, ids, true_threshold, false_threshold in cases:
            for threshold, expected in ((true_threshold, True), (false_threshold, False)):
                with self.subTest(type=watch_type, threshold=threshold):
                    predicate = WatchPredicate(type=watch_type, threshold=threshold, **ids)
                    self.assertEqual(check_watch(predicate, self.context), expected)

    def test_comms_duration_is_continuous_and_resets_on_reconnection(self):
        predicate = WatchPredicate(type=WatchType.COMMS_LOST_FOR, drone_id='d', horizon=2)
        self.assertTrue(check_watch(predicate, self.context))
        self.assertFalse(check_watch(predicate.model_copy(update={'horizon': 3}), self.context))
        self.world.drones['d'].comms = True
        self.assertFalse(check_watch(predicate, self.context))
        self.world.drones['d'].comms = False
        self.context.history[-2].comms['d'] = True
        self.assertFalse(check_watch(predicate, self.context))

    def test_weather_supports_sector_or_drone_and_named_threshold(self):
        for ids in (dict(sector_id='s'), dict(drone_id='d')):
            predicate = WatchPredicate(type=WatchType.WEATHER_AT_LEAST, threshold='stormy', **ids)
            self.assertTrue(check_watch(predicate, self.context))
            self.assertFalse(check_watch(predicate.model_copy(update={'threshold': 'blizzard'}), self.context))

    def test_deadline_ignores_rescued_targets_and_sector_completion(self):
        predicate = WatchPredicate(type=WatchType.TARGET_NOT_REACHED_BY, target_id='t', horizon=3)
        self.assertTrue(check_watch(predicate, self.context))
        self.assertFalse(check_watch(predicate, replace(self.context, step=2)))
        self.world.update_target_success('t', 2)
        self.assertFalse(check_watch(predicate, self.context))
        sector_watch = WatchPredicate(type=WatchType.SECTOR_COMPLETE, sector_id='s')
        self.assertTrue(check_watch(sector_watch, self.context))
        self.assertFalse(check_watch(sector_watch, replace(self.context, completed_sectors=set())))

    def test_unreachable_routes_and_insufficient_history(self):
        predicate = WatchPredicate(type=WatchType.ROUTE_UNREACHABLE, drone_id='d')
        self.assertFalse(check_watch(predicate, self.context))
        self.assertTrue(check_watch(predicate, replace(self.context, route_costs={'d': None})))
        self.world.no_fly.update({(0, 0)})
        predicate = WatchPredicate(type=WatchType.RETURN_RESERVE_BELOW, drone_id='d', threshold=0)
        self.assertTrue(check_watch(predicate, self.context))
        predicate = WatchPredicate(type=WatchType.BATTERY_DRAIN_ABOVE, drone_id='d', threshold=0, horizon=4)
        self.assertFalse(check_watch(predicate, self.context))

    def test_checks_do_not_change_world_or_consume_randomness(self):
        before = (dict(self.world.targets), {s: ids.copy() for s, ids in self.world.active_targets.items()},
                  self.world.drones['d'].model_dump())
        with patch('random.random', side_effect=AssertionError('Randomness consumed')):
            check_watch(WatchPredicate(type=WatchType.RETURN_RESERVE_BELOW, drone_id='d', threshold=10), self.context)
        after = (dict(self.world.targets), self.world.active_targets, self.world.drones['d'].model_dump())
        self.assertEqual(before, after)


class SimulatorWatchTest(unittest.TestCase):
    def test_free_watches_run_without_monitoring_plan_for_every_drone_and_target(self):
        world = make_world()
        world.sectors[0].visibility = 10
        world.drones['d'].comms = True
        world.targets['t2'] = (2, 3)
        world.active_targets['s'].append('t2')
        for drone_id in ('failed1', 'failed2'):
            world.drones[drone_id] = world.drones['d'].model_copy(update={'id': drone_id, 'battery': 0})
        world.drones['charging'] = world.drones['d'].model_copy(
            update={'id': 'charging', 'position': world.base_position, 'battery': 0})
        scenario = Scenario((5, 5), 1, 2, Difficulty.MEDIUM, 4, (0, 0), {}, Weather.CLOUDY, 5)
        callback = Mock()
        simulator = Simulator(world, scenario, on_alert=callback)
        with patch.object(world, 'update_sector_weather'), patch('src.simulator.random.random', return_value=0.99):
            alerts = simulator.step({})
            self.assertEqual({a.target_id for a in alerts if a.type is FreeWatchType.NEW_TARGET_DETECTED}, {'t', 't2'})
            self.assertEqual({a.drone_id for a in alerts if a.type is FreeWatchType.DRONE_HARD_FAILURE}, {'failed1', 'failed2'})
            callback.assert_called_once()
            self.assertEqual(len(callback.call_args.args[0]), 4)
            self.assertEqual(simulator.step({}), [])
        self.assertTrue(all(WatchAlert.model_validate_json(a.model_dump_json()) == a for a in alerts))

    def test_free_and_selected_watches_share_one_callback_and_free_alerts_do_not_rearm_with_plan(self):
        callback = Mock()
        simulator = self.simulator([
            WatchPredicate(type=WatchType.WEATHER_AT_LEAST, sector_id='s', threshold='stormy')
        ], callback)
        simulator.world.sectors[0].visibility = 10
        with patch('src.simulator.random.random', return_value=0.99):
            alerts = simulator.step({})
            self.assertEqual({a.type for a in alerts}, {WatchType.WEATHER_AT_LEAST, FreeWatchType.NEW_TARGET_DETECTED})
            callback.assert_called_once()
            simulator.set_monitoring_plan(simulator.monitoring_plan)
            alerts = simulator.step({})
        self.assertEqual([a.type for a in alerts], [WatchType.WEATHER_AT_LEAST])

    def test_completion_requires_coverage_and_acquisition_of_known_targets(self):
        simulator = self.simulator([])
        world = simulator.world
        simulator.explored_cells.update(reachable_cells(world))
        simulator.known_targets['t'] = world.targets['t']
        simulator.discovered_targets['t'] = world.targets['t']
        # Hidden failures must not affect either completion or its reported status.
        world.targets['hidden'] = (4, 4)
        world.failed_targets.append('hidden')
        with patch('src.simulator.random.random', return_value=0.99):
            self.assertEqual(simulator.step({}), [])
            world.drones['d'].position = (2, 1)
            alerts = simulator.step({'d': Objective(drone_id='d', objective_type=ObjectiveType.GOTO,
                                                   target_id='t')})
            self.assertEqual([a.type for a in alerts], [FreeWatchType.MISSION_COMPLETE])
            self.assertEqual(alerts[0].mission_status, MissionStatus.SUCCESS)
            self.assertEqual(simulator.acquired_targets, {'t'})
            self.assertEqual(simulator.step({}), [])

    def test_completion_excludes_obstacles_no_fly_and_unreachable_cells(self):
        simulator = self.simulator([])
        world = simulator.world
        world.obstacles.update({(1, y) for y in range(5)} | {(0, 4)})
        world.no_fly.add((0, 3))
        context = simulator.watch_context()
        self.assertEqual(context.coverage_cells, {(0, 0), (0, 1), (0, 2)})
        simulator.explored_cells.update({(0, 0), (0, 2)})
        self.assertFalse(check_free_watch(FreeWatchType.MISSION_COMPLETE, simulator.watch_context()))
        simulator.explored_cells.add((0, 1))
        self.assertTrue(check_free_watch(FreeWatchType.MISSION_COMPLETE, simulator.watch_context()))
        # The hidden target remains active behind the wall; it cannot affect the check.
        self.assertEqual(world.active_targets['s'], ['t'])
        world.targets['other_hidden'] = (3, 3)
        world.active_targets['s'].append('other_hidden')
        self.assertTrue(check_free_watch(FreeWatchType.MISSION_COMPLETE, simulator.watch_context()))
        world.obstacles.remove((1, 2))
        self.assertFalse(check_free_watch(FreeWatchType.MISSION_COMPLETE, simulator.watch_context()))

    def test_known_expired_target_does_not_count_as_acquired(self):
        simulator = self.simulator([])
        simulator.explored_cells.update(reachable_cells(simulator.world))
        simulator.known_targets['t'] = (2, 2)
        simulator.world.active_targets['s'].clear()
        simulator.world.failed_targets.append('t')
        self.assertFalse(check_free_watch(FreeWatchType.MISSION_COMPLETE, simulator.watch_context()))

    def test_free_detection_interrupts_execute_even_when_no_selected_watch_matches(self):
        callback = Mock()
        simulator = self.simulator([], callback)
        simulator.world.sectors[0].visibility = 10
        with patch('src.simulator.random.random', return_value=0.99):
            alerts, _, step = simulator.execute_plan({}, max_steps=5)
        self.assertEqual(step, 1)
        self.assertEqual([a.type for a in alerts], [FreeWatchType.NEW_TARGET_DETECTED])
        callback.assert_called_once()

    def test_turn_limit_does_not_alert_and_legacy_incidents_do_not_interrupt(self):
        callback = Mock()
        simulator = self.simulator([], callback)
        simulator.scenario.events[1] = (
            IncidentType.NEW_NO_FLY_ZONE,
            'New no-fly zone added: rows 4-4, columns 0-0',
        )
        with patch('src.simulator.random.random', return_value=0.99):
            alerts, _, step = simulator.execute_plan({}, max_steps=3)
        self.assertEqual(alerts, [])
        self.assertEqual(step, 3)
        callback.assert_not_called()
        self.assertIn((4, 0), simulator.world.no_fly)

    def test_monitoring_plan_requires_exactly_three_watches_and_reason(self):
        predicate = WatchPredicate(type=WatchType.CHARGE_ABOVE, drone_id='d', threshold=90)
        for count in (0, 1, 2, 4):
            with self.subTest(count=count), self.assertRaises(ValueError):
                MonitoringPlan(watches=[predicate] * count, reason='Constrained choice')
        with self.assertRaises(ValueError):
            MonitoringPlan(watches=[predicate] * 3, reason='')
        plan = MonitoringPlan(watches=[predicate] * 3, reason='Constrained choice')
        self.assertEqual(MonitoringPlan.model_validate_json(plan.model_dump_json()), plan)

    def test_simulator_checks_only_selected_three_and_owns_its_plan_copy(self):
        simulator = self.simulator([WatchPredicate(type=WatchType.CHARGE_ABOVE, drone_id='d', threshold=90)])
        selected = simulator.monitoring_plan
        selected.watches.clear()
        self.assertEqual(len(simulator.monitoring_plan.watches), 3)
        with patch('src.simulator.check_watch', wraps=check_watch) as hook:
            simulator.step({})
        self.assertEqual(hook.call_count, 3)
        self.assertEqual([call.args[0] for call in hook.call_args_list], simulator.monitoring_plan.watches)
        with self.assertRaises(ValueError):
            simulator.set_monitoring_plan(selected)
        self.assertEqual(len(simulator.monitoring_plan.watches), 3)

    def test_execute_requires_explicit_monitoring_plan(self):
        world = make_world()
        scenario = Scenario((5, 5), 1, 0, Difficulty.MEDIUM, 1, (0, 0), {}, Weather.CLOUDY, 5)
        simulator = Simulator(world, scenario)
        with self.assertRaises(ValueError):
            simulator.execute_plan({})
        self.assertEqual(simulator.timestep, 0)

    def test_forecast_and_prior_survival_watches_run_without_external_data(self):
        predicates = [
            WatchPredicate(type=WatchType.WEATHER_FORECAST_RISK_ABOVE,
                           sector_id='s', horizon=2, threshold=0.5),
            WatchPredicate(type=WatchType.TARGET_SURVIVAL_PROBABILITY_BELOW,
                           target_id='t', horizon=2, threshold=0.3, decay_rate_prior=log(2)),
        ]
        callback = Mock()
        simulator = self.simulator(predicates, callback)
        with patch('src.simulator.random.random', return_value=0.99):
            simulator.step({})
        callback.assert_called_once()
        self.assertEqual({p.type for p in callback.call_args.args[0]},
                         {p.type for p in predicates})

    def simulator(self, watches, callback=None):
        world = make_world()
        world.drones['d'].comms = True
        scenario = Scenario((5, 5), 1, 0, Difficulty.MEDIUM, 1, (0, 0), {}, Weather.CLOUDY, 5)
        weather = patch.object(world, 'update_sector_weather')
        weather.start()
        self.addCleanup(weather.stop)
        # Fill unused slots with deliberately nonmatching, explicitly selected watches.
        selected = watches + [WatchPredicate(type=WatchType.CHARGE_ABOVE, drone_id='d', threshold=100+i)
                              for i in range(3 - len(watches))]
        plan = MonitoringPlan(watches=selected, reason='Test monitoring selection')
        return Simulator(world, scenario, monitoring_plan=plan, on_alert=callback)

    def test_alerts_are_batched_and_rearm_only_after_condition_clears(self):
        watches = [WatchPredicate(type=WatchType.CHARGE_ABOVE, drone_id='d', threshold=19),
                   WatchPredicate(type=WatchType.WEATHER_AT_LEAST, sector_id='s', threshold='stormy')]
        callback = Mock()
        simulator = self.simulator(watches, callback)
        simulator.world.drones['d'].position = (0, 0)
        simulator.world.drones['d'].battery = 20
        with patch('src.simulator.random.random', return_value=0.99):
            simulator.step({})
            callback.assert_called_once()
            self.assertEqual(len(callback.call_args.args[0]), 2)
            simulator.step({})
            self.assertEqual(callback.call_count, 1)
            simulator.world.drones['d'].battery = 5
            simulator.step({})
            simulator.world.drones['d'].battery = 20
            simulator.step({})
        self.assertEqual(callback.call_count, 2)
        self.assertEqual(len(callback.call_args.args[0]), 1)

    def test_execute_waits_through_diagnostic_events_until_watch_fires(self):
        predicate = WatchPredicate(type=WatchType.CHARGE_ABOVE, drone_id='d', threshold=18)
        callback = Mock()
        simulator = self.simulator([predicate], callback)
        simulator.world.drones['d'].position = (0, 0)
        events, _, step = simulator.execute_plan({}, max_steps=5)
        self.assertEqual(step, 2)
        self.assertEqual(events[0].type, WatchType.CHARGE_ABOVE)
        callback.assert_called_once()
        self.assertEqual(simulator.world.drones['d'].battery, 20)

    def test_route_cost_baseline_is_captured_from_accepted_plan(self):
        predicate = WatchPredicate(type=WatchType.ROUTE_COST_INCREASED_BY, drone_id='d', threshold=1)
        simulator = self.simulator([predicate])
        with patch('src.simulator.random.random', return_value=0.99):
            simulator.step({'d': Objective(drone_id='d', objective_type=ObjectiveType.RETURN_TO_BASE)})
        self.assertEqual(simulator.watch_context().baseline_route_costs['d'], 3)
        self.assertFalse(simulator.triggered_watches)
        simulator.world.sectors[0].movement_cost = 5
        # A held order avoids consuming the final move before checking its cost.
        context = simulator.watch_context()
        self.assertTrue(check_watch(predicate, context))


if __name__ == '__main__':
    unittest.main()
