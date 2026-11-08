from types import SimpleNamespace
import unittest

from src.evaluation import evaluate


class EvaluationTest(unittest.TestCase):
    def test_empty_initial_scenario_has_finite_score(self):
        simulator = SimpleNamespace(
            world=SimpleNamespace(successful_targets=[]),
            scenario=SimpleNamespace(num_targets=0), timestep=0, drones={},
        )
        self.assertEqual(evaluate(simulator, starting_battery=0), 1201.0)

    def test_nonempty_scenario_keeps_original_scoring(self):
        simulator = SimpleNamespace(
            world=SimpleNamespace(successful_targets=['target_0']),
            scenario=SimpleNamespace(num_targets=2), timestep=10,
            drones={'0': SimpleNamespace(battery=25, comms=True),
                    '1': SimpleNamespace(battery=0, comms=False)},
        )
        self.assertEqual(evaluate(simulator, starting_battery=100), 520.75)
