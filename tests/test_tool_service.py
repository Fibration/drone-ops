import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from src.models import AgentType, DroneState, Terrain, Tool, ToolObservation
from src.scenario import Difficulty, Scenario, Sector, Weather, World
from src.services import SimulatorTools
from src.simulator import Simulator
from src.tools import ToolGateway


def simulator(service=None):
    sector = Sector('0', ((0, 0), (2, 2)), 2, 1, 0, Weather.CLEAR, 0)
    world = World([sector], 0, [], [], {})
    world.targets['hidden'] = (2, 2)
    scenario = Scenario((3, 3), 1, 1, Difficulty.MEDIUM, 0, (0, 0), {}, Weather.CLEAR, 5)
    return Simulator(world, scenario, tool_service=service)


class ToolServiceTest(unittest.TestCase):
    def test_population_estimate_easy_is_exact_without_randomness(self):
        sim = simulator(SimulatorTools())
        sim.world.sectors[0].population_rate = 4
        sim.scenario = replace(sim.scenario, difficulty=Difficulty.EASY)
        with patch('src.simulator.random.random') as draw:
            for role in (AgentType.MISSION_COMMANDER, AgentType.REVIEWER):
                tools = ToolGateway(sim.call_tool).for_role(role)
                self.assertEqual(
                    tools[Tool.ESTIMATE_SECTOR_POPULATION].invoke({'sector_id': '0'}),
                    {'estimated_population': 1},
                )
            draw.assert_not_called()
        for role in (AgentType.MISSION_PLANNER, AgentType.FLEET_MONITOR):
            self.assertNotIn(
                Tool.ESTIMATE_SECTOR_POPULATION, ToolGateway(sim.call_tool).for_role(role)
            )

    def test_population_estimate_noise_scales_with_difficulty(self):
        for difficulty, noise in (
            (Difficulty.MEDIUM, 0.05),
            (Difficulty.HARD, 0.10),
            (Difficulty.IMPOSSIBLE, 0.15),
        ):
            for draw in (0.0, 0.5, 1.0):
                sim = simulator(SimulatorTools())
                sim.world.sectors[0].population_rate = 4
                sim.scenario = replace(sim.scenario, difficulty=difficulty)
                with self.subTest(difficulty=difficulty, draw=draw), patch(
                    'src.simulator.random.random', return_value=draw
                ):
                    result = sim.call_tool(Tool.ESTIMATE_SECTOR_POPULATION, {'sector_id': '0'})
                    self.assertAlmostEqual(
                        result['estimated_population'], 1 + noise * (2 * draw - 1)
                    )

    def test_population_estimate_zero_and_invalid_rate(self):
        sim = simulator(SimulatorTools())
        sim.scenario = replace(sim.scenario, difficulty=Difficulty.EASY)
        self.assertEqual(
            sim.call_tool(Tool.ESTIMATE_SECTOR_POPULATION, {'sector_id': '0'}),
            {'estimated_population': 1},
        )
        sim = simulator(SimulatorTools())
        sim.world.sectors[0].population_rate = -1
        with self.assertRaises(ValueError):
            sim.call_tool(Tool.ESTIMATE_SECTOR_POPULATION, {'sector_id': '0'})
        with self.assertRaises(KeyError):
            sim.world.sectors[0].population_rate = 0
            sim.call_tool(Tool.ESTIMATE_SECTOR_POPULATION, {'sector_id': 'missing'})

    def test_population_estimates_are_normalized_cached_and_detached(self):
        service = SimulatorTools()
        sim = simulator(service)
        sim.world.sectors[0].population_rate = 1
        sim.world.sectors.append(Sector('1', ((3, 0), (5, 2)), 2, 1, 0, Weather.CLEAR, 3))
        sim.scenario = replace(sim.scenario, num_targets=20)
        with patch('src.simulator.random.random', return_value=1.0) as draw:
            commander = ToolGateway(sim.call_tool).for_role(AgentType.MISSION_COMMANDER)
            reviewer = ToolGateway(sim.call_tool).for_role(AgentType.REVIEWER)
            self.assertEqual(commander[Tool.ESTIMATE_SECTOR_POPULATION].invoke({'sector_id': '0'}),
                             {'estimated_population': 5.25})
            self.assertEqual(reviewer[Tool.ESTIMATE_SECTOR_POPULATION].invoke({'sector_id': '1'}),
                             {'estimated_population': 15.75})
            view = sim.observation_view()
            view[ToolObservation.POPULATION_ESTIMATES]['0'] = 999
            sim.world.targets.clear()
            for _ in range(5):
                self.assertEqual(sim.call_tool(Tool.ESTIMATE_SECTOR_POPULATION, {'sector_id': '0'}),
                                 {'estimated_population': 5.25})
            self.assertEqual(draw.call_count, 2)
            other = simulator(service)
            self.assertEqual(other.call_tool(Tool.ESTIMATE_SECTOR_POPULATION, {'sector_id': '0'}),
                             {'estimated_population': 1.05})
            self.assertEqual(draw.call_count, 3)

    def test_route_and_navigation_view_do_not_reveal_hidden_targets(self):
        sim = simulator(SimulatorTools())
        sim.world.drones['0'] = DroneState(
            id='0', position=(0, 0), battery=100, status='idle',
            comms=True, assignment=None,
        )
        arguments = {'drone_id': '0', 'destination': (2, 2)}
        view = sim.observation_view()
        route = sim.call_tool(Tool.PLAN_ROUTE, arguments)
        self.assertNotIn('hidden', repr(view))
        self.assertNotIn(
            Terrain.TARGET,
            (cell for row in view[ToolObservation.NAVIGATION_MAP] for cell in row),
        )
        sim.world.targets['hidden'] = (1, 0)
        self.assertEqual(sim.observation_view(), view)
        self.assertEqual(sim.call_tool(Tool.PLAN_ROUTE, arguments), route)
        sim.world.targets.clear()
        self.assertEqual(sim.observation_view(), view)
        self.assertEqual(sim.call_tool(Tool.PLAN_ROUTE, arguments), route)

    def test_every_tool_dispatches_to_an_explicit_stub(self):
        service = SimulatorTools()
        for tool in Tool:
            with self.subTest(tool=tool), self.assertRaisesRegex(NotImplementedError, tool.value):
                service.call(tool, {}, {})

    def test_gateway_dispatches_to_service_with_current_detached_knowledge(self):
        service = Mock()
        service.call.return_value = {'targets': []}
        sim = simulator(service)
        sim.known_targets['known'] = (1, 1)
        tools = ToolGateway(sim.call_tool).for_role(AgentType.MISSION_COMMANDER)
        self.assertEqual(tools[Tool.GET_ACTIVE_TARGETS].invoke({}), {'targets': []})
        tool, arguments, view = service.call.call_args.args
        self.assertEqual(tool, Tool.GET_ACTIVE_TARGETS)
        self.assertEqual(arguments, {})
        self.assertEqual(view['known_targets'], {'known': (1, 1)})
        self.assertNotIn('hidden', repr(view))
        view['known_targets'].clear()
        view['explored_cells'].add((2, 2))
        self.assertEqual(sim.known_targets, {'known': (1, 1)})
        self.assertFalse(sim.explored_cells)
        sim.timestep = 3
        sim.call_tool(Tool.GET_ACTIVE_TARGETS, {})
        self.assertEqual(service.call.call_args.args[2]['step'], 3)

    def test_unconfigured_service_fails_without_advancing_simulation(self):
        sim = simulator()
        with self.assertRaises(RuntimeError):
            sim.call_tool(Tool.GET_FLEET_STATUS, {})
        self.assertEqual(sim.timestep, 0)
