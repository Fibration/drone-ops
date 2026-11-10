import unittest
from unittest.mock import Mock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from src.agent import FleetMonitor, MissionCommander, MissionPlanner, Reviewer
from src.mission_graph import build_mission_graph
from src.models import AgentContext, AgentType, MissionPlan, MissionState, MissionStatus, MonitoringPlan, Plan, Tool, WatchPredicate, WatchType
from src.tools import agent_tools, ToolGateway
from src.llm_policy import ToolCallingPolicy
from src.models import CommanderRoutingDecision


class ScriptedModel(BaseChatModel):
    responses: list[AIMessage]
    bound_names: list[list[str]] = Field(default_factory=list)
    received_messages: list[list] = Field(default_factory=list)

    @property
    def _llm_type(self):
        return 'scripted-tool-test'

    def bind_tools(self, tools, **kwargs):
        self.bound_names.append([tool.name if hasattr(tool, 'name')
                                 else tool['function']['name'] for tool in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received_messages.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def tool_call(name, args, call_id):
    return AIMessage(content='', tool_calls=[{'name': name, 'args': args, 'id': call_id}])


def state():
    return MissionState(
        drone_ids=['0'], objectives=[], no_fly_zones=[], discovered_targets={}, events=[],
        explored_cells=set(), current_plan=MissionPlan(plan={}), proposed_plans=[],
        under_review=Plan(plan={}, reason='Candidate', estimated_completion_times=0),
        needs_replan=False, next_agent=None, review_status=None, human_approval=None,
        step=0, mission_status=MissionStatus.ONGOING,
    )


class AgentToolTest(unittest.TestCase):
    def test_mixed_backends_share_the_same_bound_tool_registry(self):
        clients = {}

        class HeuristicCommander:
            def evaluate(self, observation, *, tools):
                clients[AgentType.MISSION_COMMANDER] = tools
                tools[Tool.GET_FLEET_STATUS].invoke({})
                if not observation.proposed_plans:
                    return CommanderRoutingDecision(next_agent=AgentType.MISSION_PLANNER)
                return CommanderRoutingDecision(next_agent=AgentType.REVIEWER,
                                                 under_review=observation.proposed_plans[0])

        class RLPlanner:
            def __init__(self, model):
                self.model = model

            def evaluate(self, observation, *, tools):
                clients[AgentType.MISSION_PLANNER] = tools
                report = tools[Tool.GET_FLEET_STATUS].invoke({})
                action = self.model.predict([report['battery']])
                return [Plan(plan={}, reason=f'RL action {action}', estimated_completion_times=0)]

        class HeuristicMonitor:
            def evaluate(self, observation, *, tools):
                clients[AgentType.FLEET_MONITOR] = tools
                tools[Tool.GET_SECTOR_STATUS].invoke({'sector_id': '0'})
                return MonitoringPlan(
                    watches=[WatchPredicate(type=WatchType.SECTOR_COMPLETE, sector_id=str(i))
                             for i in range(3)], reason='Heuristic monitoring selection',
                )

        reviewer_model = ScriptedModel(responses=[
            tool_call('evaluate_candidate_plan', {'plan': {'plan': {}}}, 'evaluate'),
            tool_call('REVIEWERResponse', {'result': {
                'outcome': 'approve', 'critique': 'Reviewed', 'concerns': [], 'mandatory_watches': [],
            }}, 'decision'),
        ])
        rl_model = Mock()
        rl_model.predict.return_value = 1
        dispatcher = Mock(return_value={'battery': 50})
        graph = build_mission_graph(agents={
            AgentType.MISSION_COMMANDER: MissionCommander(policy=HeuristicCommander()),
            AgentType.MISSION_PLANNER: MissionPlanner(policy=RLPlanner(rl_model)),
            AgentType.REVIEWER: Reviewer(policy=ToolCallingPolicy(reviewer_model, AgentType.REVIEWER)),
            AgentType.FLEET_MONITOR: FleetMonitor(policy=HeuristicMonitor()),
        }).compile()
        initial = state()
        initial.needs_replan = True
        result = MissionState.model_validate(graph.invoke(initial, context=AgentContext(ToolGateway(dispatcher))))
        rl_model.predict.assert_called_once_with([50])
        self.assertEqual(len(result.monitoring_plan.watches), 3)
        self.assertEqual(result.current_plan.plan, {})
        for role, client in clients.items():
            self.assertEqual(set(client), set(agent_tools[role]))
            self.assertFalse(hasattr(client, 'simulator_call'))
        self.assertNotIn('battery', result.model_dump_json())

    def test_each_role_binds_only_its_registry_and_executes_through_runtime(self):
        cases = [
            (AgentType.MISSION_COMMANDER, MissionCommander, 'get_sector_status',
             {'sector_id': '0'}, Tool.GET_SECTOR_STATUS, {'sector_id': '0'},
             {'next_agent': 'MISSION_PLANNER'}),
            (AgentType.MISSION_PLANNER, MissionPlanner, 'plan_route',
             {'drone_id': '0', 'destination': [1, 2]}, Tool.PLAN_ROUTE,
             {'drone_id': '0', 'destination': (1, 2)},
             [{'plan': {}, 'reason': 'Candidate', 'estimated_completion_times': 0}]),
            (AgentType.REVIEWER, Reviewer, 'stress_test_plan',
             {'plan': {'plan': {}}, 'n_rollouts': 2}, Tool.STRESS_TEST_PLAN,
             {'plan': {'plan': {}}, 'n_rollouts': 2},
             {'outcome': 'approve', 'critique': 'Reviewed', 'concerns': [], 'mandatory_watches': []}),
            (AgentType.FLEET_MONITOR, FleetMonitor, 'get_fleet_status', {},
             Tool.GET_FLEET_STATUS, {},
             {'watches': [{'type': 'SECTOR_COMPLETE', 'sector_id': str(i)} for i in range(3)],
              'reason': 'Monitor accepted plan'}),
        ]
        for role, agent_type, name, args, expected_tool, expected_args, result in cases:
            with self.subTest(role=role):
                output_name = f'{role.value}Response'
                model = ScriptedModel(responses=[
                    tool_call(name, args, 'data'),
                    tool_call(output_name, {'result': result}, 'decision'),
                ])
                dispatcher = Mock(return_value={'reported': 'observation'})
                agent = agent_type(policy=ToolCallingPolicy(model, role))
                client = ToolGateway(dispatcher).for_role(role)
                update = agent.run(state(), tools=client)
                dispatcher.assert_called_once_with(expected_tool, expected_args)
                expected_names = {tool.name for tool in agent_tools[role].values()} | {output_name}
                self.assertTrue(model.bound_names)
                self.assertTrue(all(set(names) == expected_names for names in model.bound_names))
                self.assertTrue(any(isinstance(message, ToolMessage) and 'observation' in message.content
                                    for message in model.received_messages[-1]))
                self.assertTrue(update)
                for bound_tool in client.values():
                    properties = bound_tool.tool_call_schema.model_json_schema().get('properties', {})
                    self.assertNotIn('runtime', properties)
                    self.assertNotIn('tools', properties)
                    self.assertNotIn('simulator_call', properties)

    def test_graph_supplies_role_client_to_model_backed_handler(self):
        model = ScriptedModel(responses=[
            tool_call('get_fleet_status', {}, 'fleet'),
            tool_call('MISSION_COMMANDERResponse', {
                'result': {'next_agent': None, 'mission_status': 'success'},
            }, 'decision'),
        ])
        dispatcher = Mock(return_value={'drones': []})
        graph = build_mission_graph(nodes={
            role: Mock(side_effect=AssertionError('Unexpected node execution'))
            for role in AgentType if role is not AgentType.MISSION_COMMANDER
        }, agents={
            AgentType.MISSION_COMMANDER: MissionCommander(
                policy=ToolCallingPolicy(model, AgentType.MISSION_COMMANDER)),
        }).compile()
        result = MissionState.model_validate(graph.invoke(state(), context=AgentContext(tools=ToolGateway(dispatcher))))
        self.assertEqual(result.mission_status, MissionStatus.SUCCESS)
        dispatcher.assert_called_once_with(Tool.GET_FLEET_STATUS, {})
        self.assertNotIn('simulator_call', result.model_dump_json())

    def test_tools_outside_the_role_registry_are_not_dispatched(self):
        model = ScriptedModel(responses=[
            tool_call('stress_test_plan', {'plan': {'plan': {}}, 'n_rollouts': 1}, 'forbidden'),
            tool_call('MISSION_COMMANDERResponse', {'result': {'next_agent': 'MISSION_PLANNER'}}, 'decision'),
        ])
        dispatcher = Mock()
        MissionCommander(policy=ToolCallingPolicy(model, AgentType.MISSION_COMMANDER)).run(state(), tools=ToolGateway(dispatcher).for_role(AgentType.MISSION_COMMANDER))
        dispatcher.assert_not_called()
        messages = [message for message in model.received_messages[-1] if isinstance(message, ToolMessage)]
        self.assertTrue(any(message.status == 'error' for message in messages))

    def test_invalid_tool_arguments_do_not_reach_dispatcher(self):
        model = ScriptedModel(responses=[
            tool_call('get_weather_forecast', {'sector_id': '0', 'horizon': 0}, 'invalid'),
            tool_call('MISSION_COMMANDERResponse', {'result': {'next_agent': 'MISSION_PLANNER'}}, 'decision'),
        ])
        dispatcher = Mock()
        MissionCommander(policy=ToolCallingPolicy(model, AgentType.MISSION_COMMANDER)).run(state(), tools=ToolGateway(dispatcher).for_role(AgentType.MISSION_COMMANDER))
        dispatcher.assert_not_called()

    def test_model_backed_policy_requires_tool_registry(self):
        model = ScriptedModel(responses=[])
        with self.assertRaises(ValueError):
            MissionCommander(policy=ToolCallingPolicy(model, AgentType.MISSION_COMMANDER)).run(state())

    def test_reusing_llm_policy_does_not_reuse_another_simulators_client(self):
        model = ScriptedModel(responses=[
            message for index in range(2) for message in (
                tool_call('get_fleet_status', {}, f'fleet-{index}'),
                tool_call('MISSION_COMMANDERResponse',
                          {'result': {'next_agent': 'MISSION_PLANNER'}}, f'decision-{index}'),
            )
        ])
        agent = MissionCommander(policy=ToolCallingPolicy(model, AgentType.MISSION_COMMANDER))
        first = Mock(return_value={'fleet': 'first'})
        second = Mock(return_value={'fleet': 'second'})
        for dispatcher in (first, second):
            agent.run(state(), tools=ToolGateway(dispatcher).for_role(AgentType.MISSION_COMMANDER))
        first.assert_called_once_with(Tool.GET_FLEET_STATUS, {})
        second.assert_called_once_with(Tool.GET_FLEET_STATUS, {})


class ToolGatewayTest(unittest.TestCase):
    def test_direct_call_permissions_and_validation_are_backend_independent(self):
        dispatcher = Mock(return_value={'ok': True})
        client = ToolGateway(dispatcher).for_role(AgentType.MISSION_COMMANDER)
        with self.assertRaises(KeyError):
            client[Tool.STRESS_TEST_PLAN].invoke({'plan': {'plan': {}}, 'n_rollouts': 1})
        with self.assertRaises(ValueError):
            client[Tool.GET_WEATHER_FORECAST].invoke({'sector_id': '0', 'horizon': 0})
        dispatcher.assert_not_called()
        self.assertEqual(client[Tool.GET_FLEET_STATUS].invoke({}), {'ok': True})
        dispatcher.assert_called_once_with(Tool.GET_FLEET_STATUS, {})

    def test_tool_results_are_detached_from_simulator_data(self):
        data = {'drones': [{'battery': 50}]}
        dispatcher = Mock(return_value=data)
        client = ToolGateway(dispatcher).for_role(AgentType.MISSION_COMMANDER)
        result = client[Tool.GET_FLEET_STATUS].invoke({})
        result['drones'][0]['battery'] = 0
        self.assertEqual(data['drones'][0]['battery'], 50)


if __name__ == '__main__':
    unittest.main()
