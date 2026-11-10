"""Shared LangChain tools, bound to a role-scoped simulator gateway."""

from collections.abc import Callable
from copy import deepcopy
from typing import Annotated

from langchain_core.tools import InjectedToolArg, StructuredTool, tool
from pydantic import Field

from src.models import AgentType, MissionPlan, Point, Tool, ToolRegistry

SimulatorCall = Annotated[Callable, InjectedToolArg]


@tool
def get_fleet_status(*, simulator_call: SimulatorCall) -> dict:
    """Get the fleet's reported drone status and last known observations."""
    return simulator_call(Tool.GET_FLEET_STATUS, {})


@tool
def get_sector_status(sector_id: str, *, simulator_call: SimulatorCall) -> dict:
    """Get the known coverage and conditions of a sector."""
    return simulator_call(Tool.GET_SECTOR_STATUS, {"sector_id": sector_id})


@tool
def estimate_sector_population(sector_id: str, *, simulator_call: SimulatorCall) -> dict:
    """Estimate a sector's initial target count, with uncertainty based on difficulty.

    The estimate is fixed for this simulation run; repeated calls return the
    same value. It does not track targets remaining as the mission progresses.
    """
    return simulator_call(Tool.ESTIMATE_SECTOR_POPULATION, {"sector_id": sector_id})


@tool
def get_active_targets(*, simulator_call: SimulatorCall) -> dict:
    """Get known targets that have not been confirmed as acquired."""
    return simulator_call(Tool.GET_ACTIVE_TARGETS, {})


@tool
def get_weather_forecast(
    sector_id: str,
    horizon: Annotated[int, Field(ge=1)],
    *,
    simulator_call: SimulatorCall,
) -> dict:
    """Estimate a sector's weather risk over the requested number of turns."""
    return simulator_call(
        Tool.GET_WEATHER_FORECAST, {"sector_id": sector_id, "horizon": horizon}
    )


@tool
def plan_route(
    drone_id: str,
    destination: Point,
    *,
    simulator_call: SimulatorCall,
) -> dict:
    """Plan a route from a drone's reported position to a destination cell."""
    return simulator_call(
        Tool.PLAN_ROUTE, {"drone_id": drone_id, "destination": destination}
    )


@tool
def estimate_route_cost(
    drone_id: str,
    route: list[Point],
    *,
    simulator_call: SimulatorCall,
) -> dict:
    """Estimate a drone's battery cost for the supplied route."""
    return simulator_call(
        Tool.ESTIMATE_ROUTE_COST, {"drone_id": drone_id, "route": route}
    )


@tool
def evaluate_candidate_plan(
    plan: MissionPlan, *, simulator_call: SimulatorCall
) -> dict:
    """Evaluate candidate drone assignments using the available observations."""
    return simulator_call(
        Tool.EVALUATE_CANDIDATE_PLAN, {"plan": plan.model_dump(mode="json")}
    )


@tool
def stress_test_plan(
    plan: MissionPlan,
    n_rollouts: Annotated[int, Field(ge=1)],
    *,
    simulator_call: SimulatorCall,
) -> dict:
    """Stress-test candidate assignments under the simulator's forecast assumptions."""
    return simulator_call(
        Tool.STRESS_TEST_PLAN,
        {"plan": plan.model_dump(mode="json"), "n_rollouts": n_rollouts},
    )


commander_tools = {
    Tool.GET_FLEET_STATUS: get_fleet_status,
    Tool.GET_SECTOR_STATUS: get_sector_status,
    Tool.GET_ACTIVE_TARGETS: get_active_targets,
    Tool.GET_WEATHER_FORECAST: get_weather_forecast,
}
planner_tools = {
    **commander_tools,
    Tool.PLAN_ROUTE: plan_route,
    Tool.ESTIMATE_ROUTE_COST: estimate_route_cost,
    Tool.EVALUATE_CANDIDATE_PLAN: evaluate_candidate_plan,
}
reviewer_tools = {
    Tool.ESTIMATE_SECTOR_POPULATION: estimate_sector_population,
    Tool.GET_WEATHER_FORECAST: get_weather_forecast,
    Tool.EVALUATE_CANDIDATE_PLAN: evaluate_candidate_plan,
    Tool.STRESS_TEST_PLAN: stress_test_plan,
}
fleet_monitor_tools = dict(commander_tools)
commander_tools[Tool.ESTIMATE_SECTOR_POPULATION] = estimate_sector_population
agent_tools = {
    AgentType.MISSION_COMMANDER: commander_tools,
    AgentType.MISSION_PLANNER: planner_tools,
    AgentType.REVIEWER: reviewer_tools,
    AgentType.FLEET_MONITOR: fleet_monitor_tools,
}


def _bound_handler(
    function: Callable[..., dict], dispatcher: Callable[[Tool, dict], dict]
):
    def call(**arguments) -> dict:
        return deepcopy(function(**arguments, simulator_call=dispatcher))

    return call


class ToolGateway:
    def __init__(self, dispatcher: Callable[[Tool, dict], dict]):
        self._dispatcher = dispatcher

    def for_role(self, role: AgentType) -> ToolRegistry:
        """Return tools ready for direct invoke() or an LLM tool loop."""
        bound: ToolRegistry = {}
        for tool_id, template in agent_tools[role].items():
            assert isinstance(template, StructuredTool) and template.func is not None
            bound[tool_id] = template.model_copy(
                update={
                    "func": _bound_handler(template.func, self._dispatcher),
                    "args_schema": template.tool_call_schema,
                }
            )
        return bound
