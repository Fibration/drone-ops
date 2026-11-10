from collections.abc import Callable, Mapping
from typing import Any, Literal

from langgraph.graph import START, END, StateGraph
from langgraph.runtime import Runtime

from src.models import (
    AgentContext,
    ToolRegistry,
    Agent,
    AgentType,
    MissionPlan,
    MissionState,
    MissionStatus,
    ReviewOutcome,
)

HUMAN_REVIEW_NODE: Literal["human_reviewer"] = "human_reviewer"
NodeName = AgentType | Literal["human_reviewer"]
StateUpdate = dict[str, Any]


NodeHandler = Callable[..., StateUpdate]


def _role_tools(runtime: Runtime[AgentContext], role: AgentType) -> ToolRegistry | None:
    return runtime.context.tools.for_role(role) if runtime.context else None


def _agent_handler(role: AgentType, agent: Agent) -> NodeHandler:
    def run(state: MissionState, runtime: Runtime[AgentContext]) -> StateUpdate:
        return agent.run(state, tools=_role_tools(runtime, role))
    return run


def human_reviewer(state: MissionState) -> StateUpdate:
    assert state.under_review is not None
    human_decision = input(
        f"Plan: {state.under_review.model_dump()}\n"
        f"Review: {state.review_status}\n"
        f"Commander assessment: {state.commander_rethink}\nApprove [y/n]?"
    )
    if human_decision == "y":
        return {
            "human_approval": True,
            "current_plan": MissionPlan(plan=state.under_review.plan),
            "monitoring_plan": None,
            "needs_replan": False,
        }
    elif human_decision == "n":
        return {"human_approval": False, "needs_replan": True}
    else:
        raise ValueError("Invalid human reviewer decision.")


def route_commander(state: MissionState) -> str:
    if state.mission_status != MissionStatus.ONGOING:
        return END
    destination = state.next_agent
    if destination is None or destination not in {
        AgentType.MISSION_PLANNER,
        AgentType.REVIEWER,
        HUMAN_REVIEW_NODE,
    }:
        raise ValueError(f"Invalid commander destination: {destination!r}")
    return destination


def route_review(state: MissionState) -> str:
    routes: dict[str, str] = {
        ReviewOutcome.APPROVE: AgentType.FLEET_MONITOR,
        ReviewOutcome.REPLAN: AgentType.MISSION_COMMANDER,
        ReviewOutcome.RECONSIDER_PRIORITY: AgentType.MISSION_COMMANDER,
    }
    outcome = state.review_status
    if outcome is None or outcome.outcome not in routes:
        raise ValueError(f"Invalid review outcome: {outcome!r}")
    return routes[outcome.outcome]


def route_human_review(state: MissionState) -> str:
    if state.human_approval is True:
        return AgentType.FLEET_MONITOR
    if state.human_approval is False:
        return AgentType.MISSION_COMMANDER
    raise ValueError(f"Invalid human review outcome: {state.human_approval!r}")


def build_mission_graph(
    nodes: Mapping[NodeName, NodeHandler] | None = None,
    *,
    agents: Mapping[AgentType, Agent] | None = None,
) -> StateGraph[MissionState, AgentContext]:
    """Build the workflow with optional agent instances and overriding node handlers."""
    handlers: dict[NodeName, NodeHandler] = {HUMAN_REVIEW_NODE: human_reviewer}
    if agents is not None:
        for role, agent in agents.items():
            if role not in AgentType:
                raise ValueError(f'Unknown agent role: {role!r}')
            handlers[role] = _agent_handler(role, agent)
    if nodes is not None:
        unknown = nodes.keys() - (set(AgentType) | {HUMAN_REVIEW_NODE})
        if unknown:
            raise ValueError(f"Unknown mission graph nodes: {sorted(unknown)}")
        handlers.update(nodes)

    missing = set(AgentType) - handlers.keys()
    if missing:
        raise ValueError(f"Configure an agent or node handler for every role; missing: {sorted(missing)}")

    builder = StateGraph(MissionState, context_schema=AgentContext)
    for name, handler in handlers.items():
        builder.add_node(name.value if isinstance(name, AgentType) else name, handler)
    builder.add_edge(START, AgentType.MISSION_COMMANDER.value)
    builder.add_conditional_edges(
        AgentType.MISSION_COMMANDER.value,
        route_commander,
        {
            AgentType.MISSION_PLANNER.value: AgentType.MISSION_PLANNER.value,
            AgentType.REVIEWER.value: AgentType.REVIEWER.value,
            HUMAN_REVIEW_NODE: HUMAN_REVIEW_NODE,
            END: END,
        },
    )
    builder.add_edge(AgentType.MISSION_PLANNER.value, AgentType.MISSION_COMMANDER.value)
    builder.add_conditional_edges(
        AgentType.REVIEWER.value,
        route_review,
        {
            AgentType.FLEET_MONITOR.value: AgentType.FLEET_MONITOR.value,
            AgentType.MISSION_COMMANDER.value: AgentType.MISSION_COMMANDER.value,
        },
    )
    builder.add_conditional_edges(
        HUMAN_REVIEW_NODE,
        route_human_review,
        {
            AgentType.FLEET_MONITOR.value: AgentType.FLEET_MONITOR.value,
            AgentType.MISSION_COMMANDER.value: AgentType.MISSION_COMMANDER.value,
        },
    )
    builder.add_edge(AgentType.FLEET_MONITOR.value, END)
    return builder


if __name__ == "__main__":
    # Rendering topology does not require concrete decision policies.
    graph = build_mission_graph(nodes={role: lambda state: {} for role in AgentType}).compile()
    print(f"```mermaid\n{graph.get_graph().draw_mermaid(with_styles=False).strip()}\n```")
