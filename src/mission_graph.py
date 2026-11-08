from collections.abc import Mapping
from typing import Any, Literal, Protocol

from langgraph.graph import START, END, StateGraph

from src.agent import FleetMonitor, MissionCommander, MissionPlanner, SafetyReviewer
from src.policy import FleetMonitorPolicy
from src.models import (
    AgentType,
    MissionPlan,
    MissionState,
    MissionStatus,
    ReviewOutcome,
)

HUMAN_REVIEW_NODE: Literal["human_reviewer"] = "human_reviewer"
NodeName = AgentType | Literal["human_reviewer"]
StateUpdate = dict[str, Any]


class NodeHandler(Protocol):
    def __call__(self, state: MissionState) -> StateUpdate: ...


def fleet_monitor(state: MissionState) -> StateUpdate:
    return FleetMonitor().run(state)


def mission_commander(state: MissionState) -> StateUpdate:
    return MissionCommander().run(state)


def mission_planner(state: MissionState) -> StateUpdate:
    return MissionPlanner().run(state)


def safety_reviewer(state: MissionState) -> StateUpdate:
    return SafetyReviewer().run(state)


def human_reviewer(state: MissionState) -> StateUpdate:
    assert state.under_review is not None
    human_decision = input(
        f"Plan: {state.under_review.model_dump()}\n"
        f"Safety review: {state.safety_status}\n"
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
    if not state.needs_replan and state.current_plan is not None:
        return AgentType.FLEET_MONITOR
    destination = state.next_agent
    if destination is None or destination not in {
        AgentType.MISSION_PLANNER,
        AgentType.SAFETY_REVIEWER,
        HUMAN_REVIEW_NODE,
    }:
        raise ValueError(f"Invalid commander destination: {destination!r}")
    return destination


def route_safety_review(state: MissionState) -> str:
    routes: dict[str, str] = {
        ReviewOutcome.APPROVE: AgentType.FLEET_MONITOR,
        ReviewOutcome.REPLAN: AgentType.MISSION_COMMANDER,
        ReviewOutcome.RECONSIDER_PRIORITY: AgentType.MISSION_COMMANDER,
    }
    outcome = state.safety_status
    if outcome is None or outcome.outcome not in routes:
        raise ValueError(f"Invalid safety review outcome: {outcome!r}")
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
    fleet_monitor_policy: FleetMonitorPolicy | None = None,
) -> StateGraph[MissionState]:
    """Build the README workflow, optionally supplying concrete node handlers."""
    handlers: dict[NodeName, NodeHandler] = {
        AgentType.FLEET_MONITOR: FleetMonitor(policy=fleet_monitor_policy).run,
        AgentType.MISSION_COMMANDER: mission_commander,
        AgentType.MISSION_PLANNER: mission_planner,
        AgentType.SAFETY_REVIEWER: safety_reviewer,
        HUMAN_REVIEW_NODE: human_reviewer,
    }
    if nodes is not None:
        unknown = nodes.keys() - handlers.keys()
        if unknown:
            raise ValueError(f"Unknown mission graph nodes: {sorted(unknown)}")
        handlers.update(nodes)

    builder = StateGraph(MissionState)
    for name, handler in handlers.items():
        builder.add_node(name.value if isinstance(name, AgentType) else name, handler)
    builder.add_edge(START, AgentType.MISSION_COMMANDER.value)
    builder.add_conditional_edges(
        AgentType.MISSION_COMMANDER.value,
        route_commander,
        {
            AgentType.FLEET_MONITOR: AgentType.FLEET_MONITOR.value,
            AgentType.MISSION_PLANNER: AgentType.MISSION_PLANNER.value,
            AgentType.SAFETY_REVIEWER: AgentType.SAFETY_REVIEWER.value,
            HUMAN_REVIEW_NODE: HUMAN_REVIEW_NODE,
            END: END,
        },
    )
    builder.add_edge(AgentType.MISSION_PLANNER.value, AgentType.MISSION_COMMANDER.value)
    builder.add_conditional_edges(
        AgentType.SAFETY_REVIEWER.value,
        route_safety_review,
        {
            AgentType.FLEET_MONITOR: AgentType.FLEET_MONITOR.value,
            AgentType.MISSION_COMMANDER: AgentType.MISSION_COMMANDER.value,
        },
    )
    builder.add_conditional_edges(
        HUMAN_REVIEW_NODE,
        route_human_review,
        {
            AgentType.FLEET_MONITOR: AgentType.FLEET_MONITOR.value,
            AgentType.MISSION_COMMANDER: AgentType.MISSION_COMMANDER.value,
        },
    )
    builder.add_edge(AgentType.FLEET_MONITOR.value, END)
    return builder


graph = build_mission_graph().compile()


if __name__ == "__main__":
    print(f"```mermaid\n{graph.get_graph().draw_mermaid().strip()}\n```")
