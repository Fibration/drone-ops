from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Protocol
from enum import Enum, auto

from src.map import Terrain


class DroneState(BaseModel):
    id: str
    position: tuple[int, int]
    battery: float
    status: str
    comms: bool
    assignment: str | None


class MissionStatus(Enum):
    ONGOING = "ongoing"
    SUCCESS = "success"
    FAILED = "failed"


class ObjectiveType(Enum):
    HOLD = "hold"
    RETURN_TO_BASE = "return_to_base"
    SEARCH_SECTOR = "search_sector"
    GOTO = "goto"


class Sector(BaseModel):
    sector_id: str
    bounds: tuple[tuple[int, int], tuple[int, int]]


class Objective(BaseModel):
    drone_id: str
    objective_type: ObjectiveType
    target_id: str | None = None
    sector_id: str | None = None


class Region(BaseModel):
    coordinates: list[tuple[int, int]]


class Target(BaseModel):
    target_id: str
    position: tuple[int, int]


class MissionEvent(BaseModel):
    description: str
    drones: dict[str, DroneState]


class MissionPlan(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    plan: dict[str, Objective]


class Plan(BaseModel):
    plan: dict[str, Objective]
    reason: str
    estimated_completion_times: int


class AgentType(str, Enum):
    FLEET_MONITOR = "FLEET_MONITOR"
    MISSION_COMMANDER = "MISSION_COMMANDER"
    MISSION_PLANNER = "MISSION_PLANNER"
    SAFETY_REVIEWER = "SAFETY_REVIEWER"


class MissionState(BaseModel):
    drones: dict[str, DroneState]
    objectives: list[Objective]
    no_fly_zones: list[Region]
    discovered_targets: dict[str, Target]
    events: list[MissionEvent]
    # Events returned together by the latest simulator step; events holds history.
    latest_events: list[MissionEvent] = Field(default_factory=list)
    explored_cells: set[tuple[int, int]]
    completed_sectors: set[str] = Field(default_factory=set)
    telemetry: dict[str, str | int | float]

    current_plan: MissionPlan | None
    proposed_plans: list[Plan]
    under_review: Plan | None
    needs_replan: bool = True

    next_agent: AgentType | None
    safety_status: str | None
    human_approval: bool | None

    step: int
    mission_status: MissionStatus


# class CommanderDecision(BaseModel):
#     next_agent: Literal["fleet_monitor", "mission_planner", "safety_reviewer", "finish"]
#     objective: str


class CommanderObservation(BaseModel):
    mission_state: MissionState
    world: list[list[Terrain]]
    sectors: dict[str, Sector]


class CommanderDecision(BaseModel):
    objectives: list[Objective]
    reason: str


class SafetyDecision(BaseModel):
    outcome: Literal["approve", "revise", "human_approval"]
    violations: list[str]


class IncidentType(Enum):
    NEW_NO_FLY_ZONE = auto()
    BATTERY_DRAIN = auto()
    COMMS_FAILURE = auto()




class Agent(Protocol):
    def run(self, state: MissionState) -> dict: ...
