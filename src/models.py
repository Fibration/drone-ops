from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from enum import Enum, auto


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


class Action(BaseModel):
    action_type: str
    details: str


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

    current_plan: MissionPlan | None
    proposed_action: Action | None

    next_agent: str | None
    safety_status: str | None
    human_approval: bool | None

    step: int
    mission_status: MissionStatus


# class CommanderDecision(BaseModel):
#     next_agent: Literal["fleet_monitor", "mission_planner", "safety_reviewer", "finish"]
#     objective: str


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
