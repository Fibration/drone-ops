from pydantic import BaseModel, ConfigDict, Field
from typing import Literal, Protocol
from enum import Enum, auto


class Terrain(Enum):
    EMPTY = 0
    DRONE = 1
    OBSTACLE = 2
    NO_FLY_ZONE = 3
    TARGET = 4


class DroneState(BaseModel):
    id: str
    position: tuple[int, int]
    battery: float
    status: str
    comms: bool
    assignment: str | None


class TelemetryField(str, Enum):
    BATTERY_LEVEL = "battery_level"
    POSITION = "position"
    COMMS = "comms"
    STATUS = "status"
    ASSIGNMENT = "assignment"


class TelemetryRequest(BaseModel):
    field: TelemetryField
    drone_ids: list[str] | None = None
    at_step: int | None = Field(default=None, ge=0)
    history: bool = False


class TelemetryReading(BaseModel):
    drone_id: str
    step: int
    field: TelemetryField
    value: bool | float | tuple[int, int] | str | None


class TelemetryResult(BaseModel):
    request: TelemetryRequest
    readings: dict[str, list[TelemetryReading]]


class TelemetryCapability(BaseModel):
    field: TelemetryField
    description: str
    history_available: bool = True


class TelemetrySnapshot(BaseModel):
    step: int
    drones: dict[str, DroneState]


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
    model_config = ConfigDict(extra="forbid")

    description: str
    step: int = 0


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
    model_config = ConfigDict(extra="forbid")

    drone_ids: list[str]
    objectives: list[Objective]
    no_fly_zones: list[Region]
    discovered_targets: dict[str, Target]
    events: list[MissionEvent]
    # Events returned together by the latest simulator step; events holds history.
    latest_events: list[MissionEvent] = Field(default_factory=list)
    explored_cells: set[tuple[int, int]]
    completed_sectors: set[str] = Field(default_factory=set)
    telemetry: list[TelemetryResult] = Field(default_factory=list)
    available_telemetry: list[TelemetryCapability] = Field(default_factory=list)
    telemetry_requests: list[TelemetryRequest] = Field(default_factory=list)
    new_target_ids: set[str] = Field(default_factory=set)

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


type Point = tuple[int, int]
