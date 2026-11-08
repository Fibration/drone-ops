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


type Point = tuple[int, int]


class FreeWatchType(Enum):
    NEW_TARGET_DETECTED = "NEW_TARGET_DETECTED"
    DRONE_HARD_FAILURE = "DRONE_HARD_FAILURE"
    MISSION_COMPLETE = "MISSION_COMPLETE"


class WatchType(Enum):
    RETURN_RESERVE_BELOW = "RETURN_RESERVE_BELOW"
    BATTERY_DRAIN_ABOVE = "BATTERY_DRAIN_ABOVE"
    CHARGE_ABOVE = "CHARGE_ABOVE"
    COMMS_LOST_FOR = "COMMS_LOST_FOR"
    WEATHER_AT_LEAST = "WEATHER_AT_LEAST"
    WEATHER_FORECAST_RISK_ABOVE = "WEATHER_FORECAST_RISK_ABOVE"
    ROUTE_UNREACHABLE = "ROUTE_UNREACHABLE"
    ROUTE_COST_INCREASED_BY = "ROUTE_COST_INCREASED_BY"
    SECTOR_COMPLETE = "SECTOR_COMPLETE"
    TARGET_SURVIVAL_PROBABILITY_BELOW = "TARGET_SURVIVAL_PROBABILITY_BELOW"
    TARGET_NOT_REACHED_BY = "TARGET_NOT_REACHED_BY"


class WatchPredicate(BaseModel):
    type: WatchType

    drone_id: str | None = None
    sector_id: str | None = None
    target_id: str | None = None

    threshold: float | int | str | None = None
    horizon: int | None = None
    decay_rate_prior: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class WatchAlert(BaseModel):
    """One selected-watch or free-watch trigger, with its affected entity."""
    model_config = ConfigDict(extra="forbid")

    type: WatchType | FreeWatchType
    step: int = Field(ge=0)
    drone_id: str | None = None
    sector_id: str | None = None
    target_id: str | None = None
    position: Point | None = None
    predicate: WatchPredicate | None = None
    mission_status: MissionStatus | None = None


class MonitoringPlan(BaseModel):
    watches: list[WatchPredicate] = Field(min_length=3, max_length=3)
    reason: str = Field(min_length=1)


class ReviewOutcome(str, Enum):
    APPROVE = "approve"
    REPLAN = "replan"
    RECONSIDER_PRIORITY = "reconsider_priority"


class ReviewResult(BaseModel):
    outcome: ReviewOutcome
    critique: str
    concerns: list[str]
    mandatory_watches: list[WatchPredicate]


class CommanderReviewOutcome(Enum):
    REPLAN = "replan"
    HUMAN_REQUIRED = "human_required"


class MissionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drone_ids: list[str]
    objectives: list[Objective]
    no_fly_zones: list[Region]
    discovered_targets: dict[str, Target]
    events: list[WatchAlert]
    # Events returned together by the latest simulator step; events holds history.
    latest_events: list[WatchAlert] = Field(default_factory=list)
    explored_cells: set[tuple[int, int]]
    completed_sectors: set[str] = Field(default_factory=set)
    new_target_ids: set[str] = Field(default_factory=set)

    current_plan: MissionPlan | None
    monitoring_plan: MonitoringPlan | None = None
    proposed_plans: list[Plan]
    under_review: Plan | None
    needs_replan: bool = True

    next_agent: AgentType | Literal["human_reviewer"] | None
    safety_status: ReviewResult | None
    commander_rethink: CommanderReviewOutcome | None = None
    human_approval: bool | None

    step: int
    mission_status: MissionStatus


class Agent(Protocol):
    def run(self, state: MissionState) -> dict: ...


class CommanderObservation(BaseModel):
    mission_state: MissionState
    world: list[list[Terrain]]
    sectors: dict[str, Sector]
