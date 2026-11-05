from pydantic import BaseModel
from typing import Literal
from enum import Enum


class DroneState(BaseModel):
    id: str
    position: tuple[int, int]
    battery: float
    status: str
    comms: bool
    assignment: str | None
    route: list[tuple[int, int]]


class MissionStatus(Enum):
    ONGOING = "ongoing"
    SUCCESS = "success"
    FAILED = "failed"


class Objective(BaseModel):
    description: str


class Region(BaseModel):
    coordinates: list[tuple[int, int]]


class Target(BaseModel):
    position: tuple[int, int]


class MissionEvent(BaseModel):
    description: str
    drones: dict[str, DroneState]


class MissionPlan(BaseModel):
    plan: dict[str, list[tuple[int, int]]]


class Action(BaseModel):
    action_type: str
    details: str


class MissionState(BaseModel):
    drones: dict[str, DroneState]
    objectives: list[Objective]
    no_fly_zones: list[Region]
    discovered_targets: list[Target]
    events: list[MissionEvent]

    current_plan: MissionPlan | None
    proposed_action: Action | None

    next_agent: str | None
    safety_status: str | None
    human_approval: bool | None

    step: int
    mission_status: MissionStatus


class CommanderDecision(BaseModel):
    next_agent: Literal["fleet_monitor", "mission_planner", "safety_reviewer", "finish"]
    objective: str


class SafetyDecision(BaseModel):
    outcome: Literal["approve", "revise", "human_approval"]
    violations: list[str]
