import random
from collections.abc import Callable

from src.drone import DroneController
from src.map import no_fly_zone_from_event
from src.models import (
    DroneState,
    IncidentType,
    Objective,
    ObjectiveType,
    Point,
    MonitoringPlan,
    ToolObservation,
    ToolObservationDict,
    WatchPredicate,
    WatchType,
    FreeWatchType,
    WatchAlert,
    MissionStatus,
    Tool,
    ToolService,
)
from src.scenario import Scenario, Weather, World
from src.watch import (
    WatchSample,
    WatchContext,
    check_free_watch,
    check_watch,
    estimate_route_cost,
    reachable_cells,
)


class Simulator:
    def __init__(
        self,
        world: World,
        scenario: Scenario,
        *,
        monitoring_plan: MonitoringPlan | None = None,
        on_alert: Callable[[list[WatchAlert]], None] | None = None,
        tool_service: ToolService | None = None,
    ):
        self.world = world
        self.scenario = scenario
        self.tool_service = tool_service
        self._population_estimates: dict[str, float] | None = None
        self.timestep = 0
        self.explored_cells: set[Point] = set()
        self.discovered_targets: dict[str, Point] = {}
        self.known_targets: dict[str, Point] = {}
        self.acquired_targets: set[str] = set()
        self._coverage_starts = tuple(
            {world.base_position, *(d.position for d in world.drones.values())}
        )
        self._coverage_signature: tuple | None = None
        self._coverage_cells: set[Point] = set()
        self.completed_sectors: set[str] = set()
        self.controllers = {
            drone_id: DroneController(drone_id, world) for drone_id in world.drones
        }
        self._watch_history: list[WatchSample] = []
        self._monitoring_plan: MonitoringPlan | None = None
        self.triggered_watches: list[WatchPredicate] = []
        self._active_watch_keys: set[str] = set()
        self._active_free_watch_keys: set[tuple[FreeWatchType, str | None]] = set()
        self._new_detections: dict[str, str] = {}
        self.alerts: list[WatchAlert] = []
        self._baseline_route_costs: dict[str, float | None] = {}
        self.on_alert = on_alert
        if monitoring_plan is not None:
            self.set_monitoring_plan(monitoring_plan)
        self._record_watch_sample()

    def observation_view(self) -> ToolObservationDict:
        """Detached tool inputs; hidden target positions and decay rates are excluded."""
        if self._population_estimates is None:
            rates = {sector.sector_id: sector.population_rate for sector in self.world.sectors}
            if any(rate < 0 for rate in rates.values()):
                raise ValueError("Population rates must be nonnegative")
            total_rate = sum(rates.values())
            noise = 0.05 * (self.scenario.difficulty.value - 1)
            self._population_estimates = {
                sector_id: self.scenario.num_targets
                * (rate / total_rate if total_rate else 1 / len(rates))
                * (1 + noise * (2 * random.random() - 1) if noise else 1)
                for sector_id, rate in rates.items()
            }
        return {
            ToolObservation.POPULATION_ESTIMATES: dict(self._population_estimates),
            ToolObservation.NAVIGATION_MAP: self.world.navigation_map(None),
            ToolObservation.DRONE_POSITIONS: {
                drone_id: drone.position
                for drone_id, drone in self.world.drones.items()
            },
            ToolObservation.DRONE_BATTERIES: {
                drone_id: drone.battery
                for drone_id, drone in self.world.drones.items()
            },
            ToolObservation.STEP: self.timestep,
            ToolObservation.BASE_POSITION: self.world.base_position,
            ToolObservation.DRONE_IDS: list(self.drones),
            ToolObservation.KNOWN_TARGETS: dict(self.known_targets),
            ToolObservation.ACQUIRED_TARGETS: set(self.acquired_targets),
            ToolObservation.EXPLORED_CELLS: set(self.explored_cells),
            ToolObservation.COMPLETED_SECTORS: set(self.completed_sectors),
            ToolObservation.SECTOR_BOUNDS: {
                sector.sector_id: sector.bounds for sector in self.world.sectors
            },
            ToolObservation.WEATHER: {
                sector.sector_id: sector.weather for sector in self.world.sectors
            },
            ToolObservation.MOVEMENT_COSTS: {
                sector.sector_id: sector.movement_cost for sector in self.world.sectors
            },
        }

    def call_tool(self, tool: Tool, arguments: dict) -> dict:
        if self.tool_service is None:
            raise RuntimeError("No tool service configured for this simulator")
        return self.tool_service.call(tool, arguments, self.observation_view())

    @property
    def monitoring_plan(self) -> MonitoringPlan | None:
        return (
            self._monitoring_plan.model_copy(deep=True)
            if self._monitoring_plan
            else None
        )

    def set_monitoring_plan(self, plan: MonitoringPlan) -> None:
        """Arm exactly the three selected watches; baseline the next accepted route."""
        self._monitoring_plan = MonitoringPlan.model_validate(
            plan.model_dump()
        ).model_copy(deep=True)
        self._active_watch_keys.clear()
        self._baseline_route_costs.clear()

    def _route_cost(self, drone_id: str) -> float | None:
        controller = self.controllers[drone_id]
        objective = controller.objective
        if objective.objective_type == ObjectiveType.GOTO:
            target_id = objective.target_id
            assert target_id is not None
            if (
                target_id in self.world.successful_targets
                or target_id in self.world.failed_targets
            ):
                return 0.0
            destination = self.discovered_targets.get(target_id)
            if destination is None:
                return None
        elif objective.objective_type == ObjectiveType.RETURN_TO_BASE:
            destination = self.world.base_position
        elif controller.failure_event is not None:
            return None
        else:
            destination = controller.position
            for dx, dy in controller.route[controller.route_index :]:
                destination = destination[0] + dx, destination[1] + dy
        return estimate_route_cost(self.world, drone_id, destination)

    def watch_context(self) -> WatchContext:
        signature = (
            tuple(s.bounds for s in self.world.sectors),
            frozenset(self.world.obstacles),
            frozenset(self.world.no_fly),
            self.world.base_position,
        )
        if signature != self._coverage_signature:
            self._coverage_cells = reachable_cells(
                self.world,
                (*self._coverage_starts, self.world.base_position),
            )
            self._coverage_signature = signature
        route_ids = {
            w.drone_id
            for w in (self._monitoring_plan.watches if self._monitoring_plan else ())
            if w.type
            in (WatchType.ROUTE_UNREACHABLE, WatchType.ROUTE_COST_INCREASED_BY)
            and w.drone_id is not None
        }
        route_costs = {drone_id: self._route_cost(drone_id) for drone_id in route_ids}
        for drone_id, cost in route_costs.items():
            self._baseline_route_costs.setdefault(drone_id, cost)
        context = WatchContext(
            world=self.world,
            step=self.timestep,
            history=self._watch_history,
            completed_sectors=self.completed_sectors,
            route_costs=route_costs,
            baseline_route_costs=self._baseline_route_costs,
            scenario_weather=self.scenario.weather,
            new_target_ids=self._new_detections.keys(),
            explored_cells=self.explored_cells,
            coverage_cells=self._coverage_cells,
            known_target_ids=self.known_targets.keys(),
            acquired_target_ids=self.acquired_targets,
        )
        return context

    def _collect_alerts(self) -> list[WatchAlert]:
        context = self.watch_context()
        matches = (
            [w for w in self._monitoring_plan.watches if check_watch(w, context)]
            if self._monitoring_plan
            else []
        )
        self.triggered_watches = [
            w for w in matches if w.model_dump_json() not in self._active_watch_keys
        ]
        self._active_watch_keys = {w.model_dump_json() for w in matches}
        alerts = [
            WatchAlert(
                type=w.type,
                step=self.timestep,
                drone_id=w.drone_id,
                sector_id=w.sector_id,
                target_id=w.target_id,
                predicate=w.model_copy(deep=True),
            )
            for w in self.triggered_watches
        ]

        free_matches: dict[tuple[FreeWatchType, str | None], WatchAlert] = {}
        for target_id, position in self.world.targets.items():
            if check_free_watch(
                FreeWatchType.NEW_TARGET_DETECTED, context, target_id=target_id
            ):
                free_matches[FreeWatchType.NEW_TARGET_DETECTED, target_id] = WatchAlert(
                    type=FreeWatchType.NEW_TARGET_DETECTED,
                    step=self.timestep,
                    target_id=target_id,
                    position=position,
                    drone_id=self._new_detections[target_id],
                )
        for drone_id, drone in self.drones.items():
            if check_free_watch(
                FreeWatchType.DRONE_HARD_FAILURE, context, drone_id=drone_id
            ):
                free_matches[FreeWatchType.DRONE_HARD_FAILURE, drone_id] = WatchAlert(
                    type=FreeWatchType.DRONE_HARD_FAILURE,
                    step=self.timestep,
                    drone_id=drone_id,
                    position=drone.position,
                )
        if check_free_watch(FreeWatchType.MISSION_COMPLETE, context):
            free_matches[FreeWatchType.MISSION_COMPLETE, None] = WatchAlert(
                type=FreeWatchType.MISSION_COMPLETE,
                step=self.timestep,
                mission_status=MissionStatus.SUCCESS,
            )
        alerts.extend(
            alert
            for key, alert in free_matches.items()
            if key not in self._active_free_watch_keys
        )
        self._active_free_watch_keys = set(free_matches)
        self.alerts = alerts
        if alerts and self.on_alert is not None:
            self.on_alert([alert.model_copy(deep=True) for alert in alerts])
        return alerts

    @property
    def drones(self) -> dict[str, DroneState]:
        return self.world.drones

    def _record_watch_sample(self) -> None:
        sample = WatchSample(
            step=self.timestep,
            battery={
                drone_id: drone.battery for drone_id, drone in self.drones.items()
            },
            comms={drone_id: drone.comms for drone_id, drone in self.drones.items()},
        )
        if self._watch_history and self._watch_history[-1].step == self.timestep:
            self._watch_history[-1] = sample
        else:
            self._watch_history.append(sample)

    def _observe(self, moved: set[str] | None = None) -> None:
        for drone_id, drone in self.drones.items():
            if drone.battery <= 0 and drone_id not in (moved or ()):
                continue
            sector = self.world.sector_at(drone.position)
            if sector is None:
                continue
            visible = sector.visible_cells(drone.position)
            # A disconnected drone keeps flying but cannot report observations.
            if drone.comms:
                self.explored_cells.update(visible)
            for target_id, position in self.world.iter_active_targets():
                if (
                    drone.comms
                    and position in visible
                    and target_id not in self.discovered_targets
                ):
                    self.discovered_targets[target_id] = position
                    self.known_targets[target_id] = position
                    self._new_detections[target_id] = drone_id

    def _update_completed_sectors(self) -> None:
        blocked = self.world.obstacles | self.world.no_fly
        for sector in self.world.sectors:
            if sector.sector_id in self.completed_sectors:
                continue
            (x0, y0), (x1, y1) = sector.bounds
            if all(
                (x, y) in self.explored_cells or (x, y) in blocked
                for x in range(x0, x1 + 1)
                for y in range(y0, y1 + 1)
            ):
                self.completed_sectors.add(sector.sector_id)

    def _apply_incident(self) -> None:
        incident = self.scenario.events.get(self.timestep)
        if incident is None:
            return
        incident_type, payload = incident
        if incident_type == IncidentType.NEW_NO_FLY_ZONE:
            zone = no_fly_zone_from_event(payload)
            if zone is None or any(self.world.sector_at(cell) is None for cell in zone):
                raise ValueError("No-fly zone must describe cells within the world")
            self.world.no_fly.update(zone)
            return
        drone = self.drones[payload]
        if incident_type == IncidentType.BATTERY_DRAIN:
            drone.battery = max(0.0, drone.battery - 10)
        elif incident_type == IncidentType.COMMS_FAILURE:
            drone.comms = False

    def step(
        self,
        plan: dict[str, Objective] | None = None,
    ) -> list[WatchAlert]:
        """Advance one turn. None keeps accepted orders; an empty plan holds all drones.

        Movement uses the destination sector's cost and current weather. Rescues
        occur before target decay. Weather evolves each turn. Drones ending a
        turn at base recharge, capped at 100 battery units.
        """
        self.triggered_watches = []
        self.alerts = []
        self._new_detections = {}
        if plan is not None:
            unknown = set(plan) - self.drones.keys()
            if unknown:
                raise ValueError(f"Unknown drone IDs: {sorted(unknown)}")
            for drone_id, objective in plan.items():
                if objective.drone_id != drone_id:
                    raise ValueError("Plan key must match the objective's drone ID")
        self._observe()
        self.timestep += 1
        self._apply_incident()
        self.world.update_sector_weather(self.scenario.weather)

        moved: set[str] = set()
        for drone_id, controller in self.controllers.items():
            drone = self.drones[drone_id]
            if drone.battery <= 0:
                controller.update_objective(
                    Objective(drone_id=drone_id, objective_type=ObjectiveType.HOLD)
                )
                continue
            if drone.comms:
                controller.explored_cells = self.explored_cells
                controller.targets = self.discovered_targets
                if plan is not None:
                    controller.update_objective(
                        plan.get(drone_id)
                        or Objective(
                            drone_id=drone_id, objective_type=ObjectiveType.HOLD
                        )
                    )
                elif controller.objective.objective_type != ObjectiveType.SEARCH_SECTOR:
                    controller.calculate_route()
                drone.assignment = controller.objective.objective_type.value
            relative = controller.plan_step()
            if (
                self._monitoring_plan is not None
                and drone_id not in self._baseline_route_costs
                and any(
                    w.type == WatchType.ROUTE_COST_INCREASED_BY
                    and w.drone_id == drone_id
                    for w in self._monitoring_plan.watches
                )
            ):
                self._baseline_route_costs[drone_id] = self._route_cost(drone_id)
            if controller.failure_event is not None:
                continue
            if relative is None:
                continue
            position = drone.position[0] + relative[0], drone.position[1] + relative[1]
            sector = self.world.sector_at(position)
            occupied = position != self.world.base_position and any(
                other_id != drone_id and other.position == position
                for other_id, other in self.drones.items()
            )
            if (
                sector is None
                or position in self.world.obstacles
                or position in self.world.no_fly
                or occupied
            ):
                # Discard the uncommitted move; connected drones can reroute next turn.
                controller.clear_route()
                continue
            controller.apply_step(position)
            drone.battery = max(0.0, sector.move(relative, drone.battery))
            moved.add(drone_id)

        for drone_id, drone in self.drones.items():
            if drone.position == self.world.base_position:
                drone.battery = min(100.0, drone.battery + self.scenario.recharge_rate)
            elif drone.comms and drone.battery > 0:
                sector = self.world.sector_at(drone.position)
                comms_failure_probability = (
                    {
                        Weather.STORMY: 0.01,
                        Weather.BLIZZARD: 0.02,
                    }.get(sector.weather, 0.0)
                    if sector is not None
                    else 0.0
                )
                if (
                    comms_failure_probability
                    and random.random() < comms_failure_probability
                ):
                    drone.comms = False

        # A drone's last battery-powered move can still complete a rescue.
        for drone_id, drone in self.drones.items():
            if drone.battery <= 0 and drone_id not in moved:
                continue
            for target_id, position in list(self.world.iter_active_targets()):
                if position == drone.position:
                    self.world.update_target_success(target_id, self.timestep)
                    if drone.comms:
                        self.known_targets[target_id] = position
                        self.acquired_targets.add(target_id)
                    self.discovered_targets.pop(target_id, None)
        self._observe(moved)
        previous_failures = len(self.world.failed_targets)
        self.world.update_active_targets()
        for target_id in self.world.failed_targets[previous_failures:]:
            self.discovered_targets.pop(target_id, None)
        self._update_completed_sectors()
        self._record_watch_sample()
        return self._collect_alerts()

    def execute_plan(
        self,
        plan: dict[str, Objective],
        *,
        max_steps: int = 80,
    ) -> tuple[list[WatchAlert], set[Point], int]:
        """Run until an alert or the turn limit.

        Only the selected predicates and always-on free watches invoke the agent
        loop. Reaching max_steps returns an empty alert batch.
        """
        if self._monitoring_plan is None:
            raise ValueError("execute_plan requires a MonitoringPlan")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        for index in range(max_steps):
            alerts = self.step(plan if index == 0 else None)
            if alerts:
                return alerts, self.explored_cells.copy(), self.timestep
        return [], self.explored_cells.copy(), self.timestep
