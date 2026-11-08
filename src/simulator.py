from src.drone import DroneController
from src.map import no_fly_zone_from_event
from src.models import IncidentType, Objective, ObjectiveType, Point, TelemetrySnapshot
from src.parameters import monitoring_interval
from src.scenario import Scenario, World
from src.telemetry import RecordedTelemetry


class Simulator:
    def __init__(self, world: World, scenario: Scenario):
        self.world = world
        self.scenario = scenario
        self.timestep = 0
        self.explored_cells: set[Point] = set()
        self.discovered_targets: dict[str, Point] = {}
        self.completed_sectors: set[str] = set()
        self.controllers = {
            drone_id: DroneController(drone_id, world) for drone_id in world.drones
        }
        self._reported_empty_batteries: set[str] = set()
        self.telemetry_history: list[TelemetrySnapshot] = []
        self.telemetry_queries = RecordedTelemetry(self.telemetry_history)
        self._record_telemetry()

    @property
    def drones(self):
        return self.world.drones

    def _record_telemetry(self) -> None:
        snapshot = TelemetrySnapshot(step=self.timestep, drones={
            drone_id: drone.model_copy(deep=True) for drone_id, drone in self.drones.items()
        })
        if self.telemetry_history and self.telemetry_history[-1].step == self.timestep:
            self.telemetry_history[-1] = snapshot
        else:
            self.telemetry_history.append(snapshot)

    def _observe(self, moved: set[str] | None = None) -> list[str]:
        events = []
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
                if drone.comms and position in visible and target_id not in self.discovered_targets:
                    self.discovered_targets[target_id] = position
                    events.append(f"Drone {drone_id} detected target {target_id} at {position}")
        return events

    def _sector_completion_events(self) -> list[str]:
        events = []
        blocked = self.world.obstacles | self.world.no_fly
        for sector in self.world.sectors:
            if sector.sector_id in self.completed_sectors:
                continue
            (x0, y0), (x1, y1) = sector.bounds
            if all((x, y) in self.explored_cells or (x, y) in blocked
                   for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)):
                self.completed_sectors.add(sector.sector_id)
                events.append(f"Sector {sector.sector_id} fully explored")
        return events

    def _apply_incident(self) -> list[str]:
        incident = self.scenario.events.get(self.timestep)
        if incident is None:
            return []
        incident_type, payload = incident
        if incident_type == IncidentType.NEW_NO_FLY_ZONE:
            zone = no_fly_zone_from_event(payload)
            if zone is None or any(self.world.sector_at(cell) is None for cell in zone):
                raise ValueError("No-fly zone must describe cells within the world")
            self.world.no_fly.update(zone)
            return [payload]
        drone = self.drones[payload]
        if incident_type == IncidentType.BATTERY_DRAIN:
            drone.battery = max(0.0, drone.battery - 10)
        elif incident_type == IncidentType.COMMS_FAILURE:
            drone.comms = False
        return []

    def _monitor_events(self, battery_reserve_threshold: float) -> list[str]:
        events = []
        for drone_id, drone in self.drones.items():
            if drone.battery <= 0 and drone_id not in self._reported_empty_batteries:
                self._reported_empty_batteries.add(drone_id)
                events.append(f"Drone {drone_id} ran out of battery at {drone.position}")
            if self.timestep % monitoring_interval == 0:
                if 0 < drone.battery < battery_reserve_threshold:
                    events.append(f"Drone {drone_id} battery low {drone.battery} at {drone.position}")
                if not drone.comms:
                    events.append(f"Drone {drone_id} lost communication at {drone.position}")
        return events

    def step(
        self,
        plan: dict[str, Objective] | None = None,
        monitoring_checks: dict[str, float] | None = None,
    ) -> list[str]:
        """Advance one turn. None keeps accepted orders; an empty plan holds all drones.

        Movement uses the destination sector's cost and current weather. Rescues
        occur before target decay. Weather evolves each turn.
        """
        if plan is not None:
            unknown = set(plan) - self.drones.keys()
            if unknown:
                raise ValueError(f"Unknown drone IDs: {sorted(unknown)}")
            for drone_id, objective in plan.items():
                if objective.drone_id != drone_id:
                    raise ValueError("Plan key must match the objective's drone ID")
        events = self._observe()
        self.timestep += 1
        events.extend(self._apply_incident())
        self.world.update_sector_weather(self.scenario.weather)

        moved: set[str] = set()
        for drone_id, controller in self.controllers.items():
            drone = self.drones[drone_id]
            if drone.battery <= 0:
                controller.update_objective(Objective(drone_id=drone_id, objective_type=ObjectiveType.HOLD))
                continue
            if drone.comms:
                controller.explored_cells = self.explored_cells
                controller.targets = self.discovered_targets
                if plan is not None:
                    controller.update_objective(plan.get(drone_id) or Objective(
                        drone_id=drone_id, objective_type=ObjectiveType.HOLD
                    ))
                elif controller.objective.objective_type != ObjectiveType.SEARCH_SECTOR:
                    controller.calculate_route()
                drone.assignment = controller.objective.objective_type.value
            relative = controller.plan_step()
            if controller.failure_event is not None:
                events.append(controller.failure_event)
                continue
            if relative is None:
                continue
            position = drone.position[0] + relative[0], drone.position[1] + relative[1]
            sector = self.world.sector_at(position)
            occupied = position != self.world.base_position and any(
                other_id != drone_id and other.position == position
                for other_id, other in self.drones.items()
            )
            if (sector is None or position in self.world.obstacles
                    or position in self.world.no_fly or occupied):
                events.append(f"Drone {drone_id} route blocked at {position}")
                # Discard the uncommitted move; connected drones can reroute next turn.
                controller.clear_route()
                continue
            controller.apply_step(position)
            drone.battery = max(0.0, sector.move(drone.battery))
            moved.add(drone_id)
            if (controller.objective.objective_type == ObjectiveType.RETURN_TO_BASE
                    and position == self.world.base_position):
                events.append(f"Drone {drone_id} returned to base at {position}")

        # A drone's last battery-powered move can still complete a rescue.
        for drone_id, drone in self.drones.items():
            if drone.battery <= 0 and drone_id not in moved:
                continue
            for target_id, position in list(self.world.iter_active_targets()):
                if position == drone.position:
                    self.world.update_target_success(target_id, self.timestep)
                    self.discovered_targets.pop(target_id, None)
                    events.append(f"Drone {drone_id} rescued target {target_id} at {position}")
        events.extend(self._observe(moved))
        previous_failures = len(self.world.failed_targets)
        self.world.update_active_targets()
        for target_id in self.world.failed_targets[previous_failures:]:
            self.discovered_targets.pop(target_id, None)
            events.append(f"Target {target_id} expired at {self.world.targets[target_id]}")
        events.extend(self._sector_completion_events())
        threshold = (monitoring_checks or {}).get("battery_reserve_threshold", 10.0)
        events.extend(self._monitor_events(threshold))
        if not moved and not events:
            events.append("Mission plan exhausted")
        self._record_telemetry()
        return events

    def execute_plan(
        self,
        plan: dict[str, Objective],
        monitoring_checks: dict[str, float] | None = None,
        *,
        max_steps: int = 80,
    ) -> tuple[list[str], set[Point], int]:
        """Run until a turn produces events or max_steps is reached."""
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        for index in range(max_steps):
            events = self.step(plan if index == 0 else None, monitoring_checks)
            if events:
                return events, self.explored_cells.copy(), self.timestep
        return ["Simulation turn limit reached"], self.explored_cells.copy(), self.timestep
