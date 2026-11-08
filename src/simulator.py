from src.telemetry import RecordedTelemetry
from src.drone import DroneController
from src.models import DroneState, Objective, ObjectiveType, Sector, Target, IncidentType, TelemetrySnapshot
from src.map import Terrain, no_fly_zone_from_event
from src.parameters import BASE_POSITION, sensor_range, monitoring_interval


class Simulator:
    def __init__(
        self,
        map: list[list[Terrain]],
        drones: dict[str, DroneState],
        targets: list[tuple[int, int]],
        scenario: dict[int, tuple[IncidentType, str]] | None = None,
    ):
        self.map = map
        self.drones = drones
        self.timestep = 0
        self.targets = targets
        self.scenario = scenario or {}
        self._initial_observations_complete = False
        self._completed_sectors: set[str] = set()
        self._reported_empty_batteries = {
            drone_id for drone_id, drone in drones.items() if drone.battery <= 0
        }
        # Controllers start with HOLD and no previously delivered route.
        self.controllers = {
            drone_id: DroneController(drone_id, drone.position, map, {}, {})
            for drone_id, drone in drones.items()
        }
        self.telemetry_history: list[TelemetrySnapshot] = []
        self.telemetry_queries = RecordedTelemetry(self.telemetry_history)
        self._record_telemetry()

    def _record_telemetry(self) -> None:
        snapshot = TelemetrySnapshot(step=self.timestep, drones={
            drone_id: drone.model_copy(deep=True) for drone_id, drone in self.drones.items()
        })
        if self.telemetry_history and self.telemetry_history[-1].step == self.timestep:
            self.telemetry_history[-1] = snapshot
        else:
            self.telemetry_history.append(snapshot)

    def _observe(self, position: tuple[int, int], explored_cells: set[tuple[int, int]]):
        row, col = position
        for r in range(max(0, row - sensor_range), min(len(self.map), row + sensor_range + 1)):
            for c in range(max(0, col - sensor_range), min(len(self.map[r]), col + sensor_range + 1)):
                if (r - row) ** 2 + (c - col) ** 2 <= sensor_range**2:
                    explored_cells.add((r, c))

    def _sector_completion_events(
        self, sectors: dict[str, Sector], explored_cells: set[tuple[int, int]]
    ) -> list[str]:
        events = []
        for sector_id, sector in sectors.items():
            if sector_id in self._completed_sectors:
                continue
            (top, left), (bottom, right) = sector.bounds
            if all(
                (row, col) in explored_cells
                or self.map[row][col] in (Terrain.OBSTACLE, Terrain.NO_FLY_ZONE)
                for row in range(top, bottom + 1)
                for col in range(left, right + 1)
            ):
                self._completed_sectors.add(sector_id)
                events.append(f"Sector {sector_id} fully explored")
        return events

    def _apply_incident(self) -> list[str]:
        incident = self.scenario.get(self.timestep)
        if incident is None:
            return []
        incident_type, payload = incident
        if incident_type == IncidentType.NEW_NO_FLY_ZONE:
            zone = no_fly_zone_from_event(payload)
            if zone is not None:
                if any(row >= len(self.map) or col >= len(self.map[row]) for row, col in zone):
                    raise ValueError("No-fly zone must be within the terrain map")
                for row, col in zone:
                    self.map[row][col] = Terrain.NO_FLY_ZONE
            return [payload]
        drone = self.drones[payload]
        if incident_type == IncidentType.BATTERY_DRAIN:
            drone.battery = max(0.0, drone.battery - 10)
            if drone.battery <= 0:
                self.controllers[payload].update_objective(
                    Objective(drone_id=payload, objective_type=ObjectiveType.HOLD)
                )
        elif incident_type == IncidentType.COMMS_FAILURE:
            drone.comms = False
        return []  # Battery and communications incidents are silent.

    def _empty_battery_event(self, drone_id: str) -> list[str]:
        drone = self.drones[drone_id]
        if drone.battery <= 0 and drone_id not in self._reported_empty_batteries:
            self._reported_empty_batteries.add(drone_id)
            return [f"Drone {drone_id} ran out of battery at {drone.position}"]
        return []

    def _monitor_events(self, monitoring_checks) -> list[str]:
        if self.timestep % monitoring_interval:
            return []
        events = []
        for drone_id, drone in self.drones.items():
            events.extend(self._empty_battery_event(drone_id))
            if 0 < drone.battery < monitoring_checks["battery_reserve_threshold"]:
                events.append(f"Drone {drone_id} battery low {drone.battery} at {drone.position}")
            if not drone.comms:
                events.append(f"Drone {drone_id} lost communication at {drone.position}")
        return events

    def _movement_events(self, drone_id: str) -> list[str]:
        drone = self.drones[drone_id]
        row, col = drone.position
        if not (0 <= row < len(self.map) and 0 <= col < len(self.map[row])):
            return [f"Drone {drone_id} moved out of bounds at {drone.position}"]
        terrain = self.map[row][col]
        returning = self.controllers[drone_id].objective.objective_type == ObjectiveType.RETURN_TO_BASE
        events = []
        if terrain == Terrain.OBSTACLE:
            events.append(f"Drone {drone_id} hit an obstacle at {drone.position}")
        elif terrain == Terrain.NO_FLY_ZONE:
            events.append(f"Drone {drone_id} entered a no-fly zone at {drone.position}")
        elif terrain == Terrain.DRONE and not (returning and drone.position == BASE_POSITION):
            events.append(f"Drone {drone_id} collided with another drone at {drone.position}")
        else:
            if terrain == Terrain.TARGET:
                events.append(f"Drone {drone_id} reached the target at {drone.position}")
            else:
                events.extend(
                    f"Drone {drone_id} detected a target at {target}"
                    for target in self.targets
                    if (row - target[0]) ** 2 + (col - target[1]) ** 2 <= sensor_range**2
                )
            if returning and drone.position == BASE_POSITION:
                events.append(f"Drone {drone_id} returned to base at {BASE_POSITION}")
        events.extend(self._empty_battery_event(drone_id))
        return events

    def execute_plan(
        self,
        plan: dict[str, Objective],
        monitoring_checks,
        sectors: dict[str, Sector],
        targets: dict[str, Target],
        explored_cells: set[tuple[int, int]] | None = None,
    ) -> tuple[list[str], set[tuple[int, int]], int]:
        """Advance until a whole step produces events, returning them together."""
        if explored_cells is None:
            explored_cells = set()
        self._record_telemetry()
        if not self._initial_observations_complete:
            for drone in self.drones.values():
                if drone.battery > 0:
                    self._observe(drone.position, explored_cells)
            self._initial_observations_complete = True
        events = self._sector_completion_events(sectors, explored_cells)
        for drone_id, controller in self.controllers.items():
            drone = self.drones[drone_id]
            controller.update_position(drone.position)
            controller.update_comms(drone.comms)
            if drone.battery <= 0:
                controller.update_objective(Objective(drone_id=drone_id, objective_type=ObjectiveType.HOLD))
                continue
            if not drone.comms:
                continue
            controller.update_map(self.map)
            controller.update_explored_cells(explored_cells)
            controller.sectors = sectors
            controller.targets = targets
            controller.update_objective(plan.get(drone_id) or Objective(
                drone_id=drone_id, objective_type=ObjectiveType.HOLD
            ))
            objective = controller.objective
            if objective.objective_type == ObjectiveType.HOLD:
                drone.assignment = "hold position"
            elif objective.objective_type == ObjectiveType.RETURN_TO_BASE:
                drone.assignment = "return to base at (0, 0)"
            elif objective.objective_type == ObjectiveType.SEARCH_SECTOR:
                drone.assignment = "search for targets"
            elif objective.objective_type == ObjectiveType.GOTO:
                assert objective.target_id
                target = targets.get(objective.target_id)
                drone.assignment = f"go to {target.position}" if target else "go to target"
            if controller.failure_event is not None:
                events.append(controller.failure_event)
        self._record_telemetry()

        while not events:
            actions = {}
            for drone_id, controller in self.controllers.items():
                drone = self.drones[drone_id]
                if drone.battery <= 0:
                    continue
                controller.update_position(drone.position)
                controller.update_comms(drone.comms)
                if drone.comms:
                    controller.update_explored_cells(explored_cells)
                step = controller.plan_step()
                if controller.failure_event is not None:
                    events.append(controller.failure_event)
                elif step is not None:
                    actions[drone_id] = step
            if events:
                break
            if not actions:
                events.append("Mission plan exhausted")
                break

            self.timestep += 1
            events = self._apply_incident()
            if events:
                # A new flight restriction pauses before executing stale routes.
                self._record_telemetry()
                events.extend(self._sector_completion_events(sectors, explored_cells))
                events.extend(self._monitor_events(monitoring_checks))
                break

            moved = []
            for drone_id, step in actions.items():
                drone = self.drones[drone_id]
                if drone.battery <= 0:
                    continue
                controller = self.controllers[drone_id]
                position = drone.position[0] + step[0], drone.position[1] + step[1]
                controller.apply_step(position)
                drone.position = position
                drone.battery = max(0.0, drone.battery - 1)
                self._observe(drone.position, explored_cells)
                moved.append(drone_id)

            self._record_telemetry()
            # Collect every drone's events after all movement and observations.
            for drone_id in moved:
                events.extend(self._movement_events(drone_id))
                if self.drones[drone_id].battery <= 0:
                    self.controllers[drone_id].update_objective(
                        Objective(drone_id=drone_id, objective_type=ObjectiveType.HOLD)
                    )
            events.extend(self._monitor_events(monitoring_checks))
            events.extend(self._sector_completion_events(sectors, explored_cells))

        return events, explored_cells, self.timestep
