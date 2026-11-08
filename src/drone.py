from src.map import Terrain
from src.models import Objective, ObjectiveType, Point
from src.pathfinder import NoRouteFound, astar
from src.scenario import World


class DroneController:
    def __init__(self, drone_id: str, world: World):
        self.drone_id = drone_id
        self.world = world
        self.objective = Objective(drone_id=drone_id, objective_type=ObjectiveType.HOLD)
        self.explored_cells: set[Point] = set()
        # Only reported targets are supplied by the simulator, not world truth.
        self.targets: dict[str, Point] = {}
        self.route: list[Point] = []
        self.route_index = 0
        self.failure_event: str | None = None
        self._planned_step: Point | None = None

    @property
    def position(self) -> Point:
        return self.world.drones[self.drone_id].position

    @property
    def comms(self) -> bool:
        return self.world.drones[self.drone_id].comms

    def update_objective(self, objective: Objective):
        if objective.drone_id != self.drone_id:
            raise ValueError("Objective belongs to another drone")
        self.objective = objective
        self.calculate_route()

    def clear_route(self) -> None:
        """Discard a route and any move that has not been committed."""
        self.route = []
        self.route_index = 0
        self._planned_step = None

    def calculate_route(self) -> list[Point]:
        self.clear_route()
        self.failure_event = None
        objective_type = self.objective.objective_type
        if objective_type == ObjectiveType.HOLD:
            return self.route
        terrain = self.world.navigation_map(self.drone_id)
        if objective_type == ObjectiveType.SEARCH_SECTOR:
            sector = next((s for s in self.world.sectors
                           if s.sector_id == self.objective.sector_id), None)
            if sector is None:
                raise ValueError(f"Unknown sector: {self.objective.sector_id}")
            (x0, y0), (x1, y1) = sector.bounds
            candidates = [
                (x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
                if (x, y) not in self.explored_cells
                and (x, y) != self.position
                and terrain[x][y] is Terrain.EMPTY
            ]
            candidates.sort(key=lambda p: abs(p[0] - self.position[0])
                            + abs(p[1] - self.position[1]))
            for destination in candidates:
                result = astar(terrain, self.position, destination)
                if not isinstance(result, NoRouteFound):
                    self.route = result
                    return self.route
            if candidates:
                self.failure_event = (
                    f"Drone {self.drone_id} cannot search sector "
                    f"{sector.sector_id}: remaining cells are unreachable"
                )
            return self.route
        if objective_type == ObjectiveType.RETURN_TO_BASE:
            destination = self.world.base_position
            failure_message = f"cannot return to base at {destination}"
        elif objective_type == ObjectiveType.GOTO:
            assert self.objective.target_id is not None
            destination = self.targets.get(self.objective.target_id)
            if destination is None:
                self.failure_event = (
                    f"Drone {self.drone_id} target {self.objective.target_id} is unavailable"
                )
                return self.route
            failure_message = f"cannot reach target {self.objective.target_id} at {destination}"
        else:
            raise ValueError(f"Unsupported objective type: {objective_type}")
        result = astar(terrain, self.position, destination)
        if isinstance(result, NoRouteFound):
            self.failure_event = f"Drone {self.drone_id} {failure_message}"
        else:
            self.route = result
        return self.route

    def plan_step(self) -> Point | None:
        """Choose a relative move without consuming it; repeated calls are stable."""
        if self._planned_step is not None:
            return self._planned_step
        if self.failure_event is not None:
            return None
        if self.comms and self.objective.objective_type == ObjectiveType.SEARCH_SECTOR:
            self.calculate_route()
        if self.route_index >= len(self.route):
            return None
        self._planned_step = self.route[self.route_index]
        return self._planned_step

    def apply_step(self, position: Point) -> None:
        """Commit an accepted move to the world's drone state."""
        if self._planned_step is None:
            raise ValueError("No step has been planned")
        dx, dy = self._planned_step
        if position != (self.position[0] + dx, self.position[1] + dy):
            raise ValueError("Position does not match the planned step")
        self.world.drones[self.drone_id].position = position
        self.route_index += 1
        self._planned_step = None
