from src.map import Terrain
from src.models import Objective, ObjectiveType, Sector, Target
from src.pathfinder import NoRouteFound, astar
from src.parameters import BASE_POSITION


class DroneController:
    def __init__(
        self,
        drone_id: str,
        map: list[list[Terrain]],
        sectors: dict[str, Sector],
        targets: dict[str, Target],
    ):
        self.drone_id = drone_id
        self.objective = Objective(drone_id=drone_id, objective_type=ObjectiveType.HOLD)
        self.map = map
        self.position = (0, 0)
        self.explored_cells: set[tuple[int, int]] = set()
        self.route: list[tuple[int, int]] = []
        self.route_index = 0
        self.to_explore: list[tuple[int, int]] = []
        self.failure_event: str | None = None
        self.sectors = sectors
        self.targets = targets

    def update_objective(self, objective: Objective):
        self.objective = objective
        self.calculate_route()

    def update_position(self, position: tuple[int, int]):
        self.position = position

    def update_map(self, map: list[list[Terrain]]):
        self.map = map

    def update_explored_cells(self, explored_cells: set[tuple[int, int]]):
        self.explored_cells = explored_cells

    def calculate_route(self) -> list[tuple[int, int]]:
        self.route = []
        self.route_index = 0
        self.failure_event = None
        objective_type = self.objective.objective_type
        if objective_type == ObjectiveType.HOLD:
            return self.route
        if objective_type == ObjectiveType.SEARCH_SECTOR:
            assert self.objective.sector_id is not None
            sector = self.sectors[self.objective.sector_id]
            self.to_explore = [
                (row, col)
                for row in range(sector.bounds[0][0], sector.bounds[1][0] + 1)
                for col in range(sector.bounds[0][1], sector.bounds[1][1] + 1)
                if (row, col) not in self.explored_cells
                and (row, col) != self.position
                and self.map[row][col] == Terrain.EMPTY
            ]
            self.to_explore.sort(
                key=lambda pos: abs(pos[0] - self.position[0])
                + abs(pos[1] - self.position[1]), reverse=True,
            )
            had_candidates = bool(self.to_explore)
            while self.to_explore:
                result = astar(self.map, self.position, self.to_explore.pop())
                if not isinstance(result, NoRouteFound):
                    self.route = result
                    return self.route
            if had_candidates:
                self.failure_event = (
                    f"Drone {self.drone_id} cannot search sector "
                    f"{self.objective.sector_id}: remaining cells are unreachable"
                )
            return self.route
        if objective_type == ObjectiveType.RETURN_TO_BASE:
            destination = BASE_POSITION
            failure_message = f"cannot return to base at {destination}"
        elif objective_type == ObjectiveType.GOTO:
            assert self.objective.target_id is not None
            destination = self.targets[self.objective.target_id].position
            failure_message = f"cannot reach target {self.objective.target_id} at {destination}"
        else:
            raise ValueError(f"Unsupported objective type: {objective_type}")
        result = astar(
            self.map, self.position, destination,
            allow_occupied_end=objective_type == ObjectiveType.RETURN_TO_BASE,
        )
        if isinstance(result, NoRouteFound):
            self.failure_event = f"Drone {self.drone_id} {failure_message}"
        else:
            self.route = result
        return self.route

    def next_action(
        self, *, allow_replan: bool = True, consume: bool = True
    ) -> tuple[int, int] | None:
        if self.failure_event is not None:
            return None
        if self.route_index >= len(self.route):
            if allow_replan and self.objective.objective_type == ObjectiveType.SEARCH_SECTOR:
                self.calculate_route()
            if self.route_index >= len(self.route):
                return None
        next_step = self.route[self.route_index]
        if consume:
            self.route_index += 1
        return next_step
