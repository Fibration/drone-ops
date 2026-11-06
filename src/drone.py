from enum import Enum

from src.map import Terrain
from src.models import Objective, ObjectiveType, Sector, Target
from src.pathfinder import astar
from src.parameters import BASE_POSITION


class DroneController:
    objective: Objective
    route: list[tuple[int, int]] | None
    route_index: int
    to_explore: list[tuple[int, int]]

    def __init__(
        self,
        objective: Objective,
        sectors: dict[str, Sector],
        targets: dict[str, Target],
    ):
        self.objective = objective
        self.route = []
        self.route_index = 0
        self.to_explore = []
        self.sectors = sectors
        self.targets = targets

    def calculate_route(
        self,
        map: list[list[Terrain]],
        drone_position: tuple[int, int],
        explored_cells: set[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        if self.objective.objective_type == ObjectiveType.HOLD:
            self.route = []
            self.route_index = 0
        elif self.objective.objective_type == ObjectiveType.RETURN_TO_BASE:
            self.route = astar(map, drone_position, BASE_POSITION, allow_occupied_end=True)
            self.route_index = 0
        elif self.objective.objective_type == ObjectiveType.SEARCH_SECTOR:
            assert self.objective.sector_id is not None
            sector = self.sectors[self.objective.sector_id]
            self.to_explore.clear()
            for x in range(sector.bounds[0][0], sector.bounds[1][0] + 1):
                for y in range(sector.bounds[0][1], sector.bounds[1][1] + 1):
                    if (x, y) not in explored_cells and map[x][y] == Terrain.EMPTY:
                        self.to_explore.append((x, y))

            self.to_explore.sort(
                key=lambda pos: abs(pos[0] - drone_position[0])
                + abs(pos[1] - drone_position[1]),
                reverse=True,
            )
            self.route = (
                astar(map, drone_position, self.to_explore.pop())
                if self.to_explore
                else []
            )
            self.route_index = 0

        elif self.objective.objective_type == ObjectiveType.GOTO:
            assert self.objective.target_id is not None
            self.route = astar(
                map, drone_position, self.targets[self.objective.target_id].position
            )
            self.route_index = 0

    def next_action(self):
        if self.route is None or self.route_index >= len(self.route):
            return None
        next_step = self.route[self.route_index]
        self.route_index += 1
        return next_step
