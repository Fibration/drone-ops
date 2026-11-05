from enum import Enum

from src.map import Terrain
from src.pathfinder import astar


class ObjectiveType(Enum):
    SEARCH_SECTOR = "search sector"
    GOTO = "go to"


class Objective:
    def __init__(self, type: ObjectiveType, **kwargs):
        self.type = type
        if type == ObjectiveType.SEARCH_SECTOR:
            self.sector = kwargs.get("sector")
            self.bounds = kwargs.get("bounds")
        elif type == ObjectiveType.GOTO:
            self.target = kwargs.get("target")


class DroneController:
    objective: Objective
    route: list[tuple[int, int]] | None
    route_index: int
    to_explore: list[tuple[int, int]]

    def __init__(self, objective: Objective):
        self.objective = objective
        self.route = []
        self.route_index = 0
        self.to_explore = []

    def calculate_route(
        self,
        map: list[list[Terrain]],
        drone_position: tuple[int, int],
        explored_cells: set[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        if self.objective.type == ObjectiveType.SEARCH_SECTOR:
            self.to_explore.clear()
            assert self.objective.bounds is not None
            for x in range(
                self.objective.bounds[0][0], self.objective.bounds[1][0] + 1
            ):
                for y in range(
                    self.objective.bounds[0][1], self.objective.bounds[1][1] + 1
                ):
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

        elif self.objective.type == ObjectiveType.GOTO:
            assert self.objective.target is not None
            self.route = astar(map, drone_position, self.objective.target)
            self.route_index = 0

    def next_action(self):
        if self.route is None or self.route_index >= len(self.route):
            return None
        next_step = self.route[self.route_index]
        self.route_index += 1
        return next_step
