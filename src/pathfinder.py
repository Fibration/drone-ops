from heapq import heappop, heappush
from dataclasses import dataclass

from src.models import Terrain


@dataclass(frozen=True)
class NoRouteFound:
    """An expected outcome when no traversable route connects the endpoints."""

    start: tuple[int, int]
    end: tuple[int, int]


def astar(
    terrain_map: list[list[Terrain]],
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    allow_occupied_end: bool = False,
) -> list[tuple[int, int]] | NoRouteFound:
    """Return shortest cardinal steps, or NoRouteFound if unreachable.

    Obstacles, no-fly zones, and other drones block travel. The starting
    cell may be blocked to allow recovery after an event. An identical
    start/end returns [].
    """
    blocked = {Terrain.OBSTACLE, Terrain.NO_FLY_ZONE, Terrain.DRONE}
    for row, col in (start, end):
        if not (0 <= row < len(terrain_map) and 0 <= col < len(terrain_map[row])):
            raise ValueError("Start and end must be within the terrain map")
    if start == end:
        return []

    def is_blocked(position: tuple[int, int]) -> bool:
        row, col = position
        return terrain_map[row][col] in blocked and not (
            allow_occupied_end
            and position == end
            and terrain_map[row][col] == Terrain.DRONE
        )

    if is_blocked(end):
        return NoRouteFound(start, end)

    end_row, end_col = end
    frontier = [(abs(start[0] - end_row) + abs(start[1] - end_col), 0, start)]
    costs = {start: 0}
    parents = {}
    while frontier:
        _, negative_cost, current = heappop(frontier)
        cost = -negative_cost
        if cost != costs[current]:
            continue
        if current == end:
            route = []
            while current != start:
                parent = parents[current]
                route.append((current[0] - parent[0], current[1] - parent[1]))
                current = parent
            route.reverse()
            return route

        row, col = current
        next_cost = cost + 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            neighbor = (nr, nc)
            if (
                0 <= nr < len(terrain_map)
                and 0 <= nc < len(terrain_map[nr])
                and not is_blocked(neighbor)
                and next_cost < costs.get(neighbor, float("inf"))
            ):
                costs[neighbor] = next_cost
                parents[neighbor] = current
                priority = next_cost + abs(nr - end_row) + abs(nc - end_col)
                heappush(frontier, (priority, -next_cost, neighbor))
    return NoRouteFound(start, end)
