from enum import Enum


class Terrain(Enum):
    EMPTY = 0
    DRONE = 1
    OBSTACLE = 2
    NO_FLY_ZONE = 3
    TARGET = 4


def prepare_map(
    raw_map: list[str],
) -> list[
    list[Terrain]
]:  # . is empty, A,B,C,etc are drones, X is an obstacle, N is a no-fly zone, T is a target
    def convert(x: str) -> Terrain:
        if x == ".":
            return Terrain.EMPTY
        if x == "X":
            return Terrain.OBSTACLE
        if x == "N":
            return Terrain.NO_FLY_ZONE
        if x == "T":
            return Terrain.TARGET
        if len(x) == 1 and "A" <= x <= "Z":
            return Terrain.DRONE
        raise ValueError(f"Unknown terrain symbol: {x!r}")

    return [[convert(x) for x in row.split(" ")] for row in raw_map]


def remove_targets(terrain_map: list[list[Terrain]]) -> list[list[Terrain]]:
    return [
        [Terrain.EMPTY if cell is Terrain.TARGET else cell for cell in row]
        for row in terrain_map
    ]


def extract_from_raw(
    raw_map: list[str],
) -> tuple[
    list[list[Terrain]],
    list[list[Terrain]],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    truth_terrain = prepare_map(raw_map)
    terrain = remove_targets(truth_terrain)
    drones = []
    no_fly_zones = []
    for row_index, row in enumerate(raw_map):
        for col_index, label in enumerate(row.split()):
            if label == "N":
                no_fly_zones.append((row_index, col_index))
            elif label.isupper() and label not in {"X", "T"}:
                drones.append((row_index, col_index))
    return truth_terrain, terrain, drones, no_fly_zones
