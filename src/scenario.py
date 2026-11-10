from dataclasses import dataclass
from collections.abc import Iterator
from enum import Enum
import random
from math import ceil, floor, sqrt

from src.models import Terrain
from src.models import DroneState, IncidentType, Point
from src.utilities import points_in_bounds
from src.parameters import wind_penalty


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    IMPOSSIBLE = 4


class Weather(Enum):
    CLEAR = 0
    CLOUDY = 1
    FOGGY = 2
    STORMY = 3
    BLIZZARD = 4


def weather_transition_probabilities(
    current: int, desired: float
) -> tuple[float, float]:
    improve_probability = 0.03 if current > desired else 0.02
    worsen_probability = 0.03 if current < desired else 0.02
    return improve_probability, worsen_probability


def next_weather_level(current: int, desired: float, transition: float) -> int:
    """One persistence-biased weather transition, shared with forecasts."""
    improve_probability, worsen_probability = weather_transition_probabilities(
        current, desired
    )
    if transition < improve_probability:
        return max(Weather.CLEAR.value, current - 1)
    if transition < improve_probability + worsen_probability:
        return min(Weather.BLIZZARD.value, current + 1)
    return current


class Sector:
    def __init__(
        self,
        sector_id: str,
        bounds: tuple[Point, Point],
        visibility: float,
        movement_cost: float,
        decay_rate: float,
        weather: Weather,
        population_rate: float | None = None,
        wind_direction: tuple[int, int] = (0, 0),
    ):
        self.sector_id = sector_id
        self.bounds = bounds
        self.visibility = visibility
        self.movement_cost = movement_cost
        self.weather = weather
        self.wind_direction = wind_direction
        self.decay_rate = decay_rate
        self.population_rate = (
            population_rate if population_rate is not None else 1 / decay_rate
        )

    def move(self, direction: tuple[int, int], battery: float) -> float:
        weather_penalty = {
            Weather.STORMY: 0.2,
            Weather.BLIZZARD: 0.4,
        }.get(self.weather, 0.0)
        wind_adjustment = 0.0
        if self.wind_direction != (0, 0) and direction == self.wind_direction:
            wind_adjustment = -wind_penalty
        elif self.wind_direction != (0, 0) and direction == (-self.wind_direction[0], -self.wind_direction[1]):
            wind_adjustment = wind_penalty
        movement_cost = (
            self.movement_cost * (1 + weather_penalty) * (1 + wind_adjustment)
        )
        return battery - movement_cost

    def in_sector(self, cell: Point) -> bool:
        return points_in_bounds((cell,), self.bounds)

    def visible_cells(self, cell: Point) -> set[Point]:
        # Return all the cells within the euclidean distance of the provided
        # cell which are within (inclusive) the sector bounds
        radius = self.visibility
        weather_bonus = {
            Weather.CLEAR: 0.5,
            Weather.FOGGY: -0.5,
            Weather.STORMY: -1,
        }.get(self.weather, 0)
        radius = 0.0 if self.weather is Weather.BLIZZARD else radius + weather_bonus
        if radius < 0:
            return set()

        x, y = cell
        (min_x, min_y), (max_x, max_y) = self.bounds
        radius_squared = radius * radius
        cells: set[Point] = set()
        for visible_x in range(
            max(min_x, ceil(x - radius)), min(max_x, floor(x + radius)) + 1
        ):
            height = floor(sqrt(max(0.0, radius_squared - (visible_x - x) ** 2)))
            cells.update(
                (visible_x, visible_y)
                for visible_y in range(
                    max(min_y, y - height), min(max_y, y + height) + 1
                )
            )
        return cells

    def update_targets(self, targets: list[str]) -> list[str]:
        # use the exponential distribution with the decay rate as its parameter
        # to choose how many targets to remove, randomly choosing which
        if not targets or self.decay_rate == 0:
            return targets.copy()

        remove_count = min(len(targets), int(random.expovariate(self.decay_rate)))
        removed_indices = set(random.sample(range(len(targets)), remove_count))
        return [
            target
            for index, target in enumerate(targets)
            if index not in removed_indices
        ]


class World:
    def __init__(
        self,
        sectors: list[Sector],
        targets: int,
        obstacles: list[Point],
        no_fly: list[Point],
        drones: dict[str, DroneState],
        base_position: Point = (0, 0),
    ):
        self.sectors = sectors
        self.sector_neighbours = self._build_sector_neighbours()
        self.target_count = targets
        self.obstacles = set(obstacles)
        self.no_fly = set(no_fly)
        self.drones = dict(drones)
        self.base_position = base_position
        # Original placements are retained after rescue or expiry.
        self.targets: dict[str, Point] = {}
        self.active_targets: dict[str, list[str]] = {
            sector.sector_id: [] for sector in sectors
        }
        self.failed_targets: list[str] = []
        self.successful_targets: list[str] = []
        self.success_times: dict[str, float] = {}
        self._place_targets()

    def sector_at(self, cell: Point) -> Sector | None:
        return next((sector for sector in self.sectors if sector.in_sector(cell)), None)

    def _build_sector_neighbours(self) -> dict[str, tuple[str, ...]]:
        """Cache shared-edge adjacency, including partially overlapping edges."""
        neighbours: dict[str, list[str]] = {s.sector_id: [] for s in self.sectors}
        for index, sector in enumerate(self.sectors):
            (x0, y0), (x1, y1) = sector.bounds
            for other in self.sectors[index + 1 :]:
                (ox0, oy0), (ox1, oy1) = other.bounds
                if (
                    (x1 + 1 == ox0 or ox1 + 1 == x0) and max(y0, oy0) <= min(y1, oy1)
                ) or (
                    (y1 + 1 == oy0 or oy1 + 1 == y0) and max(x0, ox0) <= min(x1, ox1)
                ):
                    neighbours[sector.sector_id].append(other.sector_id)
                    neighbours[other.sector_id].append(sector.sector_id)
        return {sector_id: tuple(ids) for sector_id, ids in neighbours.items()}

    def navigation_map(self, drone_id: str | None) -> list[list[Terrain]]:
        """Terrain for routing, with the shared base open to returning drones."""
        width = max(sector.bounds[1][0] for sector in self.sectors) + 1
        height = max(sector.bounds[1][1] for sector in self.sectors) + 1
        terrain = [[Terrain.EMPTY] * height for _ in range(width)]
        for cells, kind in (
            (self.obstacles, Terrain.OBSTACLE),
            (self.no_fly, Terrain.NO_FLY_ZONE),
        ):
            for x, y in cells:
                terrain[x][y] = kind
        for other_id, drone in self.drones.items():
            if other_id != drone_id and drone.position != self.base_position:
                x, y = drone.position
                if terrain[x][y] is Terrain.EMPTY:
                    terrain[x][y] = Terrain.DRONE
        return terrain

    def _place_targets(self):
        occupied = (
            self.obstacles
            | self.no_fly
            | {drone.position for drone in self.drones.values()}
        )
        available = []
        weights = []
        for sector in self.sectors:
            rate = sector.population_rate
            if rate < 0:
                raise ValueError("Population rates must be nonnegative")
            # Exponential interarrival times give a Poisson count in unit time.
            count = 0
            if rate:
                elapsed = random.expovariate(rate)
                while elapsed <= 1:
                    count += 1
                    elapsed += random.expovariate(rate)
            weights.append(count)
            (min_x, min_y), (max_x, max_y) = sector.bounds
            cells = [
                (x, y)
                for x in range(min_x, max_x + 1)
                for y in range(min_y, max_y + 1)
                if (x, y) not in occupied
            ]
            random.shuffle(cells)
            available.append(cells)

        indices = [i for i, cells in enumerate(available) if cells]
        for _ in range(self.target_count):
            while indices:
                probabilities = [weights[i] for i in indices]
                index = random.choices(
                    indices,
                    weights=probabilities if any(probabilities) else None,
                    k=1,
                )[0]
                cells = available[index]
                while cells and cells[-1] in occupied:
                    cells.pop()
                if cells:
                    break
                indices.remove(index)
            else:
                break
            cell = cells.pop()
            target_id = f"target_{len(self.targets)}"
            self.targets[target_id] = cell
            self.active_targets[self.sectors[index].sector_id].append(target_id)
            occupied.add(cell)
            if not cells:
                indices.remove(index)

    def check_cell(self, cell: Point) -> Terrain:
        if cell in self.obstacles:
            return Terrain.OBSTACLE
        if cell in self.no_fly:
            return Terrain.NO_FLY_ZONE
        if any(drone.position == cell for drone in self.drones.values()):
            return Terrain.DRONE
        return Terrain.EMPTY

    def check_for_target(self, cell: Point) -> Terrain:
        terrain = self.check_cell(cell)
        if terrain is Terrain.EMPTY and any(
            position == cell for _, position in self.iter_active_targets()
        ):
            return Terrain.TARGET
        return terrain

    def iter_active_targets(self) -> Iterator[tuple[str, Point]]:
        for target_ids in self.active_targets.values():
            for target_id in target_ids:
                yield target_id, self.targets[target_id]

    def update_active_targets(self):
        for sector in self.sectors:
            previous = self.active_targets[sector.sector_id]
            remaining = sector.update_targets(previous)
            remaining_ids = set(remaining)
            for target_id in previous:
                if target_id not in remaining_ids:
                    self.failed_targets.append(target_id)
            self.active_targets[sector.sector_id] = remaining

    def update_target_success(self, target_id: str, success_time: float):
        """Rescue an active target, recording its elapsed simulation time.

        Raises KeyError if the target is unknown or already resolved.
        """
        for target_ids in self.active_targets.values():
            if target_id in target_ids:
                target_ids.remove(target_id)
                break
        else:
            raise KeyError(target_id)
        self.successful_targets.append(target_id)
        self.success_times[target_id] = success_time

    def update_drone(self, drone_id: str, drone_state: DroneState):
        self.drones[drone_id] = drone_state

    def update_sector_weather(self, scenario_weather: Weather):
        levels = {sector.sector_id: sector.weather.value for sector in self.sectors}
        neighbour_averages = {
            sector_id: (
                sum(levels[neighbour] for neighbour in neighbours) / len(neighbours)
                if neighbours
                else scenario_weather.value
            )
            for sector_id, neighbours in self.sector_neighbours.items()
        }
        for sector in self.sectors:
            current = levels[sector.sector_id]
            neighbour_average = neighbour_averages[sector.sector_id]
            desired = 0.5 * scenario_weather.value + 0.5 * neighbour_average
            sector.weather = Weather(
                next_weather_level(current, desired, random.random())
            )


@dataclass(frozen=True)
class Scenario:
    world_size: tuple[int, int]
    num_sectors: int
    num_targets: int
    difficulty: Difficulty
    num_drones: int
    base_position: Point
    events: dict[int, tuple[IncidentType, str]]
    weather: Weather
    recharge_rate: float
    wind_direction: tuple[int, int] = (0, 0)


def generate_world(scenario: Scenario) -> World:
    width, height = scenario.world_size
    if width <= 0 or height <= 0:
        raise ValueError("World dimensions must be positive")
    if not 1 <= scenario.num_sectors <= width * height:
        raise ValueError("Sector count must be between one and the world area")
    if scenario.num_targets < 0 or scenario.num_drones < 0:
        raise ValueError("Target and drone counts must be nonnegative")
    base_x, base_y = scenario.base_position
    if not points_in_bounds(
        (scenario.base_position,), ((0, 0), (width - 1, height - 1))
    ):
        raise ValueError("Base position must be within the world")
    if scenario.num_targets > width * height - 1:
        raise ValueError("Not enough cells to place targets outside the base")

    difficulty = scenario.difficulty.value
    bounds = [((0, 0), (width - 1, height - 1))]
    while len(bounds) < scenario.num_sectors:
        index = max(
            range(len(bounds)),
            key=lambda i: (bounds[i][1][0] - bounds[i][0][0] + 1)
            * (bounds[i][1][1] - bounds[i][0][1] + 1),
        )
        (x0, y0), (x1, y1) = bounds.pop(index)
        if x1 - x0 >= y1 - y0:
            middle = (x0 + x1) // 2
            bounds.extend([((x0, y0), (middle, y1)), ((middle + 1, y0), (x1, y1))])
        else:
            middle = (y0 + y1) // 2
            bounds.extend([((x0, y0), (x1, middle)), ((x0, middle + 1), (x1, y1))])

    sectors = [
        Sector(
            sector_id=str(index),
            bounds=sector_bounds,
            visibility=3 if random.random() < (5 - difficulty) / 5 else 2,
            movement_cost=random.uniform(
                0.5 + (difficulty - 1) * 0.3, 1.1 + (difficulty - 1) * 0.3
            ),
            decay_rate=4.5 - difficulty * 0.4,
            weather=Weather(
                random.randint(
                    max(0, scenario.weather.value - 1),
                    min(4, scenario.weather.value + 1),
                )
            ),
            population_rate=4.0,
            wind_direction=scenario.wind_direction,
        )
        for index, sector_bounds in enumerate(bounds)
    ]

    # Reserve room for every target and keep the launch area clear.
    protected = {
        (x, y)
        for x in range(max(0, base_x - 1), min(width, base_x + 2))
        for y in range(max(0, base_y - 1), min(height, base_y + 2))
    }
    blocked_budget = width * height - 1 - scenario.num_targets
    no_fly: set[Point] = set()
    for _ in range(difficulty):
        rectangle_width = random.randint(1, min(width, difficulty + 1))
        rectangle_height = random.randint(1, min(height, difficulty + 1))
        x0 = random.randrange(width - rectangle_width + 1)
        y0 = random.randrange(height - rectangle_height + 1)
        rectangle = {
            (x, y)
            for x in range(x0, x0 + rectangle_width)
            for y in range(y0, y0 + rectangle_height)
        }
        if not rectangle & protected and len(no_fly | rectangle) <= blocked_budget:
            no_fly.update(rectangle)
    available = [
        (x, y)
        for x in range(width)
        for y in range(height)
        if (x, y) not in protected and (x, y) not in no_fly
    ]
    obstacle_count = min(
        int(width * height * difficulty * 0.025),
        blocked_budget - len(no_fly),
        len(available),
    )
    obstacles = random.sample(available, obstacle_count)
    drones = {
        str(index): DroneState(
            id=str(index),
            position=scenario.base_position,
            battery=80.0,
            status="idle",
            comms=True,
            assignment=None,
        )
        for index in range(scenario.num_drones)
    }
    return World(
        sectors,
        scenario.num_targets,
        obstacles,
        list(no_fly),
        drones,
        base_position=scenario.base_position,
    )
