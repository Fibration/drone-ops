"""Pure watch predicates; simulation history and estimates are explicit inputs."""
from collections.abc import Callable, Mapping, Sequence, Set
from collections import deque
from dataclasses import dataclass, field
from math import expm1, inf, isfinite

from src.models import FreeWatchType, Point, WatchPredicate, WatchType
from src.pathfinder import NoRouteFound, astar
from src.scenario import Weather, World, weather_transition_probabilities


@dataclass(frozen=True)
class WatchSample:
    """Internal history needed by battery-drain and comms-duration predicates."""
    step: int
    battery: dict[str, float]
    comms: dict[str, bool]


@dataclass(frozen=True)
class WatchContext:
    world: World
    step: int
    history: Sequence[WatchSample] = ()
    completed_sectors: Set[str] = field(default_factory=frozenset)
    route_costs: Mapping[str, float | None] = field(default_factory=dict)
    baseline_route_costs: Mapping[str, float | None] = field(default_factory=dict)
    scenario_weather: Weather | None = None
    new_target_ids: Set[str] = field(default_factory=frozenset)
    explored_cells: Set[Point] = field(default_factory=frozenset)
    coverage_cells: Set[Point] | None = None
    known_target_ids: Set[str] = field(default_factory=frozenset)
    acquired_target_ids: Set[str] = field(default_factory=frozenset)


def new_target_detected(context: WatchContext, entity_id: str | None) -> bool:
    if entity_id is None:
        raise ValueError('NEW_TARGET_DETECTED requires target_id')
    context.world.targets[entity_id]
    return entity_id in context.new_target_ids


def drone_hard_failure(context: WatchContext, entity_id: str | None) -> bool:
    if entity_id is None:
        raise ValueError('DRONE_HARD_FAILURE requires drone_id')
    drone = context.world.drones[entity_id]
    return drone.battery <= 0 and drone.position != context.world.base_position


def mission_complete(context: WatchContext, entity_id: str | None = None) -> bool:
    required = context.coverage_cells
    if required is None:
        required = reachable_cells(context.world)
    return (required <= context.explored_cells
            and context.known_target_ids <= context.acquired_target_ids)


def reachable_cells(world: World, starts: Sequence[Point] | None = None) -> set[Point]:
    """Static cardinal reachability, ignoring temporary drone occupancy and fuel."""
    blocked = world.obstacles | world.no_fly
    walkable = {
        (x, y) for sector in world.sectors
        for x in range(sector.bounds[0][0], sector.bounds[1][0] + 1)
        for y in range(sector.bounds[0][1], sector.bounds[1][1] + 1)
        if (x, y) not in blocked
    }
    if starts is None:
        starts = (world.base_position, *(drone.position for drone in world.drones.values()))
    reached = set(starts) & walkable
    frontier = deque(reached)
    while frontier:
        x, y = frontier.popleft()
        for cell in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if cell in walkable and cell not in reached:
                reached.add(cell)
                frontier.append(cell)
    return reached


FREE_WATCH_HOOKS: dict[FreeWatchType, Callable[[WatchContext, str | None], bool]] = {
    FreeWatchType.NEW_TARGET_DETECTED: new_target_detected,
    FreeWatchType.DRONE_HARD_FAILURE: drone_hard_failure,
    FreeWatchType.MISSION_COMPLETE: mission_complete,
}


def check_free_watch(
    watch_type: FreeWatchType, context: WatchContext, *,
    drone_id: str | None = None, target_id: str | None = None,
) -> bool:
    entity_id = drone_id if watch_type is FreeWatchType.DRONE_HARD_FAILURE else target_id
    return FREE_WATCH_HOOKS[watch_type](context, entity_id)


def estimate_route_cost(world: World, drone_id: str, destination: Point) -> float | None:
    """Battery cost of a shortest cardinal route at current sector weather.

    None means no route exists. This estimate does not predict future weather.
    """
    position = world.drones[drone_id].position
    route = astar(world.navigation_map(drone_id), position, destination)
    if isinstance(route, NoRouteFound):
        return None
    cost = 0.0
    for dx, dy in route:
        position = position[0] + dx, position[1] + dy
        sector = world.sector_at(position)
        if sector is None:
            return None
        cost -= sector.move(0.0)
    return cost


def _threshold(predicate: WatchPredicate) -> float:
    if predicate.threshold is None:
        raise ValueError(f'{predicate.type.value} requires a numeric threshold')
    value = float(predicate.threshold)
    if not isfinite(value):
        raise ValueError('Watch thresholds must be finite')
    return value


def _horizon(predicate: WatchPredicate, default: int | None = None) -> int:
    value = predicate.horizon if predicate.horizon is not None else default
    if value is None or value < 1:
        raise ValueError(f'{predicate.type.value} requires a positive horizon')
    return value


def _drone_id(predicate: WatchPredicate, context: WatchContext) -> str:
    if predicate.drone_id is None:
        raise ValueError(f'{predicate.type.value} requires drone_id')
    context.world.drones[predicate.drone_id]
    return predicate.drone_id


def _sector_id(predicate: WatchPredicate, context: WatchContext) -> str:
    if predicate.sector_id is not None:
        if not any(s.sector_id == predicate.sector_id for s in context.world.sectors):
            raise KeyError(predicate.sector_id)
        return predicate.sector_id
    sector = context.world.sector_at(context.world.drones[_drone_id(predicate, context)].position)
    if sector is None:
        raise ValueError('Drone is outside all sectors')
    return sector.sector_id


def _target_id(predicate: WatchPredicate, context: WatchContext) -> str:
    if predicate.target_id is None:
        raise ValueError(f'{predicate.type.value} requires target_id')
    context.world.targets[predicate.target_id]
    return predicate.target_id


def return_reserve_below(predicate: WatchPredicate, context: WatchContext) -> bool:
    drone_id = _drone_id(predicate, context)
    cost = estimate_route_cost(context.world, drone_id, context.world.base_position)
    reserve = context.world.drones[drone_id].battery - (cost if cost is not None else inf)
    return reserve < _threshold(predicate)


def battery_drain_above(predicate: WatchPredicate, context: WatchContext) -> bool:
    drone_id = _drone_id(predicate, context)
    since = context.step - _horizon(predicate, 1)
    threshold = _threshold(predicate)
    for snapshot in reversed(context.history):
        if snapshot.step <= since and drone_id in snapshot.battery:
            return (snapshot.battery[drone_id]
                    - context.world.drones[drone_id].battery) > threshold
    return False


def charge_above(predicate: WatchPredicate, context: WatchContext) -> bool:
    drone = context.world.drones[_drone_id(predicate, context)]
    return drone.position == context.world.base_position and drone.battery > _threshold(predicate)


def comms_lost_for(predicate: WatchPredicate, context: WatchContext) -> bool:
    drone_id = _drone_id(predicate, context)
    duration = _horizon(predicate)
    if context.world.drones[drone_id].comms:
        return False
    since = context.step
    for snapshot in reversed(context.history):
        if snapshot.step > context.step:
            continue
        comms = snapshot.comms.get(drone_id)
        if comms is None or comms:
            break
        since = snapshot.step
    return context.step - since >= duration


def weather_at_least(predicate: WatchPredicate, context: WatchContext) -> bool:
    sector_id = _sector_id(predicate, context)
    sector = next(s for s in context.world.sectors if s.sector_id == sector_id)
    threshold = predicate.threshold
    if isinstance(threshold, str) and threshold.upper() in Weather.__members__:
        level = Weather[threshold.upper()].value
    else:
        level = _threshold(predicate)
    return sector.weather.value >= level


def weather_forecast_risk_above(predicate: WatchPredicate, context: WatchContext) -> bool:
    if context.scenario_weather is None:
        raise ValueError('Weather forecasting requires scenario_weather')
    risk = forecast_weather_risk(context.world, context.scenario_weather,
                                 _sector_id(predicate, context), _horizon(predicate))
    return risk > _threshold(predicate)


def forecast_weather_risk(
    world: World, scenario_weather: Weather, sector_id: str, horizon: int,
) -> float:
    """P(stormy or worse within horizon), ignoring neighbouring weather.

    This scenario-only forecast is intentionally approximate relative to the
    coupled simulation. Propagation over the three safe levels is exact.
    """
    if horizon < 1:
        raise ValueError('Forecast horizon must be positive')
    sector = next(s for s in world.sectors if s.sector_id == sector_id)
    if sector.weather.value >= Weather.STORMY.value:
        return 1.0
    if sector.weather.value + horizon < Weather.STORMY.value:
        return 0.0
    safe = [0.0] * Weather.STORMY.value
    safe[sector.weather.value] = 1.0
    for _ in range(horizon):
        updated = [0.0] * len(safe)
        for current, mass in enumerate(safe):
            improve, worsen = weather_transition_probabilities(current, scenario_weather.value)
            updated[max(0, current - 1)] += mass * improve
            updated[current] += mass * (1 - improve - worsen)
            if current + 1 < Weather.STORMY.value:
                updated[current + 1] += mass * worsen
        safe = updated
    return min(1.0, max(0.0, 1.0 - sum(safe)))


def route_unreachable(predicate: WatchPredicate, context: WatchContext) -> bool:
    return context.route_costs[_drone_id(predicate, context)] is None


def route_cost_increased_by(predicate: WatchPredicate, context: WatchContext) -> bool:
    drone_id = _drone_id(predicate, context)
    current = context.route_costs[drone_id]
    baseline = context.baseline_route_costs[drone_id]
    threshold = _threshold(predicate)
    return current is not None and baseline is not None and current - baseline > threshold


def sector_complete(predicate: WatchPredicate, context: WatchContext) -> bool:
    return _sector_id(predicate, context) in context.completed_sectors


def target_survival_probability_below(predicate: WatchPredicate, context: WatchContext) -> bool:
    target_id = _target_id(predicate, context)
    horizon = _horizon(predicate)
    if predicate.decay_rate_prior is None:
        raise ValueError('Target survival requires a supplied decay_rate_prior')
    probability = target_survival_probability(
        target_id, context.world, predicate.decay_rate_prior, horizon,
    )
    return probability < _threshold(predicate)


def target_survival_probability(
    target_id: str, world: World, decay_rate_prior: float, horizon: int,
) -> float:
    """Forecast survival for one target, independent of other target counts.

    Under the supplied removal prior, P(floor(Exp(prior)) == 0) is
    1 - exp(-prior) per turn. Only this target's resolved status is consulted;
    neither the true decay rate nor the surrounding population is used.
    """
    world.targets[target_id]
    if horizon < 1 or not isfinite(decay_rate_prior) or decay_rate_prior < 0:
        raise ValueError('Survival needs a positive horizon and a nonnegative finite prior')
    if target_id in world.successful_targets:
        return 1.0
    if target_id in world.failed_targets:
        return 0.0
    if decay_rate_prior == 0:
        return 1.0  # Sector.update_targets treats zero as disabling removals.
    return (-expm1(-decay_rate_prior)) ** horizon


def target_not_reached_by(predicate: WatchPredicate, context: WatchContext) -> bool:
    target_id = _target_id(predicate, context)
    # horizon is an absolute simulation turn for this deadline predicate.
    deadline = _horizon(predicate)
    return context.step >= deadline and target_id not in context.world.successful_targets


WATCH_HOOKS: dict[WatchType, Callable[[WatchPredicate, WatchContext], bool]] = {
    WatchType.RETURN_RESERVE_BELOW: return_reserve_below,
    WatchType.BATTERY_DRAIN_ABOVE: battery_drain_above,
    WatchType.CHARGE_ABOVE: charge_above,
    WatchType.COMMS_LOST_FOR: comms_lost_for,
    WatchType.WEATHER_AT_LEAST: weather_at_least,
    WatchType.WEATHER_FORECAST_RISK_ABOVE: weather_forecast_risk_above,
    WatchType.ROUTE_UNREACHABLE: route_unreachable,
    WatchType.ROUTE_COST_INCREASED_BY: route_cost_increased_by,
    WatchType.SECTOR_COMPLETE: sector_complete,
    WatchType.TARGET_SURVIVAL_PROBABILITY_BELOW: target_survival_probability_below,
    WatchType.TARGET_NOT_REACHED_BY: target_not_reached_by,
}


def check_watch(predicate: WatchPredicate, context: WatchContext) -> bool:
    """Evaluate one watch without changing the world or consuming randomness."""
    return WATCH_HOOKS[predicate.type](predicate, context)
