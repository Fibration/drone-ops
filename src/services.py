"""Simulator tool implementations, injected into Simulator as a ToolService."""

from random import random

from src.models import Tool, ToolObservation, ToolObservationDict
from src.pathfinder import NoRouteFound, astar
from src.scenario import Weather, next_weather_level
from src.utilities import points_in_bounds

type Obs = ToolObservation
type ObsDict = ToolObservationDict


class SimulatorTools:
    def __init__(self):
        self._handlers = {
            Tool.ESTIMATE_SECTOR_POPULATION: self.estimate_sector_population,
            Tool.GET_FLEET_STATUS: self.get_fleet_status,
            Tool.GET_SECTOR_STATUS: self.get_sector_status,
            Tool.GET_ACTIVE_TARGETS: self.get_active_targets,
            Tool.GET_WEATHER_FORECAST: self.get_weather_forecast,
            Tool.PLAN_ROUTE: self.plan_route,
            Tool.ESTIMATE_ROUTE_COST: self.estimate_route_cost,
            Tool.EVALUATE_CANDIDATE_PLAN: self.evaluate_candidate_plan,
            Tool.STRESS_TEST_PLAN: self.stress_test_plan,
        }

    def call(self, tool: Tool, arguments: dict, observation: ObsDict) -> dict:
        return self._handlers[tool](arguments, observation)

    def estimate_sector_population(self, arguments: dict, observation: ObsDict) -> dict:
        """Return the sector's fixed initial population estimate for this run."""
        return {
            "estimated_population": observation[ToolObservation.POPULATION_ESTIMATES][
                arguments["sector_id"]
            ]
        }

    def get_fleet_status(self, arguments: dict, observation: ObsDict) -> dict:
        return {
            "drone_positions": dict(observation[ToolObservation.DRONE_POSITIONS]),
            "drone_batteries": dict(observation[ToolObservation.DRONE_BATTERIES]),
        }

    def get_sector_status(self, arguments: dict, observation: ObsDict) -> dict:
        sector_id = arguments["sector_id"]
        if sector_id in observation[ToolObservation.COMPLETED_SECTORS]:
            sector_completion = 1
        else:
            bounds = observation[ToolObservation.SECTOR_BOUNDS][sector_id]
            (x0, y0), (x1, y1) = bounds
            explored = sum(
                points_in_bounds(
                    observation[ToolObservation.EXPLORED_CELLS], bounds, return_mask=True
                )
            )
            sector_completion = explored / ((x1 - x0 + 1) * (y1 - y0 + 1))
        return {
            "completion": sector_completion,
            "weather": observation[ToolObservation.WEATHER][sector_id],
        }

    def get_active_targets(self, arguments: dict, observation: ObsDict) -> dict:
        return {
            "active_targets": set(observation[ToolObservation.KNOWN_TARGETS].values())
            - set(observation[ToolObservation.ACQUIRED_TARGETS])
        }

    def get_weather_forecast(self, arguments: dict, observation: ObsDict) -> dict:
        desired = Weather(arguments["scenario_weather"]).value
        horizon = arguments.get("horizon", 1)
        if horizon < 1:
            raise ValueError("Forecast horizon must be positive")
        weather = observation[ToolObservation.WEATHER]
        sector_ids = (arguments["sector_id"],) if "sector_id" in arguments else weather
        forecast = {}
        for sector_id in sector_ids:
            current = weather[sector_id].value
            levels = []
            for _ in range(horizon):
                current = next_weather_level(current, desired, random())
                levels.append(Weather(current))
            forecast[sector_id] = levels
        return {"forecast": forecast}

    def plan_route(self, arguments: dict, observation: ObsDict) -> dict:
        start = observation[ToolObservation.DRONE_POSITIONS][arguments["drone_id"]]
        destination = tuple(arguments["destination"])
        terrain = observation[ToolObservation.NAVIGATION_MAP]
        result = astar(terrain, start, destination)
        return {"route": None if isinstance(result, NoRouteFound) else result}

    def estimate_route_cost(self, arguments: dict, observation: ObsDict) -> dict:
        x, y = observation[ToolObservation.DRONE_POSITIONS][arguments["drone_id"]]
        weather = observation[ToolObservation.WEATHER]
        desired_weather = arguments.get("weather") or {}
        movement_costs = observation[ToolObservation.MOVEMENT_COSTS]
        penalties = {Weather.STORMY: 0.5, Weather.BLIZZARD: 1.0}
        sectors = [
            (
                sector_id,
                bounds,
                movement_costs[sector_id]
                + penalties.get(Weather(desired_weather.get(sector_id, weather[sector_id])), 0.0),
            )
            for sector_id, bounds in observation[ToolObservation.SECTOR_BOUNDS].items()
        ]
        sector_costs = {}
        for dx, dy in arguments["route"]:
            x += dx
            y += dy
            for sector_id, ((x0, y0), (x1, y1)), cost in sectors:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    sector_costs[sector_id] = sector_costs.get(sector_id, 0.0) + cost
                    break
            else:
                raise ValueError(f"Route leaves known sectors at {(x, y)}")
        return {"sector_costs": sector_costs, "cost": sum(sector_costs.values())}

    def evaluate_candidate_plan(self, arguments: dict, observation: ObsDict) -> dict:
        raise NotImplementedError("EVALUATE_CANDIDATE_PLAN is not implemented")

    def stress_test_plan(self, arguments: dict, observation: ObsDict) -> dict:
        raise NotImplementedError("STRESS_TEST_PLAN is not implemented")
