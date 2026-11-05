from src.drone import DroneController, ObjectiveType
from src.models import DroneState
from src.map import Terrain
from src.parameters import sensor_range, monitoring_interval


class Simulator:
    def __init__(
        self,
        map: list[list[Terrain]],
        drones: dict[str, DroneState],
        targets: list[tuple[int, int]],
    ):
        self.map = map
        self.drones = drones
        self.event = None
        self.timestep = 0
        self.targets = targets

    def execute_plan(self, plan, monitoring_checks, explored_cells=None):
        self.event = None
        if explored_cells is None:
            explored_cells = set()
        controllers = {}
        for drone_id, objective in plan.items():
            drone = self.drones.get(drone_id)
            if drone is None:
                continue
            controller = DroneController(objective)
            if isinstance(objective, list):
                controller.route = objective
            else:
                controller.calculate_route(self.map, drone.position, explored_cells)
            controllers[drone_id] = controller

        while not self.event:
            actions = {}
            for drone_id, controller in controllers.items():
                step = controller.next_action()
                if (
                    step is None
                    and not isinstance(controller.objective, list)
                    and controller.objective.type == ObjectiveType.SEARCH_SECTOR
                ):
                    controller.calculate_route(
                        self.map, self.drones[drone_id].position, explored_cells
                    )
                    step = controller.next_action()
                if step is not None:
                    actions[drone_id] = step
            if not actions:
                self.event = "Mission plan exhausted"
                break

            self.timestep += 1
            monitor = self.timestep % monitoring_interval == 0

            for drone_id, step in actions.items():
                drone = self.drones[drone_id]
                if drone:
                    drone.position = (
                        drone.position[0] + step[0],
                        drone.position[1] + step[1],
                    )
                    drone.battery -= 1  # Decrease battery for each move
                    row, col = drone.position
                    for r in range(
                        max(0, row - sensor_range),
                        min(len(self.map), row + sensor_range + 1),
                    ):
                        for c in range(
                            max(0, col - sensor_range),
                            min(len(self.map[r]), col + sensor_range + 1),
                        ):
                            if (r - row) ** 2 + (c - col) ** 2 <= sensor_range**2:
                                explored_cells.add((r, c))
            # All moves and sensor observations belong to the same timestep.
            for drone_id in actions:
                drone = self.drones[drone_id]
                if drone:
                    row, col = drone.position
                    if not (0 <= row < len(self.map) and 0 <= col < len(self.map[row])):
                        self.event = (
                            f"Drone {drone_id} moved out of bounds at {drone.position}"
                        )
                        break
                    elif (
                        self.map[drone.position[0]][drone.position[1]]
                        == Terrain.OBSTACLE
                    ):
                        self.event = (
                            f"Drone {drone_id} hit an obstacle at {drone.position}"
                        )
                        break
                    elif (
                        self.map[drone.position[0]][drone.position[1]]
                        == Terrain.NO_FLY_ZONE
                    ):
                        self.event = f"Drone {drone_id} entered a no-fly zone at {drone.position}"
                        break
                    elif (
                        self.map[drone.position[0]][drone.position[1]] == Terrain.TARGET
                    ):
                        self.event = (
                            f"Drone {drone_id} reached the target at {drone.position}"
                        )
                        break
                    elif (
                        self.map[drone.position[0]][drone.position[1]] == Terrain.DRONE
                    ):
                        self.event = f"Drone {drone_id} collided with another drone at {drone.position}"
                        break
                    elif drone.battery <= 0:
                        self.event = (
                            f"Drone {drone_id} ran out of battery at {drone.position}"
                        )
                        break
                    else:
                        detected_target = next(
                            (
                                target
                                for target in self.targets
                                if (drone.position[0] - target[0]) ** 2
                                + (drone.position[1] - target[1]) ** 2
                                <= sensor_range**2
                            ),
                            None,
                        )
                        if detected_target is not None:
                            self.event = f"Drone {drone_id} detected a target at {detected_target}"
                            break

                    if monitor:
                        if drone.battery < monitoring_checks["battery_reserve_threshold"]:
                            self.event = f"Drone {drone_id} battery low {drone.battery} at {drone.position}"
                            break
                        elif drone.comms is False:
                            self.event = f"Drone {drone_id} lost communication at {drone.position}"
                            break

        return self.event, self.drones, explored_cells, self.timestep
