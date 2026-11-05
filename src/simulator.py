from models import DroneState
from map import Terrain
from parameters import sensor_range, monitoring_interval


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

    def execute_plan(self, plan, monitoring_checks):
        self.event = None
        while not self.event:

            self.timestep += 1
            monitor = self.timestep % monitoring_interval == 0

            for drone_id, route in plan.items():
                drone = self.drones.get(drone_id)
                if drone and route:
                    step = route.pop(0)
                    drone.position = (
                        drone.position[0] + step[0],
                        drone.position[1] + step[1],
                    )
                    drone.battery -= 1  # Decrease battery for each move
                    # Check for events
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
                        if drone.battery < monitoring_checks.battery_reserve_threshold:
                            self.event = f"Drone {drone_id} battery low {drone.battery} at {drone.position}"
                            break
                        elif drone.comms is False:
                            self.event = f"Drone {drone_id} lost communication at {drone.position}"
                            break

        return self.event, self.drones
