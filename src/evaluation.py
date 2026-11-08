from src.simulator import Simulator


def evaluate(simulator: Simulator, starting_battery: float):
    # lexicographic scoring:
    # 1. percentage targets acquired
    # 2. reciprocal time elapsed
    # 3. fraction fleet survived + fraction energy remaining

    fraction_targets_acquired = (
        len(simulator.world.successful_targets) / simulator.scenario.num_targets
    )

    reciprocal_time_elapsed = 1 / simulator.timestep

    fraction_fleet_survived = len(
        [x for x in simulator.drones.values() if x.battery > 0 and x.comms]
    ) / len(simulator.drones.keys())

    fraction_energy_remaining = (
        sum([x.battery for x in simulator.drones.values()]) / starting_battery
    )

    return (
        1000 * fraction_targets_acquired
        + 200 * reciprocal_time_elapsed
        + fraction_fleet_survived
        + fraction_energy_remaining
    )
