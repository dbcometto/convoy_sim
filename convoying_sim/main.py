# Runs the convoy simulation: `python convoying_sim/main.py`.  Everything tunable lives in config.yaml.
import time
from enum import IntEnum

import numpy as np

from core import plotting
from core.config import load_config
from core.sim import Simulation
from core.live import LiveView


#========================================#
# CONFIG
#========================================#

class VISUALIZATION_MODES(IntEnum):
    LIVE = 0  # draw every step, paced to real time
    END_PLOT = 1  # run as fast as possible, then plot the world view and errors






#========================================#
# RUNNING
#========================================#

def print_progress(simulation: Simulation, num_steps: int, wall_start: float, bar_width: int) -> None:
    """Print a progress bar with the elapsed and estimated remaining wall time"""
    step = simulation.step_count
    fraction = step / num_steps
    elapsed = time.time() - wall_start  # s
    eta = elapsed / fraction - elapsed  # s
    bar = '#' * int(bar_width * fraction)
    print(f"\r[{bar:<{bar_width}}] {100*fraction:5.1f}%  sim {simulation.time:6.1f}/{simulation.config.sim.duration}s  "
          f"elapsed {elapsed:5.1f}s  eta {eta:5.1f}s", end='', flush=True)
    if step == num_steps:
        print()


def print_debug(simulation: Simulation) -> None:
    """Print the true and estimated states"""
    print("VEHICLES")
    print("id | true x | true y | true theta | true vel")
    for vehicle_id, vehicle in simulation.vehicles.items():
        print(f"{vehicle_id} | {vehicle.x} | {vehicle.y} | {vehicle.theta} | {vehicle.vel}")

    print("\nFILTERS")
    print("id | ekf x | ekf y | ekf theta | ekf vel")
    for follower_id, ekf in simulation.filters.items():
        print(f"{follower_id} | {ekf.x[0]} | {ekf.x[1]} | {ekf.x[2]} | {ekf.x[3]}")


def run(simulation: Simulation, live_view: LiveView = None) -> None:
    """Step the simulation to the end of its duration, drawing live if a live view is given"""
    run_config = simulation.config.run
    num_steps = int(round(simulation.config.sim.duration / simulation.dt))
    wall_start = time.time()  # s

    while simulation.step_count < num_steps and (live_view is None or live_view.is_open()):
        step_start = time.time()
        simulation.step()
        print_progress(simulation, num_steps, wall_start, run_config.progress_bar_width)

        if run_config.do_debug_print:
            print_debug(simulation)

        if live_view is not None:
            live_view.draw()
            live_view.pace(step_start)






#========================================#
# MAIN
#========================================#

def main() -> None:
    """Load the config, run the simulation, then save and plot the results"""
    config = load_config()
    simulation = Simulation(config)
    is_live = VISUALIZATION_MODES[config.run.visualization_mode] == VISUALIZATION_MODES.LIVE
    live_view = LiveView(simulation, config) if is_live else None

    try:
        run(simulation, live_view)

    except KeyboardInterrupt:
        print("\nInterrupted, saving results so far")

    finally:
        print("Shutting down sim")

    if simulation.history["t"]:
        results = simulation.build_results()

        if config.run.do_save_results:
            config.run.output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(config.run.output_file, **results)
            print(f"Saved results to {config.run.output_file}")

        if is_live:
            live_view.close()
        else:
            plotting.plot_results(results, config)


if __name__ == "__main__":
    main()
