# A live matplotlib view of a running Simulation, with arrow-key control of the leader in MANUAL mode
import math
import time

import matplotlib.pyplot as plt

from . import plotting


#========================================#
# LIVE VIEW
#========================================#

class LiveView():
    """Draws the simulation every step and paces it to real time"""

    def __init__(self, simulation, config) -> None:
        """Open the live window and hook up the keyboard"""
        self.simulation = simulation
        self.config = config
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=config.plot.fig_size)
        # NOTE: laid out once, with room for the legend; re-running tight_layout every frame dominated the draw time
        self.fig.subplots_adjust(right=config.plot.live_right_margin)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)


    def is_open(self) -> bool:
        """Whether the window is still open; closing it stops the run"""
        return bool(plt.get_fignums())


    def close(self) -> None:
        """Close the live window"""
        plt.close(self.fig)


    def on_key_press(self, event) -> None:
        """Manual leader control from the arrow keys"""
        manual = self.config.manual
        command = self.simulation.leader_manual

        if event.key == 'up':
            command['vel'] = min(command['vel'] + manual.vel_step, manual.vel_max)
        elif event.key == 'down':
            command['vel'] = max(command['vel'] - manual.vel_step, 0.0)
        elif event.key == 'left':
            command['delta'] = min(command['delta'] + manual.delta_step_rad, manual.delta_max_rad)
        elif event.key == 'right':
            command['delta'] = max(command['delta'] - manual.delta_step_rad, -manual.delta_max_rad)


    def draw(self) -> None:
        """Redraw the world view with the estimates and target points over the vehicles"""
        simulation = self.simulation
        colors = self.config.plot.vehicle_colors
        arrow_len = self.config.plot.arrow_len
        self.ax.clear()

        # Estimates and arrows
        for ekf in simulation.filters.values():
            vehicle = simulation.vehicles[ekf.veh_id]
            x, y, theta = body_to_world(vehicle, *ekf.x[:3])
            self.ax.plot(x, y, marker='+', linestyle="", color=colors[vehicle.id],
                         label=f"veh {ekf.target_id} from veh {ekf.veh_id}")
            self.ax.annotate("", xytext=(x, y), xy=(x + arrow_len*math.cos(theta), y + arrow_len*math.sin(theta)),
                             arrowprops=dict(arrowstyle="->", color=colors[vehicle.id]))

        # Target points
        for follower_id, controller in simulation.controllers.items():
            x, y, _ = body_to_world(simulation.vehicles[follower_id], *controller.target)
            self.ax.plot(x, y, marker='x', linestyle="", color=colors[follower_id], label=f"veh {follower_id} target pt")

        plotting.draw_world(self.ax, simulation.build_results(), self.config)


    def pace(self, step_start: float) -> None:
        """Show the frame, waiting out the rest of the sim step so the view runs at real time"""
        self.fig.canvas.draw()  # NOTE: render before timing; otherwise plt.pause renders on top of the wait

        remaining = self.simulation.dt - (time.time() - step_start)  # s
        plt.pause(max(remaining, self.config.run.live_min_pause))






#========================================#
# HELPERS
#========================================#

def body_to_world(vehicle, x: float, y: float, theta: float) -> tuple:
    """Convert a pose in a vehicle's body frame to the world frame"""
    c, s = math.cos(vehicle.theta), math.sin(vehicle.theta)
    return vehicle.x + c*x - s*y, vehicle.y + s*x + c*y, vehicle.theta + theta
