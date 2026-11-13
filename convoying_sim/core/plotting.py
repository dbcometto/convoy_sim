# Plotting for the convoy sim: the world view (live or final) and the end-of-run error plots
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from .vehicle import LEADER_ID


#========================================#
# CONFIG
#========================================#

METRICS = [("est", "Estimator Error"), ("trk", "Tracking Error"), ("lat", "Lateral Deviation")]
Z_VEHICLE_MARKER = 10  # draw offsets so markers sit above trails, and time labels above markers
Z_TIME_LABEL = 20






#========================================#
# WORLD VIEW
#========================================#

def world_z(vehicle_id: int, num_vehicles: int, base: int = 0) -> int:
    """Draw order for the world view: lower vehicle ids on top"""
    return base + num_vehicles - vehicle_id


def draw_world(ax, results, config) -> None:
    """Draw trails, the latest vehicle poses, and leader time labels from results (poses is (steps, vehicles, 3))"""
    style = config.plot
    poses = results["poses"]
    vehicle_ids = list(results["vehicle_ids"])
    sim_dt = float(results["sim_dt"])  # s
    lidar_fov = float(results["lidar_fov"])  # rad
    num_vehicles = len(vehicle_ids)

    # Trails
    for index, vehicle_id in enumerate(vehicle_ids):
        ax.plot(poses[:, index, 0], poses[:, index, 1], linestyle='-', alpha=style.trail_alpha,
                color=style.vehicle_colors[vehicle_id], zorder=world_z(vehicle_id, num_vehicles))

    # Latest poses
    for index, vehicle_id in enumerate(vehicle_ids):
        x, y, theta = poses[-1, index]
        color = style.vehicle_colors[vehicle_id]

        if style.do_visualize_lidar_fov:
            ax.add_patch(Wedge((x, y), style.fov_radius, math.degrees(theta - lidar_fov/2), math.degrees(theta + lidar_fov/2),
                               color=color, alpha=style.fov_alpha, zorder=world_z(vehicle_id, num_vehicles)))

        ax.plot(x, y, marker='o', linestyle="", color=color, label=f"veh {vehicle_id}",
                zorder=world_z(vehicle_id, num_vehicles, Z_VEHICLE_MARKER))
        ax.annotate("", xytext=(x, y), xy=(x + style.arrow_len*math.cos(theta), y + style.arrow_len*math.sin(theta)),
                    arrowprops=dict(arrowstyle="->"), zorder=world_z(vehicle_id, num_vehicles, Z_VEHICLE_MARKER))

    # Time labels along the leader's trail
    leader = vehicle_ids.index(LEADER_ID)
    label_every = int(round(style.time_label_interval / sim_dt))  # steps
    for step in range(label_every, len(poses) + 1, label_every):
        mark_x, mark_y, _ = poses[step - 1, leader]
        ax.plot(mark_x, mark_y, marker='o', markersize=style.time_label_marker_size, linestyle="", color='k',
                zorder=world_z(LEADER_ID, num_vehicles, Z_TIME_LABEL))
        ax.annotate(f"{step * sim_dt:g}s", (mark_x, mark_y), xytext=style.time_label_offset, textcoords='offset points',
                    fontsize=style.time_label_font_size, zorder=world_z(LEADER_ID, num_vehicles, Z_TIME_LABEL))

    # Limits
    if style.do_autoscale_world:
        points = poses[:, :, :2].reshape(-1, 2)  # m
        low, high = points.min(axis=0), points.max(axis=0)
        center = (low + high) / 2
        half_width = max((high - low).max() / 2, style.min_world_half_width) * (1 + style.autoscale_margin)
        ax.set_xlim(center[0] - half_width, center[0] + half_width)
        ax.set_ylim(center[1] - half_width, center[1] + half_width)
    else:
        ax.set_xlim(-style.world_size, style.world_size)
        ax.set_ylim(-style.world_size, style.world_size)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Simulator")
    ax.legend(loc=style.legend_loc, bbox_to_anchor=style.legend_anchor, borderaxespad=0,
              fontsize=style.legend_font_size, framealpha=style.legend_alpha)






#========================================#
# END PLOTS
#========================================#

def print_error_summary(results) -> None:
    """Print max, median, and mean of each error metric per follower and overall"""
    follower_ids = list(results["follower_ids"])

    for key, title in METRICS:
        print(f"{title} (m)")
        for column, follower_id in enumerate(follower_ids):
            values = results[key][:, column]
            print(f"  veh {follower_id}: max {np.nanmax(values):.4f}  median {np.nanmedian(values):.4f}  mean {np.nanmean(values):.4f}")
        all_values = results[key]
        print(f"  overall: max {np.nanmax(all_values):.4f}  median {np.nanmedian(all_values):.4f}  mean {np.nanmean(all_values):.4f}")


def plot_results(results, config) -> None:
    """One window: world view on the left, error metrics stacked on the right.  `results` is a results dict or npz."""
    style = config.plot
    t = results["t"]  # s
    follower_ids = list(results["follower_ids"])

    fig = plt.figure(figsize=style.fig_size)
    grid = fig.add_gridspec(len(METRICS), 2, width_ratios=style.metrics_width_ratios)
    ax_world = fig.add_subplot(grid[:, 0])
    ax_metrics = [fig.add_subplot(grid[0, 1])]
    ax_metrics += [fig.add_subplot(grid[row, 1], sharex=ax_metrics[0]) for row in range(1, len(METRICS))]

    # Metrics
    for ax_metric, (key, title) in zip(ax_metrics, METRICS):
        for column, follower_id in enumerate(follower_ids):
            ax_metric.plot(t, results[key][:, column], color=style.vehicle_colors[follower_id], label=f"veh {follower_id}",
                           zorder=len(follower_ids) - follower_id + 2)
        ax_metric.set_title(title)
        ax_metric.set_ylabel("Error (m)")
        ax_metric.grid(alpha=style.grid_alpha)
        ax_metric.tick_params(labelbottom=(ax_metric is ax_metrics[-1]))
    ax_metrics[-1].set_xlabel("Time (s)")

    # World view
    draw_world(ax_world, results, config)

    print_error_summary(results)
    fig.tight_layout()
    plt.show()
