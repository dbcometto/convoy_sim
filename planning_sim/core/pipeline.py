# Chains the stages (heightmap -> terrain -> driveability -> costmap -> planner -> speed profile) with no entry point
from types import SimpleNamespace

import numpy as np

from .costmap import build_cost_map
from .driveability_analyzer import build_driveability_map
from .heightmap import load_height_map
from .planner import plan_path
from .speed_profile import build_trajectory
from .terrain_analyzer import build_terrain_map
from .types import PlanningResult


#========================================#
# CONFIG
#========================================#

LAYER_PREFIX = "layer_"  # results keys for the layers the plotter can switch between
DRIVEABILITY_ANY = "driveability_any"  # layer: driveable at one or more headings
DRIVEABILITY_ALL = "driveability_all"  # layer: driveable at every heading
DRIVEABILITY_HEADING = "driveability_heading"  # layer: driveable at the heading picked on the plot slider
HEADINGS_KEY = "headings_rad"  # results key: the heading of each bin of the 3D layer






#========================================#
# PIPELINE
#========================================#

def run_pipeline(config: SimpleNamespace) -> PlanningResult:
    """Run every stage in order and keep each stage's output"""
    height_map = load_height_map(config.map.folder)
    terrain_map = build_terrain_map(height_map, config.terrain)
    driveability_map = build_driveability_map(height_map, terrain_map, config.driveability, config.vehicle)
    cost_map = build_cost_map(driveability_map, terrain_map, config.costmap)
    path = plan_path(cost_map, config.start, config.goal, config.planner)
    trajectory = build_trajectory(path, config.speed_profile)
    return PlanningResult(height_map, terrain_map, driveability_map, cost_map, path, trajectory)


def build_results(result: PlanningResult, config: SimpleNamespace) -> dict:
    """Package the layers, trajectory, and start/goal as arrays so they can be saved and plotted later"""
    driveability_map = result.driveability_map
    layers = {
        **result.height_map.layers,
        **result.terrain_map.layers,
        DRIVEABILITY_ANY: driveability_map.any_heading(),
        DRIVEABILITY_ALL: driveability_map.all_headings(),
        DRIVEABILITY_HEADING: driveability_map.driveable,  # 3D (bins, rows, cols): the plot picks one bin with its slider
        "cost": result.cost_map.cost,
    }
    results = {LAYER_PREFIX + name: array for name, array in layers.items()}
    results[HEADINGS_KEY] = driveability_map.headings_rad  # rad

    results["extent"] = np.array(result.height_map.grid.extent())  # m, (x_min, x_max, y_min, y_max)
    results["trajectory_xy"] = result.trajectory.xy  # m
    results["start_xy"] = np.array([config.start.x, config.start.y])  # m
    results["goal_xy"] = np.array([config.goal.x, config.goal.y])  # m
    return results
