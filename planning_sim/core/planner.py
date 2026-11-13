# Stage 5: CostMap + start + goal -> Path
from types import SimpleNamespace

import numpy as np

from .types import CostMap, Path


def plan_path(cost_map: CostMap, start: SimpleNamespace, goal: SimpleNamespace, config: SimpleNamespace) -> Path:
    """Find a path from start (x, y, theta_rad) to goal (x, y) through the cost map (config is `config.planner`)"""
    # TODO: implement a real planner that avoids infinite-cost cells and respects the vehicle's turn radius;
    # placeholder is a straight line that ignores the cost map
    xy = np.array([[start.x, start.y], [goal.x, goal.y]])  # m
    heading = np.arctan2(goal.y - start.y, goal.x - start.x)  # rad
    theta = np.array([start.theta_rad, heading])  # rad
    return Path(xy=xy, theta=theta)
