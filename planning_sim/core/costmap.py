# Stage 4: DriveabilityMap (+ TerrainMap) -> CostMap
from types import SimpleNamespace

import numpy as np

from .types import CostMap, DriveabilityMap, TerrainMap


def build_cost_map(driveability_map: DriveabilityMap, terrain_map: TerrainMap, config: SimpleNamespace) -> CostMap:
    """Soft cost per cell (slope, roughness, obstacle proximity), infinite where not driveable (config is `config.costmap`)"""
    # TODO: implement the weighted terms; placeholder is zero cost where driveable
    # NOTE: heading-independent for now, so a cell is open if any heading is driveable; a kinematic planner should use the stack
    cost = np.where(driveability_map.any_heading(), 0.0, np.inf)
    return CostMap(grid=driveability_map.grid, cost=cost)
