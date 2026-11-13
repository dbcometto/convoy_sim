# Data passed between the pipeline stages: grid layers, the path, and the trajectory
from dataclasses import dataclass

import numpy as np


#========================================#
# GRID LAYERS
#========================================#

@dataclass
class Grid:
    """Geometry shared by every layer.  Arrays are indexed [row, col] with row 0 at the lowest world y (+y is up)."""
    cell_size_m: float  # m
    origin_cell: tuple  # (col, row) of the cell centered on world (0, 0)
    shape: tuple  # (rows, cols)

    def cell_to_world(self, col: float, row: float) -> tuple:
        """World (x, y) in m of a cell center"""
        return ((col - self.origin_cell[0]) * self.cell_size_m, (row - self.origin_cell[1]) * self.cell_size_m)

    def world_to_cell(self, x: float, y: float) -> tuple:
        """Fractional (col, row) of a world point in m"""
        return (x / self.cell_size_m + self.origin_cell[0], y / self.cell_size_m + self.origin_cell[1])

    def extent(self) -> tuple:
        """(x_min, x_max, y_min, y_max) in m of the outer cell edges, for plotting"""
        rows, cols = self.shape
        x_min, y_min = self.cell_to_world(-0.5, -0.5)
        x_max, y_max = self.cell_to_world(cols - 0.5, rows - 0.5)
        return (x_min, x_max, y_min, y_max)


@dataclass
class GridMap:
    """Generic map: named (rows, cols) layers on one shared grid, with a unit string per layer"""
    grid: Grid
    layers: dict  # layer name -> (rows, cols) array
    units: dict  # layer name -> unit string ("" for unitless)

    def __getitem__(self, name: str) -> np.ndarray:
        """The array of one layer"""
        return self.layers[name]


# Layer names, so producers and consumers share one spelling
HEIGHT = "height"  # m
VALID = "valid"  # bool, False where the height is unknown
GX = "gx"  # m/m, height gain per meter along world +x (positive = uphill toward +x)
GY = "gy"  # m/m, height gain per meter along world +y
GRADE = "grade"  # m/m, magnitude of the uphill gradient (tan of the slope angle)
STEP_UP = "step_up"  # m, largest rise from the cell to a valid neighbor
STEP_DOWN = "step_down"  # m, largest drop from the cell to a valid neighbor
STEP_HEIGHT = "step_height"  # m, larger of step_up and step_down
ROUGHNESS = "roughness"  # m, RMS residual of the local plane fit
CURVATURE = "curvature"  # 1/m, mean second derivative of the height; positive = dip, negative = crest
SUPPORT = "support"  # fraction of the fit window that is valid

# Both are GridMaps; the names say which layers a map holds
HeightMap = GridMap  # layers: HEIGHT, VALID
TerrainMap = GridMap  # vehicle-independent layers derived from the height map: GX .. SUPPORT above


@dataclass
class DriveabilityMap:
    """Where the vehicle's center may be at each heading, after the footprint, clearance, and slope limits are accounted for"""
    grid: Grid
    headings_rad: np.ndarray  # (bins,), rad, world frame, evenly spaced from 0
    driveable: np.ndarray  # (bins, rows, cols), bool

    def any_heading(self) -> np.ndarray:
        """(rows, cols) bool, True where at least one heading is driveable"""
        return self.driveable.any(axis=0)

    def all_headings(self) -> np.ndarray:
        """(rows, cols) bool, True where every heading is driveable"""
        return self.driveable.all(axis=0)


@dataclass
class CostMap:
    """Cost of traversing each cell; infinite where not driveable"""
    grid: Grid
    cost: np.ndarray  # (rows, cols), unitless




#========================================#
# PATH AND TRAJECTORY
#========================================#

@dataclass
class Path:
    """Geometric route from start to goal"""
    xy: np.ndarray  # (points, 2), m, world frame
    theta: np.ndarray  # (points,), rad, world frame


@dataclass
class Trajectory:
    """The path with timing: what the leader vehicle follows"""
    t: np.ndarray  # (points,), s
    xy: np.ndarray  # (points, 2), m, world frame
    theta: np.ndarray  # (points,), rad, world frame
    vel: np.ndarray  # (points,), m/s
    curvature: np.ndarray  # (points,), 1/m


@dataclass
class PlanningResult:
    """Every stage's output from one run of the pipeline"""
    height_map: HeightMap
    terrain_map: TerrainMap
    driveability_map: DriveabilityMap
    cost_map: CostMap
    path: Path
    trajectory: Trajectory
