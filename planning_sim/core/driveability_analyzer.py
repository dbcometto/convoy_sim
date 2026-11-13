# Stage 3: HeightMap + TerrainMap -> DriveabilityMap.  For each heading bin, marks the poses (cell center + heading) where
# the vehicle's footprint rests on known, gentle, smooth terrain that clears the underbody.
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import correlate

from .terrain_analyzer import fit_plane_over_kernel, kernel_offsets_m
from .types import (GX, GY, HEIGHT, ROUGHNESS, STEP_HEIGHT, VALID, DriveabilityMap, Grid, HeightMap, TerrainMap)


#========================================#
# CONFIG
#========================================#

SYMMETRY_BINS = 2  # the rectangle and the slope limits are unchanged by a 180 deg turn, so the headings repeat every half circle






#========================================#
# STAGE
#========================================#

def build_driveability_map(height_map: HeightMap, terrain_map: TerrainMap, config: SimpleNamespace,
                           vehicle_config: SimpleNamespace) -> DriveabilityMap:
    """Mark where the vehicle center may be for each heading bin (config is `config.driveability`)"""
    if config.heading_bins % SYMMETRY_BINS != 0:
        raise ValueError(f"driveability.heading_bins must be even, got {config.heading_bins}")

    grid = height_map.grid
    height, valid = height_map[HEIGHT], height_map[VALID]  # m
    headings_rad = np.arange(config.heading_bins) * 2 * np.pi / config.heading_bins  # rad, world frame

    # Heading-independent: unknown or unreliable terrain, steps, and roughness.  NaN fails the comparison, so it is bad too.
    heading_free_bad = (~valid | ~(terrain_map[STEP_HEIGHT] <= vehicle_config.max_step_height)
                        | ~(terrain_map[ROUGHNESS] <= vehicle_config.max_roughness))

    # Only the first half circle is computed; the second half is the same masks
    unique_bins = config.heading_bins // SYMMETRY_BINS
    unique = [pose_is_driveable(heading_rad, height, valid, heading_free_bad, terrain_map, grid, vehicle_config)
              for heading_rad in headings_rad[:unique_bins]]
    driveable = np.concatenate([unique] * SYMMETRY_BINS)  # (bins, rows, cols)
    return DriveabilityMap(grid=grid, headings_rad=headings_rad, driveable=driveable)






#========================================#
# HELPERS
#========================================#

def pose_is_driveable(heading_rad: float, height: np.ndarray, valid: np.ndarray, heading_free_bad: np.ndarray,
                      terrain_map: TerrainMap, grid: Grid, vehicle_config: SimpleNamespace) -> np.ndarray:
    """(rows, cols) bool: the vehicle center may sit in the cell at this heading"""
    kernel, dx, dy = footprint_kernel(heading_rad, grid.cell_size_m, vehicle_config)

    # Any bad cell under the footprint blocks the pose; outside the map counts as bad
    bad = heading_free_bad | slope_exceeds_limits(heading_rad, terrain_map, vehicle_config)
    blocked = correlate(bad.astype(float), kernel, mode="constant", cval=1.0) > 0.0

    return ~blocked & ~belly_hits_terrain(height, valid, kernel, dx, dy, vehicle_config.clearance)


def footprint_kernel(heading_rad: float, cell_size_m: float, vehicle_config: SimpleNamespace) -> tuple:
    """Weights (1 inside, 0 outside) of the cells whose centers fall under the margin-inflated rectangle at this heading,
    on an odd square grid, plus each cell's (dx, dy) in m from the center (world frame)"""
    half_length = vehicle_config.footprint_length / 2 + vehicle_config.footprint_margin  # m
    half_width = vehicle_config.footprint_width / 2 + vehicle_config.footprint_margin  # m
    half_cells = int(np.ceil(np.hypot(half_length, half_width) / cell_size_m))  # the half diagonal bounds any heading
    dx, dy = kernel_offsets_m(2 * half_cells + 1, cell_size_m)  # m

    along = dx * np.cos(heading_rad) + dy * np.sin(heading_rad)  # m, forward of the center
    across = -dx * np.sin(heading_rad) + dy * np.cos(heading_rad)  # m, left of the center
    kernel = ((np.abs(along) <= half_length) & (np.abs(across) <= half_width)).astype(float)
    return kernel, dx, dy


def slope_exceeds_limits(heading_rad: float, terrain_map: TerrainMap, vehicle_config: SimpleNamespace) -> np.ndarray:
    """(rows, cols) bool: the grade along the heading (pitch) or across it (roll) is over the limit; NaN grade counts as over"""
    gx, gy = terrain_map[GX], terrain_map[GY]  # m/m
    pitch_grade = np.abs(gx * np.cos(heading_rad) + gy * np.sin(heading_rad))  # m/m, uphill along the heading
    roll_grade = np.abs(-gx * np.sin(heading_rad) + gy * np.cos(heading_rad))  # m/m, uphill across the heading
    return ~((pitch_grade <= np.tan(vehicle_config.max_pitch_rad)) & (roll_grade <= np.tan(vehicle_config.max_roll_rad)))


def belly_hits_terrain(height: np.ndarray, valid: np.ndarray, kernel: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                       clearance: float) -> np.ndarray:
    """(rows, cols) bool: some terrain under the footprint rises more than `clearance` above the plane fit through it.
    Slopes alone do not trip this, only bumps (a rock or crest between the wheels)."""
    fit = fit_plane_over_kernel(height, valid, kernel, dx, dy)
    rows, cols = height.shape
    half = kernel.shape[0] // 2

    # NOTE: -inf marks unknown cells, which can never be the highest point.  They are already blocked as bad cells.
    padded = np.pad(np.where(valid, fit.centered, -np.inf), half, constant_values=-np.inf)
    plane_a, plane_gx, plane_gy = fit.plane[..., 0], fit.plane[..., 1], fit.plane[..., 2]

    highest_residual = np.full(height.shape, -np.inf)  # m
    for kernel_row, kernel_col in zip(*np.nonzero(kernel)):
        shifted = padded[kernel_row:kernel_row + rows, kernel_col:kernel_col + cols]  # footprint cell under each center
        plane_height = plane_a + plane_gx * dx[kernel_row, kernel_col] + plane_gy * dy[kernel_row, kernel_col]
        highest_residual = np.maximum(highest_residual, shifted - plane_height)

    # NOTE: a degenerate (collinear) footprint has no plane, so it is not checked here
    return fit.solvable & (highest_residual > clearance)
