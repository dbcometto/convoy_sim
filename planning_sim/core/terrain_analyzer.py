# Stage 2: HeightMap -> TerrainMap.  Vehicle-independent features: gradient, grade, steps, roughness, curvature, support.
from dataclasses import dataclass
from itertools import product
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import correlate

from .types import (CURVATURE, GRADE, GX, GY, HEIGHT, ROUGHNESS, STEP_DOWN, STEP_HEIGHT, STEP_UP, SUPPORT, VALID,
                    Grid, GridMap, HeightMap, TerrainMap)


#========================================#
# CONFIG
#========================================#

MIN_WINDOW_CELLS = 3  # a plane needs at least a 3x3 window to be over-determined
SINGULAR_TOLERANCE = 1e-9  # relative determinant below this means the window cells are (nearly) collinear
NEIGHBOR_OFFSETS = [offset for offset in product((-1, 0, 1), repeat=2) if offset != (0, 0)]  # (row, col), 8-neighborhood

UNITS = {GX: "m/m", GY: "m/m", GRADE: "m/m", STEP_UP: "m", STEP_DOWN: "m", STEP_HEIGHT: "m", ROUGHNESS: "m",
         CURVATURE: "1/m", SUPPORT: ""}






#========================================#
# STAGE
#========================================#

def build_terrain_map(height_map: HeightMap, config: SimpleNamespace) -> TerrainMap:
    """Compute the terrain layers from the height map (config is `config.terrain`)"""
    height = height_map[HEIGHT]  # m
    valid = height_map[VALID]

    window_cells = window_size_in_cells(config.window_size_m, height_map.grid)
    layers = fit_local_plane(height, valid, window_cells, height_map.grid.cell_size_m, config.min_support)
    layers.update(compute_steps(height, valid))
    return GridMap(grid=height_map.grid, layers=layers, units=UNITS)






#========================================#
# HELPERS
#========================================#

def window_size_in_cells(window_size_m: float, grid: Grid) -> int:
    """Odd number of cells (at least MIN_WINDOW_CELLS) closest to the window size, so the window has a center cell"""
    cells = round(window_size_m / grid.cell_size_m)
    cells += 1 - cells % 2  # even -> next odd
    return max(cells, MIN_WINDOW_CELLS)



#----------# PLANE FIT #----------#

@dataclass
class PlaneFit:
    """Per-cell least-squares plane z = a + gx*x + gy*y over a kernel footprint (x, y in m from the kernel center)"""
    plane: np.ndarray  # (rows, cols, 3): a (m, relative to `centered`), gx, gy (m/m)
    solvable: np.ndarray  # (rows, cols), bool; False where the kernel's valid cells are (nearly) collinear
    count: np.ndarray  # (rows, cols), valid kernel cells
    sum_xx: np.ndarray  # (rows, cols), m^2
    sum_yy: np.ndarray  # (rows, cols), m^2
    residual_sum_sq: np.ndarray  # (rows, cols), m^2
    centered: np.ndarray  # (rows, cols), m, height minus the map mean (0 where unknown); the frame `a` is in


def kernel_offsets_m(size_cells: int, cell_size_m: float) -> tuple:
    """(dx, dy) in m of each cell of an odd square kernel from its center; columns run along +x, rows along +y"""
    offsets_m = (np.arange(size_cells) - size_cells // 2) * cell_size_m  # m
    return np.meshgrid(offsets_m, offsets_m)


def fit_plane_over_kernel(height: np.ndarray, valid: np.ndarray, kernel: np.ndarray, dx: np.ndarray,
                          dy: np.ndarray) -> PlaneFit:
    """Fit a plane over the valid cells under the kernel (weights, usually 0/1) centered on every cell.
    Outside the map counts as unknown."""
    # Subtracting the mean keeps the sums of squares well conditioned; unknown cells are zeroed and carry weight 0
    mean_height = height[valid].mean() if valid.any() else 0.0  # m
    mask = valid.astype(float)
    centered = np.where(valid, height - mean_height, 0.0)  # m

    def window_sum(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Sum of values*weights*kernel over every window; outside the map counts as zero (invalid)"""
        return correlate(values, weights * kernel, mode="constant", cval=0.0)

    ones = np.ones_like(kernel)
    count = window_sum(mask, ones)  # valid cells in the window
    sum_x, sum_y = window_sum(mask, dx), window_sum(mask, dy)
    sum_xx, sum_xy, sum_yy = window_sum(mask, dx * dx), window_sum(mask, dx * dy), window_sum(mask, dy * dy)
    sum_h, sum_xh, sum_yh = window_sum(centered, ones), window_sum(centered, dx), window_sum(centered, dy)
    sum_hh = window_sum(centered * centered, ones)

    # Normal equations A @ [a, gx, gy] = b, one 3x3 system per cell
    normal_matrix = np.stack([np.stack([count, sum_x, sum_y], axis=-1),
                              np.stack([sum_x, sum_xx, sum_xy], axis=-1),
                              np.stack([sum_y, sum_xy, sum_yy], axis=-1)], axis=-2)  # (rows, cols, 3, 3)
    rhs = np.stack([sum_h, sum_xh, sum_yh], axis=-1)  # (rows, cols, 3)

    determinant = np.linalg.det(normal_matrix)
    scale = count * sum_xx * sum_yy  # determinant magnitude of a well-spread window
    solvable = np.abs(determinant) > SINGULAR_TOLERANCE * np.maximum(scale, np.finfo(float).tiny)
    safe_matrix = np.where(solvable[..., None, None], normal_matrix, np.eye(3))  # placeholder so solve never sees a singular matrix
    plane = np.linalg.solve(safe_matrix, rhs[..., None])[..., 0]  # (rows, cols, 3): a, gx, gy

    # Residual sum of squares: sum(h^2) - beta . b, valid because beta solves the normal equations
    residual_sum_sq = sum_hh - np.einsum("...i,...i", plane, rhs)  # m^2
    return PlaneFit(plane, solvable, count, sum_xx, sum_yy, residual_sum_sq, centered)


def fit_local_plane(height: np.ndarray, valid: np.ndarray, window_cells: int, cell_size_m: float,
                    min_support: float) -> dict:
    """Least-squares plane over the valid cells of each cell's square window (x, y in m from the window center).
    Gives gx, gy, grade, roughness (RMS residual), curvature, and support; NaN where the fit is not trustworthy."""
    dx, dy = kernel_offsets_m(window_cells, cell_size_m)
    fit = fit_plane_over_kernel(height, valid, np.ones((window_cells, window_cells)), dx, dy)

    support = fit.count / window_cells**2
    usable = valid & fit.solvable & (support >= min_support)

    safe_count = np.maximum(fit.count, 1.0)
    roughness = np.sqrt(np.maximum(fit.residual_sum_sq, 0.0) / safe_count)  # m

    # The center cell sits at the window origin, so its plane value is `a`.  For z = k/2 * r^2 the center residual is
    # -k/2 * mean(r^2), which gives the curvature k.
    center_residual = fit.centered - fit.plane[..., 0]  # m
    mean_radius_sq = (fit.sum_xx + fit.sum_yy) / safe_count  # m^2
    curvature = -2.0 * center_residual / np.maximum(mean_radius_sq, np.finfo(float).tiny)  # 1/m

    gx, gy = fit.plane[..., 1], fit.plane[..., 2]  # m/m
    fit_layers = {GX: gx, GY: gy, GRADE: np.hypot(gx, gy), ROUGHNESS: roughness, CURVATURE: curvature}
    layers = {name: np.where(usable, values, np.nan) for name, values in fit_layers.items()}
    layers[SUPPORT] = support
    return layers



#----------# STEPS #----------#

def compute_steps(height: np.ndarray, valid: np.ndarray) -> dict:
    """Largest rise and drop (m, both >= 0) from each cell to its valid 8-neighbors, and the larger of the two.
    NaN where the cell or all of its neighbors are unknown."""
    rows, cols = height.shape
    padded = np.pad(np.where(valid, height, np.nan), 1, constant_values=np.nan)  # outside the map is unknown

    differences = []  # neighbor height minus cell height, m
    for row_offset, col_offset in NEIGHBOR_OFFSETS:
        neighbor = padded[1 + row_offset:1 + row_offset + rows, 1 + col_offset:1 + col_offset + cols]
        differences.append(neighbor - np.where(valid, height, np.nan))
    differences = np.stack(differences)  # (neighbors, rows, cols)

    known = np.isfinite(differences)
    step_up = np.where(known, differences, -np.inf).max(axis=0)  # m, a neighbor higher than the cell
    step_down = np.where(known, -differences, -np.inf).max(axis=0)  # m, a neighbor lower than the cell
    has_neighbor = known.any(axis=0)

    # Clip at 0 so a cell on a pure slope (neighbors only one way) reports a zero rise or drop the other way
    step_up = np.where(has_neighbor, np.maximum(step_up, 0.0), np.nan)
    step_down = np.where(has_neighbor, np.maximum(step_down, 0.0), np.nan)
    return {STEP_UP: step_up, STEP_DOWN: step_down, STEP_HEIGHT: np.fmax(step_up, step_down)}
