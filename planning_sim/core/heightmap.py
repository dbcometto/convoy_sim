# Stage 1: builds the HeightMap from a grayscale image and the yaml beside it
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .types import HEIGHT, VALID, Grid, GridMap, HeightMap


#========================================#
# CONFIG
#========================================#

GRAY_MAX = 255  # 8-bit image, so gray 0..255 spans height_min_m..height_max_m






#========================================#
# STAGE
#========================================#

def load_height_map(map_folder: Path) -> HeightMap:  # a GridMap with HEIGHT and VALID layers
    """Load `<name>.png` and `<name>.yaml` from a map folder named `<name>` into a HeightMap
    (gray 0..255 maps linearly to the yaml's height range)"""
    image_path = map_folder / f"{map_folder.name}.png"
    with open(map_folder / f"{map_folder.name}.yaml") as file:
        meta = yaml.safe_load(file)

    # Flip rows so that row 0 is the lowest world y
    gray = np.flipud(np.array(Image.open(image_path).convert("L"), dtype=float))
    height = meta["height_min_m"] + gray / GRAY_MAX * (meta["height_max_m"] - meta["height_min_m"])  # m

    rows, cols = gray.shape
    origin_col, origin_row_from_top = meta["origin_px"]
    grid = Grid(cell_size_m=meta["cell_size_m"], origin_cell=(origin_col, rows - 1 - origin_row_from_top), shape=(rows, cols))

    # TODO: PNGs have no unknown cells; RTAB-Map maps will, so mark them invalid when that source exists
    return GridMap(grid=grid, layers={HEIGHT: height, VALID: np.ones(gray.shape, dtype=bool)},
                   units={HEIGHT: "m", VALID: ""})
