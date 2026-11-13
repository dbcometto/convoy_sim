# Plotting for the planning sim: one map view with a switchable layer and a toggleable trajectory
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, RadioButtons, Slider

from .pipeline import DRIVEABILITY_ALL, DRIVEABILITY_ANY, DRIVEABILITY_HEADING, HEADINGS_KEY, LAYER_PREFIX


#========================================#
# CONFIG
#========================================#

LAYER_STYLES = {  # layer name: (stage, title, colormap); layers not listed here fall back to DEFAULT_LAYER_STYLE
    "height": ("Height", "Height (m)", "cividis"),
    "valid": ("Height", "Valid cells", "gray"),
    "gx": ("Terrain", "Gradient x (m/m)", "RdBu_r"),
    "gy": ("Terrain", "Gradient y (m/m)", "RdBu_r"),
    "grade": ("Terrain", "Grade (m/m)", "magma"),
    "step_height": ("Terrain", "Step height (m)", "magma"),
    "step_up": ("Terrain", "Step up (m)", "magma"),
    "step_down": ("Terrain", "Step down (m)", "magma"),
    "roughness": ("Terrain", "Roughness (m)", "magma"),
    "curvature": ("Terrain", "Curvature (1/m)", "RdBu_r"),
    "support": ("Terrain", "Window support", "gray"),
    DRIVEABILITY_ANY: ("Driveability", "Any heading", "RdYlGn"),
    DRIVEABILITY_ALL: ("Driveability", "All headings", "RdYlGn"),
    DRIVEABILITY_HEADING: ("Driveability", "At heading", "RdYlGn"),
    "cost": ("Cost", "Cost", "viridis"),
}
BOOL_LIMITS = (0.0, 1.0)  # fixed color scale for bool layers, so the colors keep their meaning when the data is constant
DEFAULT_LAYER_STYLE = ("Other", "Layer", "viridis")
TRAJECTORY_LABEL = "Trajectory"
GRID_LABEL = "Grid"
TOGGLE_LABELS = [TRAJECTORY_LABEL, GRID_LABEL]  # check button order: indices below follow it
TRAJECTORY_INDEX, GRID_INDEX = 0, 1






#========================================#
# LAYER VIEW
#========================================#

class LayerView:
    """Map window.  Radio buttons (or the up/down keys) pick the layer; a checkbox (or the trajectory key) toggles the trajectory.
    Layers with one map per heading also get a heading slider (or the left/right keys)."""

    def __init__(self, results, config) -> None:
        """Build the figure from a results dict or npz"""
        self.style = config.plot
        self.results = results
        self.layer_names = [key[len(LAYER_PREFIX):] for key in results.keys() if key.startswith(LAYER_PREFIX)]
        self.layer_titles = [self.layer_label(name) for name in self.layer_names]
        self.headings_deg = np.degrees(results[HEADINGS_KEY])  # deg, one per bin of the 3D layers

        self.fig = plt.figure(figsize=self.style.fig_size)
        grid = self.fig.add_gridspec(3, 2, width_ratios=self.style.controls_width_ratios,
                                     height_ratios=self.style.controls_height_ratios)
        self.ax_map = self.fig.add_subplot(grid[:, 1])
        self.layer_selector = RadioButtons(self.fig.add_subplot(grid[0, 0]), self.layer_titles)
        self.trajectory_toggle = CheckButtons(self.fig.add_subplot(grid[1, 0]), TOGGLE_LABELS,
                                              [self.style.do_show_trajectory, self.style.do_show_grid])
        heading_grid = grid[2, 0].subgridspec(2, 1, height_ratios=self.style.heading_height_ratios)
        self.ax_heading = self.fig.add_subplot(heading_grid[0])
        self.ax_compass = self.fig.add_subplot(heading_grid[1])
        self.heading_step_deg = self.headings_deg[1] - self.headings_deg[0]  # deg
        self.heading_slider = Slider(self.ax_heading, "Heading", self.headings_deg[0], self.headings_deg[-1],
                                     valinit=self.headings_deg[0], valstep=self.heading_step_deg, valfmt="%.1f deg")
        self.heading_slider.label.set_position(self.style.heading_label_position)  # above the bar, so it is not cut off
        self.heading_slider.label.set_horizontalalignment("center")
        self.set_control_font_size()

        self.draw_compass()
        self.draw_map()
        self.layer_selector.on_clicked(self.on_layer_selected)
        self.trajectory_toggle.on_clicked(self.on_toggled)
        self.heading_slider.on_changed(self.on_heading_changed)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.show_layer(0)  # before tight_layout, so the title is included in the layout
        self.fig.tight_layout()

    @staticmethod
    def layer_label(name: str) -> str:
        """Radio button text: the stage the layer comes from, then its name without units"""
        stage, title, _ = LAYER_STYLES.get(name, DEFAULT_LAYER_STYLE)
        return f"{stage}: {title.split(' (')[0]}"

    def set_control_font_size(self) -> None:
        """Apply the control font size to the radio, check button, and slider labels"""
        for label in self.layer_selector.labels + self.trajectory_toggle.labels + [self.heading_slider.label,
                                                                                    self.heading_slider.valtext]:
            label.set_fontsize(self.style.control_font_size)

    def draw_compass(self) -> None:
        """Circle with an arrow from its center; the arrow follows the heading in the map's frame (0 deg = +x = right)"""
        limit = self.style.compass_limit
        self.ax_compass.add_patch(plt.Circle((0, 0), 1, fill=False, color=self.style.compass_color))
        self.compass_arrow = self.ax_compass.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(
            arrowstyle="-|>", color=self.style.compass_color, linewidth=self.style.compass_width))
        self.ax_compass.set_xlim(-limit, limit)
        self.ax_compass.set_ylim(-limit, limit)
        self.ax_compass.set_aspect("equal")
        self.ax_compass.axis("off")

    def update_compass(self) -> None:
        """Point the compass arrow at the slider's heading"""
        heading_rad = np.radians(self.headings_deg[self.heading_index()])  # rad, world frame
        self.compass_arrow.xy = (np.cos(heading_rad), np.sin(heading_rad))

    def heading_index(self) -> int:
        """Heading bin picked on the slider"""
        return round((self.heading_slider.val - self.headings_deg[0]) / self.heading_step_deg)

    def draw_map(self) -> None:
        """Create the image, colorbar, trajectory line, and start/goal markers once; layers then only swap the image data"""
        extent = self.results["extent"]  # m, (x_min, x_max, y_min, y_max)
        first_layer = self.results[LAYER_PREFIX + self.layer_names[0]]
        self.image = self.ax_map.imshow(np.zeros(first_layer.shape), origin="lower", extent=extent, interpolation="nearest")
        self.colorbar = self.fig.colorbar(self.image, ax=self.ax_map)

        trajectory_xy = self.results["trajectory_xy"]  # m
        (self.trajectory_line,) = self.ax_map.plot(trajectory_xy[:, 0], trajectory_xy[:, 1], color=self.style.trajectory_color,
                                                   linewidth=self.style.trajectory_width, label=TRAJECTORY_LABEL,
                                                   visible=self.style.do_show_trajectory)
        start_x, start_y = self.results["start_xy"]  # m
        goal_x, goal_y = self.results["goal_xy"]  # m
        self.ax_map.plot(start_x, start_y, marker="o", linestyle="", color=self.style.start_color,
                         markersize=self.style.marker_size, label="Start")
        self.ax_map.plot(goal_x, goal_y, marker="*", linestyle="", color=self.style.goal_color,
                         markersize=self.style.marker_size, label="Goal")
        self.ax_map.legend(loc=self.style.legend_loc, bbox_to_anchor=self.style.legend_anchor, borderaxespad=0,
                           ncols=self.style.legend_columns, fontsize=self.style.legend_font_size)
        self.ax_map.grid(True, color=self.style.grid_color, linewidth=self.style.grid_width)  # style set once
        self.ax_map.grid(self.style.do_show_grid)  # then only visibility changes
        self.ax_map.set_xlabel("x (m)")
        self.ax_map.set_ylabel("y (m)")

    def show_layer(self, index: int) -> None:
        """Display one layer (the slider's heading bin for 3D layers): swap the image data, color map, and color limits"""
        name = self.layer_names[index]
        stage, title, colormap = LAYER_STYLES.get(name, DEFAULT_LAYER_STYLE)
        title = f"{stage}: {title}"
        layer = self.results[LAYER_PREFIX + name]
        has_headings = layer.ndim == 3
        if has_headings:
            layer = layer[self.heading_index()]
            title = f"{title} {self.headings_deg[self.heading_index()]:.1f} deg"
        data = np.ma.masked_invalid(layer.astype(float))  # inf (blocked cells) shows as blank

        if layer.dtype == bool:
            low, high = BOOL_LIMITS
        else:
            low, high = float(data.min()), float(data.max())
        if low == high:
            low, high = low - self.style.flat_layer_margin, high + self.style.flat_layer_margin  # constant layer: avoid a zero-width scale

        self.ax_heading.set_visible(has_headings)
        self.ax_compass.set_visible(has_headings)
        self.update_compass()
        self.image.set_data(data)
        self.image.set_cmap(colormap)
        self.image.set_clim(low, high)
        self.ax_map.set_title(title)
        self.fig.canvas.draw_idle()

    def on_layer_selected(self, label: str) -> None:
        """Radio button callback"""
        self.show_layer(self.layer_titles.index(label))

    def on_heading_changed(self, value: float) -> None:
        """Slider callback: redraw the current layer at the new heading"""
        self.show_layer(self.layer_titles.index(self.layer_selector.value_selected))

    def on_toggled(self, label: str) -> None:
        """Checkbox callback: show or hide the trajectory line and the grid"""
        status = self.trajectory_toggle.get_status()
        self.trajectory_line.set_visible(status[TRAJECTORY_INDEX])
        self.ax_map.grid(status[GRID_INDEX])
        self.fig.canvas.draw_idle()

    def on_key(self, event) -> None:
        """Keyboard shortcuts, routed through the buttons so the widgets stay in sync"""
        if event.key == self.style.trajectory_key:
            self.trajectory_toggle.set_active(TRAJECTORY_INDEX)
        elif event.key == self.style.grid_key:
            self.trajectory_toggle.set_active(GRID_INDEX)
        elif event.key in (self.style.layer_up_key, self.style.layer_down_key):
            step = -1 if event.key == self.style.layer_up_key else 1
            current = self.layer_titles.index(self.layer_selector.value_selected)
            self.layer_selector.set_active((current + step) % len(self.layer_names))
        elif event.key in (self.style.heading_left_key, self.style.heading_right_key):
            step = -1 if event.key == self.style.heading_left_key else 1
            next_index = (self.heading_index() + step) % len(self.headings_deg)
            self.heading_slider.set_val(self.headings_deg[next_index])






#========================================#
# ENTRY
#========================================#

def plot_results(results, config) -> None:
    """Open the layer view.  `results` is a results dict or npz."""
    view = LayerView(results, config)  # held so the widgets stay alive until the window closes
    plt.show()
