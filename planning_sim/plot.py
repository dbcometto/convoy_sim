# Plots a previously saved results file: `python planning_sim/plot.py`.  The file is `run.output_file` in config.yaml.
import numpy as np

from core.config import load_config
from core.plotting import plot_results


#========================================#
# MAIN
#========================================#

def main() -> None:
    """Load the saved results and plot them"""
    config = load_config()
    plot_results(np.load(config.run.output_file), config)


if __name__ == "__main__":
    main()
