# Runs the planning pipeline: `python planning_sim/main.py`.  Everything tunable lives in config.yaml.
import numpy as np

from core import plotting
from core.config import load_config
from core.pipeline import build_results, run_pipeline


#========================================#
# MAIN
#========================================#

def main() -> None:
    """Load the config, run the pipeline, then save and plot the results"""
    config = load_config()
    results = build_results(run_pipeline(config), config)

    if config.run.do_save_results:
        config.run.output_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(config.run.output_file, **results)
        print(f"Saved results to {config.run.output_file}")

    plotting.plot_results(results, config)


if __name__ == "__main__":
    main()
