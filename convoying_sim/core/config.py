# Loads config.yaml into nested attribute-style namespaces
import math
from pathlib import Path
from types import SimpleNamespace

import yaml


#========================================#
# CONFIG
#========================================#

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"  # one level up, next to main.py
DEGREE_SUFFIX = "_deg"
RADIAN_SUFFIX = "_rad"






#========================================#
# LOADING
#========================================#

def to_namespace(value):
    """Recursively turn dicts into namespaces, converting `_deg` keys to `_rad`"""
    if not isinstance(value, dict):
        return value

    converted = {}
    for key, item in value.items():
        if key.endswith(DEGREE_SUFFIX):
            converted[key[:-len(DEGREE_SUFFIX)] + RADIAN_SUFFIX] = math.radians(item)
        else:
            converted[key] = to_namespace(item)
    return SimpleNamespace(**converted)


def load_config(path: Path = CONFIG_PATH) -> SimpleNamespace:
    """Load the yaml config, with the output file resolved relative to the yaml's folder"""
    with open(path) as file:
        config = to_namespace(yaml.safe_load(file))

    config.run.output_file = Path(path).parent / config.run.output_file
    return config
