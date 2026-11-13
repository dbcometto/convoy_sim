# Stage 6: Path -> Trajectory
from types import SimpleNamespace

import numpy as np

from .types import Path, Trajectory


def build_trajectory(path: Path, config: SimpleNamespace) -> Trajectory:
    """Attach time, speed, and curvature to the path (config is `config.speed_profile`)"""
    # TODO: implement curvature and lateral-acceleration speed limits; placeholder is constant speed, zero curvature
    segment_lengths = np.linalg.norm(np.diff(path.xy, axis=0), axis=1)  # m
    distance = np.concatenate([[0.0], np.cumsum(segment_lengths)])  # m, along the path
    vel = np.full(len(path.xy), config.vel_max)  # m/s
    return Trajectory(t=distance / config.vel_max, xy=path.xy, theta=path.theta, vel=vel, curvature=np.zeros(len(path.xy)))
