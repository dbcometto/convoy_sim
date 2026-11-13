# Changelog Against Hardware

Each simulator has its own section.





## Convoying Sim

Hardware is at the level of the 2026 SSRR Paper: the original no_track control implementation.

### Needs to be added
1. **Covariance time scaling**: process noise is scaled by elapsed time, `P = F P F^T + dt*Q`, with `Q` a per-second rate.  See `convoying_sim/core/ekf.py` (`EKF.update_predict`).  Hardware uses `P = F P F^T + Q`, so its uncertainty growth depends on the update rate.  Retune `q` as a rate (sim: `ekf.q_*` in `convoying_sim/config.yaml`, 0.3 /s).
2. **MPC tuning**: weights `w_vel=0.01`, `w_delta=0.03`, and following distance (`mpc` and `sim.following_distance` in `convoying_sim/config.yaml`).
3. **MPC reachability check**: hold (zero command) if the target point bearing is beyond `ahead_angle`.  See `MPC.solve` in `convoying_sim/core/mpc.py`.
4. **MPC warm start**: reuse the previous plan shifted by one step.  See `MPC.solve` in `convoying_sim/core/mpc.py`.
5. **MPC analytic gradient** (speed only, same objective): `MPC.cost_and_gradient` in `convoying_sim/core/mpc.py` passes the exact gradient to the optimizer instead of finite differences, cutting the solve time several-fold.  Worth adopting if the hardware solve time is a limit.






## Planning Sim

### Needs to be added
1. **Terrain analyzer**: height map → gx, gy, grade, step up/down/height, roughness, curvature, support.  See `planning_sim/core/terrain_analyzer.py` (`build_terrain_map`, `fit_local_plane`, `compute_steps`) and `terrain` in `planning_sim/config.yaml`.
2. **Driveability analyzer**: height + terrain maps → per-heading driveable mask (oriented footprint, step/roughness/pitch/roll limits, underbody clearance from a footprint plane fit).  See `planning_sim/core/driveability_analyzer.py` (`build_driveability_map`, `pose_is_driveable`, `belly_hits_terrain`), the shared `fit_plane_over_kernel` in `terrain_analyzer.py`, and `driveability` and `vehicle` in `planning_sim/config.yaml`.  Tune the vehicle limits on the real F1Tenth.

Other stages are still unimplemented.  Record each stage's changes here as it is implemented.
