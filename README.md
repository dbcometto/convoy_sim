# Convoy Simulation with No Track

This repo simulates off-track convoy operations: a leader vehicle drives a planned trajectory over rough terrain, and followers each estimate and track the vehicle ahead.  The target platform is an F1Tenth-scale vehicle.

There are two simulators, one for each half of the problem:

| Simulator | Question it answers | Status |
|---|---|---|
| `convoying_sim/` | How do followers estimate and track the vehicle ahead (EKF + MPC)? | Complete |
| `planning_sim/` | How does the leader turn a height map into a trajectory? | Partly built |

See `docs/spec.md` for the architecture, `docs/notes.md` for the design history, and `docs/to_hardware.md` for the differences versus the hardware.

Install the dependencies with `pip install -r requirements.txt`.


## Convoying Sim

`python convoying_sim/main.py`

All parameters are in `convoying_sim/config.yaml`.  Set `run.visualization_mode` to `LIVE` (watch it run) or `END_PLOT` (run fast, plot world view and errors at the end).
`python convoying_sim/plot.py` plots a previously saved results file.


## Planning Sim

`python planning_sim/main.py`

Runs the leader's planning pipeline, then saves and plots the result:

1. Height map (image + yaml in `planning_sim/maps/<name>/`) is loaded.
2. Terrain analyzer: gradient, steps, roughness, curvature.
3. Driveability analyzer: for each heading, where the vehicle's footprint may sit, given its slope, step, roughness, and clearance limits.
4. Costmap, planner, and speed profile: stubs with placeholder outputs for now.

All parameters, including the map choice and the vehicle's dimensions and limits, are in `planning_sim/config.yaml`.  In the plot window, pick a layer with the radio buttons or the up/down keys, toggle the trajectory (`t`) and grid (`g`), and step through headings with the slider or the left/right keys.
`python planning_sim/plot.py` plots a previously saved results file.
