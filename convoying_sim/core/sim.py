# The simulation logic for off-track convoy operations.  Holds the state and steps it; see main.py to run it.
import math
from collections import deque
from enum import IntEnum

import numpy as np

from .vehicle import Vehicle, LEADER_ID
from .ekf import EKF
from .mpc import MPC


#========================================#
# CONFIG
#========================================#

class LEADER_MODES(IntEnum):
    MANUAL = 0  # driven through `leader_manual`, which a live view updates from the keyboard
    CIRCLE = 1
    RANDOM = 2






#========================================#
# SIMULATION
#========================================#

def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]"""
    return math.atan2(math.sin(angle), math.cos(angle))


class Simulation():
    """A convoy of one leader and several followers, each follower estimating and tracking the vehicle ahead"""

    def __init__(self, config) -> None:
        """Build the vehicles, filters, and controllers from the config"""
        self.config = config
        self.dt = config.sim.dt  # s
        self.following_distance = config.sim.following_distance  # m
        self.leader_mode = LEADER_MODES[config.leader.mode]
        self.step_count = 0

        # Vehicles, filters, and controllers, with the leader first and each follower tracking the vehicle ahead
        self.follower_ids = list(range(1, config.sim.num_followers + 1))
        self.vehicles = {LEADER_ID: self.make_vehicle(LEADER_ID, x=0)}
        self.filters = {}
        self.controllers = {}
        for follower_id in self.follower_ids:
            self.vehicles[follower_id] = self.make_vehicle(follower_id, x=-self.following_distance*follower_id)
            self.filters[follower_id] = EKF(time=0.0, veh_id=follower_id, target_id=follower_id - 1,
                                            x0=-self.following_distance, ekf_config=config.ekf,
                                            sensors_config=config.sensors, wheelbase=config.vehicle.wheelbase)
            self.controllers[follower_id] = MPC(config.mpc, dt=self.dt, wheelbase=config.vehicle.wheelbase)

        self.inputs = {vehicle_id: {'vel': 0.0, 'delta': 0.0} for vehicle_id in self.vehicles}
        self.leader_manual = {'vel': 0.0, 'delta': 0.0}  # NOTE: only used in MANUAL mode, which needs a live view to set it

        # Independent streams so sensor noise draws (which depend on FOV) never shift the random leader's commands
        sensor_seed, leader_seed = np.random.SeedSequence(config.sim.seed).spawn(2)
        self.sensor_random = np.random.default_rng(sensor_seed)
        self.leader_random = np.random.default_rng(leader_seed)

        self.trails = {vehicle_id: deque(maxlen=config.sim.max_past_poses) for vehicle_id in self.vehicles}

        # Point behind the leader's start along its initial heading; used only for lateral deviation, never plotted
        leader_start = self.vehicles[LEADER_ID]
        path_prefix_length = self.following_distance * config.sim.num_followers + config.sim.path_prefix_margin  # m
        self.path_prefix = np.array([[leader_start.x - path_prefix_length*math.cos(leader_start.theta),
                                      leader_start.y - path_prefix_length*math.sin(leader_start.theta)]])
        self.lateral_progress = {}  # last matched trail segment per follower

        self.history = {
            "t": [],
            "poses": [],  # per step: (x, y, theta) for each vehicle, ordered by id
            "est": {follower_id: [] for follower_id in self.follower_ids},
            "trk": {follower_id: [] for follower_id in self.follower_ids},
            "lat": {follower_id: [] for follower_id in self.follower_ids},
        }


    @property
    def time(self) -> float:
        """Simulated time in seconds, independent of the wall clock so runs are reproducible"""
        return self.step_count * self.dt


    def make_vehicle(self, vehicle_id: int, x: float) -> Vehicle:
        """Create a stationary vehicle at the start line"""
        return Vehicle(vehicle_id=vehicle_id, x=x, y=self.config.sim.start_y, theta=0, vel=0,
                       wheelbase=self.config.vehicle.wheelbase)




    #===============# SENSORS #===============#

    def get_lidar_meas(self, veh_id_center: int, veh_id_target: int) -> tuple:
        """Generate a noisy lidar centroid in the center vehicle's body frame, or None if dropped or out of FOV"""
        sensors = self.config.sensors
        if sensors.do_drop_lidar:
            return None, None

        vehicle_target = self.vehicles[veh_id_target]
        vehicle_center = self.vehicles[veh_id_center]

        dx = vehicle_target.x - vehicle_center.x  # m, world frame
        dy = vehicle_target.y - vehicle_center.y  # m, world frame

        # Check if target is within FOV before adding noise
        bearing = wrap_angle(math.atan2(dy, dx) - vehicle_center.theta)
        if abs(bearing) > sensors.lidar_fov_rad / 2:
            return None, None

        dx += self.sensor_random.normal(0, sensors.lidar_noise_x)
        dy += self.sensor_random.normal(0, sensors.lidar_noise_y)

        # Rotate into follower body frame
        c, s = math.cos(-vehicle_center.theta), math.sin(-vehicle_center.theta)
        x_meas = c*dx - s*dy
        y_meas = s*dx + c*dy

        return x_meas, y_meas


    def get_pozyx_meas(self, veh_id_center: int, veh_id_target: int) -> float:
        """Generate a noisy pozyx range, or None if dropped"""
        sensors = self.config.sensors
        if sensors.do_drop_pozyx:
            return None

        vehicle_target = self.vehicles[veh_id_target]
        vehicle_center = self.vehicles[veh_id_center]

        dx = vehicle_target.x - vehicle_center.x  # m
        dy = vehicle_target.y - vehicle_center.y  # m
        return math.hypot(dx, dy) + self.sensor_random.normal(0, sensors.pozyx_noise_r)




    #===============# METRICS #===============#

    def true_relative(self, veh_id_center: int, veh_id_target: int) -> tuple:
        """True pose of the target in the center vehicle's body frame"""
        vehicle_target = self.vehicles[veh_id_target]
        vehicle_center = self.vehicles[veh_id_center]

        dx = vehicle_target.x - vehicle_center.x
        dy = vehicle_target.y - vehicle_center.y
        c, s = math.cos(-vehicle_center.theta), math.sin(-vehicle_center.theta)
        return c*dx - s*dy, s*dx + c*dy, wrap_angle(vehicle_target.theta - vehicle_center.theta)


    def lateral_deviation(self, point: tuple, trail: deque, key: int) -> float:
        """Min distance from a point to the polyline of a trail, searching only near the previous match
        so a path that loops back on itself doesn't match a far-away (in time) portion"""
        if len(trail) < 1:
            return float('nan')

        # Hidden extension of the path backwards from its start so followers behind the start measure cross-track
        path_points = np.vstack([self.path_prefix, np.asarray(trail)])
        segment_starts, segment_ends = path_points[:-1], path_points[1:]

        # Restrict to a window of segments around the last match
        window = self.config.sim.lateral_window
        last_match = self.lateral_progress.get(key, 0)
        low = max(0, last_match - window)
        high = min(len(segment_starts), last_match + window + 1)
        segment_starts, segment_ends = segment_starts[low:high], segment_ends[low:high]

        segments = segment_ends - segment_starts
        to_point = np.asarray(point) - segment_starts
        segment_length_sq = np.einsum('ij,ij->i', segments, segments)
        fraction = np.divide(np.einsum('ij,ij->i', to_point, segments), segment_length_sq,
                             out=np.zeros_like(segment_length_sq), where=segment_length_sq > np.finfo(float).eps)
        closest = segment_starts + np.clip(fraction, 0, 1)[:, None] * segments
        distances = np.linalg.norm(closest - point, axis=1)
        best = int(np.argmin(distances))
        self.lateral_progress[key] = low + best
        return float(distances[best])


    def compute_metrics(self) -> tuple:
        """Per-follower estimator error, tracking error, and lateral deviation"""
        est, trk, lat = {}, {}, {}
        for follower_id, ekf in self.filters.items():
            true_x, true_y, true_theta = self.true_relative(ekf.veh_id, ekf.target_id)

            # Estimator error over position components
            est[follower_id] = math.hypot(ekf.x[0] - true_x, ekf.x[1] - true_y)

            # Tracking error: distance to the target point computed from true state
            controller = self.controllers[follower_id]
            saved_target = controller.target
            goal_x, goal_y, _ = controller.get_target_point(true_x, true_y, true_theta, distance=self.following_distance)
            controller.target = saved_target
            trk[follower_id] = math.hypot(goal_x, goal_y)

            # Lateral deviation from the leader's trail
            vehicle = self.vehicles[follower_id]
            lat[follower_id] = self.lateral_deviation((vehicle.x, vehicle.y), self.trails[LEADER_ID], key=follower_id)
        return est, trk, lat




    #===============# CONTROL #===============#

    def get_leader_inputs(self) -> dict:
        """Leader command for the current time, or None to keep the previous command"""
        leader = self.config.leader
        time_in_period = self.time % leader.period  # s

        if self.leader_mode == LEADER_MODES.CIRCLE:
            return {'vel': leader.vel, 'delta': leader.delta_rad}

        if self.leader_mode == LEADER_MODES.RANDOM and time_in_period < leader.resample_window:
            delta = self.leader_random.uniform(-leader.max_angle_rad, leader.max_angle_rad)  # drawn first to keep the seed's sequence
            vel = self.leader_random.uniform(0, leader.vel)
            return {'vel': vel, 'delta': delta}

        return None


    def get_crosstalk(self, ekf: EKF) -> dict:
        """Vehicle and target inputs shared with an EKF, or none if crosstalk is disabled"""
        if self.config.sim.do_enable_crosstalk:
            return {'vel': self.inputs[ekf.veh_id]['vel'], 'delta': self.inputs[ekf.veh_id]['delta'],
                    'vel_t': self.inputs[ekf.target_id]['vel'], 'delta_t': self.inputs[ekf.target_id]['delta']}
        return {'vel': None, 'delta': None, 'vel_t': None, 'delta_t': None}




    #===============# STEPPING #===============#

    def step(self) -> None:
        """Advance the estimators, controllers, and true dynamics by one sim step, and record the results"""
        self.step_count += 1
        current_time = self.time  # s

        # State estimate updates
        for ekf in self.filters.values():
            crosstalk = self.get_crosstalk(ekf)

            # Lidar update
            x_meas_lidar, y_meas_lidar = self.get_lidar_meas(ekf.veh_id, ekf.target_id)
            ekf.update_lidar(x_meas_lidar, y_meas_lidar, current_time, **crosstalk)

            # POZYX update
            r_meas_pozyx = self.get_pozyx_meas(ekf.veh_id, ekf.target_id)
            ekf.update_pozyx(r_meas_pozyx, current_time, **crosstalk)

            # Final predict update
            ekf.update_predict(current_time, **crosstalk)

        # Leader control
        if self.leader_mode == LEADER_MODES.MANUAL:
            self.inputs[LEADER_ID] = self.leader_manual
        else:
            new_input = self.get_leader_inputs()
            if new_input is not None:
                self.inputs[LEADER_ID] = new_input

        # Follower control
        for follower_id, controller in self.controllers.items():
            x_targ, y_targ, theta_targ, _ = self.filters[follower_id].x
            vel, delta = controller.solve(x_targ, y_targ, theta_targ, distance=self.following_distance)
            self.inputs[follower_id] = {'vel': vel, 'delta': delta}

        if current_time < self.config.sim.sit_time:
            for vehicle_id in self.inputs:
                self.inputs[vehicle_id] = {'vel': self.config.sim.sit_vel, 'delta': 0}

        # True dynamics
        for vehicle in self.vehicles.values():
            vehicle.step(self.inputs[vehicle.id]['vel'], self.inputs[vehicle.id]['delta'], self.dt)
            self.trails[vehicle.id].append((vehicle.x, vehicle.y))

        self.record_step()


    def record_step(self) -> None:
        """Record poses and metrics for the plots"""
        est, trk, lat = self.compute_metrics()
        self.history["t"].append(self.time)
        self.history["poses"].append([(vehicle.x, vehicle.y, vehicle.theta) for vehicle in self.vehicles.values()])
        for follower_id in self.follower_ids:
            self.history["est"][follower_id].append(est[follower_id])
            self.history["trk"][follower_id].append(trk[follower_id])
            self.history["lat"][follower_id].append(lat[follower_id])


    def build_results(self) -> dict:
        """Package the history as arrays for saving and plotting"""
        ids = self.follower_ids
        return {
            "t": np.array(self.history["t"]),
            "poses": np.array(self.history["poses"]),  # (steps, vehicles, 3)
            "vehicle_ids": np.array(list(self.vehicles)),
            "follower_ids": np.array(ids),
            "est": np.array([self.history["est"][k] for k in ids]).T,  # (steps, followers)
            "trk": np.array([self.history["trk"][k] for k in ids]).T,
            "lat": np.array([self.history["lat"][k] for k in ids]).T,
            "sim_dt": self.dt,
            "lidar_fov": self.config.sensors.lidar_fov_rad,
        }
