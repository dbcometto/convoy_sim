# The EKF class
import math

import numpy as np


#========================================#
# CONFIG
#========================================#

STATE_SIZE = 4  # x, y, theta, vel of the target in the follower's body frame






#========================================#
# EKF
#========================================#

class EKF():
    """An EKF for Convoy Operations"""

    def __init__(self, time: float, veh_id: int, target_id: int, x0: float, ekf_config, sensors_config,
                 wheelbase: float, y0: float = 0, theta0: float = 0, vel0: float = 0) -> None:
        """Init an EKF estimating the target's pose in the follower's body frame"""
        self.veh_id = veh_id
        self.target_id = target_id
        self.last_time = time  # s

        # State
        self.x = np.array([x0, y0, theta0, vel0])
        self.P = np.eye(STATE_SIZE)
        self.Q = np.diag([ekf_config.q_x, ekf_config.q_y, ekf_config.q_theta, ekf_config.q_vel])  # per second
        self.R_lidar = np.diag([sensors_config.lidar_noise_x, sensors_config.lidar_noise_y])
        self.R_pozyx = sensors_config.pozyx_noise_r

        # Parameters
        self.wheelbase = wheelbase  # m


    def update_lidar(self, centroid_x: float, centroid_y: float, time: float,
                     vel: float = None, delta: float = None, vel_t: float = None, delta_t: float = None) -> None:
        """Update the EKF based on the lidar centroid"""
        if centroid_x is not None and centroid_y is not None:
            self.update_predict(time, vel, delta, vel_t, delta_t)  # predict to measurement time

            H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
            z = np.array([centroid_x, centroid_y])
            y = z - H @ self.x
            S = H @ self.P @ H.T + self.R_lidar
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(STATE_SIZE) - K @ H) @ self.P


    def update_pozyx(self, range_meas: float, time: float,
                     vel: float = None, delta: float = None, vel_t: float = None, delta_t: float = None) -> None:
        """Update the EKF based on the pozyx distance"""
        if range_meas is not None:
            self.update_predict(time, vel, delta, vel_t, delta_t)  # predict to measurement time

            r_pred = math.sqrt(self.x[0]**2 + self.x[1]**2)
            H = np.array([[self.x[0]/r_pred, self.x[1]/r_pred, 0, 0]])
            y = range_meas - r_pred
            S = H @ self.P @ H.T + self.R_pozyx
            K = self.P @ H.T / S
            self.x = self.x + K.flatten() * y
            self.P = (np.eye(STATE_SIZE) - K @ H) @ self.P


    def update_predict(self, time: float,
                       vel: float = None, delta: float = None, vel_t: float = None, delta_t: float = None) -> None:
        """Predict forward using the dynamics"""
        dt = time - self.last_time  # s
        self.last_time = time

        # Unpack state
        x, y, theta, target_vel = self.x

        # Current vehicle inputs
        vel = vel if vel is not None else 0
        delta = delta if delta is not None else 0
        theta_dot = vel / self.wheelbase * math.tan(delta)

        # Target (leader) inputs
        target_vel = vel_t if vel_t is not None else target_vel
        delta_t = delta_t if delta_t is not None else 0

        # State propagation
        self.x[0] += dt * (target_vel * math.cos(theta) - vel + y * theta_dot)
        self.x[1] += dt * (target_vel * math.sin(theta) - x * theta_dot)
        self.x[2] += dt * (target_vel / self.wheelbase * math.tan(delta_t) - theta_dot)
        self.x[3] = target_vel

        # Wrap state
        self.x[2] = math.atan2(math.sin(self.x[2]), math.cos(self.x[2]))

        # Jacobian F
        F = np.array([
            [1,             dt*theta_dot, -dt*target_vel*math.sin(theta), dt*math.cos(theta)                ],
            [-dt*theta_dot, 1,             dt*target_vel*math.cos(theta), dt*math.sin(theta)                ],
            [0,             0,             1,                             dt*math.tan(delta_t)/self.wheelbase],
            [0,             0,             0,                             1                                 ]
        ])

        # Covariance propagation
        # NOTE: Q is a rate, so it is scaled by the elapsed time; otherwise the uncertainty growth depends on the
        # update rate.  This is not yet in the hardware version (see to_hardware.md).
        self.P = F @ self.P @ F.T + dt*self.Q
