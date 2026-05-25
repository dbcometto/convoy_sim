# The EKF class
import numpy as np
import math

class EKF():
    """An EKF for Convoy Operations"""
    def __init__(self, time, veh_id = 1, target_id = 0, x0 = 0, y0 = 0, theta0 = 0, v0 = 0, 
                 qx = 0.001, qy = 0.001, qtheta = 0.001, qv = 0.001, rx = 0.05, ry = 0.05, rr = 0.05,
                 L = 0.33) -> None:
        """Init a vehicle with true state"""
        self.veh_id = veh_id
        self.target_id = target_id
        self.last_time = time
        
        # State
        self.x = np.array([x0, y0, theta0, v0])
        self.P = np.eye(4)
        self.Q = np.diag([qx, qy, qtheta, qv])
        self.R_lidar = np.diag([rx, ry])
        self.R_pozyx = rr

        # Parameters
        self.L = L



    def update_lidar(self, centroid_x, centroid_y, time, vel=None, delta=None, vel_t=None, delta_t=None) -> None:
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
            self.P = (np.eye(4) - K @ H) @ self.P



    def update_pozyx(self, range, time, vel=None, delta=None, vel_t=None, delta_t=None) -> None:
        """Update the EKF based on the pozyx distance"""
        if range is not None:
            self.update_predict(time, vel, delta, vel_t, delta_t)  # predict to measurement time

            r_pred = math.sqrt(self.x[0]**2 + self.x[1]**2)
            H = np.array([[self.x[0]/r_pred, self.x[1]/r_pred, 0, 0]])
            z = range
            y = z - r_pred
            S = H @ self.P @ H.T + self.R_pozyx
            K = self.P @ H.T / S
            self.x = self.x + K.flatten() * y
            self.P = (np.eye(4) - K @ H) @ self.P





    def update_predict(self, time, vel=None, delta=None, vel_t=None, delta_t=None) -> None:
        """Predict forward using the dynamics"""
        dt = time - self.last_time
        self.last_time = time

        # Unpack state
        x, y, theta, v = self.x

        # Current vehicle inputs
        vel = vel if vel is not None else 0
        delta = delta if delta is not None else 0
        theta_dot = vel / self.L * math.tan(delta)

        # Target (leader) inputs
        v = vel_t if vel_t is not None else v
        delta_t = delta_t if delta_t is not None else 0

        # State propagation
        self.x[0] += dt * (v * math.cos(theta) - vel + y * theta_dot)
        self.x[1] += dt * (v * math.sin(theta) - x * theta_dot)
        self.x[2] += dt * (v / self.L * math.tan(delta_t) - theta_dot)
        self.x[3] = v

        # Wrap state
        self.x[2] = math.atan2(math.sin(self.x[2]), math.cos(self.x[2]))

        # Jacobian F
        F = np.array([
            [1,           dt*theta_dot, -dt*v*math.sin(theta), dt*math.cos(theta)         ],
            [-dt*theta_dot, 1,           dt*v*math.cos(theta),  dt*math.sin(theta)        ],
            [0,           0,             1,                     dt*math.tan(delta_t)/self.L],
            [0,           0,             0,                     1                         ]
        ])

        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q


