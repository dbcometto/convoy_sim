# The MPC class
import math

import numpy as np
from scipy.optimize import minimize


#========================================#
# CONFIG
#========================================#

INPUTS_PER_STEP = 2  # the flat input array alternates vel/delta






#========================================#
# HELPERS
#========================================#

def suffix_sum(values: np.ndarray) -> np.ndarray:
    """Sum of each entry and every entry after it"""
    return np.cumsum(values[::-1])[::-1]






#========================================#
# MPC
#========================================#

class MPC():
    """An MPC controller for Convoy Operations"""

    def __init__(self, mpc_config, dt: float, wheelbase: float) -> None:
        """Init an MPC controller for Convoy Operations"""
        self.horizon = mpc_config.horizon
        self.dt = dt  # s, must match the simulation step
        self.target_method = mpc_config.target_method
        self.degenerate_denominator = mpc_config.degenerate_denominator
        self.last_u = None
        self.target = None  # (x, y, theta) in the follower's body frame

        # Weights
        self.w_x = mpc_config.w_x
        self.w_y = mpc_config.w_y
        self.w_theta = mpc_config.w_theta
        self.w_vel = mpc_config.w_vel
        self.w_delta = mpc_config.w_delta

        # Constraints
        self.vel_min = mpc_config.vel_min  # m/s
        self.vel_max = mpc_config.vel_max  # m/s
        self.delta_min = mpc_config.delta_min_rad  # rad
        self.delta_max = mpc_config.delta_max_rad  # rad

        # Parameters
        self.wheelbase = wheelbase  # m
        self.ahead_angle = mpc_config.ahead_angle_rad  # rad, target bearing beyond this is unreachable (can't reverse)


    def get_target_point(self, x_targ: float, y_targ: float, theta_targ: float, distance: float, method: str = None) -> tuple:
        """Calculate the target point, `distance` behind the target.  Methods include `linear`, `hitch`, `curved`, and `offhooked`."""
        method = method if method is not None else self.target_method
        position = np.array([x_targ, y_targ])
        heading = np.array([math.cos(theta_targ), math.sin(theta_targ)])

        if method == 'linear':
            # Calculate target point d backwards along line from follower to leader and match heading to that line
            direction = position/np.linalg.norm(position)
            x, y = position - distance*direction
            theta = math.atan2(y_targ, x_targ)

        elif method == 'hitch':
            # Calculate target point d backwards along leader heading
            x, y = position - distance*heading
            theta = theta_targ

        elif method == 'curved':
            # Circle tangent to leader heading at leader, passing through follower
            denom = x_targ * math.sin(theta_targ) - y_targ * math.cos(theta_targ)

            if abs(denom) < self.degenerate_denominator:
                # Degenerate case - fall back to linear
                return self.get_target_point(x_targ, y_targ, theta_targ, distance, method='linear')

            radius = (x_targ**2 + y_targ**2) / (2 * denom)

            # Circle center
            center_x = x_targ - radius * math.sin(theta_targ)
            center_y = y_targ + radius * math.cos(theta_targ)

            # Angle from center to leader, then target is `distance` meters back along arc from leader
            phi_leader = math.atan2(y_targ - center_y, x_targ - center_x)
            phi_target = phi_leader - distance / radius

            x = center_x + radius * math.cos(phi_target)
            y = center_y + radius * math.sin(phi_target)
            theta = phi_target + math.pi/2 * math.copysign(1, radius)

        elif method == 'offhooked':
            # Calculate linear point distance/2 backwards from distance/2 behind the vehicle (half at the hitch, half at the link)

            # Calculate hitch point
            hitch = position - distance/2*heading

            # Calculate link point
            hitch_direction = hitch/np.linalg.norm(hitch)
            x, y = hitch - distance/2*hitch_direction
            theta = math.atan2(y, x)

        else:
            raise ValueError(f"Unknown target method: {method}")

        self.target = x, y, theta
        return self.target




    #===============# HELPERS #===============#

    def generate_rollout(self, u: np.ndarray) -> tuple:
        """Rollout a control input trajectory and return the (x, y, theta) arrays, one entry per step"""
        # NOTE: u is a flat array of alternating vel/delta pairs; vectorized because the optimizer calls this thousands of times
        vel = u[::INPUTS_PER_STEP]
        delta = u[1::INPUTS_PER_STEP]

        theta = np.cumsum(self.dt * vel / self.wheelbase * np.tan(delta))

        # Position advances along the heading from before the step
        previous_theta = np.concatenate(([0.0], theta[:-1]))
        x = np.cumsum(self.dt * vel * np.cos(previous_theta))
        y = np.cumsum(self.dt * vel * np.sin(previous_theta))
        return x, y, theta


    def cost_and_gradient(self, u: np.ndarray, x_t: float, y_t: float, theta_t: float) -> tuple:
        """Calculate the cost of a control input trajectory and its gradient with respect to u"""
        # NOTE: the analytic gradient replaces scipy's finite differences, which needed ~20 cost calls per gradient
        vel = u[::INPUTS_PER_STEP]
        delta = u[1::INPUTS_PER_STEP]
        x, y, theta = self.generate_rollout(u)
        previous_theta = np.concatenate(([0.0], theta[:-1]))

        cost = (np.sum(self.w_x*(x-x_t)**2 + self.w_y*(y-y_t)**2 + self.w_theta*(theta-theta_t)**2)
                + np.sum(self.w_vel*vel**2 + self.w_delta*delta**2))

        # Sensitivity of the cost to each state, summed over every later step the input still affects
        x_sens = suffix_sum(2*self.w_x*(x-x_t))
        y_sens = suffix_sum(2*self.w_y*(y-y_t))
        theta_sens = suffix_sum(2*self.w_theta*(theta-theta_t))

        # A turn also rotates the heading used by every later position step
        heading_effect = self.dt * vel * (-np.sin(previous_theta)*x_sens + np.cos(previous_theta)*y_sens)
        turn_sens = theta_sens + np.append(suffix_sum(heading_effect)[1:], 0.0)  # sensitivity to this step's heading change

        vel_grad = (self.dt*(np.cos(previous_theta)*x_sens + np.sin(previous_theta)*y_sens)
                    + turn_sens*self.dt*np.tan(delta)/self.wheelbase + 2*self.w_vel*vel)
        delta_grad = turn_sens*self.dt*vel/self.wheelbase/np.cos(delta)**2 + 2*self.w_delta*delta

        gradient = np.empty_like(u)
        gradient[::INPUTS_PER_STEP] = vel_grad
        gradient[1::INPUTS_PER_STEP] = delta_grad
        return float(cost), gradient






    #===============# MAIN SOLVER #===============#

    def solve(self, x_targ: float, y_targ: float, theta_targ: float, distance: float, method: str = None) -> tuple:
        """Given target vehicle state, output control inputs (vel, delta)"""

        # Find target pose in relative frame
        x_t, y_t, theta_t = self.get_target_point(x_targ, y_targ, theta_targ, distance, method)

        # Reachability check: a target behind the vehicle can't be reached without reversing, so hold
        if abs(math.atan2(y_t, x_t)) >= self.ahead_angle:
            return 0.0, 0.0

        # Run scipy optimization
        u0 = self.last_u if self.last_u is not None else np.zeros(INPUTS_PER_STEP * self.horizon)
        bounds = [(self.vel_min, self.vel_max), (self.delta_min, self.delta_max)] * self.horizon
        result = minimize(self.cost_and_gradient, u0, args=(x_t, y_t, theta_t), jac=True, bounds=bounds)

        # Warm start the next solve with this plan shifted forward one step (last step repeated)
        self.last_u = np.concatenate([result.x[INPUTS_PER_STEP:], result.x[-INPUTS_PER_STEP:]])

        # Return first step
        vel0, delta0 = result.x[0], result.x[1]
        return vel0, delta0
