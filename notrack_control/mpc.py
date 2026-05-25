import numpy as np
import math
from scipy.optimize import minimize

class MPC():
    """An MPC controller for Convoy Operations"""
    def __init__(self, N=10, dt=0.1,
                 wx=1, wy=1, wtheta=1, wv=0.1, wdelta=0.1,
                 vmin=0, vmax=1, deltamin=-math.pi/4, deltamax=math.pi/4,
                 L = 0.33):
        """Init an MPC controller for Convoy Operations"""
        self.N = N
        self.dt = dt
        self.last_u = None
        self.target = None

        # Weights
        self.wx = wx
        self.wy = wy
        self.wtheta = wtheta
        self.wv = wv
        self.wdelta = wdelta

        # Constraints
        self.vmin = vmin
        self.vmax = vmax
        self.deltamin = deltamin
        self.deltamax = deltamax

        # Parameters
        self.L = L
        

    def get_target_point(self, x_targ, y_targ, theta_targ, d=0.5, method='offhooked') -> tuple:
        """Calculate the target point.  Methods include `linear`, `hitch`, `curved`, and `offhooked`."""

        if method == 'linear':
            # Calculate target point d backwards along line from follower to leader and match heading to that line
            r = np.array([x_targ, y_targ])
            r_norm = r/np.linalg.norm(r)
            targ = r - d*r_norm

            x,y = targ
            theta = math.atan2(y_targ, x_targ) 

            self.target = x, y, theta
            return self.target
        
        elif method == 'hitch':
            # Calculate target point d backwards along leader heading
            r = np.array([x_targ, y_targ])
            r_norm = np.array([math.cos(theta_targ), math.sin(theta_targ)])
            targ = r - d*r_norm

            x,y = targ
            theta = theta_targ 

            self.target = x, y, theta
            return self.target
        
        elif method == 'curved':
            # Circle tangent to leader heading at leader, passing through follower
            denom = x_targ * math.sin(theta_targ) - y_targ * math.cos(theta_targ)
            
            if abs(denom) < 1e-6:
                # Degenerate case - fall back to linear
                return self.get_target_point(x_targ, y_targ, theta_targ, d, method='linear')
            
            R = (x_targ**2 + y_targ**2) / (2 * denom)
            
            # Circle center
            cx = x_targ - R * math.sin(theta_targ)
            cy = y_targ + R * math.cos(theta_targ)
            
            # Angles from center to follower and leader
            phi_l = math.atan2(y_targ - cy, x_targ - cx)
            
            # Target is d meters back along arc from leader
            phi_t = phi_l - d / R
            
            x = cx + R * math.cos(phi_t)
            y = cy + R * math.sin(phi_t)
            theta = phi_t + math.pi/2 * math.copysign(1, R)
            
            self.target = x, y, theta
            return self.target
        
        elif method == 'offhooked':
            # Calculate linear point d/2 backwards from d/2 behind the vehicle

            # Calculate hitch point
            r = np.array([x_targ, y_targ])
            r_norm = np.array([math.cos(theta_targ), math.sin(theta_targ)])
            hitch = r - d/2*r_norm

            # Calculate link point
            hitch_norm = hitch/np.linalg.norm(hitch)
            targ = hitch - d/2*hitch_norm

            x,y = targ
            theta = math.atan2(y, x) 

            self.target = x, y, theta
            return self.target
        




    # Helpers

    def generate_rollout(self, u):
        """Rollout a control input trajectory and return the states"""
        # Note that u is a flat array of alternating v/delta pairs
        x, y, theta = 0.0, 0.0, 0.0

        states = []
        for k in range(self.N):
            v = u[2*k]
            delta = u[2*k + 1]
            x += self.dt * v * math.cos(theta)
            y += self.dt * v * math.sin(theta)
            theta += self.dt * v / self.L * math.tan(delta)
            states.append((x, y, theta))
        return states
    


    def cost(self, u, x_t, y_t, theta_t):
        """Calculate the cost of a control input trajectory"""
        # Note that u is a flat array of alternating v/delta pairs
        states = self.generate_rollout(u)
        J = 0
        for x, y, theta in states:
            J += self.wx*(x-x_t)**2 + self.wy*(y-y_t)**2 + self.wtheta*(theta-theta_t)**2
        for k in range(self.N):
            J += self.wv*u[2*k]**2 + self.wdelta*u[2*k+1]**2
        return J
        


    # Main solver

    def solve(self, x_targ, y_targ, theta_targ, d=0.5, method='offhooked') -> tuple:
        """Given target vehicle state, output control inputs"""
        
        # Find target pose in relative frame
        x_t, y_t, theta_t = self.get_target_point(x_targ, y_targ, theta_targ, d, method)

        # run scipy optimization
        u0 = self.last_u if self.last_u is not None else np.zeros(2 * self.N)
        bounds = [(self.vmin, self.vmax), (self.deltamin, self.deltamax)] * self.N
        result = minimize(self.cost, u0, args=(x_t, y_t, theta_t), bounds=bounds)

        # Return first step
        v0, delta0 = result.x[0], result.x[1]
        return v0, delta0
