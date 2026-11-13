# The base vehicle class
import math


#========================================#
# CONFIG
#========================================#

LEADER_ID = 0  # the vehicle every other vehicle follows, directly or indirectly






#========================================#
# VEHICLE
#========================================#

class Vehicle():
    """A vehicle based on bicycle dynamics"""

    def __init__(self, vehicle_id: int, x: float, y: float, theta: float, vel: float, wheelbase: float) -> None:
        """Init a vehicle with true state"""
        self.id = vehicle_id

        # State, in the world frame
        self.x = x  # m
        self.y = y  # m
        self.theta = theta  # rad
        self.vel = vel  # m/s

        # Parameters
        self.wheelbase = wheelbase  # m


    def step(self, vel: float, delta: float, dt: float) -> tuple:
        """Step a vehicle given control inputs and a dt and return true state"""
        self.x += dt * vel * math.cos(self.theta)
        self.y += dt * vel * math.sin(self.theta)
        self.theta += dt * vel / self.wheelbase * math.tan(delta)
        self.vel = vel

        # Wrap theta from -pi to pi
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        return (self.x, self.y, self.theta, self.vel)
