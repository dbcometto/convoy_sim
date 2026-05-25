# The base vehicle class
import numpy as np
import math


class Vehicle():
    """A vehicle based on bicycle dynamics"""
    def __init__(self, id = 0, x = 0, y = 0, theta = 0, v = 0, L = 0.33) -> None:
        """Init a vehicle with true state"""
        self.id = id

        # State
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v

        # Parameters
        self.L = L



    def step(self, v, delta, dt) -> tuple:
        """Step a vehicle given control inputs and a dt and return true state"""
        self.x += dt * v * math.cos(self.theta)
        self.y += dt * v * math.sin(self.theta)
        self.theta += dt * v/self.L * math.tan(delta)
        self.v = v

        # Wrap theta from -pi to pi
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        return (self.x, self.y, self.theta, self.v)