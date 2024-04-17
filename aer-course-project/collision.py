import numpy as np


class CollisionObject:
    def __init__(self):
        pass


class CollisionEnvironment:
    '''environment class to manage any collision query'''

    def __init__(self,
            x_bound: np.array,
            y_bound: np.array,
            z_bound: np.array,
            clearance: float):

        # set boundary
        self.x_bound = np.array(x_bound, dtype=float)
        self.y_bound = np.array(y_bound, dtype=float)
        self.z_bound = np.array(z_bound, dtype=float)
        self.clearance = clearance

        # initialize
        self.obstacles = []


    def add_obstacles(self, position):
        pass


    def add_gate(self, position, orientation):
        pass


    def is_in_bound(self, position):
        pass


    def is_in_collision(self, position):
        pass


    def is_in_collision(self, start, end):
        pass