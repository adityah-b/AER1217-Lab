'''path planning implementation'''

import numpy as np
from matplotlib import pyplot as plt

from constants import *
from collision import *


class PathPlanner3D:
    def __init__(self,
            initial_obs: list,
            initial_info: dict,
            x_bound: list = [-3.5, 3.5],
            y_bound: list = [-3.5, 3.5],
            z_bound: list = [ 0.0, 2.0],
            drone_radius: float = 0.05):

        # set drone size (used as collision clearance and search resolution)
        self.drone_radius = drone_radius
        self.step_size = drone_radius
        self.clearance = drone_radius * 2.0

        # initialize collision environment
        self.env = CollisionEnvironment(x_bound, y_bound, z_bound, self.clearance)

        # set start and end states
        self.start_state = np.array([initial_obs[0], initial_obs[2], initial_obs[4]])
        self.end_state   = np.array([initial_info['x_reference'][0], initial_info['x_reference'][2], 0.0])

        # add all obstacles
        for obstacle_position in initial_info['nominal_obstacles_pos']:
            self.env.add_obstacles(obstacle_position)

        # add all goal posts
        self.gates = []
        for [x, y, z, r, p, yaw, _] in initial_info['nominal_gates_pos_and_type']:
            self.__add_gate([x, y], yaw)
            self.env.add_gate([x, y], yaw)

        self.initial_waypoints = []
        self.final_waypoints = []


    def __add_gate(self, position, orientation):
        # TODO::to be updated to generate waypoints
        self.gates.append((position[0], position[1], orientation))


    def initialize_trajectory(self, bidirectional=False):
        '''initialize trajectory by connecting the dots'''

        waypoints = []
        # additional waypoints for smooth take-off (vertical)
        waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT]))
        waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 2]))
        waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 3]))
        waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 4]))
        waypoints.append(self.start_state + np.array([0.0, 0.0, 1.0]))

        prev_point_1 = self.start_state + np.array([0.0, 0.0, 1.0])
        prev_point_2 = self.start_state + np.array([0.0, 0.0, 1.0])

        for (x, y, theta) in self.gates:
            center = np.array([x, y, GATE_CENTER_HEIGHT])
            offset = GATE_CENTER_OFFSET
            offset = offset * np.array([np.sin(theta), np.cos(theta), 0.0])
            front  = center + offset
            back   = center - offset

            if bidirectional:
                pass
            else:
                if np.linalg.norm(prev_point_1 - front) < np.linalg.norm(prev_point_1 - back):
                    waypoints.append(front)
                    waypoints.append(back)
                    prev_point_1 = back
                else:
                    waypoints.append(back)
                    waypoints.append(front)
                    prev_point_1 = front

        # vertical landing
        waypoints.append(self.end_state + np.array([0.0, 0.0, 1.0]))
        waypoints.append(self.end_state + np.array([0.0, 0.0, LANDING_HEIGHT]))

        self.initial_waypoints = waypoints
        return waypoints


    def plan_trajectory(self):
        pass


    def plot_trajectory(self):
        pass


    def plot_potential_feild(self):
        pass