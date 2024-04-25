"""Example utility module.

Please use a file like this one to add extra functions.

"""

'''
constants.py
'''

# debug flags
DEBUG_PATH_PLANNING = False
DEBUG_WAYPOINT_TRACKING = True
DEBUG_COLLISIONS = False

# bounds
X_BOUND_LOWER = -3.5
X_BOUND_UPPER =  3.5
Y_BOUND_LOWER = -3.5
Y_BOUND_UPPER =  3.5
Z_BOUND_LOWER =  0.0
Z_BOUND_UPPER =  2.0

# dimensions
DRONE_RADIUS = 0.05
OBSTACLE_RADIUS = 0.06
OBSTACLE_HEIGHT = 1.3

GATE_CENTER_HEIGHT = 1.0
GATE_CENTER_OFFSET = 0.2

GATE_RIM_WIDTH = 0.1
GATE_OPENING_SIZE = 0.4

# take-off and landing
TAKE_OFF_HEIGHT = 0.2
LANDING_HEIGHT  = 0.1

# tracking
WAYPOINT_TRACKING_THRES = 0.2
WAYPOINT_TRACKING_STEP_SIZE = 0.05

SMOOTH_TRACKING_SPEED_MAX = 0.8
SMOOTH_TRACKING_SPEED_MID = 0.5
SMOOTH_TRACKING_SPEED_MIN = 0.3

WAYPOINT_TRACKING_SPEED_MAX = 1.0
WAYPOINT_TRACKING_SPEED_MIN = 0.5

# state machine
FLIGHT_STATE_READY = 0
FLIGHT_STATE_TRACK = 1
FLIGHT_STATE_LANDING = 4
FLIGHT_STATE_OFF = 8

SUB_SAMPLE_DISTANCE = 0.2
NUM_SUB_SAMPLE = 5 # driven by sample distance)

WEIGHT_OF_LENGTH = 3.0
WEIGHT_OF_COLLISION = 2.0
WEIGHT_OF_ACCELERATION = 0.5

MAX_ACCELERATION = 0.5 # (m/s)

POT_PLOT_SUB_TRAJECTORY = False
POT_PLOT_FINAL_TRAJECTORY = True

# robot dimension
ROBOT_RADIUS = 0.05

# clearances
OBSTACLE_SAFETY_FACTOR = 1.2
GATE_SAFETY_FACTOR = 1.1
OBSTACLE_CLEARANCE = (4 * OBSTACLE_RADIUS + ROBOT_RADIUS) * OBSTACLE_SAFETY_FACTOR
GATE_EDGE_CLEARANCE = (GATE_RIM_WIDTH + ROBOT_RADIUS) * GATE_SAFETY_FACTOR

# Planner configs

# path planning (potential field)
USE_SMOOTH_TRAJECTORY = True
USE_BIDIRECTIONAL_EXIT = True
USE_LINEAR_COLLISION_LOSS = True

# path planning (FMT)
USE_FMT_SMOOTH_TRAJECTORY = True
USE_FMT_SLIDING_WINDOW = False
USE_FMT_BIDIRECTIONAL_EXIT = True

'''
collision.py
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class CircleObject:
    def __init__(self, x, y, radius):
        self.center = np.array([x, y])
        self.radius = radius


    def is_in_collision(self, position):
        if np.linalg.norm(position - self.center) <= self.radius:
            return True
        return False


class Obstacle:
    def __init__(self, x, y, clearance):
        self.center = np.array([x, y])
        self.radius = OBSTACLE_RADIUS + clearance


    def is_in_collision(self, position):
        if np.linalg.norm(position - self.center) <= self.radius:
            return True
        return False


    def compute_cost(self, position):
        diff = np.array([position[0] - self.center[0], position[1] - self.center[1]])
        dist = np.linalg.norm(diff)
        if USE_LINEAR_COLLISION_LOSS:
            cost = max(1 - dist / self.radius, 0.0)
        else:
            cost = np.exp(-dist / self.radius)
        return cost


    def render(self, figure, ax):
        column = plt.Circle(self.center, OBSTACLE_RADIUS, color='r', fill=False)
        circle = plt.Circle(self.center, self.radius, color='b', fill=False)
        ax.add_patch(column)
        ax.add_patch(circle)


class Gate:
    def __init__(self, x, y, phi, clearance):
        self.center = np.array([x, y])
        self.radius = (GATE_OPENING_SIZE + GATE_RIM_WIDTH) / 2 + clearance
        self.minor_radius = clearance
        self.phi = phi


    def is_in_collision(self, position):
        if np.linalg.norm(position - self.center) <= self.radius:
            return True
        return False


    def compute_cost(self, position):
        '''circular version'''
        diff = np.array([position[0] - self.center[0], position[1] - self.center[1]])
        dist = np.linalg.norm(diff)
        if USE_LINEAR_COLLISION_LOSS:
            cost = max(1 - dist / self.radius, 0.0)
        else:
            cost = np.exp(-dist / self.radius)
        return cost


    # def compute_cost(self, position):
    #     '''ellipital version'''
    #     x_scale = self.minor_radius * np.sin(self.phi) + self.radius * np.cos(self.phi)
    #     y_scale = self.minor_radius * np.cos(self.phi) + self.radius * np.sin(self.phi)
    #     diff = np.array([(position[0] - self.center[0]) / x_scale, (position[1] - self.center[1]) / y_scale])
    #     dist = np.linalg.norm(diff)
    #     cost = np.exp(-dist)
    #     return cost


    def render(self, figure, ax):
        '''circular version'''
        sin_phi = np.sin(self.phi)
        cos_phi = np.cos(self.phi)

        rim_offset  = (GATE_OPENING_SIZE + GATE_RIM_WIDTH) / 2
        rim_radius  = GATE_RIM_WIDTH

        # find vertices
        center_l = np.array([self.center[0] - rim_offset * cos_phi, self.center[1] - rim_offset * sin_phi])
        center_r = np.array([self.center[0] + rim_offset * cos_phi, self.center[1] + rim_offset * sin_phi])

        # draw shapes
        column_l = plt.Circle(center_l, rim_radius, color='r', fill=False)
        column_r = plt.Circle(center_r, rim_radius, color='r', fill=False)
        circle = plt.Circle(self.center, self.radius, color='b', fill=False)
        ax.add_patch(column_l)
        ax.add_patch(column_r)
        ax.add_patch(circle)


    # def render(self, figure, ax):
    #     '''ellipital version'''
    #     sin_phi = np.sin(self.phi)
    #     cos_phi = np.cos(self.phi)

    #     rim_offset  = (GATE_OPENING_SIZE + GATE_RIM_WIDTH) / 2
    #     rim_radius  = GATE_RIM_WIDTH

    #     # find vertices
    #     center_l = np.array([self.center[0] - rim_offset * cos_phi, self.center[1] - rim_offset * sin_phi])
    #     center_r = np.array([self.center[0] + rim_offset * cos_phi, self.center[1] + rim_offset * sin_phi])

    #     # draw shapes
    #     column_l = plt.Circle(center_l, rim_radius, color='r', fill=False)
    #     column_r = plt.Circle(center_r, rim_radius, color='r', fill=False)
    #     ellipse = Ellipse(self.center, self.radius * 2, self.minor_radius * 2, angle=self.phi / np.pi * 180, color='b', fill=False)
    #     ax.add_patch(column_l)
    #     ax.add_patch(column_r)
    #     ax.add_patch(ellipse)


class CollisionEnvironment:
    '''environment class to manage any collision check'''

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
        self.shapes = []


    def add_obstacles(self, position):
        obstacle = Obstacle(position[0], position[1], self.clearance)
        self.shapes.append(obstacle)


    def add_gate(self, position, orientation):
        gate = Gate(position[0], position[1], orientation, self.clearance)
        self.shapes.append(gate)


    def is_in_bound(self, position):
        if position[0] < X_BOUND_LOWER or position[0] > X_BOUND_UPPER:
            return False
        if position[1] < Y_BOUND_LOWER or position[1] > Y_BOUND_UPPER:
            return False
        if position[2] < Z_BOUND_LOWER or position[2] > Z_BOUND_UPPER:
            return False
        return True


    def is_in_collision(self, position):
        for shape in self.shapes:
            if shape.is_in_collision(position) == True:
                return True
        return False


    def is_reachable(self, start, end):
        seg = end - start
        len = np.linalg.norm(seg)
        dir = seg / len

        num = int(np.ceil(len / self.clearance))

        for i in range(num):
            if self.is_in_collision(start + i * self.clearance * dir):
                return False
        return True


    def compute_cost(self, position):
        cost = 0
        for shape in self.shapes:
            cost += shape.compute_cost(position)
        return cost


    def render(self):
        figure, ax = plt.subplots(figsize=(X_BOUND_UPPER - X_BOUND_LOWER, Y_BOUND_UPPER - Y_BOUND_LOWER))
        ax.scatter(-1.0, -3.0, s=5, color="green", marker="o")
        ax.scatter(-0.5,  2.0, s=5, color="red",   marker="o")

        for shape in self.shapes:
            shape.render(figure, ax)

        return figure, ax

'''
path_planner_potential.py
'''

'''path planning implementation'''

import numpy as np
from scipy.optimize import minimize

from matplotlib import pyplot as plt


class PathPlannerPotential:
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
        self.clearance = drone_radius * 5.0

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

        # # plot environment
        # self.figure, self.ax = self.env.render()
        # plt.axis('equal')
        # plt.show()

        # # plot potential field
        # self.plot_potential_feild()

        self.initial_waypoints = []
        self.final_waypoints = []
        self.speeds = []


    def __add_gate(self, position, orientation):
        # TODO::to be updated to generate waypoints
        self.gates.append((position[0], position[1], orientation))


    def plan_trajectory(self):
        if USE_BIDIRECTIONAL_EXIT:
            return self.plan_trajectory_bidirectional()
        else:
            return self.plan_trajectory_unidirectional()


    def plan_trajectory_unidirectional(self):
        waypoints = []

        # waypoints for smooth take-off (vertical)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT]))
        self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 2]))
        self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 3]))
        self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 4]))
        self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, 1.0]))
        self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)

        prev_entry_point = self.start_state + np.array([0.0, 0.0, 1.0])
        prev_exit_point  = self.start_state + np.array([0.0, 0.0, 1.0])
        curr_entry_point = None
        curr_exit_point  = None

        for (x, y, theta) in self.gates:
            center = np.array([x, y, GATE_CENTER_HEIGHT])
            offset = GATE_CENTER_OFFSET
            offset = offset * np.array([np.sin(theta), np.cos(theta), 0.0])
            front  = center + offset
            back   = center - offset

            # determine entry direction
            if np.linalg.norm(prev_exit_point - front) < np.linalg.norm(prev_exit_point - back):
                curr_entry_point = front
                curr_exit_point  = back
            else:
                curr_entry_point = back
                curr_exit_point  = front

            # optimize trajectory and velocity
            points = self.path_planning(prev_entry_point, prev_exit_point, curr_entry_point, curr_exit_point)
            speeds = self.compute_speeds(prev_entry_point, prev_exit_point, curr_entry_point, curr_exit_point, points, self.speeds[-1])
            self.final_waypoints.append(prev_exit_point)
            self.final_waypoints.extend(points)
            self.final_waypoints.append(curr_entry_point)
            self.speeds.extend(speeds)

            prev_entry_point = curr_entry_point
            prev_exit_point  = curr_exit_point

        # find optimal trajectory for reaching end state
        points = self.path_planning(prev_entry_point, prev_exit_point, self.end_state, self.end_state)
        speeds = self.compute_speeds(prev_entry_point, prev_exit_point, self.end_state, self.end_state, points, self.speeds[-1])
        self.final_waypoints.append(prev_exit_point)
        self.final_waypoints.extend(points)
        self.speeds.extend(speeds[0: -1])

        # vertical landing
        self.final_waypoints.append(self.end_state + np.array([0.0, 0.0, 1.0]))
        self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)
        self.final_waypoints.append(self.end_state + np.array([0.0, 0.0, LANDING_HEIGHT]))
        self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)

        # plot final trajectory
        if POT_PLOT_FINAL_TRAJECTORY: self.plot_trajectory()
        return self.final_waypoints, self.speeds


    def plan_trajectory_bidirectional(self):
        waypoints = []

        # waypoints for smooth take-off (vertical)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT]))
        self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 2]))
        self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 3]))
        self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 4]))
        self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)
        # self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, 1.0]))
        # self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)

        prev_front_point = self.start_state + np.array([0.0, 0.0, 1.0])
        prev_back_point  = self.start_state + np.array([0.0, 0.0, 1.0])
        prev_entry_point = prev_front_point
        prev_exit_point  = prev_back_point
        prev_entry_point_key = "front"
        prev_exit_point_key = "back"
        curr_front_point = None
        curr_back_point  = None
        curr_entry_point = None
        curr_exit_point  = None
        curr_entry_point_key = None
        curr_exit_point_key = None

        for (x, y, theta) in self.gates:
            center = np.array([x, y, GATE_CENTER_HEIGHT])
            offset = GATE_CENTER_OFFSET
            offset = offset * np.array([np.sin(theta), np.cos(theta), 0.0])
            curr_front_point  = center + offset
            curr_back_point   = center - offset

            # determine entry direction
            shortest_distance = 1000
            shortest_connection = None
            dist_front_to_front = np.linalg.norm(prev_front_point - curr_front_point)
            dist_front_to_back  = np.linalg.norm(prev_front_point - curr_back_point)
            dist_back_to_front  = np.linalg.norm(prev_back_point  - curr_front_point)
            dist_back_to_back   = np.linalg.norm(prev_back_point  - curr_back_point)

            if dist_front_to_front < shortest_distance:
                shortest_distance = dist_front_to_front
                shortest_connection = ("front", "front")
                prev_exit_point_key = "front"
                curr_entry_point_key = "front"
            if dist_front_to_back < shortest_distance:
                shortest_distance = dist_front_to_back
                shortest_connection = ("front", "back")
                prev_exit_point_key = "front"
                curr_entry_point_key = "back"
            if dist_back_to_front < shortest_distance:
                shortest_distance = dist_back_to_front
                shortest_connection = ("back", "front")
                prev_exit_point_key = "back"
                curr_entry_point_key = "front"
            if dist_back_to_back < shortest_distance:
                shortest_distance = dist_back_to_back
                shortest_connection = ("back", "back")
                prev_exit_point_key = "back"
                curr_entry_point_key = "back"

            # add the opposite point if entry and exit is the same
            if prev_exit_point_key == prev_entry_point_key:
                self.speeds.append(SMOOTH_TRACKING_SPEED_MID)
                if prev_exit_point_key == "front":
                    self.final_waypoints.append(prev_back_point)
                if prev_exit_point_key == "back":
                    self.final_waypoints.append(prev_front_point)

            if prev_exit_point_key == "front":
                prev_entry_point = prev_back_point
                prev_exit_point  = prev_front_point
            else:
                prev_entry_point = prev_front_point
                prev_exit_point  = prev_back_point

            if curr_entry_point_key == "front":
                curr_entry_point = curr_front_point
                curr_exit_point  = curr_back_point
            else:
                curr_entry_point = curr_back_point
                curr_exit_point  = curr_front_point

            # prev_entry_point = prev_front_point if prev_entry_point_key == "front" else prev_back_point
            # prev_exit_point  = prev_front_point if prev_exit_point_key  == "front" else prev_back_point

            # curr_entry_point = curr_front_point if curr_entry_point_key == "front" else curr_back_point
            # curr_exit_point  = curr_front_point if curr_exit_point_key  == "front" else curr_back_point

            # optimize trajectory and velocity
            points = self.path_planning(prev_entry_point, prev_exit_point, curr_entry_point, curr_exit_point)
            speeds = self.compute_speeds(prev_entry_point, prev_exit_point, curr_entry_point, curr_exit_point, points, self.speeds[-1])
            self.final_waypoints.append(prev_exit_point)
            self.final_waypoints.extend(points)
            self.final_waypoints.append(curr_entry_point)
            self.speeds.extend(speeds)

            prev_front_point = curr_front_point
            prev_back_point = curr_back_point
            prev_entry_point = curr_entry_point
            prev_exit_point  = curr_exit_point
            prev_entry_point_key = curr_entry_point_key
            prev_exit_point_key = curr_exit_point_key

        # find optimal trajectory for reaching end state
        dist_to_front = np.linalg.norm(self.end_state - prev_front_point)
        dist_to_back  = np.linalg.norm(self.end_state - prev_back_point)

        if dist_to_front < dist_to_back:
            prev_exit_point_key = "front"
            prev_entry_point = prev_back_point
            prev_exit_point = prev_front_point
        else:
            prev_exit_point_key = "back"
            prev_entry_point = prev_front_point
            prev_exit_point = prev_back_point

        if prev_exit_point_key == prev_entry_point_key:
            self.speeds.append(SMOOTH_TRACKING_SPEED_MID)
            if prev_exit_point_key == "front":
                self.final_waypoints.append(prev_back_point)
            else:
                self.final_waypoints.append(prev_front_point)

        points = self.path_planning(prev_entry_point, prev_exit_point, self.end_state, self.end_state)
        speeds = self.compute_speeds(prev_entry_point, prev_exit_point, self.end_state, self.end_state, points, self.speeds[-1])
        self.final_waypoints.append(prev_exit_point)
        self.final_waypoints.extend(points)
        self.speeds.extend(speeds[0: -1])

        # vertical landing
        self.final_waypoints.append(self.end_state + np.array([0.0, 0.0, 1.0]))
        self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)
        self.final_waypoints.append(self.end_state + np.array([0.0, 0.0, LANDING_HEIGHT]))
        self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)

        # plot final trajectory
        if POT_PLOT_FINAL_TRAJECTORY: self.plot_trajectory()
        return self.final_waypoints, self.speeds


    def path_planning(self, prev_entry_point, prev_exit_point, curr_entry_point, curr_exit_point):
        '''find path from start position to goal position'''

        # initialize waypoints
        if SUB_SAMPLE_DISTANCE == None:
            num_sub_sample = NUM_SUB_SAMPLE
        else:
            num_sub_sample = int(np.linalg.norm(curr_entry_point - prev_exit_point) / SUB_SAMPLE_DISTANCE)
            print(f"Subsample {num_sub_sample} points from {prev_exit_point} to {curr_entry_point}")

        scales = np.linspace(0.0, 1.0, num=(num_sub_sample + 2))
        scales = scales[1: -1].reshape([-1, 1])

        diff = curr_entry_point - prev_exit_point
        points = np.tile(diff[0:2], (num_sub_sample, 1)) * scales + prev_exit_point[0:2]

        # optimize waypoints
        def waypoint_loss(points):
            loss = 0.0
            loss += WEIGHT_OF_LENGTH       * length_loss(points)
            loss += WEIGHT_OF_COLLISION    * collision_loss(points)
            loss += WEIGHT_OF_ACCELERATION * turning_loss(points)
            return loss

        def length_loss(points):
            loss = 0.0
            loss += (points[0] - prev_exit_point[0])**2 + (points[1] - prev_exit_point[1])**2
            for i in range(num_sub_sample - 1):
                loss += (points[2 * (i + 1)] - points[2 * i])**2 + (points[2 * (i + 1) + 1] - points[2 * i + 1])**2
            loss += (curr_entry_point[0] - points[-2])**2 + (curr_entry_point[1] - points[-1])**2
            return loss

        def collision_loss(points):
            loss = 0.0
            for i in range(num_sub_sample):
                loss += self.env.compute_cost(np.array([points[2 * i], points[2 * i + 1], 1.0]))
            return loss

        def turning_loss(points):
            loss = 0.0
            v1 = np.array([prev_exit_point[0] - prev_entry_point[0], prev_exit_point[1] - prev_entry_point[1]])
            v2 = np.array([points[0] - prev_exit_point[0], points[1] - prev_exit_point[1]])
            loss -= np.dot(v1, v2) / (np.linalg.norm(v1) + 1e-6) / (np.linalg.norm(v2) + 1e-6)

            v1 = np.array([points[0] - prev_exit_point[0], points[1] - prev_exit_point[1]])
            v2 = np.array([points[2] - points[0], points[3] - points[1]])
            loss -= np.dot(v1, v2) / (np.linalg.norm(v1) + 1e-6) / (np.linalg.norm(v2) + 1e-6)

            for i in range(1, num_sub_sample - 1):
                v1 = np.array([points[2 * i] - points[2 * (i - 1)], points[2 * i + 1] - points[2 * (i - 1) + 1]])
                v2 = np.array([points[2 * (i + 1)] - points[2 * i], points[2 * (i + 1) + 1] - points[2 * i + 1]])
                loss -= np.dot(v1, v2) / (np.linalg.norm(v1) + 1e-6) / (np.linalg.norm(v2) + 1e-6)

            v1 = np.array([curr_entry_point[0] - points[-2], curr_entry_point[1] - points[-1]])
            v2 = np.array([points[-2] - points[-4], points[-1] - points[-3]])
            loss -= np.dot(v1, v2) / (np.linalg.norm(v1) + 1e-6) / (np.linalg.norm(v2) + 1e-6)

            v1 = np.array([curr_exit_point[0] - curr_entry_point[0], curr_exit_point[1] - curr_entry_point[1]])
            v2 = np.array([curr_entry_point[0] - points[-2], curr_entry_point[1] - points[-1]])
            loss -= np.dot(v1, v2) / (np.linalg.norm(v1) + 1e-6) / (np.linalg.norm(v2) + 1e-6)

            return loss

        res = minimize(waypoint_loss, points, method='BFGS', options={'xatol': 1e-6, 'disp': True})

        print(res.x)
        points = res.x.reshape([num_sub_sample, 2])

        # visualize
        if POT_PLOT_SUB_TRAJECTORY:
            x_arr = points[:, 0]
            y_arr = points[:, 1]
            plt.plot([prev_entry_point[0], prev_exit_point[0]], [prev_entry_point[1], prev_exit_point[1]], c='r')
            plt.plot([prev_exit_point[0], points[0, 0]], [prev_exit_point[1], points[0, 1]], c='b')
            plt.plot(x_arr, y_arr, c='b')
            plt.plot([points[-1, 0], curr_entry_point[0]], [points[-1, 1], curr_entry_point[1]], c='b')
            plt.plot([curr_entry_point[0], curr_exit_point[0]], [curr_entry_point[1], curr_exit_point[1]], c='r')
            plt.scatter(x_arr, y_arr, s=10, label="o", c="b")
            plt.axis('equal')
            plt.show()

        waypoints = []
        for i in range(num_sub_sample):
            waypoints.append(np.array([points[i, 0], points[i, 1], 1.0]))

        return waypoints


    def compute_speeds(self, prev_entry_point, prev_exit_point, curr_entry_point, curr_exit_point, points, v_prev):
        speeds = []

        # compute speed for prev_exit_point
        speed = self.__compute_speeds(prev_entry_point, prev_exit_point, points[0], self.speeds[-1])
        speeds.append(speed)

        # compute speed for first point
        speed = self.__compute_speeds(prev_exit_point, points[0], points[1], speeds[-1])
        speeds.append(speed)

        # compute speed for each point
        for i in range(1, len(points) - 1):
            speed = self.__compute_speeds(points[i - 1], points[i], points[i + 1], speeds[-1])
            speeds.append(speed)

        # compute speed for last point
        speed = self.__compute_speeds(points[-2], points[-1], curr_entry_point, speeds[-1])
        speeds.append(speed)

        # compute speed for curr_entry_point
        speed = self.__compute_speeds(points[-1], curr_entry_point, curr_exit_point, speeds[-1])
        speeds.append(speed)

        return speeds


    def __compute_speeds(self, p_prev, p_curr, p_next, v_prev):
        # compute the geometric quantity
        e1 = p_curr - p_prev
        e2 = p_next - p_curr
        a  = np.linalg.norm(e1)
        b  = np.linalg.norm(e2)
        c  = np.linalg.norm(e1 + e2)

        # compute the next possible speed
        speed_next = np.sqrt(v_prev**2 + 0.5 * MAX_ACCELERATION * a)

        # compute max speed given curvature
        if a == 0 or b == 0:
            r = 1e3
        else:
            r = a * b * c / 2 / np.linalg.norm(np.cross(e1, e2))
        speed_max = np.sqrt(MAX_ACCELERATION * r)

        return min(speed_next, speed_max, SMOOTH_TRACKING_SPEED_MAX)


    def plot_trajectory(self):
        self.figure, self.ax = self.env.render()
        waypoints = self.final_waypoints

        self.ax.plot([self.start_state[0], waypoints[0][0]], [self.start_state[1], waypoints[0][1]], c='b')
        for i in range(len(waypoints) - 1):
            self.ax.plot([waypoints[i][0], waypoints[i + 1][0]], [waypoints[i][1], waypoints[i + 1][1]], c='b')
        self.ax.plot([waypoints[-1][0], self.end_state[0]], [waypoints[-1][1], self.end_state[1]], c='b')
        x_arr = [waypoint[0] for waypoint in waypoints]
        y_arr = [waypoint[1] for waypoint in waypoints]
        plt.scatter(x_arr, y_arr, s=10, label="o", c="b")
        plt.axis('equal')
        plt.show()


    def plot_potential_feild(self):
        GRID_RESOLUTION = DRONE_RADIUS
        Z = np.zeros([int((X_BOUND_UPPER - X_BOUND_LOWER) / GRID_RESOLUTION), int((Y_BOUND_UPPER - Y_BOUND_LOWER) / GRID_RESOLUTION)])

        X, Y = np.meshgrid(range(Z.shape[0]), range(Z.shape[1]))
        X = X * GRID_RESOLUTION + X_BOUND_LOWER
        Y = Y * GRID_RESOLUTION + Y_BOUND_LOWER

        for x in range(Z.shape[0]):
            for y in range(Z.shape[1]):
                Z[x, y] = self.env.compute_cost(np.array([x * GRID_RESOLUTION + X_BOUND_LOWER, y * GRID_RESOLUTION + Y_BOUND_LOWER, 1.0]))

        # show hight map in 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z)
        plt.title('z as 3d height map')
        plt.show()

'''
path_planner.py
'''

import numpy as np

import heapq
from scipy.spatial import KDTree

from matplotlib import pyplot as plt

from scipy.interpolate import CubicSpline


class MinHeap:
    def __init__(self):
        self.min_heap = []
        self.entry_finder = {}
        self.REMOVED = '<removed>'
        self.counter = 0

    def add_or_update_node(self, node_index, cost):
        if node_index in self.entry_finder:
            self.remove_node(node_index)
        entry = [cost, self.counter, node_index]
        self.entry_finder[node_index] = entry
        heapq.heappush(self.min_heap, entry)
        self.counter += 1

    def remove_node(self, node_index):
        entry = self.entry_finder.pop(node_index)
        entry[-1] = self.REMOVED

    def top(self):
        while self.min_heap:
            cost, count, node_index = self.min_heap[0]
            if node_index is not self.REMOVED:
                return node_index, cost
            heapq.heappop(self.min_heap)

    def keys(self):
        return self.entry_finder.keys()

    def __bool__(self):
        return bool(self.entry_finder)

    def is_empty(self):
        return not self.entry_finder


class Node:
    def __init__(
            self,
            point,
            parent_idx = -1,
            cost = 0):
        self.point = point
        self.parent = parent_idx
        self.cost = cost
        self.children = []

    # Return L2 norm to get from Node A to Node B
    def GetCostToNode(self, nodeB):
        return np.linalg.norm(self.point - nodeB.point)


class PathPlanner:
    def __init__(
            self,
            initial_obs: list,
            initial_info: dict,
            x_bound: list = [-3.5, 3.5],
            y_bound: list = [-3.5, 3.5],
            robot_radius: float = 0.05,
            safety_factor: float = 1.25) -> None:

        ###############################
        # Map boundaries
        ###############################
        self.X_BOUND = np.array(x_bound, dtype=np.float32)
        if DEBUG_PATH_PLANNING: print(f'X_BOUND: {self.X_BOUND}')

        self.Y_BOUND = np.array(y_bound, dtype=np.float32)
        if DEBUG_PATH_PLANNING: print(f'Y_BOUND: {self.Y_BOUND}')

        ###############################
        # Desired robot states
        ###############################
        self.START_STATE = np.array([
            initial_obs[0],
            initial_obs[2],
            initial_obs[4]
        ], dtype=np.float32)
        if DEBUG_PATH_PLANNING: print(f'START STATE: {self.START_STATE}')

        self.GOAL_STATE = np.array([
            initial_info['x_reference'][0],
            initial_info['x_reference'][2],
            initial_info['x_reference'][4]
        ], dtype=np.float32)
        if DEBUG_PATH_PLANNING: print(f'GOAL STATE: {self.GOAL_STATE}')

        ###############################
        # Gate properties
        ###############################
        self.GATE_INNER_EDGE_LEN = 0.4
        self.GATE_EDGE_LEN = 0.05
        self.GATE_EDGE_WID = 0.05

        # Get x, y, z and yaw of gates (rad)
        self.GATE_LOCATIONS = np.array(initial_info['nominal_gates_pos_and_type'], dtype=np.float32)
        self.GATE_LOCATIONS = self.GATE_LOCATIONS[:, [0, 1, 2, 5]]
        if DEBUG_PATH_PLANNING: print(f'GATE_LOCATIONS: {self.GATE_LOCATIONS}')

        ###############################
        # Obstacle properties
        ###############################
        self.OBSTACLE_RADIUS = 0.06
        # if DEBUG_PATH_PLANNING: print(f'OBSTACLE_RADIUS: {self.OBSTACLE_RADIUS}')

        self.OBSTACLE_LOCATIONS = np.array([
            obs_loc[:3] for obs_loc in initial_info['nominal_obstacles_pos']
        ], dtype=np.float32)
        if DEBUG_PATH_PLANNING: print(f'OBSTACLE_LOCATIONS: {self.OBSTACLE_LOCATIONS}')

        ###############################
        # Controller properties
        ###############################
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        # if DEBUG_PATH_PLANNING: print(f'CTRL_TIMESTEP: {self.CTRL_TIMESTEP}')

        self.CTRL_FREQ = initial_info["ctrl_freq"]
        # if DEBUG_PATH_PLANNING: print(f'CTRL_FREQ: {self.CTRL_FREQ}')

        ####################
        # Robot properties
        ####################
        self.ROBOT_RADIUS = robot_radius

        #################################
        # Planning algorithm properties
        #################################
        self.GATE_SHAPES, self.GATE_EDGE_CENTERS = self.__constructGateShapes()

        self.start_points, self.goal_points, self.gate_indices_dict = self.__setupInitialTraj()

        self.SAFETY_FACTOR = safety_factor

    def runFMT(
            self,
            num_points: int = 5000,
            max_iters: int = 10000,
            rn: float = 0.3,
            filter_nodes: bool = True,
            sample_local: bool = False,
            start_points: list = None,
            goal_points: list = None,
            center_point = None,
            local_radius: float = None
            ):
        if not start_points or not goal_points:
            start_points = self.start_points
            goal_points = self.goal_points
        gate_indices_dict = self.gate_indices_dict
        path = []
        if sample_local:
            sampled_points = np.array(self.__sampleNPointsLocal(num_points=num_points, center_point=center_point, local_radius=local_radius))
            if DEBUG_PATH_PLANNING: print(f'LOCAL SAMPLING center_point: {center_point}; local_radius: {local_radius}')
        else:
            sampled_points = np.array(self.__sampleNPoints(num_points=num_points))
        for i in range(len(start_points)):
            start_point = start_points[i]
            goal_point = goal_points[i]
            active_gate_indices = []

            if USE_FMT_BIDIRECTIONAL_EXIT:
                if tuple(start_point) in gate_indices_dict:
                    active_gate_indices.append(gate_indices_dict[tuple(start_point)])

            if tuple(goal_point) in gate_indices_dict:
                active_gate_indices.append(gate_indices_dict[tuple(goal_point)])

            if DEBUG_PATH_PLANNING: print(f'START POINT: {start_point}')
            if DEBUG_PATH_PLANNING: print(f'GOAL POINT: {goal_point}')

            if len(active_gate_indices) > 1 and active_gate_indices[0] == active_gate_indices[1]:
                path += [start_point, goal_point]
                continue

            nodes = self.fmt(
                sampled_points=sampled_points.copy(),
                start_point=start_point,
                goal_point=goal_point,
                max_iters=max_iters,
                rn=rn,
                active_gate_indices=active_gate_indices)

            if filter_nodes:
                if USE_FMT_SLIDING_WINDOW:
                    sub_path_nodes = [nodes[-1]]
                    path_end_idx = nodes[-1].parent
                    while path_end_idx > -1:
                        sub_path_nodes.append(nodes[path_end_idx])
                        path_end_idx = nodes[path_end_idx].parent
                    sub_path_nodes.reverse()
                    sub_path = []

                    start_idx = 0
                    sub_path.append(sub_path_nodes[start_idx].point)

                    while start_idx < len(sub_path_nodes) - 1:
                        end_offset = 1
                        # Keep track of the last valid non-collision node index
                        last_valid = start_idx
                        while start_idx + end_offset < len(sub_path_nodes) and self.__checkCollision(sub_path_nodes[start_idx], sub_path_nodes[start_idx + end_offset], active_gate_indices) and np.linalg.norm(sub_path_nodes[start_idx].point - sub_path_nodes[start_idx + end_offset].point) <= 1.5:
                            last_valid = start_idx + end_offset
                            end_offset += 1

                        sub_path.append(sub_path_nodes[last_valid].point)
                        # Move start index to the node after the last valid
                        start_idx = last_valid
                    path += sub_path

                elif USE_FMT_SMOOTH_TRAJECTORY:
                    sub_path = [nodes[-1].point]
                    path_end_idx = nodes[-1].parent
                    while path_end_idx > -1:
                        sub_path.append(nodes[path_end_idx].point)
                        path_end_idx = nodes[path_end_idx].parent

                    sub_path.reverse()
                    path += self.filterPath(sub_path, start_point, goal_point)
            else:
                sub_path = [nodes[-1].point]
                path_end_idx = nodes[-1].parent
                while path_end_idx > -1:
                    sub_path.append(nodes[path_end_idx].point)
                    path_end_idx = nodes[path_end_idx].parent
                sub_path.reverse()
                path += sub_path
        return path

    def computeSpeeds(self, waypoints, lookahead_distance=10):
        n = len(waypoints)
        speeds = np.zeros(n)

        if n > 1:
            speeds[0] = SMOOTH_TRACKING_SPEED_MIN

        for i in range(1, n):
            if i < n - 1:
                p_prev = np.array(waypoints[i - 1])
                p_curr = np.array(waypoints[i])
                p_next = np.array(waypoints[i + 1])

                e1 = p_curr - p_prev
                e2 = p_next - p_curr
                a = np.linalg.norm(e1)
                b = np.linalg.norm(e2)
                c = np.linalg.norm(e1 + e2)

                speed_next = np.sqrt(speeds[i - 1]**2 + 2 * MAX_ACCELERATION * a)

                if a == 0 or b == 0:
                    r = 1e3
                else:
                    cross_product = np.cross(e1, e2)
                    r = a * b * c / (2 * np.linalg.norm(cross_product)) if np.linalg.norm(cross_product) != 0 else 1e3

                speed_max = np.sqrt(MAX_ACCELERATION * r)

                lookahead_speed = float('inf')
                for j in range(1, min(lookahead_distance, n - i)):
                    future_point = np.array(waypoints[i + j])
                    distance_to_future = np.linalg.norm(future_point - p_curr)
                    future_speed = np.sqrt(speeds[i - 1]**2 + 2 * MAX_ACCELERATION * distance_to_future)
                    lookahead_speed = min(lookahead_speed, future_speed)

                speeds[i] = min(speed_next, speed_max, lookahead_speed, SMOOTH_TRACKING_SPEED_MAX)
            else:
                speeds[i] = min(speeds[i - 1], SMOOTH_TRACKING_SPEED_MIN)

        return speeds

    def initTrajectory(self, path):
        final_waypoints = []
        waypoints = []
        speeds = []

        waypoints.append(self.START_STATE)

        for point in path:
            waypoints.append(np.array([point[0], point[1], 1.0]))

        waypoints.append([self.GOAL_STATE[0], self.GOAL_STATE[1], LANDING_HEIGHT])

        for i in range(0, len(waypoints)):
            if i > 0:
                if np.all(waypoints[i - 1] == waypoints[i]):
                    continue
            final_waypoints.append(waypoints[i])

        speeds = self.computeSpeeds(final_waypoints)

        return final_waypoints, speeds

    def filterPath(self, path, start_point, goal_point, waypoint_spacing = 0.3):
        intermediate_path = []
        for i in range(1, len(path) - 1):
            intermediate_path.append(path[i])

        intermediate_filtered_path = self.movingAvg(intermediate_path, window_size=3)
        filtered_path = [start_point]
        filtered_path += intermediate_filtered_path
        filtered_path += [goal_point]

        waypoints = np.array(filtered_path)
        x = waypoints[:, 0]
        y = waypoints[:, 1]

        cum_dist = np.zeros(len(x))
        for i in range(1, len(cum_dist)):
            cum_dist[i] = cum_dist[i-1] + np.linalg.norm(waypoints[i] - waypoints[i-1])

        spline_x = CubicSpline(cum_dist, x, bc_type='natural')
        spline_y = CubicSpline(cum_dist, y, bc_type='natural')

        num_samples = int(cum_dist[-1] / waypoint_spacing)
        sample_steps = np.linspace(cum_dist[0], cum_dist[-1], num_samples)
        x_filtered = spline_x(sample_steps)
        y_filtered = spline_y(sample_steps)

        filtered_path = [np.array([x, y]) for x, y in zip(x_filtered, y_filtered)]
        filtered_path_no_collisions = [filtered_path[0]]

        i = 1
        while i < len(filtered_path):
            # Locally resample and run fmt
            active_gate_indices = []

            if USE_FMT_BIDIRECTIONAL_EXIT:
                if tuple(filtered_path[i-1]) in self.gate_indices_dict:
                    active_gate_indices.append(self.gate_indices_dict[tuple(filtered_path[i-1])])

            if tuple(filtered_path[i]) in self.gate_indices_dict:
                active_gate_indices.append(self.gate_indices_dict[tuple(filtered_path[i])])

            if not self.__checkCollision(Node(filtered_path[i-1]), Node(filtered_path[i]), active_gate_indices):
                print(f'LOCAL RESAMPLING')
                start_idx = i - 1

                goal_idx = i
                sampling_offset = 1

                while goal_idx < len(filtered_path) - 1 and self.__isPointInObstacle(filtered_path[goal_idx]):
                    goal_idx += 1
                    sampling_offset += 1

                tmp_start_point = filtered_path[start_idx]
                tmp_goal_point = filtered_path[goal_idx]
                vec_goal_start = tmp_goal_point - tmp_start_point
                vec_goal_start_mag = np.linalg.norm(tmp_goal_point - tmp_start_point)
                vec_goal_start_unit = vec_goal_start / vec_goal_start_mag

                mid_point = tmp_start_point + vec_goal_start_unit * (vec_goal_start_mag / 2)
                intermediate_path = self.runFMT(
                    num_points=2500,
                    start_points=[tmp_start_point],
                    goal_points=[tmp_goal_point],
                    filter_nodes=False,
                    rn=0.05,
                    sample_local=True,
                    center_point=mid_point,
                    local_radius=sampling_offset * waypoint_spacing)
                print(f'INTERMEDIATE PATH LEN: {len(intermediate_path)}')
                filtered_path_no_collisions += intermediate_path
                i = goal_idx
            else:
                filtered_path_no_collisions.append(filtered_path[i])
            i += 1

        return filtered_path_no_collisions

    def movingAvg(self, points, window_size):
        if len(points) < window_size:
            return points

        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])

        x_filtered = np.convolve(x, np.ones(window_size) / window_size, mode='valid')
        y_filtered = np.convolve(y, np.ones(window_size) / window_size, mode='valid')

        filtered_points = np.column_stack((x_filtered, y_filtered)).tolist()

        return filtered_points

    def plotPath(self, path, speeds=None):
        fig, ax = plt.subplots()
        ax.set_xlim((-3.5, 3.5))
        ax.set_ylim((-3.5, 3.5))
        markersize = 10

        x = list()
        y = list()

        for point in path:
            x.append(point[0])
            y.append(point[1])
        ax.plot(x, y, color='C1', linewidth=1)
        ax.scatter(x, y, color='b', s=markersize)

        ax.scatter(path[0][0], path[0][1], color='g', s=markersize)
        ax.scatter(path[-1][0], path[-1][1], color='g', s=markersize)

        theta = np.linspace(0, 2 * np.pi, 100)
        for gate_key in self.GATE_SHAPES.keys():
            rotated_gate_corners, (rotated_gate_edge_center_left, rotated_gate_edge_center_right) = self.GATE_SHAPES[gate_key]

            corners = np.vstack([rotated_gate_corners, rotated_gate_corners[0]])
            x, y = corners[:, 0], corners[:, 1]

            ax.plot(x, y, color='g', linewidth=2)

            x = rotated_gate_edge_center_left[0] + self.GATE_EDGE_LEN * np.cos(theta)
            y = rotated_gate_edge_center_left[1] + self.GATE_EDGE_LEN * np.sin(theta)

            ax.plot(x, y, color='r')
            ax.scatter(rotated_gate_edge_center_left[0], rotated_gate_edge_center_left[1], color='r', s=markersize)

            x = rotated_gate_edge_center_left[0] + GATE_EDGE_CLEARANCE * np.cos(theta)
            y = rotated_gate_edge_center_left[1] + GATE_EDGE_CLEARANCE * np.sin(theta)
            ax.plot(x, y, color='r', linestyle='dashed', linewidth=1)

            x = rotated_gate_edge_center_right[0] + self.GATE_EDGE_LEN * np.cos(theta)
            y = rotated_gate_edge_center_right[1] + self.GATE_EDGE_LEN * np.sin(theta)

            ax.plot(x, y, color='r', linewidth=1)
            ax.scatter(rotated_gate_edge_center_right[0], rotated_gate_edge_center_right[1], color='r', s=markersize)

            x = rotated_gate_edge_center_right[0] + GATE_EDGE_CLEARANCE * np.cos(theta)
            y = rotated_gate_edge_center_right[1] + GATE_EDGE_CLEARANCE * np.sin(theta)

            ax.plot(x, y, color='r', linestyle='dashed', linewidth=1)

        for obstacle_center in self.OBSTACLE_LOCATIONS:
            obstacle_center_2d = obstacle_center[:2]

            x = obstacle_center_2d[0] + self.OBSTACLE_RADIUS * np.cos(theta)
            y = obstacle_center_2d[1] + self.OBSTACLE_RADIUS * np.sin(theta)

            ax.plot(x, y, color='r', linewidth=1)

            x = obstacle_center_2d[0] + OBSTACLE_CLEARANCE * np.cos(theta)
            y = obstacle_center_2d[1] + OBSTACLE_CLEARANCE * np.sin(theta)

            ax.plot(x, y, color='r', linestyle='dashed', linewidth=1)

        ax.scatter(self.GATE_LOCATIONS[:, 0], self.GATE_LOCATIONS[:, 1], color='b', s=markersize)
        ax.scatter(self.OBSTACLE_LOCATIONS[:, 0], self.OBSTACLE_LOCATIONS[:, 1], color='r', s=markersize)

        plt.gca().set_aspect('equal', adjustable='box')

        if speeds is not None:
            fig, ax = plt.subplots()
            ax.plot(np.array(speeds), label='Velocity Profile')
            ax.set_title('Velocity Profile Along Path')
            ax.set_xlabel('Point Index')
            ax.set_ylabel('Velocity (units/s)')
            ax.grid(True)
            ax.legend()

        plt.show()

    def __addGoalStates(self, waypoint_tolerance: float = 0.2):
        gate_locations = self.GATE_LOCATIONS

        gate_goal_states = []
        for gate_location in gate_locations:
            gate_goal_1 = np.array([gate_location[0], gate_location[1] + waypoint_tolerance])
            gate_goal_2 = np.array([gate_location[0], gate_location[1] - waypoint_tolerance])

            # if DEBUG_PATH_PLANNING: print(f'GATE GOAL 1: {gate_goal_1}')
            # if DEBUG_PATH_PLANNING: print(f'GATE GOAL 2: {gate_goal_2}')

            gate_goals = np.vstack([
                [self.__rotatePoint(gate_goal_1, gate_location[:2], gate_location[3])],
                [self.__rotatePoint(gate_goal_2, gate_location[:2], gate_location[3])]
            ])
            # if DEBUG_PATH_PLANNING: print(f'GATE GOALS SHAPE: {gate_goals.shape}')
            gate_goal_states.append(gate_goals)

        return gate_goal_states

    def __setupInitialTraj(self):
        if USE_FMT_BIDIRECTIONAL_EXIT:
            return self.__setupInitialTrajBidirectional()
        else:
            return self.__setupInitialTrajUnidirectional()

    def __setupInitialTrajUnidirectional(self):
        gate_indices_dict = {}
        gate_goal_states = self.__addGoalStates()
        # if DEBUG_PATH_PLANNING: print(f'GATE GOAL STATES: {gate_goal_states}')

        start_point = self.START_STATE[:2]
        goal_point = self.GOAL_STATE[:2]

        start_states = []
        goal_states = []

        cur_point = start_point
        idx = 0
        for i in range(len(gate_goal_states)):
            # if DEBUG_PATH_PLANNING: print(f'CUR_POINT: {cur_point}')
            gate = self.GATE_LOCATIONS[i, :2].copy()
            gate_goals = gate_goal_states[i]

            gate_indices_dict[tuple(gate_goals[0])] = i
            gate_indices_dict[tuple(gate_goals[1])] = i

            # if DEBUG_PATH_PLANNING: print(f'GATE_GOALS: {gate_goals}')

            dists = np.linalg.norm(gate_goals - cur_point, axis=1)
            # if DEBUG_PATH_PLANNING: print(f'DISTS: {dists}')
            closest_point = gate_goals[np.argmin(dists)]
            # if DEBUG_PATH_PLANNING: print(f'CLOSEST_POINT: {closest_point}')

            start_states.append(cur_point)
            goal_states.append(closest_point)

            cur_point = gate_goals[np.argmax(dists)]

        # if DEBUG_PATH_PLANNING: print(f'START_STATES SHAPE: {len(start_states)}')
        # if DEBUG_PATH_PLANNING: print(f'GOAL_STATES SHAPE: {len(goal_states)}')
        start_states.append(cur_point)
        goal_states.append(goal_point)

        return start_states, goal_states, gate_indices_dict

    def __setupInitialTrajBidirectional(self):
        gate_indices_dict = {}
        gate_goal_states = self.__addGoalStates()
        # if DEBUG_PATH_PLANNING: print(f'GATE GOAL STATES: {gate_goal_states}')

        start_point = self.START_STATE[:2]
        goal_point = self.GOAL_STATE[:2]

        start_states = []
        goal_states = []

        cur_point = start_point
        closest_point = start_point
        dists = None

        prev_gate_idx = -1
        for i in range(len(gate_goal_states)):
            # if DEBUG_PATH_PLANNING: print(f'CUR_POINT: {cur_point}')
            gate_goals = gate_goal_states[i]

            gate_indices_dict[tuple(gate_goals[0])] = i
            gate_indices_dict[tuple(gate_goals[1])] = i

            # if DEBUG_PATH_PLANNING: print(f'GATE_GOALS: {gate_goals}')

            if i > 0:
                prev_gate_goals = gate_goal_states[prev_gate_idx]
                dists_0 = np.linalg.norm(gate_goals - prev_gate_goals[0], axis=1)
                dists_1 = np.linalg.norm(gate_goals - prev_gate_goals[1], axis=1)

                min_dist_0 = np.min(dists_0)
                min_dist_1 = np.min(dists_1)

                if min_dist_0 < min_dist_1:
                    new_cur_point = prev_gate_goals[0]
                    if not np.all(cur_point == new_cur_point):
                        start_states.append(cur_point)
                        goal_states.append(new_cur_point)
                    cur_point = new_cur_point
                    dists = dists_0
                else:
                    new_cur_point = prev_gate_goals[1]
                    if not np.all(cur_point == new_cur_point):
                        start_states.append(cur_point)
                        goal_states.append(new_cur_point)
                    cur_point = new_cur_point
                    dists = dists_1
            else:
                dists = np.linalg.norm(gate_goals - cur_point, axis=1)
                # if DEBUG_PATH_PLANNING: print(f'DISTS: {dists}')
                # if DEBUG_PATH_PLANNING: print(f'CLOSEST_POINT: {closest_point}')
            closest_point = gate_goals[np.argmin(dists)]

            start_states.append(cur_point)
            goal_states.append(closest_point)

            cur_point = gate_goals[np.argmax(dists)]

            start_states.append(closest_point)
            goal_states.append(cur_point)

            prev_gate_idx = i

        # if DEBUG_PATH_PLANNING: print(f'START_STATES SHAPE: {len(start_states)}')
        # if DEBUG_PATH_PLANNING: print(f'GOAL_STATES SHAPE: {len(goal_states)}')
        prev_gate_goals = gate_goal_states[prev_gate_idx]
        dists = np.linalg.norm(goal_point - prev_gate_goals, axis=1)
        closest_point = prev_gate_goals[np.argmin(dists)]
        if not np.all(closest_point == cur_point):
            start_states.append(cur_point)
            goal_states.append(closest_point)
            cur_point = closest_point

        start_states.append(cur_point)
        goal_states.append(goal_point)

        return start_states, goal_states, gate_indices_dict

    def plotObstacles(self):
        fig, ax = plt.subplots()
        ax.set_xlim((-3.5, 3.5))
        ax.set_ylim((-3.5, 3.5))
        markersize = 10

        theta = np.linspace(0, 2 * np.pi, 100)
        for gate_key in self.GATE_SHAPES.keys():
            rotated_gate_corners, (rotated_gate_edge_center_left, rotated_gate_edge_center_right) = self.GATE_SHAPES[gate_key]

            corners = np.vstack([rotated_gate_corners, rotated_gate_corners[0]])
            x, y = corners[:, 0], corners[:, 1]

            ax.plot(x, y, color='g', linewidth=2)

            x = rotated_gate_edge_center_left[0] + self.GATE_EDGE_LEN * np.cos(theta)
            y = rotated_gate_edge_center_left[1] + self.GATE_EDGE_LEN * np.sin(theta)

            ax.plot(x, y, color='r')
            ax.scatter(rotated_gate_edge_center_left[0], rotated_gate_edge_center_left[1], color='r', s=markersize)

            x = rotated_gate_edge_center_right[0] + self.GATE_EDGE_LEN * np.cos(theta)
            y = rotated_gate_edge_center_right[1] + self.GATE_EDGE_LEN * np.sin(theta)

            ax.plot(x, y, color='r')
            ax.scatter(rotated_gate_edge_center_right[0], rotated_gate_edge_center_right[1], color='r', s=markersize)

        for obstacle_center in self.OBSTACLE_LOCATIONS:
            obstacle_center_2d = obstacle_center[:2]

            x = obstacle_center_2d[0] + self.OBSTACLE_RADIUS * np.cos(theta)
            y = obstacle_center_2d[1] + self.OBSTACLE_RADIUS * np.sin(theta)

            ax.plot(x, y, color='r')

        ax.scatter(self.GATE_LOCATIONS[:, 0], self.GATE_LOCATIONS[:, 1], color='b', s=markersize)
        ax.scatter(self.OBSTACLE_LOCATIONS[:, 0], self.OBSTACLE_LOCATIONS[:, 1], color='r', s=markersize)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def __constructGateShapes(self):
        gate_shapes = {}
        gate_edge_centers = []

        for gate_pose in self.GATE_LOCATIONS:
            # Get gate center and orientation
            gate_center = gate_pose[:2]
            gate_yaw = gate_pose[3]
            # if DEBUG_PATH_PLANNING: print(f'GATE_YAW: {gate_yaw}')

            # Construct full area of gate using 4 corners
            gate_len = 2.0 * self.GATE_EDGE_LEN + self.GATE_INNER_EDGE_LEN
            gate_width = self.GATE_EDGE_WID

            # Gate limits
            gate_x_min = gate_center[0] - gate_len / 2.0
            gate_x_max = gate_center[0] + gate_len / 2.0
            gate_y_min = gate_center[1] - gate_width / 2.0
            gate_y_max = gate_center[1] + gate_width / 2.0

            # Gate corners
            gate_corners = np.array([
                [gate_x_min, gate_y_min], # bottom-left
                [gate_x_max, gate_y_min], # bottom-right
                [gate_x_max, gate_y_max], # top-right
                [gate_x_min, gate_y_max], # top-left
            ], dtype=np.float32)

            # Gate edge centers
            gate_edge_center_left = [gate_x_min + self.GATE_EDGE_LEN / 2, gate_center[1]]
            gate_edge_center_right = [gate_x_max - self.GATE_EDGE_LEN / 2, gate_center[1]]

            # if DEBUG_PATH_PLANNING: print(f'GATE EDGE CENTER LEFT: {gate_edge_center_left}')
            # if DEBUG_PATH_PLANNING: print(f'GATE EDGE CENTER RIGHT: {gate_edge_center_right}')
            # if DEBUG_PATH_PLANNING: print(f'GATE CORNERS: {gate_corners.shape}')

            # Rotate gate corners based on orientation
            rotated_gate_corners = np.array([
                self.__rotatePoint(gate_corner, gate_center, gate_yaw) for gate_corner in gate_corners])
            if DEBUG_PATH_PLANNING: print(f'ROTATED GATE CORNERS: {rotated_gate_corners}')

            # Rotate gate edge centers based on orientation
            rotated_gate_edge_center_left = self.__rotatePoint(gate_edge_center_left, gate_center, gate_yaw)
            rotated_gate_edge_center_right = self.__rotatePoint(gate_edge_center_right, gate_center, gate_yaw)
            if DEBUG_PATH_PLANNING: print(f'ROTATED GATE EDGE CENTER LEFT: {rotated_gate_edge_center_left}')
            if DEBUG_PATH_PLANNING: print(f'ROTATED GATE EDGE CENTER RIGHT: {rotated_gate_edge_center_right}')

            gate_shapes[tuple(gate_center)] = (rotated_gate_corners, (rotated_gate_edge_center_left, rotated_gate_edge_center_right))

            gate_edge_centers.append([rotated_gate_edge_center_left, rotated_gate_edge_center_right])

        return gate_shapes, np.vstack(gate_edge_centers, dtype=np.float32)

    def __rotatePoint(self, point, center, angle):
        s, c = np.sin(angle), np.cos(angle)
        x_rot = c * (point[0] - center[0]) - s * (point[1] - center[1]) + center[0]
        y_rot = s * (point[0] - center[0]) + c * (point[1] - center[1]) + center[1]
        return np.array([x_rot, y_rot])

    #########################################################################################################
    # Function: __samplePoint(self, sample_goal: bool = False)
    #
    # Samples a point within the map boundaries and returns a node in free space
    #
    # Inputs: sample_goal (Optional)
    #   sample_goal - Bool to control whether to sample near the goal state
    #
    # Outputs: sample_point
    #   sample_point - 1 x 2 array representing sampled point in free space
    #########################################################################################################
    def __samplePoint(self, sample_goal: bool = False):
        sample_point = None
        sample_in_free_space = False
        while not sample_in_free_space:
            x = np.random.uniform(self.X_BOUND[0], self.X_BOUND[1])
            y = np.random.uniform(self.Y_BOUND[0], self.Y_BOUND[1])

            sample_point = np.array([x, y], dtype=np.float32)

            # Check if point is in obstacle region
            if not self.__isPointInObstacle(sample_point):
                sample_in_free_space = True

        return sample_point

    def __samplePointLocal(self, center_point, local_radius: float = 0.5):
        sample_point = None
        sample_in_free_space = False
        while not sample_in_free_space:
            x_bound_low = max(self.X_BOUND[0], center_point[0] - local_radius)
            x_bound_hi = min(self.X_BOUND[1], center_point[0] + local_radius)

            y_bound_low = max(self.Y_BOUND[0], center_point[1] - local_radius)
            y_bound_hi = max(self.Y_BOUND[1], center_point[1] + local_radius)

            x = np.random.uniform(x_bound_low, x_bound_hi)
            y = np.random.uniform(y_bound_low, y_bound_hi)

            sample_point = np.array([x, y], dtype=np.float32)

            # Check if point is in obstacle region
            if not self.__isPointInObstacle(sample_point) and np.linalg.norm(sample_point - center_point) <= local_radius:
                sample_in_free_space = True

        return sample_point

    #########################################################################################################
    # Function: __sampleNPoints(self, start_point = None, goal_point = None, num_points: int = 5000)
    #
    # Samples N points within the map boundaries and returns N nodes in free space
    #
    # Inputs: start_point (Optional), goal_point (Optional), num_points (Optional)
    #   start_point - 1 x 2 array representing starting point
    #   goal_point - 1 x 2 array representing goal point
    #   num_points - int denoting number of samples desired
    #
    # Outputs: sampled_points
    #   sampled_points - N x 2 array representing N sampled points in free space
    #########################################################################################################
    def __sampleNPoints(self, start_point = None, goal_point = None, num_points: int = 5000):
        sampled_points_list = []

        if start_point is not None:
            sampled_points_list.append(start_point)

        for _ in range(num_points):
            sampled_points_list.append(self.__samplePoint())

        if goal_point is not None:
            sampled_points_list.append(goal_point)

        sampled_points = np.array(sampled_points_list)
        return sampled_points

    def __sampleNPointsLocal(self, num_points: int = 5000, center_point = None, local_radius: float = 0.5):
        sampled_points_list = []

        for _ in range(num_points):
            sampled_points_list.append(self.__samplePointLocal(center_point=center_point, local_radius=local_radius))

        sampled_points = np.array(sampled_points_list)
        return sampled_points

    #########################################################################################################
    # Function: __isPointInObstacle(self, sample_point)
    #
    # Checks whether a sampled point is inside an obstacle region
    #
    # Inputs: sample_point
    #   sample_point - 1 x 2 array representing the sampled point
    #
    # Outputs: bool
    #   Returns true if point in obstacle region, else false
    #########################################################################################################
    def __isPointInObstacle(self, sample_point):
        # Check obstacles
        obstacle_centers_2d = self.OBSTACLE_LOCATIONS[:, :2]
        dists_to_obstacles = np.linalg.norm(sample_point - obstacle_centers_2d, axis=1)
        # if np.any(dists_to_obstacles <= self.OBSTACLE_RADIUS):
        if np.any(dists_to_obstacles <= OBSTACLE_CLEARANCE):
            return True

        # Check gate edges
        gate_edge_centers = self.GATE_EDGE_CENTERS
        dists_to_gate_edges = np.linalg.norm(sample_point - gate_edge_centers, axis=1)
        # if np.any(dists_to_gate_edges <= self.GATE_EDGE_LEN):
        if np.any(dists_to_gate_edges <= GATE_EDGE_CLEARANCE):
            return True

        return False

    def __lineLineIntersection(self, p1, p2, q1, q2):
        # Line connecting nodes
        x1, y1 = p1
        x2, y2 = p2

        a1 = y2 - y1
        b1 = x1 - x2
        c1 = a1 * x1 + b1 * y1

        # Line connecting 2 gate corners
        x3, y3 = q1
        x4, y4 = q2

        a2 = y4 - y3
        b2 = x3 - x4
        c2 = a2 * x3 + b2 * y3

        # Matrix form of equations
        A_mat = np.array([[a1, b1], [a2, b2]])
        b_vec = np.array([c1, c2])

        det = np.linalg.det(A_mat)
        # Parallel lines
        if det == 0:
            return None

        intersection_point = np.linalg.solve(A_mat, b_vec)
        return tuple(intersection_point)

    def __isPointInLine(self, p1, p2, q):
        x, y = q
        x1, y1 = p1
        x2, y2 = p2
        return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

    def __lineRectangleIntersection(self, line_points, rectangle_corners):
        p1, p2 = line_points
        intersection_points = []
        for i in range(4):
            intersection = self.__lineLineIntersection(p1, p2, rectangle_corners[i, :], rectangle_corners[(i+1) % 4, :])
            if intersection and self.__isPointInLine(rectangle_corners[i, :], rectangle_corners[(i+1) % 4, :], intersection):
                intersection_points.append(intersection)

        # Check if intersection points are on the node line segment
        segment_intersections = [pt for pt in intersection_points if self.__isPointInLine(p1, p2, pt)]

        if segment_intersections:
            return True
        return False

    def __checkCollision(self, nodeA, nodeB, active_gate_indices):
        if DEBUG_COLLISIONS: print(f'NODE A: {nodeA.point}')
        if DEBUG_COLLISIONS: print(f'NODE B: {nodeB.point}')

        # Start to end point vector and magnitude
        vec_a_to_b = nodeB.point - nodeA.point
        vec_a_to_b_mag = np.linalg.norm(vec_a_to_b)
        vec_a_to_b_unit = vec_a_to_b / vec_a_to_b_mag

        if DEBUG_COLLISIONS: print(f'VEC_AB: {vec_a_to_b}')
        if DEBUG_COLLISIONS: print(f'VEC_AB_MAG: {vec_a_to_b_mag}')
        if DEBUG_COLLISIONS: print(f'VEC_AB_UNIT: {vec_a_to_b_unit}')

        # Obstacles
        if DEBUG_COLLISIONS: print(f'----------------------OBSTACLES------------------------')
        for obstacle_center in self.OBSTACLE_LOCATIONS:
            # Start to obstacle center vector
            obstacle_center_2d = obstacle_center[:2]
            vec_a_to_obs = obstacle_center_2d - nodeA.point

            # Project obstacle center to line
            proj_obs_scale = np.dot(vec_a_to_b_unit, vec_a_to_obs)
            if proj_obs_scale >= 0 and proj_obs_scale <= vec_a_to_b_mag:
                proj_obs_point = nodeA.point + vec_a_to_b_unit * proj_obs_scale

                # Calculate distances of projected points and obstacle centers
                dist_to_obs_center = np.linalg.norm(proj_obs_point - obstacle_center_2d)
                if DEBUG_COLLISIONS: print(f'PROJECTED OBS POINT: {proj_obs_point}')
                if DEBUG_COLLISIONS: print(f'OBS CENTER: {obstacle_center_2d}')
                if DEBUG_COLLISIONS: print(f'DISTANCE: {dist_to_obs_center}')

                if dist_to_obs_center <= OBSTACLE_CLEARANCE:
                    if DEBUG_COLLISIONS: print(f'OBSTACLE COLLISION')
                    return False

            dist_start_to_obstacle_center = np.linalg.norm(obstacle_center_2d - nodeA.point)
            dist_end_to_obstacle_center = np.linalg.norm(obstacle_center_2d - nodeB.point)

            if dist_start_to_obstacle_center <= OBSTACLE_CLEARANCE or dist_end_to_obstacle_center <= OBSTACLE_CLEARANCE:
                if DEBUG_COLLISIONS: print(f'OBSTACLE START/END COLLISION')
                return False
        if DEBUG_COLLISIONS: print(f'----------------------END OBSTACLES------------------------')

        # Gates
        if DEBUG_COLLISIONS: print(f'----------------------GATES------------------------')
        for gate_edge_center in self.GATE_EDGE_CENTERS:

            # Start to gate edge centers vector
            vec_a_to_gate_edge = gate_edge_center - nodeA.point
            if DEBUG_COLLISIONS: print(f'VEC_A_EDGE: {vec_a_to_gate_edge}')

            # Project gate edge center to line
            proj_gate_edge_scale = np.dot(vec_a_to_b_unit, vec_a_to_gate_edge)
            if proj_gate_edge_scale >= 0 and proj_gate_edge_scale <= vec_a_to_b_mag:
            # proj_gate_edge_scale = np.clip(proj_gate_edge_scale, 0, vec_a_to_b_mag)
                proj_gate_edge_point = nodeA.point + vec_a_to_b_unit * proj_gate_edge_scale

                # Calculate distances of projected points and gate edge centers
                dist_to_gate_edge_center = np.linalg.norm(proj_gate_edge_point - gate_edge_center)
                if DEBUG_COLLISIONS: print(f'PROJECTED GATE EDGE POINT: {proj_gate_edge_point}')
                if DEBUG_COLLISIONS: print(f'GATE EDGE CENTER: {gate_edge_center}')
                if DEBUG_COLLISIONS: print(f'DISTANCE: {dist_to_gate_edge_center}')

                if dist_to_gate_edge_center <= GATE_EDGE_CLEARANCE:
                    if DEBUG_COLLISIONS: print(f'GATE COLLISION')
                    return False

            dist_start_to_gate_edge_center = np.linalg.norm(gate_edge_center - nodeA.point)
            dist_end_to_gate_edge_center = np.linalg.norm(gate_edge_center - nodeB.point)

            if dist_start_to_gate_edge_center <= GATE_EDGE_CLEARANCE or dist_end_to_gate_edge_center <= GATE_EDGE_CLEARANCE:
                if DEBUG_COLLISIONS: print(f'GATE START/END COLLISION')
                return False
        if DEBUG_COLLISIONS: print(f'----------------------END GATES------------------------')

        # Inactive gate collisions
        for i in range(self.GATE_LOCATIONS.shape[0]):
            if i not in active_gate_indices:
                gate_center = tuple(self.GATE_LOCATIONS[i, :2])
                gate_corners, _ = self.GATE_SHAPES[gate_center]
                if self.__lineRectangleIntersection((nodeA.point, nodeB.point), gate_corners):
                    if DEBUG_COLLISIONS: print(f'INACTIVE GATE COLLISION')
                    return False

        if DEBUG_COLLISIONS: print(f'NO COLLISION')
        return True

    def fmt(self,
            sampled_points,
            start_point,
            goal_point,
            max_iters,
            rn,
            active_gate_indices):

        sampled_points[0] = start_point
        sampled_points[-1] = goal_point
        nodes = np.array([Node(sampled_point) for sampled_point in sampled_points])
        if DEBUG_PATH_PLANNING: print(f'NODE SHAPE: {nodes.shape}')

        nn_kd_tree = KDTree(sampled_points)

        path_found = True

        V_open = MinHeap()
        V_open.add_or_update_node(0, nodes[0].cost)
        V_unvisited = np.arange(0, len(nodes), 1)
        V_unvisited[0] = -1

        z_idx = 0
        z = nodes[z_idx]
        if DEBUG_PATH_PLANNING: print(f"Z START: {z.point}")

        while not np.allclose(z.point, goal_point, atol=1e-3):
            # Find all nodes within radius rn from z
            N_z_indices = nn_kd_tree.query_ball_point(z.point, rn)
            if DEBUG_PATH_PLANNING: print(f'Z NEIGHBOURS: {N_z_indices}')

            # Filter nodes only belonging to unvisited set
            X_near_indices = np.intersect1d(N_z_indices, V_unvisited)
            X_near = nodes[X_near_indices]
            # if DEBUG_PATH_PLANNING: print(f'X_NEAR: {[node.point for node in X_near]}')
            if DEBUG_PATH_PLANNING: print(f'X_NEAR SHAPE: {X_near.shape}')

            for x, x_idx in zip(X_near, X_near_indices):
                # Find all nodes within radius rn from x
                N_x_indices = nn_kd_tree.query_ball_point(x.point, rn)
                # if DEBUG_PATH_PLANNING: print(f'X NEIGHBOURS: {N_x_indices}')

                # Filter nodes only belonging to open set
                V_open_indices = np.array(list(V_open.keys()))
                Y_near_indices = np.intersect1d(N_x_indices, V_open_indices)
                Y_near = nodes[Y_near_indices]
                if DEBUG_PATH_PLANNING: print(f'Y_NEAR SHAPE: {Y_near.shape}')

                # for y in Y_near:
                #     if DEBUG_PATH_PLANNING: print(f'Y COST TO X: {y.GetCostToNode(x)}')

                Y_costs = np.array([nodes[y_idx].cost + nodes[y_idx].GetCostToNode(x) for y_idx in Y_near_indices])
                y_min_cost = np.min(Y_costs)
                y_min_idx = Y_near_indices[np.argmin(Y_costs)]

                if DEBUG_PATH_PLANNING: print(f'Y_MIN_COST: {y_min_cost}')
                if DEBUG_PATH_PLANNING: print(f'Y_MIN_IDX: {y_min_idx}')

                if self.__checkCollision(nodes[y_min_idx], nodes[x_idx], active_gate_indices):
                    nodes[y_min_idx].children.append(x)
                    nodes[x_idx].parent = y_min_idx
                    nodes[x_idx].cost = y_min_cost

                    V_open.add_or_update_node(x_idx, y_min_cost)
                    V_unvisited[x_idx] = -1

            V_open.remove_node(z_idx)

            if V_open.is_empty():
                if DEBUG_PATH_PLANNING: print(f'OUT OF NODES')
                path_found = False
                break

            z_idx, _ = V_open.top()
            z = nodes[z_idx]

        if path_found:
            if DEBUG_PATH_PLANNING: print(f'GOAL POINT FOUND')
        else:
            if DEBUG_PATH_PLANNING: print(f'NO PATH FOUND FAIL')

        return nodes if path_found else []


def exampleFunction():
    """Example of user-defined function.

    """
    x = -1
    return x
