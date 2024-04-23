'''path planning implementation'''

import numpy as np
from scipy.optimize import minimize

from matplotlib import pyplot as plt

from constants import *
from collision import *


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
        self.speeds.append(0.0)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 2]))
        self.speeds.append(0.0)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 3]))
        self.speeds.append(0.0)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 4]))
        self.speeds.append(0.0)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, 1.0]))
        self.speeds.append(0.0)

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
        self.speeds.append(0.0)
        self.final_waypoints.append(self.end_state + np.array([0.0, 0.0, LANDING_HEIGHT]))
        self.speeds.append(0.0)

        # plot final trajectory
        if POT_PLOT_FINAL_TRAJECTORY: self.plot_trajectory()
        return self.final_waypoints, self.speeds


    def plan_trajectory_bidirectional(self):
        waypoints = []

        # waypoints for smooth take-off (vertical)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT]))
        self.speeds.append(0.0)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 2]))
        self.speeds.append(0.0)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 3]))
        self.speeds.append(0.0)
        self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, TAKE_OFF_HEIGHT * 4]))
        self.speeds.append(0.0)
        # self.final_waypoints.append(self.start_state + np.array([0.0, 0.0, 1.0]))
        # self.speeds.append(0.0)

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
                self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)
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
            self.speeds.append(SMOOTH_TRACKING_SPEED_MIN)
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
        self.speeds.append(0.0)
        self.final_waypoints.append(self.end_state + np.array([0.0, 0.0, LANDING_HEIGHT]))
        self.speeds.append(0.0)

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
        speed_next = np.sqrt(v_prev**2 + 2 * MAX_ACCELERATION * a)

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