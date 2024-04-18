'''
PathPlanner class implementation

Can perform RRT* and FMT* path planning
'''

import numpy as np

from pqdict import pqdict
from scipy.spatial import KDTree

from constants import *
from matplotlib import pyplot as plt

from shapely.geometry import Point, LineString, Polygon

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
            robot_radius: float = 0.05) -> None:

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
        print(f'GATE EDGE CENTERS SHAPE: {self.GATE_EDGE_CENTERS.shape}')

    def runFMT(
            self,
            num_points: int = 10000,
            max_iters: int = 10000,
            rn: float = 0.25):
        start_points, goal_points = self.__setupInitialTraj()
        path = []
        sampled_points = np.array(self.__sampleNPoints(num_points=num_points))
        for i in range(len(start_points)):
            start_point = start_points[i]
            goal_point = goal_points[i]
            if DEBUG_PATH_PLANNING: print(f'START POINT: {start_point}')
            if DEBUG_PATH_PLANNING: print(f'GOAL POINT: {goal_point}')

            nodes = self.fmt(
                sampled_points=sampled_points.copy(),
                start_point=start_point,
                goal_point=goal_point,
                max_iters=max_iters,
                rn=rn)
            sub_path = [nodes[-1].point]

            path_end_idx = nodes[-1].parent
            while path_end_idx > -1:
                sub_path.append(nodes[path_end_idx].point)
                path_end_idx = nodes[path_end_idx].parent
            sub_path.reverse()
            path += sub_path

        return path

    def plotPath(self, path):
        fig, ax = plt.subplots()
        ax.set_xlim((-3.5, 3.5))
        ax.set_ylim((-3.5, 3.5))
        markersize = 10

        x = list()
        y = list()

        for point in path:
            x.append(point[0])
            y.append(point[1])
        ax.plot(x, y, color='r', linewidth=1)
        ax.scatter(x, y, color='b', s=markersize)

        ax.scatter(path[0][0], path[0][1], color='g', s=markersize)
        ax.scatter(path[-1][0], path[-1][1], color='g', s=markersize)

        theta = np.linspace(0, 2 * np.pi, 100)  # Create an array of angular coordinates
        for gate_key in self.GATE_SHAPES.keys():
            rotated_gate_corners, (rotated_gate_edge_center_left, rotated_gate_edge_center_right) = self.GATE_SHAPES[gate_key]

            gate_object = Polygon(rotated_gate_corners)
            x,y = gate_object.exterior.xy
            
            ax.plot(x, y, color='g', linewidth=2)

            x = rotated_gate_edge_center_left[0] + self.GATE_EDGE_LEN * np.cos(theta)  # X coordinates of circle
            y = rotated_gate_edge_center_left[1] + self.GATE_EDGE_LEN * np.sin(theta)  # Y coordinates of circle
            
            ax.plot(x, y, color='r')  # Plot the circle
            ax.scatter(rotated_gate_edge_center_left[0], rotated_gate_edge_center_left[1], color='r', s=markersize)

            x = rotated_gate_edge_center_right[0] + self.GATE_EDGE_LEN * np.cos(theta)  # X coordinates of circle
            y = rotated_gate_edge_center_right[1] + self.GATE_EDGE_LEN * np.sin(theta)  # Y coordinates of circle
            
            ax.plot(x, y, color='r')  # Plot the circle
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

            # if DEBUG_PATH_PLANNING: print(f'GATE_GOALS: {gate_goals}')

            dists = np.linalg.norm(gate_goals - cur_point, axis=1)
            # if DEBUG_PATH_PLANNING: print(f'DISTS: {dists}')
            closest_point = gate_goals[np.argmin(dists)]
            # if DEBUG_PATH_PLANNING: print(f'CLOSEST_POINT: {closest_point}')

            start_states.append(cur_point)
            goal_states.append(closest_point)

            start_states.append(closest_point)
            goal_states.append(gate)

            cur_point = gate_goals[np.argmax(dists)]

            start_states.append(gate)
            goal_states.append(cur_point)

        # if DEBUG_PATH_PLANNING: print(f'START_STATES SHAPE: {len(start_states)}')
        # if DEBUG_PATH_PLANNING: print(f'GOAL_STATES SHAPE: {len(goal_states)}')
        start_states.append(cur_point)
        goal_states.append(goal_point)

        return start_states, goal_states

    def plotObstacles(self):
        fig, ax = plt.subplots()
        ax.set_xlim((-3.5, 3.5))
        ax.set_ylim((-3.5, 3.5))
        markersize = 10

        theta = np.linspace(0, 2 * np.pi, 100)  # Create an array of angular coordinates
        for gate_key in self.GATE_SHAPES.keys():
            rotated_gate_corners, (rotated_gate_edge_center_left, rotated_gate_edge_center_right) = self.GATE_SHAPES[gate_key]

            gate_object = Polygon(rotated_gate_corners)
            x,y = gate_object.exterior.xy
            
            ax.plot(x, y, color='g', linewidth=2)

            x = rotated_gate_edge_center_left[0] + self.GATE_EDGE_LEN * np.cos(theta)  # X coordinates of circle
            y = rotated_gate_edge_center_left[1] + self.GATE_EDGE_LEN * np.sin(theta)  # Y coordinates of circle
            
            ax.plot(x, y, color='r')  # Plot the circle
            ax.scatter(rotated_gate_edge_center_left[0], rotated_gate_edge_center_left[1], color='r', s=markersize)

            x = rotated_gate_edge_center_right[0] + self.GATE_EDGE_LEN * np.cos(theta)  # X coordinates of circle
            y = rotated_gate_edge_center_right[1] + self.GATE_EDGE_LEN * np.sin(theta)  # Y coordinates of circle
            
            ax.plot(x, y, color='r')  # Plot the circle
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
        if np.any(dists_to_obstacles <= self.OBSTACLE_RADIUS):
            return True
            
        # Check gate edges
        gate_edge_centers = self.GATE_EDGE_CENTERS
        dists_to_gate_edges = np.linalg.norm(sample_point - gate_edge_centers, axis=1)
        if np.any(dists_to_gate_edges <= self.GATE_EDGE_LEN):
            return True

        return False

    def __checkCollision(self, nodeA, nodeB):
        if DEBUG_PATH_PLANNING: print(f'NODE A: {nodeA.point}')
        if DEBUG_PATH_PLANNING: print(f'NODE B: {nodeB.point}')
        
        # Start to end point vector and magnitude
        vec_a_to_b = (nodeB.point - nodeA.point).reshape(2, 1)
        vec_a_to_b_mag = np.linalg.norm(vec_a_to_b)
        vec_a_to_b_unit = vec_a_to_b / vec_a_to_b_mag

        # Obstacles

        # Start to obstacle centers vector
        obstacle_centers_2d = self.OBSTACLE_LOCATIONS[:, :2]
        obstacle_centers_2d = obstacle_centers_2d.reshape(obstacle_centers_2d.shape[0], 2)
        vec_a_to_obs = obstacle_centers_2d - nodeA.point

        # Project obstacle centers to line
        # proj_obs_scale = np.dot(vec_a_to_b, vec_a_to_obs)
        proj_obs_scale = np.dot(vec_a_to_obs, vec_a_to_b)
        proj_obs_scale = np.clip(proj_obs_scale, 0, vec_a_to_b_mag)
        proj_obs_point = nodeA.point + np.outer(proj_obs_scale, vec_a_to_b_unit)

        # Calculate distances of projected points and obstacle centers
        dists_to_obstacle_centers = np.linalg.norm(proj_obs_point - obstacle_centers_2d, axis=1)
        if np.any(dists_to_obstacle_centers <= self.OBSTACLE_RADIUS + self.ROBOT_RADIUS):
            if DEBUG_PATH_PLANNING: print(f'OBSTACLE COLLISION')
            return False

        dist_start_to_obstacle_centers = np.linalg.norm(nodeA.point - obstacle_centers_2d, axis=1)
        dist_end_to_obstacle_centers = np.linalg.norm(nodeB.point - obstacle_centers_2d, axis=1)

        if np.any(dist_start_to_obstacle_centers <= self.OBSTACLE_RADIUS + self.ROBOT_RADIUS) or np.any(dist_end_to_obstacle_centers <= self.OBSTACLE_RADIUS + self.ROBOT_RADIUS):
            if DEBUG_PATH_PLANNING: print(f'OBSTACLE START/END COLLISION')
            return False

        # Gate edges
        # Start to gate edge centers vector
        gate_edge_centers_2d = self.GATE_EDGE_CENTERS[:, :2]
        gate_edge_centers_2d = gate_edge_centers_2d.reshape(gate_edge_centers_2d.shape[0], 2)
        vec_a_to_gate_edges = gate_edge_centers_2d - nodeA.point

        # Project gate edge centers to line
        proj_gate_edge_scale = np.dot(vec_a_to_gate_edges, vec_a_to_b)
        proj_gate_edge_scale = np.clip(proj_gate_edge_scale, 0, vec_a_to_b_mag)
        proj_gate_edge_point = nodeA.point + np.outer(proj_gate_edge_scale, vec_a_to_b_unit)

        # Calculate distances of projected points and gate edge centers
        dists_to_gate_edge_centers = np.linalg.norm(proj_gate_edge_point - gate_edge_centers_2d, axis=1)
        if np.any(dists_to_gate_edge_centers <= self.GATE_EDGE_LEN + self.ROBOT_RADIUS):
            if DEBUG_PATH_PLANNING: print(f'GATE COLLISION')
            return False

        dist_start_to_gate_edge_centers = np.linalg.norm(nodeA.point - gate_edge_centers_2d, axis=1)
        dist_end_to_gate_edge_centers = np.linalg.norm(nodeB.point - gate_edge_centers_2d, axis=1)

        if np.any(dist_start_to_gate_edge_centers <= self.GATE_EDGE_LEN + self.ROBOT_RADIUS) or np.any(dist_end_to_gate_edge_centers <= self.GATE_EDGE_LEN + self.ROBOT_RADIUS):
            if DEBUG_PATH_PLANNING: print(f'GATE START/END COLLISION')
            return False
        
        if DEBUG_PATH_PLANNING: print(f'NO COLLISION')
        return True

    def fmt(self,
            sampled_points,
            start_point,
            goal_point,
            max_iters,
            rn):

        sampled_points[0] = start_point
        sampled_points[-1] = goal_point
        nodes = np.array([Node(sampled_point) for sampled_point in sampled_points])
        if DEBUG_PATH_PLANNING: print(f'NODE SHAPE: {nodes.shape}')

        nn_kd_tree = KDTree(sampled_points)

        path_found = False

        V_open = pqdict({
            0 : nodes[0].cost
        })
        V_closed = []
        V_unvisited = np.arange(0, len(nodes), 1)
        V_unvisited[0] = -1
        if DEBUG_PATH_PLANNING: print(f'V_OPEN SHAPE: {len(V_open)}')
        # if DEBUG_PATH_PLANNING: print(f'V_UNVISITED SHAPE: {V_unvisited.shape}')

        z_idx = V_open[0]
        z = nodes[z_idx]
        if DEBUG_PATH_PLANNING: print(f"Z START: {z.point}")

        for _ in range(max_iters):
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

                Y_costs = np.array([V_open[y_idx] + nodes[y_idx].GetCostToNode(x) for y_idx in Y_near_indices])
                y_min_cost = np.min(Y_costs)
                y_min_idx = Y_near_indices[np.argmin(Y_costs)]

                if DEBUG_PATH_PLANNING: print(f'Y_MIN_COST: {y_min_cost}')
                if DEBUG_PATH_PLANNING: print(f'Y_MIN_IDX: {y_min_idx}')

                if self.__checkCollision(nodes[y_min_idx], nodes[x_idx]):
                    nodes[y_min_idx].children.append(x)
                    nodes[x_idx].parent = y_min_idx
                    nodes[x_idx].cost = y_min_cost

                    if x_idx in V_open:
                        V_open.updateitem(x_idx, y_min_cost)
                    else:
                        V_open.additem(x_idx, y_min_cost)

                    V_unvisited[x_idx] = -1

            V_open.pop(z_idx)
            V_closed.append(z_idx)

            if len(V_open) == 0:
                if DEBUG_PATH_PLANNING: print(f'OUT OF NODES')
                break

            z_idx = V_open.top()
            z = nodes[z_idx]

            # if np.all(z.point == goal_point):
            if np.linalg.norm(z.point - goal_point) <= 1e-5:
            # if z_idx == len(nodes) - 1:
                if DEBUG_PATH_PLANNING: print(f'GOAL POINT FOUND')
                path_found = True
                break

        if path_found:
            if DEBUG_PATH_PLANNING: print(f'PATH FOUND')
        else:
            if DEBUG_PATH_PLANNING: print(f'NO PATH FOUND FAIL')

        return nodes if path_found else []
