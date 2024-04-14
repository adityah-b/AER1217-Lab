'''
PathPlanner class implementation

Can perform RRT* and FMT* path planning
'''

import numpy as np

from pqdict import PQDict
from scipy.spatial import KDTree

from matplotlib import pyplot as plt

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
            grid_resolution: float = 0.005,
            robot_radius: float = 0.05) -> None:

        ###############################
        # Map boundaries
        ###############################
        self.X_BOUND = np.array(x_bound, dtype=np.float32)
        print(f'X_BOUND: {self.X_BOUND}')

        self.Y_BOUND = np.array(y_bound, dtype=np.float32)
        print(f'Y_BOUND: {self.Y_BOUND}')

        ###############################
        # Desired robot states
        ###############################
        self.START_STATE = np.array([
            initial_obs[0],
            initial_obs[2],
            initial_obs[4]
        ], dtype=np.float32)
        print(f'START STATE: {self.START_STATE}')
        
        self.GOAL_STATE = np.array([
            initial_info['x_reference'][0],
            initial_info['x_reference'][2],
            initial_info['x_reference'][4]
        ], dtype=np.float32)
        print(f'GOAL STATE: {self.GOAL_STATE}')

        ###############################
        # Gate properties
        ###############################
        # self.GATE_DIMS = initial_info['gate_dimensions']
        # print(f'GATE DIMS: {self.GATE_DIMS}')
        # self.GATE_EDGE_LEN = (initial_info['gate_dimensions']['tall']['edge'] - 0.4) / 2.0
        self.GATE_INNER_EDGE_LEN = 0.4
        self.GATE_EDGE_LEN = 0.05
        self.GATE_EDGE_WID = 0.05

        # Get x, y, z and yaw of gates (rad)
        self.GATE_LOCATIONS = np.array(initial_info['nominal_gates_pos_and_type'], dtype=np.float32)
        self.GATE_LOCATIONS = self.GATE_LOCATIONS[:, [0, 1, 2, 5]]
        print(f'GATE_LOCATIONS: {self.GATE_LOCATIONS}')

        ###############################
        # Obstacle properties
        ###############################
        # self.OBSTACLE_DIMS = initial_info['obstacle_dimensions']
        # print(f'OBSTACLE_DIMS: {self.OBSTACLE_DIMS}')
        self.OBSTACLE_RADIUS = 0.06
        print(f'OBSTACLE_RADIUS: {self.OBSTACLE_RADIUS}')

        self.OBSTACLE_LOCATIONS = np.array([
            obs_loc[:3] for obs_loc in initial_info['nominal_obstacles_pos']
        ], dtype=np.float32)
        print(f'OBSTACLE_LOCATIONS: {self.OBSTACLE_LOCATIONS}')

        ###############################
        # Controller properties
        ###############################
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        print(f'CTRL_TIMESTEP: {self.CTRL_TIMESTEP}')

        self.CTRL_FREQ = initial_info["ctrl_freq"]
        print(f'CTRL_FREQ: {self.CTRL_FREQ}')

        ####################
        # Robot properties
        ####################
        self.ROBOT_RADIUS = robot_radius

        ##################
        # Occupancy grid
        ##################
        self.GRID_RESOLUTION = grid_resolution

        # Construct sparse occupancy grid points
        grid_obstacles, grid_gates = self.constructOccupancyGrid()
        self.OCCUPANCY_GRID_POINTS = np.vstack([grid_obstacles, grid_gates])
        # self.OCCUPANCY_GRID_GATE_POINTS = grid_gates

        # Construct occupancy grid KD-Tree
        self.OCCUPANCY_KD_TREE = KDTree(self.OCCUPANCY_GRID_POINTS)

        #################################
        # Planning algorithm properties
        #################################
        # np.random.seed(42)
        # self.fmt(
        #     start_point=self.START_STATE[:2],
        #     goal_point=self.GOAL_STATE[:2],
        #     max_iters=1)

        # start_points = [
        #     self.START_STATE[:2], 
        #     self.GATE_LOCATIONS[0][:2], 
        #     self.GATE_LOCATIONS[1][:2],
        #     self.GATE_LOCATIONS[2][:2],
        #     self.GATE_LOCATIONS[3][:2]
        # ]
        # goal_points = [
        #     self.GATE_LOCATIONS[0][:2], 
        #     self.GATE_LOCATIONS[1][:2],
        #     self.GATE_LOCATIONS[2][:2],
        #     self.GATE_LOCATIONS[3][:2],
        #     self.GOAL_STATE[:2]
        # ]

        # print(f'PATH: {path}')

    # def runFMT(self):
    #     start_points, goal_points = self.__setupInitialTraj()
    #     path = []
    #     for i in range(len(start_points)):
    #         nodes = self.fmt(start_point=start_points[i], goal_point=goal_points[i])
    #         sub_path = [nodes[-1].point]

    #         path_end_idx = nodes[-1].parent
    #         while path_end_idx > -1:
    #             sub_path.append(nodes[path_end_idx].point)
    #             path_end_idx = nodes[path_end_idx].parent
    #         sub_path.reverse()
    #         path += sub_path

    #     return path
    
    def runFMT(self):
        start_points, goal_points = self.__setupInitialTraj()
        path = []
        for i in range(len(start_points)):
            nodes = self.fmt(start_point=start_points[i], goal_point=goal_points[i])
            sub_path = [nodes[-1].point * self.GRID_RESOLUTION]

            path_end_idx = nodes[-1].parent
            while path_end_idx > -1:
                sub_path.append(nodes[path_end_idx].point * self.GRID_RESOLUTION)
                path_end_idx = nodes[path_end_idx].parent
            sub_path.reverse()
            path += sub_path

        return path
    
    # def plotPath(self, path):
    #     fig, ax = plt.subplots()
    #     ax.set_xlim((-3.5, 3.5))
    #     ax.set_ylim((-3.5, 3.5))

    #     x = list()
    #     y = list()

    #     for point in path:
    #         x.append(point[0] * self.GRID_RESOLUTION)
    #         y.append(point[1] * self.GRID_RESOLUTION)
    #     # ax.plot(x, y, color='r', linewidth=1)
    #     ax.scatter(x, y, color='b')

    #     ax.scatter(path[0][0] * self.GRID_RESOLUTION, path[0][1] * self.GRID_RESOLUTION, color='g')
    #     ax.scatter(path[-1][0] * self.GRID_RESOLUTION, path[-1][1] * self.GRID_RESOLUTION, color='g')
    #     # ax.scatter(self.GATE_LOCATIONS[:, 0], self.GATE_LOCATIONS[:, 1], color='b')

    #     print(f'OCCUPANCY SHAPE: {self.OCCUPANCY_GRID_POINTS.shape}')
    #     ax.scatter(self.OCCUPANCY_GRID_POINTS[:, 0] * self.GRID_RESOLUTION, self.OCCUPANCY_GRID_POINTS[:, 1] * self.GRID_RESOLUTION, color='r')
    #     plt.show()
    def plotPath(self, path):
        fig, ax = plt.subplots()
        ax.set_xlim((-3.5, 3.5))
        ax.set_ylim((-3.5, 3.5))

        x = list()
        y = list()

        for point in path:
            x.append(point[0])
            y.append(point[1])
        ax.plot(x, y, color='r', linewidth=1)
        ax.scatter(x, y, color='b')

        ax.scatter(path[0][0], path[0][1], color='g')
        ax.scatter(path[-1][0], path[-1][1], color='g')
        # ax.scatter(self.GATE_LOCATIONS[:, 0], self.GATE_LOCATIONS[:, 1], color='b')

        print(f'OCCUPANCY SHAPE: {self.OCCUPANCY_GRID_POINTS.shape}')
        ax.scatter(self.OCCUPANCY_GRID_POINTS[:, 0] * self.GRID_RESOLUTION, self.OCCUPANCY_GRID_POINTS[:, 1] * self.GRID_RESOLUTION, color='r')
        plt.show()

    def __addGoalStates(self, waypoint_tolerance: float = 0.2):
        gate_locations = self.GATE_LOCATIONS

        gate_goal_states = []
        for gate_location in gate_locations:
            gate_goal_1 = np.array([gate_location[0], gate_location[1] + waypoint_tolerance])
            gate_goal_2 = np.array([gate_location[0], gate_location[1] - waypoint_tolerance])

            # print(f'GATE GOAL 1: {gate_goal_1}')
            # print(f'GATE GOAL 2: {gate_goal_2}')

            gate_goals = np.vstack([
                [self.__rotatePoint(gate_goal_1, gate_location[:2], gate_location[3])],
                [self.__rotatePoint(gate_goal_2, gate_location[:2], gate_location[3])]
            ])
            # print(f'GATE GOALS SHAPE: {gate_goals.shape}')
            gate_goal_states.append(gate_goals)

        return gate_goal_states
    
    def __setupInitialTraj(self):
        gate_goal_states = self.__addGoalStates()
        # print(f'GATE GOAL STATES: {gate_goal_states}')
        
        start_point = self.START_STATE[:2]
        goal_point = self.GOAL_STATE[:2]

        start_states = []
        goal_states = []

        cur_point = start_point
        idx = 0
        for i in range(len(gate_goal_states)):
            print(f'CUR_POINT: {cur_point}')
            gate = self.GATE_LOCATIONS[i, :2].copy()
            gate_goals = gate_goal_states[i]

            print(f'GATE_GOALS: {gate_goals}')

            dists = np.linalg.norm(gate_goals - cur_point, axis=1)
            print(f'DISTS: {dists}')
            closest_point = gate_goals[np.argmin(dists)]
            print(f'CLOSEST_POINT: {closest_point}')
            
            start_states.append(cur_point)
            goal_states.append(closest_point)

            start_states.append(closest_point)
            goal_states.append(gate)

            cur_point = gate_goals[np.argmax(dists)]

            start_states.append(gate)
            goal_states.append(cur_point)

        print(f'START_STATES SHAPE: {len(start_states)}')
        print(f'GOAL_STATES SHAPE: {len(goal_states)}')
        start_states.append(cur_point)
        goal_states.append(goal_point)

        return start_states, goal_states
    
    # TODO: Add obstacles for gate edges
    def constructOccupancyGrid(self):
        # Get 2D x-y coordinates of obstacle positions
        obstacle_points = self.OBSTACLE_LOCATIONS[:, :2]
        print(f'2D OBS POINTS: {obstacle_points}')

        # Get discretized grid points of filled obstacle regions
        grid_obstacles = self.__fillGridObstacles(obstacle_points)
        
        grid_gates = self.__fillGridGates(self.GATE_LOCATIONS)

        return grid_obstacles, grid_gates
    
    #########################################################################################################
    # Function: __fillGridObstacles(self, obstacle_points)
    # 
    # Fill area bounded by circular obstacle with discretized points
    # 
    # Inputs: obstacle_points
    #   obstacle_points - N x 2 set of 2D Cartesian centroids of circular obstacles
    #
    # Outputs: discrete_obstacles_grid
    #   discrete_obstacles_grid - N * M x 2 set of i, j indices representing locations of points in grid coordinates
    #########################################################################################################
    def __fillGridObstacles(self, obstacle_points):
        # Fill in obstacle regions
        discrete_obstacles_list = []
        for obstacle_point in obstacle_points:
            discrete_obstacles_list.append(self.__fillCircularObstacle(obstacle_point))

        discrete_obstacles = np.vstack(discrete_obstacles_list, dtype=np.float32)

        # Transform obstacle points to corresponding grid coordinates
        discrete_obstacles_grid = self.__pointsToGrid(discrete_obstacles)

        return discrete_obstacles_grid
    
    #########################################################################################################
    # Function: __fillGridGates(self, gate_poses)
    # 
    # Fill area bounded by rectangular obstacle with discretized points
    # 
    # Inputs: gate_poses
    #   gate_poses - N x 4 set of 3D Cartesian coordinates and yaw of rectangular gates
    #
    # Outputs: discrete_gates_grid
    #   discrete_gates_grid - N * M x 2 set of i, j indices representing locations of points in grid coordinates
    #########################################################################################################
    def __fillGridGates(self, gate_poses):
        # Fill in gate edge regions
        discrete_gates_list = []
        for gate_pose in gate_poses:
            discrete_gates_list.append(self.__fillRectangularObstacle(gate_pose))

        discrete_gates = np.vstack(discrete_gates_list, dtype=np.float32)

        # Transform obstacle points to corresponding grid coordinates
        discrete_gates_grid = self.__pointsToGrid(discrete_gates)

        return discrete_gates_grid

    #########################################################################################################
    # Function: __pointsToGrid(self, points)
    # 
    # Convert set of [x, y] coordinates to their corresponding locations in the occupancy grid
    # 
    # Inputs: points
    #   points - N x 2 set of 2D Cartesian points
    #
    # Outputs: grid_points
    #   grid_points - N x 2 set of i, j indices representing locations of points in grid
    #########################################################################################################
    def __pointsToGrid(self, points, grid_origin=[-3.5, -3.5]):
        # Transform points from inertial frame to grid frame
        # points -= np.array(grid_origin)
    
        # Subdivide transformed grid points based on cell resolution with grid origin corresponding to [0, 0]
        grid_points = np.floor(points / self.GRID_RESOLUTION)

        return grid_points
    
    #########################################################################################################
    # Function: __fillCircularObstacle(self, obstacle_center)
    # 
    # Fills region enclosed by circular obstacle with points based on grid resolution
    # 
    # Inputs: obstacle_center
    #   obstacle_center - 1 x 2 array representing centroid of circular obstacle
    #
    # Outputs: circle_points
    #   circle_points - N x 2 set of 2D Cartesian points representing all points within circle
    #       N is determined by the diameter of the circle and the grid resolution
    #########################################################################################################
    def __fillCircularObstacle(self, obstacle_center):
        x, y = np.meshgrid(
            np.arange(obstacle_center[0] - self.OBSTACLE_RADIUS, obstacle_center[0] + self.OBSTACLE_RADIUS + self.GRID_RESOLUTION, self.GRID_RESOLUTION),
            np.arange(obstacle_center[1] - self.OBSTACLE_RADIUS, obstacle_center[1] + self.OBSTACLE_RADIUS + self.GRID_RESOLUTION, self.GRID_RESOLUTION),
        )

        # Filter points belonging to circle
        circle_eq = (x - obstacle_center[0]) ** 2 + (y - obstacle_center[1]) ** 2 <= self.OBSTACLE_RADIUS ** 2
        x_circle = x[circle_eq]
        y_circle = y[circle_eq]
        circle_points = np.transpose(np.vstack((x_circle, y_circle)))

        return circle_points
    
    def __rotatePoint(self, point, center, angle):
        s, c = np.sin(angle), np.cos(angle)
        x_rot = c * (point[0] - center[0]) - s * (point[1] - center[1]) + center[0]
        y_rot = s * (point[0] - center[0]) + c * (point[1] - center[1]) + center[1]
        return np.array([x_rot, y_rot])
    
    #########################################################################################################
    # Function: __fillRectangularObstacle(self, obstacle_center)
    # 
    # Fills region enclosed by circular obstacle with points based on grid resolution
    # 
    # Inputs: obstacle_center
    #   obstacle_center - 1 x 2 array representing centroid of circular obstacle
    #
    # Outputs: circle_points
    #   circle_points - N x 2 set of 2D Cartesian points representing all points within circle
    #       N is determined by the diameter of the circle and the grid resolution
    #########################################################################################################
    def __fillRectangularObstacle(self, gate_pose):
        gate_center = gate_pose[:2]
        gate_yaw = gate_pose[3]
        print(f'GATE_YAW: {gate_yaw}')

        gate_len = 2.0 * self.GATE_EDGE_LEN + self.GATE_INNER_EDGE_LEN
        gate_width = self.GATE_EDGE_WID
        
        gate_corners = np.array([
            [gate_center[0] - gate_len / 2, gate_center[1] - gate_width / 2],
            [gate_center[0] - gate_len / 2, gate_center[1] + gate_width / 2],
            [gate_center[0] + gate_len / 2, gate_center[1] - gate_width / 2],
            [gate_center[0] + gate_len / 2, gate_center[1] + gate_width / 2],
        ], dtype=np.float32)

        print(f'GATE CORNERS: {gate_corners}')

        rotated_corners = np.array([
            self.__rotatePoint(gate_corner, gate_center, gate_yaw) for gate_corner in gate_corners])
        print(f'ROTATED GATE CORNERS: {rotated_corners}')

        x_min = np.min(rotated_corners[:, 0])
        x_max = np.max(rotated_corners[:, 0])

        y_min = np.min(rotated_corners[:, 1])
        y_max = np.max(rotated_corners[:, 1])

        x_edge_low, y_edge_low = np.meshgrid(
            np.arange(x_min, x_min + self.GATE_EDGE_LEN + self.GRID_RESOLUTION, self.GRID_RESOLUTION),
            np.arange(y_min, y_min + self.GATE_EDGE_WID + self.GRID_RESOLUTION, self.GRID_RESOLUTION),
        )

        x_edge_hi, y_edge_hi = np.meshgrid(
            np.arange(x_max - self.GATE_EDGE_LEN, x_max + self.GRID_RESOLUTION, self.GRID_RESOLUTION),
            np.arange(y_max - self.GATE_EDGE_WID, y_max + self.GRID_RESOLUTION, self.GRID_RESOLUTION),
        )

        edge_points_low = np.transpose(np.vstack([x_edge_low.ravel(), y_edge_low.ravel()]))
        edge_points_hi = np.transpose(np.vstack([x_edge_hi.ravel(), y_edge_hi.ravel()]))

        # from matplotlib.path import Path
        # path = Path(rotated_corners)
        
        # grid_points_low = edge_points_low[path.contains_points(edge_points_low)]
        # grid_points_hi = edge_points_hi[path.contains_points(edge_points_hi)]
        
        # grid_points = np.vstack([grid_points_low, grid_points_hi])
        grid_points = np.vstack([edge_points_low, edge_points_hi])

        return grid_points

    #########################################################################################################
    # Function: __samplePoint(self, sample_goal: bool = False)
    # 
    # Samples a point within the map boundaries and returns a node in free space
    # 
    # Inputs: sample_goal (Optional)
    #   sample_goal - Bool to control whether to sample near the goal state
    #
    # Outputs: sample_point_grid
    #   sample_point_grid - 1 x 2 array representing sampled point in free space
    #########################################################################################################
    def __samplePoint(self, sample_goal: bool = False):
        sample_point_grid = None
        sample_in_free_space = False
        while not sample_in_free_space:
            x = np.random.uniform(self.X_BOUND[0], self.X_BOUND[1])
            y = np.random.uniform(self.Y_BOUND[0], self.Y_BOUND[1])

            sample_point = np.array([x, y], dtype=np.float32)

            # Transform sampled point to grid coordinates
            sample_point_grid = self.__pointsToGrid(sample_point)

            # Check if point is in obstacle region
            if not self.__isPointInObstacle(sample_point_grid):
                sample_in_free_space = True

        return sample_point_grid

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
    # Function: __isPointInObstacle(self, sample_point_grid)
    # 
    # Checks whether a sampled point is inside an obstacle region
    # 
    # Inputs: sample_point_grid
    #   sample_point_grid - 1 x 2 array representing the sampled point in grid coordinates
    #
    # Outputs: bool
    #   Returns true if point in obstacle region, else false
    #########################################################################################################
    def __isPointInObstacle(self, sample_point_grid):
        distance, _ = self.OCCUPANCY_KD_TREE.query(sample_point_grid)
        return distance == 0
    
    # def __findNearestNode(self, node):

    # def rrtStar(self, max_iters: int = 5000):
    #     for i in range(max_iters):
    #         rand_node = self.__samplePoint()

    #         nearest_node = self.findNearestNode(rand_node)
    #         new_node = self.connectNodes(nearest_node, rand_node)

    #         if self.checkCollision(nearest_node, new_node):
    #             continue

    #         neighbours = self.findNeighbours(new_node)
    #         self.rewireParent(new_node)
    #         self.rewire(new_node)

            
    #     pass

    def __checkCollision(self, nodeA, nodeB):
        print(f'NODE A: {nodeA.point}')
        print(f'NODE B: {nodeB.point}')
        points = []
        dx = abs(nodeB.point[0] - nodeA.point[0])
        dy = abs(nodeB.point[1] - nodeA.point[1])
        x, y = nodeA.point[0], nodeA.point[1]
        sx = -1 if nodeA.point[0] > nodeB.point[0] else 1
        sy = -1 if nodeA.point[1] > nodeB.point[1] else 1
        if dx > dy:
            err = dx / 2.0
            while x != nodeB.point[0]:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != nodeB.point[1]:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x, y))

        nn_dists, _ = self.OCCUPANCY_KD_TREE.query(points)
        no_collision = bool(np.min(nn_dists) > (self.ROBOT_RADIUS / self.GRID_RESOLUTION))

        if no_collision:
            print(f'NO COLLISION')
        else:
            print(f'COLLISION')

        return no_collision

    def fmt(self,
            num_points: int = 10000,
            start_point = None,
            goal_point = None,
            max_iters: int = 5000,
            rn: float = 100.0):
        
        start_point_grid = self.__pointsToGrid(start_point)
        goal_point_grid = self.__pointsToGrid(goal_point)
        print(f'START GRID: {start_point_grid}')
        print(f'GOAL GRID: {goal_point_grid}')

        sampled_points = self.__sampleNPoints(
            start_point=start_point_grid,
            goal_point=goal_point_grid,
            num_points=num_points)
        nodes = np.array([Node(sampled_point) for sampled_point in sampled_points])
        print(f'NODE SHAPE: {nodes.shape}')

        nn_kd_tree = KDTree(sampled_points)

        path_found = False

        V_open = PQDict({
            0 : nodes[0].cost
        })
        V_closed = []
        V_unvisited = list(range(len(nodes)))
        V_unvisited.remove(0)
        print(f'V_OPEN SHAPE: {len(V_open)}')
        # print(f'V_UNVISITED SHAPE: {V_unvisited.shape}')

        z_idx = V_open[0]
        z = nodes[z_idx]
        print(f"Z START: {z.point}")

        # while not np.all(z.point == goal_point_grid):
        for _ in range(max_iters):
            # Find all nodes within radius rn from z
            N_z_indices = nn_kd_tree.query_ball_point(z.point, rn)
            print(f'Z NEIGHBOURS: {N_z_indices}')

            # Filter nodes only belonging to unvisited set
            X_near_indices = np.intersect1d(N_z_indices, V_unvisited)
            X_near = nodes[X_near_indices]
            # print(f'X_NEAR: {[node.point for node in X_near]}')
            print(f'X_NEAR SHAPE: {X_near.shape}')

            for x, x_idx in zip(X_near, X_near_indices):
                # Find all nodes within radius rn from x
                N_x_indices = nn_kd_tree.query_ball_point(x.point, rn)
                # print(f'X NEIGHBOURS: {N_x_indices}')

                # Filter nodes only belonging to open set
                V_open_indices = np.array(list(V_open.keys()))
                Y_near_indices = np.intersect1d(N_x_indices, V_open_indices)
                Y_near = nodes[Y_near_indices]
                print(f'Y_NEAR SHAPE: {Y_near.shape}')

                # for y in Y_near:
                #     print(f'Y COST TO X: {y.GetCostToNode(x)}')

                Y_costs = np.array([V_open[y_idx] + nodes[y_idx].GetCostToNode(x) for y_idx in Y_near_indices])
                y_min_cost = np.min(Y_costs)
                y_min_idx = Y_near_indices[np.argmin(Y_costs)]

                print(f'Y_MIN_COST: {y_min_cost}')
                print(f'Y_MIN_IDX: {y_min_idx}')

                if self.__checkCollision(nodes[y_min_idx], nodes[x_idx]):
                    nodes[y_min_idx].children.append(x)
                    nodes[x_idx].parent = y_min_idx
                    nodes[x_idx].cost = y_min_cost

                    if x_idx in V_open:
                        V_open.updateitem(x_idx, y_min_cost)
                    else:
                        V_open.additem(x_idx, y_min_cost)

                    # V_unvisited[x_idx] = -1
                    V_unvisited.remove(x_idx)
                
            V_open.pop(z_idx)
            V_closed.append(z_idx)

            if len(V_open) == 0:
                print(f'OUT OF NODES')
                break
            
            z_idx = V_open.top()
            z = nodes[z_idx]

            if np.all(z.point == goal_point_grid):
            # if z_idx == len(nodes) - 1:
                print(f'GOAL POINT FOUND')
                path_found = True
                break

        if path_found:
            print(f'PATH FOUND')
        else:
            print(f'NO PATH FOUND FAIL')

        return nodes if path_found else []
