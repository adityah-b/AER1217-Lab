import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from constants import *


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


    # def compute_cost(self, position):
    #     '''circular version'''
    #     diff = np.array([position[0] - self.center[0], position[1] - self.center[1]])
    #     dist = np.linalg.norm(diff)
    #     cost = np.exp(-dist / self.radius)
    #     return cost


    def compute_cost(self, position):
        '''ellipital version'''
        x_scale = self.minor_radius * np.sin(self.phi) + self.radius * np.cos(self.phi)
        y_scale = self.minor_radius * np.cos(self.phi) + self.radius * np.sin(self.phi)
        diff = np.array([(position[0] - self.center[0]) / x_scale, (position[1] - self.center[1]) / y_scale])
        dist = np.linalg.norm(diff)
        cost = np.exp(-dist)
        return cost


    # def render(self, figure, ax):
    #     '''circular version'''
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
    #     circle = plt.Circle(self.center, self.radius, color='b', fill=False)
    #     ax.add_patch(column_l)
    #     ax.add_patch(column_r)
    #     ax.add_patch(circle)


    def render(self, figure, ax):
        '''ellipital version'''
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
        ellipse = Ellipse(self.center, self.radius * 2, self.minor_radius * 2, angle=self.phi / np.pi * 180, color='b', fill=False)
        ax.add_patch(column_l)
        ax.add_patch(column_r)
        ax.add_patch(ellipse)


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