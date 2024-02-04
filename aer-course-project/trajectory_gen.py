import numpy as np

def genEquidistantPoints(point1, point2, num_waypoints=10):
    """
    Generates equidistant points between two points
    """
    return list(
        zip(
            np.linspace(point1[0], point2[0], num_waypoints),
            np.linspace(point1[1], point2[1], num_waypoints),
            np.linspace(point2[2], point2[2], num_waypoints)
        )
    )

def genCirclularTrajectory(center, radius, num_waypoints=10):
    """
    Generates circular trajectory given center and radius
    """
    x_center, y_center, z_center = center
    thetas = np.linspace(0.0, 2.0 * np.pi, num_waypoints)

    waypoints = []
    for theta in thetas:
        x = x_center + radius * np.cos(theta)
        y = y_center + radius * np.sin(theta)
        z = z_center

        waypoints.append((x, y, z))

    return waypoints