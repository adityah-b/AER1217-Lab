import os
import re
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# constants
mat_intrisnic = np.array([
    [698.86,   0.00, 306.91],
    [  0.00, 699.13, 150.34],
    [  0.00,   0.00,   1.00]
])

distortion_coefs = np.array([0.191887, -0.563680, -0.003676, -0.002037, 0.000000])

# the symbol is defined differently from the lab handout
# in the lab the subscript CB is for coordinate frame transformation from
# body frame to camera frame. In my convention, subscript CB is for transforming
# point in body frame to point in camera frame (or camera frame to body frame)
T_bc = np.array([
    [ 0.0, -1.0,  0.0,  0.0],
    [-1.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0]
])


def read_poses(filename):
    '''read poses from csv file'''

    poses = []
    lines = None

    with open(filename) as f:
        lines = f.readlines()

    for line in lines[1:]:
        num = np.fromstring(line, dtype=float, count=8, sep=',')

        q = np.zeros(4)
        q[0] = num[5] # qx
        q[1] = num[6] # qy
        q[2] = num[7] # qz
        q[3] = num[4] # qw
        R_ib = Rotation.from_quat(q).as_matrix()
        T_ib = np.eye(4)
        T_ib[0:3, 0:3] = R_ib
        T_ib[0, 3] = num[1] # x
        T_ib[1, 3] = num[2] # y
        T_ib[2, 3] = num[3] # z
        poses.append(T_ib)

    return poses


def read_image(filename):
    '''read and undistort input image'''

    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return cv2.undistort(src=img, cameraMatrix=mat_intrisnic, distCoeffs=distortion_coefs)


def detect_landmark(image):
    '''detect the landmark in the image and return the image coordinate of landmarks (N, 2)'''

    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=30,
              minRadius=10, maxRadius=20)

    if circles is None:
        return None

    center = np.zeros(2)
    center[0] = circles[0][0][0]
    center[1] = circles[0][0][1]

    return center


def estimate_landmark_position(obs, K, T_bc, T_ib):
    '''
    estimate 3d position of landmark given:
    obs  - 2D pixel coordinate
    K    - camera intrinsic matrix
    T_bc - transformation from body frame to camera frame
    T_ib - transformation from inertial frame to body frame
    '''

    # ray of the pixel
    r_c = np.array([(obs[0] - K[0, 2]) / K[0, 0], (obs[1] - K[1, 2]) / K[1, 1], 1, 1]).reshape(4, 1)

    # coordinate of focal point
    T_ic = np.matmul(T_ib, T_bc)
    p_focal = T_ic[0:3, 3].reshape(3, 1)

    # ray point in inertial frame
    p_ray = np.matmul(T_ic, r_c)
    vec_dir = p_ray[0:3] - p_focal
    vec_dir = vec_dir / np.linalg.norm(vec_dir)

    # find length of intersection with Z = 0
    rho = -p_focal[2] / vec_dir[2]

    # landmark is the point of intersection
    p_landmark = p_focal + rho * vec_dir

    return p_landmark


def cluster_landmark_observations(landmarks, observations, poses):
    '''cluster the landmark position and observation into groups'''

    group_id = 0
    group_mean = {}
    position_groups = {}
    observation_groups = {}
    pose_groups = {}

    for landmark, obs, pose in zip(landmarks, observations, poses):
        found = False

        for id in position_groups:
            mean, N = group_mean[id]
            dist = np.linalg.norm(landmark - mean)
            if dist < 0.2:
                position_groups[id].append(landmark)
                observation_groups[id].append(obs)
                pose_groups[id].append(pose)
                mean = (mean * N + landmark) / (N + 1)
                N += 1
                group_mean[id] = (mean, N)
                found = True
                break

        if found: continue

        # no neighbor found add as new landmark group
        group_id += 1
        group_mean[group_id] = (landmark, 1)
        position_groups[group_id] = []
        observation_groups[group_id] = []
        pose_groups[group_id] = []
        position_groups[group_id].append(landmark)
        observation_groups[group_id].append(obs)
        pose_groups[group_id].append(pose)

    return group_mean, position_groups, observation_groups, pose_groups


def solve_least_square(x0, Y, poses):
    '''
    solve the least square problem for each points
    x0    - initial point coordinate (px, py, pz)
    Y     - array of observation (N, 2)
    poses - array of poses of the observation (N, 4x4)
    '''

    def cost_function(x):
        J = 0
        for y, T_ib in zip(Y, poses):
            T_ic = np.matmul(T_ib, T_bc)
            T_ci = np.linalg.inv(T_ic)
            R = T_ci[0:3, 0:3]
            r = T_ci[0:3, 3]
            x_c = np.matmul(R, x) + r
            x_n = x_c / x_c[2]
            uv  = np.matmul(mat_intrisnic, x_n)
            err = y - uv[0:2]
            J += np.dot(err, err)
        return J

    res = least_squares(cost_function, x0.flatten())
    return res.x


def get_img_index(filename):
    matches = re.findall(r'\d+', filename)

    if matches is None:
        exit(f"[FAIL]: '{filename}' doesn't contain index")

    return int(matches[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", help="Path to images.", default="./images")
    parser.add_argument("--poses",  help="Pose csv file.",  default="lab3_pose.csv")
    args = parser.parse_args()

    # read all poses
    poses = read_poses(args.poses)

    # find landmark in images and estimate 3D coordinate
    observations = []
    obs_poses = []
    landmarks = []
    image_dir = os.path.abspath(args.images)
    sorted_list = os.listdir(image_dir)
    sorted_list.sort()
    for image_file in sorted_list:
        # print(image_file)
        index = get_img_index(image_file)

        # read image
        img = read_image(os.path.join(image_dir, image_file))

        # detect landmark
        obs = detect_landmark(img)
        if obs is None: continue

        observations.append(obs)
        obs_poses.append(poses[index])

        # estimate landmark coordinate
        position = estimate_landmark_position(obs, mat_intrisnic, T_bc, poses[index])
        landmarks.append(position)

    # plot scatter plot of landmark
    X = [landmark[0] for landmark in landmarks]
    Y = [landmark[1] for landmark in landmarks]
    plt.scatter(X, Y, c='b', alpha=0.3)
    plt.show()

    # clustering
    group_mean, position_groups, observation_groups, pose_groups = cluster_landmark_observations(landmarks, observations, obs_poses)

    # solve optimization
    X_opt = []
    for id in position_groups:
        mean, N = group_mean[id]
        if N < 10: continue

        print(f"[INFO]: landmark {id} - initial estimate = {mean.flatten()}")
        x_opt = solve_least_square(mean, observation_groups[id], pose_groups[id])
        X_opt.append(x_opt)
        print(f"[INFO]: landmark {id} -   final estimate = {x_opt}")

    # plot final result
    X_c = [x_opt[0] for x_opt in X_opt]
    Y_c = [x_opt[1] for x_opt in X_opt]
    plt.scatter(X, Y, c='b', alpha=0.3)
    plt.scatter(X_c, Y_c, c='r', alpha=0.8)
    plt.show()
    print()

if __name__ == "__main__":
    main()
