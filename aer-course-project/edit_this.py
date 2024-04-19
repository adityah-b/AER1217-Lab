"""Write your proposed algorithm.
[NOTE]: The idea for the final project is to plan the trajectory based on a sequence of gates
while considering the uncertainty of the obstacles. The students should show that the proposed
algorithm is able to safely navigate a quadrotor to complete the task in both simulation and
real-world experiments.

Then run:

    $ python3 final_project.py --overrides ./getting_started.yaml

Tips:
    Search for strings `INSTRUCTIONS` and `REPLACE THIS (START)` in this file.

    Change the code between the 5 blocks starting with
        #########################
        # REPLACE THIS (START) ##
        #########################
    and ending with
        #########################
        # REPLACE THIS (END) ####
        #########################
    with your own code.

    They are in methods:
        1) planning
        2) cmdFirmware

"""
import numpy as np

from collections import deque

try:
    from project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory_in_gui
except ImportError:
    # PyTest import.
    from .project_utils import Command, PIDController, timing_step, timing_ep, plot_trajectory, draw_trajectory_in_gui

#########################
# REPLACE THIS (START) ##
#########################

# Optionally, create and import modules you wrote.
# Please refrain from importing large or unstable 3rd party packages.
try:
    import example_custom_utils as ecu
except ImportError:
    # PyTest import.
    from . import example_custom_utils as ecu

from constants import *
import path_planner as pp
import path_planner_3D as pp3


#########################
# REPLACE THIS (END) ####
#########################

class Controller():
    """Template controller class.

    """

    def __init__(self,
                 initial_obs,
                 initial_info,
                 use_firmware: bool = False,
                 buffer_size: int = 100,
                 verbose: bool = False
                 ):
        """Initialization of the controller.

        INSTRUCTIONS:
            The controller's constructor has access the initial state `initial_obs` and the a priori infromation
            contained in dictionary `initial_info`. Use this method to initialize constants, counters, pre-plan
            trajectories, etc.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary with keys
                'symbolic_model', 'nominal_physical_parameters', 'nominal_gates_pos_and_type', etc.
            use_firmware (bool, optional): Choice between the on-board controll in `pycffirmware`
                or simplified software-only alternative.
            buffer_size (int, optional): Size of the data buffers used in method `learn()`.
            verbose (bool, optional): Turn on and off additional printouts and plots.

        """
        # Save environment and control parameters.
        self.CTRL_TIMESTEP = initial_info["ctrl_timestep"]
        self.CTRL_FREQ = initial_info["ctrl_freq"]
        self.initial_obs = initial_obs
        self.VERBOSE = verbose
        self.BUFFER_SIZE = buffer_size

        # Store a priori scenario information.
        # plan the trajectory based on the information of the (1) gates and (2) obstacles.
        self.NOMINAL_GATES = initial_info["nominal_gates_pos_and_type"]
        self.NOMINAL_OBSTACLES = initial_info["nominal_obstacles_pos"]

        # set start and end states
        self.start_state = np.array([initial_obs[0], initial_obs[2], initial_obs[4]])
        self.end_state   = np.array([initial_info['x_reference'][0], initial_info['x_reference'][2], LANDING_HEIGHT])

        # Check for pycffirmware.
        if use_firmware:
            self.ctrl = None
        else:
            # Initialize a simple PID Controller for debugging and test.
            # Do NOT use for the IROS 2022 competition.
            self.ctrl = PIDController()
            # Save additonal environment parameters.
            self.KF = initial_info["quadrotor_kf"]

        # Reset counters and buffers.
        self.reset()
        self.interEpisodeReset()

        # perform trajectory planning
        t_scaled = self.planning(use_firmware, initial_info)

        self.curr_waypoint_idx = 0
        self.curr_waypoint = None
        self.prev_waypoint = None

        ## visualization
        # # Plot trajectory in each dimension and 3D.
        # plot_trajectory(t_scaled, self.waypoints)

        # Draw the trajectory on PyBullet's GUI.
        draw_trajectory_in_gui(initial_info, self.waypoints)


    def planning(self, use_firmware, initial_info):
        """Trajectory planning algorithm"""

        #########################
        # REPLACE THIS (START) ##
        #########################

        # generate trajectory
        self.use_3d_path_planning = True
        if self.use_3d_path_planning:
            planner = pp3.PathPlanner3D(self.initial_obs, initial_info)
            waypoints = planner.initialize_trajectory()
        else:
            planner = pp.PathPlanner(self.initial_obs, initial_info)
            planner.constructOccupancyGrid()
            path = planner.runFMT()
            planner.plotPath(path)

            # initial waypoint
            if use_firmware:
                waypoints = [(self.initial_obs[0], self.initial_obs[2], initial_info["gate_dimensions"]["tall"]["height"])]  # Height is hardcoded scenario knowledge.
            else:
                waypoints = [(self.initial_obs[0], self.initial_obs[2], self.initial_obs[4])]

            for point in path:
                waypoints.append(np.array([point[0], point[1], 1.0]))
            print(f'WAYPOINTS: {waypoints}')

        self.waypoints = waypoints
        self.num_waypoints = len(waypoints)
        self.flight_state = FLIGHT_STATE_READY

        #########################
        # REPLACE THIS (END) ####
        #########################

        return None


    def cmdFirmware(self,
                    time,
                    obs,
                    reward=None,
                    done=None,
                    info=None
                    ):
        """Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        INSTRUCTIONS:
            Re-implement this method to return the target position, velocity, acceleration, attitude, and attitude rates to be sent
            from Crazyswarm to the Crazyflie using, e.g., a `cmdFullState` call.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)

        """
        if self.ctrl is not None:
            raise RuntimeError("[ERROR] Using method 'cmdFirmware' but Controller was created with 'use_firmware' = False.")

        # [INSTRUCTIONS]
        # self.CTRL_FREQ is 30 (set in the getting_started.yaml file)
        # control input iteration indicates the number of control inputs sent to the quadrotor
        iteration = int(time*self.CTRL_FREQ)

        #########################
        # REPLACE THIS (START) ##
        #########################

        # initialize
        command_type = Command(0) # NONE
        args = []

        # get current position
        curr_pos = np.array([obs[0], obs[2], obs[4]])

        # state machine
        if self.flight_state == FLIGHT_STATE_TRACK:
            err = np.linalg.norm(curr_pos - self.curr_waypoint)
            if err < WAYPOINT_TRACKING_THRES:
                self.curr_waypoint_idx += 1

                if self.curr_waypoint_idx < self.num_waypoints:
                    self.prev_waypoint = self.curr_waypoint
                    self.curr_waypoint = self.waypoints[self.curr_waypoint_idx]
                else:
                    self.flight_state = FLIGHT_STATE_LANDING
                    command_type = Command(3) # Land
                    height = 0.0
                    duration = 1.0
                    args = [height, duration]

            command_type = Command(1) # track
            err_curr = self.curr_waypoint - curr_pos
            err_prev = curr_pos - self.prev_waypoint
            err_curr_norm = np.linalg.norm(err_curr)
            err_prev_norm = np.linalg.norm(err_prev)
            err_dir = err_curr / err_curr_norm
            vel_norm =  err_prev_norm * err_curr_norm * WAYPOINT_TRACKING_SPEED_MAX
            vel_norm = min(vel_norm, WAYPOINT_TRACKING_SPEED_MAX)
            vel_norm = max(vel_norm, WAYPOINT_TRACKING_SPEED_MIN)
            if DEBUG_PATH_PLANNING: print(vel_norm)
            velocity = err_dir * vel_norm
            position = curr_pos + velocity * WAYPOINT_TRACKING_STEP_SIZE
            # [position, velocity, acceleration, yaw, rpy_rates]
            args = [position, velocity, np.zeros(3), 0, np.zeros(3)]

        elif self.flight_state == FLIGHT_STATE_READY:
            # transition state
            self.flight_state = FLIGHT_STATE_TRACK
            self.prev_waypoint = self.start_state
            self.curr_waypoint = self.waypoints[0]

        elif self.flight_state == FLIGHT_STATE_LANDING:
            if np.linalg.norm(curr_pos - self.end_state) < WAYPOINT_TRACKING_THRES:
                self.flight_state = FLIGHT_STATE_OFF
                command_type = Command(4)  # STOP
                args = []

        return command_type, args



        if iteration == 0:
            height = 1
            duration = 2

            command_type = Command(2)  # Take-off.
            args = [height, duration]

        # [INSTRUCTIONS] Example code for using cmdFullState interface
        elif iteration >= 3*self.CTRL_FREQ and iteration < 20*self.CTRL_FREQ:
            step = min(iteration-3*self.CTRL_FREQ, len(self.ref_x) -1)
            target_pos = np.array([self.ref_x[step], self.ref_y[step], self.ref_z[step]])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

        elif iteration == 20*self.CTRL_FREQ:
            command_type = Command(6)  # Notify setpoint stop.
            args = []

       # [INSTRUCTIONS] Example code for using goTo interface
        elif iteration == 20*self.CTRL_FREQ+1:
            x = self.ref_x[-1]
            y = self.ref_y[-1]
            z = 1.5
            yaw = 0.
            duration = 2.5

            command_type = Command(5)  # goTo.
            args = [[x, y, z], yaw, duration, False]

        elif iteration == 23*self.CTRL_FREQ:
            x = self.initial_obs[0]
            y = self.initial_obs[2]
            z = 1.5
            yaw = 0.
            duration = 6

            command_type = Command(5)  # goTo.
            args = [[x, y, z], yaw, duration, False]

        elif iteration == 30*self.CTRL_FREQ:
            height = 0.
            duration = 3

            command_type = Command(3)  # Land.
            args = [height, duration]

        elif iteration == 33*self.CTRL_FREQ-1:
            command_type = Command(4)  # STOP command to be sent once the trajectory is completed.
            args = []

        else:
            command_type = Command(0)  # None.
            args = []

        #########################
        # REPLACE THIS (END) ####
        #########################

        return command_type, args

    def cmdSimOnly(self,
                   time,
                   obs,
                   reward=None,
                   done=None,
                   info=None
                   ):
        """PID per-propeller thrusts with a simplified, software-only PID quadrotor controller.

        INSTRUCTIONS:
            You do NOT need to re-implement this method for the project.
            Only re-implement this method when `use_firmware` == False to return the target position and velocity.

        Args:
            time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's state [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            List: target position (len == 3).
            List: target velocity (len == 3).

        """
        if self.ctrl is None:
            raise RuntimeError("[ERROR] Attempting to use method 'cmdSimOnly' but Controller was created with 'use_firmware' = True.")

        iteration = int(time*self.CTRL_FREQ)

        #########################
        if iteration < len(self.ref_x):
            target_p = np.array([self.ref_x[iteration], self.ref_y[iteration], self.ref_z[iteration]])
        else:
            target_p = np.array([self.ref_x[-1], self.ref_y[-1], self.ref_z[-1]])
        target_v = np.zeros(3)
        #########################

        return target_p, target_v

    def reset(self):
        """Initialize/reset data buffers and counters.

        Called once in __init__().

        """
        # Data buffers.
        self.action_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.obs_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.reward_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.done_buffer = deque([], maxlen=self.BUFFER_SIZE)
        self.info_buffer = deque([], maxlen=self.BUFFER_SIZE)

        # Counters.
        self.interstep_counter = 0
        self.interepisode_counter = 0

    # NOTE: this function is not used in the course project.
    def interEpisodeReset(self):
        """Initialize/reset learning timing variables.

        Called between episodes in `getting_started.py`.

        """
        # Timing stats variables.
        self.interstep_learning_time = 0
        self.interstep_learning_occurrences = 0
        self.interepisode_learning_time = 0
