'''define all constant here'''


# debug flags
DEBUG_PATH_PLANNING = False
DEBUG_WAYPOINT_TRACKING = True

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
GATE_CENTER_OFFSET = 0.3

GATE_RIM_WIDTH = 0.1
GATE_OPENING_SIZE = 0.4

# take-off and landing
TAKE_OFF_HEIGHT = 0.2
LANDING_HEIGHT  = 0.1

# tracking
WAYPOINT_TRACKING_THRES = 0.2
WAYPOINT_TRACKING_STEP_SIZE = 0.1

SMOOTH_TRACKING_SPEED_MAX = 0.8
SMOOTH_TRACKING_SPEED_MIN = 0.5

WAYPOINT_TRACKING_SPEED_MAX = 1.5
WAYPOINT_TRACKING_SPEED_MIN = 0.5

# state machine
FLIGHT_STATE_READY = 0
FLIGHT_STATE_TRACK = 1
FLIGHT_STATE_LANDING = 4
FLIGHT_STATE_OFF = 8

# path planning (potential field)
USE_SMOOTH_TRAJECTORY = True
USE_BIDIRECTIONAL_EXIT = True
USE_LINEAR_COLLISION_LOSS = True

SUB_SAMPLE_DISTANCE = 0.2
NUM_SUB_SAMPLE = 5 # driven by sample distance)

WEIGHT_OF_LENGTH = 3.0
WEIGHT_OF_COLLISION = 2.0
WEIGHT_OF_ACCELERATION = 0.5

MAX_ACCELERATION = 0.5 # (m/s)

POT_PLOT_SUB_TRAJECTORY = False
POT_PLOT_FINAL_TRAJECTORY = True