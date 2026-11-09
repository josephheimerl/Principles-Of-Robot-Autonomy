import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X


def add_prior(graph, noise):
    """
    Adds a gtsam.PriorFactorPose2 on X(0) at the origin, gtsam.Pose2(0, 0, 0).
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########


def odometry_pose(speed, turn):
    """
    Relative pose of the robot after one step, expressed in the frame of the pose before the step.
    In each step the robot drives `speed` meters straight along its heading, then turns by
    `turn` radians.
    Output:
        gtsam.Pose2
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########


def add_odometry(graph, speeds, turns, noise):
    """
    Adds a gtsam.BetweenFactorPose2 from X(i) to X(i + 1) with the measured relative pose
    odometry_pose(speeds[i], turns[i]), for i = 0, ..., len(speeds) - 1.
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########


def add_measurements(graph, ranges, bearings, noise):
    """
    Adds a gtsam.BearingRangeFactor2D between X(t) and L(j) for every measurement that exists
    (ranges[t, j] is not NaN), with bearing gtsam.Rot2.fromDegrees(bearings[t, j]) and range
    ranges[t, j]. Add the factors in order of t, then j.
    Inputs:
        ranges, bearings: (T, M) arrays, bearings in degrees, NaN where landmark j was not seen at time t
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########


def landmark_from_measurement(pose, range_m, bearing_deg):
    """
    Position in the world frame of a landmark measured at range range_m and bearing bearing_deg
    (degrees, relative to the robot heading) from pose.
    Inputs:
        pose: gtsam.Pose2
    Output:
        (2,) numpy array
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########


def initial_estimate(speeds, turns, ranges, bearings):
    """
    Initial guess for the optimizer, as a gtsam.Values containing
        - X(0) = Pose2(0, 0, 0) and X(i + 1) = X(i) composed with odometry_pose(speeds[i], turns[i])
          (dead reckoning), and
        - L(j) for every landmark that is seen at least once, placed with landmark_from_measurement
          using its first measurement (smallest t) and the dead-reckoned pose at that time.
    Insert poses with values.insert(X(i), pose) and landmarks with values.insert(L(j), point).
    """
    values = gtsam.Values()
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return values
