import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X


def add_prior(graph : gtsam.NonlinearFactorGraph, noise):
    """
    Adds a gtsam.PriorFactorPose2 on X(0) at the origin, gtsam.Pose2(0, 0, 0).
    """
    ########## Code starts here ##########
    init_pose = gtsam.Pose2(0,0,0)
    prior_factor = gtsam.PriorFactorPose2(X(0),init_pose,noise)
    graph.add(prior_factor)
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
    return gtsam.Pose2(speed,0,turn)
    ########## Code ends here ##########


def add_odometry(graph, speeds, turns, noise):
    """
    Adds a gtsam.BetweenFactorPose2 from X(i) to X(i + 1) with the measured relative pose
    odometry_pose(speeds[i], turns[i]), for i = 0, ..., len(speeds) - 1.
    """
    ########## Code starts here ##########
    for i, ( speed, turn ) in enumerate( zip(speeds,turns) ):
        rel_pose = odometry_pose(speed,turn)
        factor = gtsam.BetweenFactorPose2(X(i), X(i+1), rel_pose, noise)
        graph.add(factor)
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
    for t in range( ranges.shape[0] ):
        for j in range( ranges.shape[1] ):
            if not np.isnan(ranges[t,j]):
                factor = gtsam.BearingRangeFactor2D(X(t), L(j), gtsam.Rot2.fromDegrees(bearings[t,j]), ranges[t,j], noise)
                graph.add(factor)
    ########## Code ends here ##########


def landmark_from_measurement(pose : gtsam.Pose2, range_m, bearing_deg):
    """
    Position in the world frame of a landmark measured at range range_m and bearing bearing_deg
    (degrees, relative to the robot heading) from pose.
    Inputs:
        pose: gtsam.Pose2
    Output:
        (2,) numpy array
    """
    ########## Code starts here ##########
    pos = range_m*np.array([ np.cos(np.deg2rad(bearing_deg)), np.sin(np.deg2rad(bearing_deg)) ])
    return pose.transformFrom(pos)
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
    poses = [ gtsam.Pose2(0,0,0) ]
    values.insert(X(0), poses[0])
    for i, ( speed, turn ) in enumerate( zip(speeds,turns) ):
        rel_pose = odometry_pose(speed,turn)
        abs_pose = poses[i].compose(rel_pose)
        values.insert(X(i+1), abs_pose)
        poses.append(abs_pose)

    for j in range(ranges.shape[1]):
        for t in range(ranges.shape[0]):
            if not np.isnan(ranges[t,j]):
                values.insert(L(j), landmark_from_measurement(poses[t], ranges[t,j], bearings[t,j]))
                break
    ########## Code ends here ##########
    return values
