import numpy as np

from P1_extrinsics import compute_homography


def homography_residuals(H, world_xy, uv):
    """
    Inputs:
        H: (3, 3), world_xy: (N, 2), uv: (N, 2)
    Output:
        (N,) pixel distance between uv and the projection of world_xy through H
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########


def ransac_num_iterations(p, w, s):
    """
    Number of RANSAC iterations needed to draw at least one all-inlier sample
    with probability p, given inlier ratio w and sample size s.
    Output:
        int, rounded up
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########


def ransac_homography(world_xy, uv, num_iters, threshold, rng):
    """
    Robust homography estimate with RANSAC.
    Inputs:
        world_xy, uv: (N, 2) correspondences, some of which are outliers
        num_iters: number of random samples to try
        threshold: a correspondence is an inlier if its residual is below this (pixels)
        rng: np.random.Generator, use rng.choice(N, 4, replace=False) to draw samples
    Output:
        H: (3, 3) homography refit on all inliers of the best sample
        inliers: (N,) boolean mask of correspondences with residual below threshold under H

    HINT: the best sample is the one with the most inliers. Use compute_homography
          (imported above) and homography_residuals.
    """
    N = len(uv)
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return H, inliers
