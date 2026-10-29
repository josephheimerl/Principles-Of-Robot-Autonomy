import numpy as np
from scipy.spatial import cKDTree


def apply_transform(T, X):
    """Applies a (4, 4) homogeneous transform to (N, 3) points."""
    return X @ T[:3, :3].T + T[:3, 3]


def nearest_neighbors(src, dst):
    """
    Inputs:
        src: (N, 3), dst: (M, 3)
    Output:
        dist: (N,) distance from each src point to its nearest dst point
        idx: (N,) index into dst of that nearest point

    HINT: build a scipy.spatial.cKDTree on dst once and query it.
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return dist, idx


def best_fit_transform(A, B):
    """
    Least-squares rigid transform mapping corresponding points A onto B,
    i.e. minimizing sum_i ||R a_i + t - b_i||^2.
    Inputs:
        A, B: (N, 3) corresponding points
    Output:
        (4, 4) homogeneous transform T = [[R, t], [0, 1]] with det(R) = +1

    HINT: center both sets, take the SVD of the 3x3 cross-covariance, and
          flip the sign of the last singular direction if det(R) < 0.
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return T


def icp(src, dst, T_init, max_iters=50, tol=1e-6, reject_dist=None):
    """
    Point-to-point ICP aligning src to dst.
    Inputs:
        src: (N, 3) source points, dst: (M, 3) target points
        T_init: (4, 4) initial guess for the transform from src to dst
        max_iters: maximum number of iterations
        tol: stop when the mean nearest-neighbor distance changes by less than tol
        reject_dist: if not None, only pairs closer than reject_dist are used to
            fit the transform at each iteration (outlier rejection)
    Output:
        T: (4, 4) final transform from the original src to dst
        errors: list of the mean nearest-neighbor distance (over all src points)
            at the start of each iteration

    Each iteration: transform src by the current T, find nearest neighbors,
    fit the incremental transform with best_fit_transform on the kept pairs,
    and compose it with T.
    """
    T = T_init.copy()
    errors = []
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return T, errors
