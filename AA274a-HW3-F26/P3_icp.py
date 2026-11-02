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
    kd = cKDTree(dst)
    dist,idx = kd.query(src)

    # N = len(src)
    # dist = np.zeros(N)
    # idx = np.zeros(N,int)
    # for i in range(N):
    #     idxi,disti = kd.query(src[i,:])
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
    abar = np.mean(A,axis=0)
    bbar = np.mean(B,axis=0)

    Cov = (A - abar).T @ (B-bbar)

    

    U, _, Vt = np.linalg.svd(Cov)

    Sig = np.eye(3)
    Sig[2,2] = np.linalg.det(Vt.T@U.T)
    R = Vt.T@Sig@U.T

    t = bbar-R@abar

    T = np.zeros([4,4])
    T[3,3] = 1
    T[:3,:3] = R
    T[:3,3] = t

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
    srcH = np.column_stack([src,np.ones(len(src))])
    for i in range(max_iters):
        # Transform src by current T
        TsrcH = (T @ srcH.T).T

        # Find nearest neighbors
        dist, idx = nearest_neighbors(TsrcH[:,:3],dst)

        if reject_dist is not None:
            dist_mask = dist < reject_dist
            idx = idx[dist_mask]
            TsrcH = TsrcH[dist_mask,:]

        # Fit transform based on nearest neighbor correspondence
        dT = best_fit_transform(TsrcH[:,:3], dst[idx,:])

        # Compose with T
        T = dT@T

        mean_dist = np.mean(dist)
        errors.append(mean_dist)
        if i > 0 and np.abs(errors[i-1] - mean_dist)< tol:
            break


    ########## Code ends here ##########
    return T, errors
