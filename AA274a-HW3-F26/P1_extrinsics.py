import numpy as np


def checkerboard_points(nx, ny, square_size):
    """
    World coordinates of the checkerboard corners on the Z = 0 plane.
    Inputs:
        nx, ny: number of inner corners along x and y
        square_size: side length of one square (m)
    Output:
        (nx * ny, 2) array of (X, Y), with X varying fastest:
        (0, 0), (s, 0), ..., ((nx-1)s, 0), (0, s), ...
    """
    ########## Code starts here ##########
    points = np.zeros((nx,ny,2))
    xrange = np.arange(nx) * square_size
    yrange = np.arange(ny) * square_size

    # broadcast ranges
    points[:,:,0] = points[:,:,0] + xrange[:,None]
    points[:,:,1] = points[:,:,1] + yrange[None,:]

    # stack results, x incrementing first
    points = points.transpose(1,0,2) # rearrange so reshape gives proper order
    stacked_points = points.reshape(-1,2)
    return stacked_points
    ########## Code ends here ##########


def compute_homography(world_xy, uv):
    """
    Homography H with [u, v, 1]^T ~ H [X, Y, 1]^T, estimated with the DLT and the SVD.
    Inputs:
        world_xy: (N, 2) points on the board, N >= 4
        uv: (N, 2) corresponding pixel coordinates
    Output:
        (3, 3) homography, scaled so that H[2, 2] = 1
    """

    ########## Code starts here ##########
    # H = K[R,t]
    # ph = H * PhW

    X = world_xy[:,0]
    Y = world_xy[:,1]
    u = uv[:,0]
    v = uv[:,1]
    
    Mi = lambda i: np.array([
        [-X[i], -Y[i], -1,      0,      0,   0,  u[i]*X[i], u[i]*Y[i], u[i]],
        [    0,     0,  0,  -X[i],  -Y[i],  -1,  v[i]*X[i], v[i]*Y[i], v[i]]
    ])

    M = np.vstack([Mi(i) for i in range(len(X))])

    _,_,Vt = np.linalg.svd(M)

    h = Vt[-1,:]
    H = h.reshape(3,3)
    H = H/H[2,2]
    return H
    ########## Code ends here ##########


def pose_from_homography(K, H):
    """
    Camera pose from a board homography, so that P_cam = R P_world + t.
    Inputs:
        K: (3, 3) intrinsics, H: (3, 3) homography
    Output:
        R: (3, 3) rotation matrix (orthonormal, det = +1)
        t: (3,) translation, with t[2] > 0 (board in front of the camera)

    HINT: H is only known up to scale, including its sign. Normalize by the
          norm of the first column, pick the sign that gives t[2] > 0, and
          replace [r0, r1, r0 x r1] by the closest rotation matrix (SVD).
    """
    ########## Code starts here ##########
    # rescale
    
    Rt = np.linalg.inv(K)@H

    scale = np.linalg.norm(Rt[:,0])
    Rt = Rt/scale
    Rt = Rt * Rt[2,2]/abs(Rt[2,2])

    # construct closest orthogonal matrix
    R0 = np.column_stack([Rt[:,:2], np.cross(Rt[:,0],Rt[:,1])])
    U,_,Vt = np.linalg.svd(R0)
    Sig = np.eye(3)
    Sig[2,2] = np.linalg.det(U@Vt)
    R = U@Sig@Vt
    

    t = Rt[:,2]
    ########## Code ends here ##########
    return R, t


def project_points(K, R, t, X):
    """
    Inputs:
        K, R: (3, 3), t: (3,)
        X: (N, 3) points in the world frame
    Output:
        (N, 2) pixel coordinates (u, v)
    """
    ########## Code starts here ##########
    Xh = np.column_stack([X, np.ones((len(X), 1))])

    ph = K @ np.column_stack([R,t]) @ Xh.T
    p = ph[:2,:] / ph[2,:]
    ########## Code ends here ##########
    return p.T

def reprojection_rmse(uv, uv_proj):
    """
    Output:
        Root-mean-square pixel distance between (N, 2) arrays uv and uv_proj
    """
    ########## Code starts here ##########
    diff2 = (uv - uv_proj)**2

    rmse = np.sqrt(np.mean(np.sum(diff2, axis=1)))
    return rmse

    ########## Code ends here ##########
