import numpy as np


def dynamics(x, u, dt):
    """
    Robot motion model g(u_t, x_{t-1}) without noise.
    Inputs:
        x: (3,) pose (x, y, theta), u: (2,) controls (V, omega), dt: time step
    Output:
        (3,) next pose
    """
    return np.array([x[0] + u[0] * np.cos(x[2]) * dt,
                     x[1] + u[0] * np.sin(x[2]) * dt,
                     x[2] + u[1] * dt])


def measurement(x, landmarks):
    """
    Measurement model h(x_t) without noise: the position of every landmark in the robot frame.
    Inputs:
        x: (3,) pose, landmarks: (n, 2) landmark positions in the world frame
    Output:
        (2n,) stacked [x_1, y_1, ..., x_n, y_n] in the robot frame
    """
    c, s = np.cos(x[2]), np.sin(x[2])
    Rot = np.array([[c, s], [-s, c]])
    return ((landmarks - x[:2]) @ Rot.T).ravel()


def jacobian_G(x, u, dt):
    """
    Jacobian of dynamics with respect to the pose x.
    Output:
        (3, 3) array
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return G


def jacobian_H(x, landmarks):
    """
    Jacobian of measurement with respect to the pose x.
    Output:
        (2n, 3) array, rows in the same order as measurement(x, landmarks)
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return H


def ekf_predict(mu, Sigma, u, dt, R):
    """
    EKF prediction step. G is evaluated at the previous mean mu.
    Output:
        mu_bar: (3,), Sigma_bar: (3, 3)
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return mu_bar, Sigma_bar


def ekf_update(mu_bar, Sigma_bar, z, landmarks, Q):
    """
    EKF correction step with measurement z. H is evaluated at the predicted mean mu_bar.
    Output:
        mu: (3,), Sigma: (3, 3)
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return mu, Sigma


def run_ekf(mu0, Sigma0, us, zs, landmarks, dt, R, Q):
    """
    Runs the EKF: at step t, predict with the control us[t] and then correct with zs[t].
    theta does not need to be wrapped to [-pi, pi).
    Output:
        mus: (T + 1, 3) with mus[0] = mu0, Sigmas: (T + 1, 3, 3) with Sigmas[0] = Sigma0
    """
    T = len(us)
    mus = np.zeros((T + 1, 3))
    Sigmas = np.zeros((T + 1, 3, 3))
    mus[0], Sigmas[0] = mu0, Sigma0
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return mus, Sigmas
