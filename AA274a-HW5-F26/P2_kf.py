import numpy as np


def landmark_model(num_landmarks=4):
    """
    Linear model of the stacked landmark positions x = [x_1, y_1, ..., x_n, y_n], n = num_landmarks.
        x_t = A x_{t-1} + noise with covariance R,   z_t = C x_t + noise with covariance Q
    Output:
        A: (2n, 2n) dynamics matrix
        C: (2n, 2n) measurement matrix
        R: (2n, 2n) process noise covariance
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return A, C, R


def kf_predict(mu, Sigma, A, R):
    """
    Kalman filter prediction step (there is no control input).
    Output:
        mu_bar: (d,), Sigma_bar: (d, d)
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return mu_bar, Sigma_bar


def kf_update(mu_bar, Sigma_bar, z, C, Q):
    """
    Kalman filter correction step with measurement z.
    Output:
        mu: (d,), Sigma: (d, d)
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return mu, Sigma


def run_kf(mu0, Sigma0, zs, A, C, R, Q):
    """
    Runs the Kalman filter on the measurements zs[0], ..., zs[T - 1], predicting and then
    correcting at every step.
    Inputs:
        mu0: (d,), Sigma0: (d, d) initial belief
        zs: (T, m) measurements
    Output:
        mus: (T + 1, d) with mus[0] = mu0, Sigmas: (T + 1, d, d) with Sigmas[0] = Sigma0
    """
    T = len(zs)
    mus = np.zeros((T + 1, len(mu0)))
    Sigmas = np.zeros((T + 1, len(mu0), len(mu0)))
    mus[0], Sigmas[0] = mu0, Sigma0
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return mus, Sigmas
