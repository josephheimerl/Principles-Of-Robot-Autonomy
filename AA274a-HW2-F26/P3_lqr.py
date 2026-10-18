import numpy as np
from scipy.linalg import solve_continuous_are
from planar_quad import *


def find_closest_nominal_state(state, nominal_states):
    """
    Inputs:
        state: current state, shape (x_dim,)
        nominal_states: nominal trajectory, shape (N + 1, x_dim)
    Output:
        Integer index j of the nominal state closest to state, using the Euclidean
        norm of the full state difference
    """
    ########## Code starts here ##########
    dist = nominal_states - state[None,:]
    distnorm = np.linalg.norm(dist,axis=1)
    return np.argmin(distnorm)
    ########## Code ends here ##########


def compute_lqr_gain(A, B, Q, R):
    """
    Infinite-horizon continuous-time LQR gain for xdot = A x + B u.
    Inputs:
        A: (x_dim, x_dim), B: (x_dim, u_dim), Q: (x_dim, x_dim), R: (u_dim, u_dim)
    Output:
        K: (u_dim, x_dim), so that u = -K x

    HINT: use solve_continuous_are (imported above).
    """
    ########## Code starts here ##########
    P = solve_continuous_are(A,B,Q,R)
    K = np.linalg.inv(R)@np.transpose(B)@P
    ########## Code ends here ##########
    return K


def nominal_control_at(j, nominal_controls, planar_quad):
    """Nominal control for nominal state j. The last state has no control, so hover thrust is used."""
    if j < len(nominal_controls):
        return nominal_controls[j]
    return planar_quad.hover_control


def compute_gain_schedule(planar_quad : PlanarQuadrotor, nominal_states, nominal_controls, Q, R):
    """
    Inputs:
        planar_quad: PlanarQuadrotor object
        nominal_states: (N + 1, x_dim), nominal_controls: (N, u_dim)
        Q, R: LQR weights
    Output:
        List of N + 1 gain matrices, where entry j is the LQR gain for the dynamics
        linearized about nominal state j and its nominal control

    HINT: use planar_quad.get_continuous_jacobians and nominal_control_at.
    """
    gains = []
    ########## Code starts here ##########
    N = nominal_states.shape[0]
    for j in range(N):
        nom_cont = nominal_control_at(j,nominal_controls,planar_quad)
        nom_state = nominal_states[j,:]
        A,B = planar_quad.get_continuous_jacobians(nom_state, nom_cont)
        gains.append(compute_lqr_gain(A,B,Q,R))
        
    ########## Code ends here ##########
    return gains


def compute_hover_gain(planar_quad, Q, R):
    """
    Output:
        Single LQR gain K for the dynamics linearized about a level hover
        (zero velocity and pitch, thrust planar_quad.hover_control).
    """
    ########## Code starts here ##########
    nom_cont = planar_quad.hover_control
    nom_state = np.zeros(planar_quad.x_dim)
    A,B = planar_quad.get_continuous_jacobians(nom_state, nom_cont)
    K = compute_lqr_gain(A,B,Q,R)
    ########## Code ends here ##########
    return K


def lqr_control(state, j, nominal_states, nominal_controls, gains, planar_quad):
    """
    Gain-scheduled LQR about nominal point j: u = u_j - K_j (x - x_j).
    Inputs:
        state: (x_dim,)
        j: index of the nominal state / gain to use
        gains: list of N + 1 gain matrices
    Output:
        Control (u_dim,). Do not clip; the simulator applies the thrust limits.
    """
    ########## Code starts here ##########
    K = gains[j]
    xstar = nominal_states[j,:]
    ustar = nominal_control_at(j,nominal_controls, planar_quad)
    control = ustar - K @ (state-xstar)
    ########## Code ends here ##########
    return control
