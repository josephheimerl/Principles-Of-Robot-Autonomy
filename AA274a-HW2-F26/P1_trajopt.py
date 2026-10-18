import typing as T

import numpy as np
from scipy.optimize import minimize, Bounds

N = 50  # Number of time discretization nodes (0, 1, ... N).
s_dim = 3  # State dimension; 3 for (x, y, th).
u_dim = 2  # Control dimension; 2 for (V, om).
v_max = 0.5  # Maximum linear velocity.
om_max = 1.0  # Maximum angular velocity.

# Default problem setup
S_0 = np.array([0.0, 0.0, np.pi / 2])  # Initial state.
S_F = np.array([5.0, 5.0, np.pi / 2])  # Final state.
OBS_POS = np.array([2.5, 2.5])
OBS_RADIUS = 0.3
EGO_RADIUS = 0.1


def pack_decision_variables(t_f: float, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Packs t_f, states (N + 1, s_dim) and controls (N, u_dim) into a 1D vector z."""
    return np.concatenate([[t_f], s.ravel(), u.ravel()])


def unpack_decision_variables(z: np.ndarray) -> T.Tuple[float, np.ndarray, np.ndarray]:
    """Unpacks z into t_f, states (N + 1, s_dim) and controls (N, u_dim)."""
    t_f = float(z[0])
    s = z[1:1 + (N + 1) * s_dim].reshape(N + 1, s_dim)
    u = z[-N * u_dim:].reshape(N, u_dim)
    return t_f, s, u


def cost(z: np.ndarray, alpha: float) -> float:
    """
    Discretized cost J = int_0^tf (alpha + V^2 + om^2) dt, using dt = t_f / N.
    Output:
        Float cost value
    """
    ########## Code starts here ##########
    t_f, s, u = unpack_decision_variables(z)
    N = u.shape[0]
    costmat = np.concatenate((np.full((N,1),alpha), np.square(u)), axis=1)
    return np.sum(costmat)*t_f/N
    ########## Code ends here ##########


def dynamics_constraints(z: np.ndarray) -> np.ndarray:
    """
    Euler-discretized unicycle dynamics as equality constraints.
    Output:
        Array of shape (N, s_dim) whose row i is s_{i+1} - (s_i + dt * f(s_i, u_i)).
        All entries are zero for a dynamically feasible z.
    """
    ########## Code starts here ##########
    t_f, s, u = unpack_decision_variables(z)
    s_i1 = s[1:,:]
    s_i0 = s[:-1,:]
    N = u.shape[0]
    dt = t_f/N

    f = lambda s,u: np.column_stack((
        u[:,0]*np.cos(s[:,2]),
        u[:,0]*np.sin(s[:,2]),
        u[:,1]
    ))
    return s_i1 - s_i0 - dt*f(s_i0,u)
    ########## Code ends here ##########


def boundary_constraints(z: np.ndarray, s_0: np.ndarray, s_f: np.ndarray) -> np.ndarray:
    """
    Output:
        Array of shape (2, s_dim): row 0 is states[0] - s_0 and row 1 is states[N] - s_f.
    """
    ########## Code starts here ##########
    t_f, s, u = unpack_decision_variables(z)
    return np.stack((
        s[0,:] - s_0,
        s[-1,:] - s_f
    ))
    ########## Code ends here ##########


def obstacle_constraints(z: np.ndarray, obs_pos: np.ndarray, obs_radius: float,
                         ego_radius: float) -> np.ndarray:
    """
    Collision avoidance as inequality constraints, one per state s_0, ..., s_N.
    Output:
        Array of shape (N + 1,), entry i is ||(x_i, y_i) - obs_pos|| - (obs_radius + ego_radius).
        Nonnegative entries mean no collision.
    """
    ########## Code starts here ##########
    t_f, s, u = unpack_decision_variables(z)
    dist = s[:,0:2] - obs_pos

    distance = np.linalg.norm(dist,axis=1)  - (obs_radius + ego_radius)
    return distance
    ########## Code ends here ##########


def optimize_trajectory(alpha: float = 1.0, s_0: np.ndarray = S_0, s_f: np.ndarray = S_F,
                        obs_pos: np.ndarray = OBS_POS, obs_radius: float = OBS_RADIUS,
                        ego_radius: float = EGO_RADIUS, verbose: bool = False):
    """
    Solves the trajectory optimization problem with scipy.optimize.minimize.
    Output:
        t_f, s (N + 1, s_dim), u (N, u_dim)
    """
    s_0, s_f, obs_pos = np.asarray(s_0, float), np.asarray(s_f, float), np.asarray(obs_pos, float)

    # Straight line initial guess
    z_guess = pack_decision_variables(
        20, s_0 + np.linspace(0, 1, N + 1)[:, np.newaxis] * (s_f - s_0),
        np.ones(N * u_dim))

    # Speed and turn rate limits
    bounds = Bounds(
        pack_decision_variables(
            0., -np.inf * np.ones((N + 1, s_dim)),
            np.array([0.01, -om_max]) * np.ones((N, u_dim))),
        pack_decision_variables(
            np.inf, np.inf * np.ones((N + 1, s_dim)),
            np.array([v_max, om_max]) * np.ones((N, u_dim)))
    )

    def eq_constraints(z):
        return np.concatenate([dynamics_constraints(z).ravel(),
                               boundary_constraints(z, s_0, s_f).ravel()])
    def ineq_constraints(z):
        return obstacle_constraints(z, obs_pos, obs_radius, ego_radius)

    result = minimize(lambda z: cost(z, alpha),
                      z_guess,
                      bounds=bounds,
                      constraints=[{'type': 'eq', 'fun': eq_constraints},
                                   {'type': 'ineq', 'fun': ineq_constraints}])
    if verbose:
        print(result.message)

    return unpack_decision_variables(result.x)
