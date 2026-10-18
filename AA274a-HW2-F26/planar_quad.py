import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize


class BasePlanarQuadrotor:

    def __init__(self):
        # Dynamics constants
        self.x_dim = 6         # state dimension (see dynamics below)
        self.u_dim = 2         # control dimension (see dynamics below)
        self.g = 9.807         # gravity (m / s**2)
        self.m = 2.5           # mass (kg)
        self.l = 1.0           # half-length (m)
        self.Iyy = 1.0         # moment of inertia about the out-of-plane axis (kg * m**2)
        self.Cd_v = 0.25       # translational drag coefficient
        self.Cd_phi = 0.02255  # rotational drag coefficient

        # Control constraints
        self.max_thrust_per_prop = 0.75 * self.m * self.g  # total thrust-to-weight ratio = 1.5
        self.min_thrust_per_prop = 0

        # Thrust per prop that holds the quadrotor in a level hover
        self.hover_control = 0.5 * self.m * self.g * np.ones(self.u_dim)

    def ode(self, state, control):
        """Continuous-time dynamics of a planar quadrotor expressed as an ODE."""
        x, v_x, y, v_y, phi, omega = state
        T_1, T_2 = control
        return np.array([
            v_x,
            (-(T_1 + T_2) * np.sin(phi) - self.Cd_v * v_x) / self.m,
            v_y,
            ((T_1 + T_2) * np.cos(phi) - self.Cd_v * v_y) / self.m - self.g,
            omega,
            ((T_2 - T_1) * self.l - self.Cd_phi * omega) / self.Iyy,
        ])

    def discrete_step(self, state, control, dt):
        """Discrete-time dynamics (Euler-integrated) of a planar quadrotor."""
        # Euler integration is used for both planning and simulation so that the
        # nominal plan is reproduced exactly in the absence of disturbances.
        return state + dt * self.ode(state, control)

    def get_continuous_jacobians(self, state_nominal, control_nominal):
        """Continuous-time Jacobians A, B of the dynamics about (state_nominal, control_nominal)."""
        x, v_x, y, v_y, phi, omega = state_nominal
        T_1, T_2 = control_nominal
        A = np.array([[0., 1., 0., 0., 0., 0.],
                      [0., -self.Cd_v / self.m, 0., 0., -(T_1 + T_2) * np.cos(phi) / self.m, 0.],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., -self.Cd_v / self.m, -(T_1 + T_2) * np.sin(phi) / self.m, 0.],
                      [0., 0., 0., 0., 0., 1.],
                      [0., 0., 0., 0., 0., -self.Cd_phi / self.Iyy]])
        B = np.array([[0., 0.],
                      [-np.sin(phi) / self.m, -np.sin(phi) / self.m],
                      [0., 0.],
                      [np.cos(phi) / self.m, np.cos(phi) / self.m],
                      [0., 0.],
                      [-self.l / self.Iyy, self.l / self.Iyy]])
        return A, B


class PlanarQuadrotor(BasePlanarQuadrotor):
    """Adds plotting of quadrotor trajectories on top of BasePlanarQuadrotor."""

    def generate_patch_group(self, frame_height, frame_width, axle_height, axle_width, prop_width, prop_height, hub_width, hub_height):
        frame = matplotlib.patches.Rectangle((-frame_width / 2, -frame_height / 2),
                                             frame_width,
                                             frame_height,
                                             facecolor="tab:blue",
                                             edgecolor="k")
        hub = matplotlib.patches.FancyBboxPatch((-hub_width / 2, -hub_height / 2),
                                                hub_width,
                                                hub_height,
                                                facecolor="tab:blue",
                                                edgecolor="k",
                                                boxstyle="Round,pad=0.,rounding_size=0.05")
        axle_left = matplotlib.patches.Rectangle((-frame_width / 2, frame_height / 2),
                                                 axle_width,
                                                 axle_height,
                                                 facecolor="tab:blue",
                                                 edgecolor="k")
        axle_right = matplotlib.patches.Rectangle((frame_width / 2 - axle_width, frame_height / 2),
                                                  axle_width,
                                                  axle_height,
                                                  facecolor="tab:blue",
                                                  edgecolor="k")
        prop_left = matplotlib.patches.Ellipse(((axle_width - frame_width) / 2, frame_height / 2 + axle_height),
                                               prop_width,
                                               prop_height,
                                               facecolor="tab:gray",
                                               edgecolor="k",
                                               alpha=0.7)
        prop_right = matplotlib.patches.Ellipse(((frame_width - axle_width) / 2, frame_height / 2 + axle_height),
                                                prop_width,
                                                prop_height,
                                                facecolor="tab:gray",
                                                edgecolor="k",
                                                alpha=0.7)
        bubble = matplotlib.patches.Circle((0, 0), 1.5 * self.l, facecolor="None", edgecolor="red", linestyle="--")
        return (frame, hub, axle_left, axle_right, prop_left, prop_right, bubble)

    def animate(self, states, dt, ax=None):
        x, y, phi = states[:, 0], states[:, 2], states[:, 4]

        frame_width = 2 * self.l
        frame_height = 0.15
        axle_height = 0.2
        axle_width = 0.05
        prop_width = 0.5 * frame_width
        prop_height = 1.5 * frame_height
        hub_width = 0.3 * frame_width
        hub_height = 2.5 * frame_height

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        x_min, x_max = np.min(x), np.max(x)
        x_pad = (frame_width + prop_width) / 2 + 0.1 * (x_max - x_min)
        y_min, y_max = np.min(y), np.max(y)
        y_pad = (frame_width + prop_width) / 2 + 0.1 * (y_max - y_min)
        ax.set_xlim([x_min - x_pad, x_max + x_pad])
        ax.set_ylim([y_min - y_pad, y_max + y_pad])
        ax.set_aspect(1.)

        ax.plot(x, y)

        for k in range(0, len(states), 10):
            patch_group = self.generate_patch_group(frame_height, frame_width, axle_height, axle_width, prop_width, prop_height, hub_width, hub_height)
            transform = matplotlib.transforms.Affine2D().rotate_around(0., 0., phi[k])
            transform += matplotlib.transforms.Affine2D().translate(x[k], y[k])
            transform += ax.transData
            for patch in patch_group:
                patch.set_transform(transform)
                ax.add_patch(patch)
        return ax


OBSTACLE_CENTER = np.array([5., 5.])
OBSTACLE_CLEARANCE = 3.


def optimize_nominal_trajectory(N=50):
    """Direct method for a nominal trajectory from hover at (0, 5) to hover at (10, 7) around the obstacle."""
    planar_quad = PlanarQuadrotor()
    x_0 = np.array([0., 0., 5., 0., 0., 0.])
    x_f = np.array([10., 0., 7., 0., 0., 0.])

    equilibrium_thrust = 0.5 * planar_quad.m * planar_quad.g
    x_dim = planar_quad.x_dim
    u_dim = planar_quad.u_dim

    def pack_decision_variables(final_time, states, controls):
        return np.concatenate([[final_time], states.ravel(), controls.ravel()])

    def unpack_decision_variables(z):
        final_time = z[0]
        states = z[1:1 + (N + 1) * x_dim].reshape(N + 1, x_dim)
        controls = z[-N * u_dim:].reshape(N, u_dim)
        return final_time, states, controls

    def cost(z):
        final_time, states, controls = unpack_decision_variables(z)
        dt = final_time / N
        return final_time + dt * np.sum(np.square(controls - equilibrium_thrust))

    z_guess = pack_decision_variables(10, x_0 + np.linspace(0, 1, N + 1)[:, np.newaxis] * (x_f - x_0),
                                      equilibrium_thrust * np.ones((N, u_dim)))

    bounds = scipy.optimize.Bounds(
        pack_decision_variables(0., -np.inf * np.ones((N + 1, x_dim)),
                                planar_quad.min_thrust_per_prop * np.ones((N, u_dim))),
        pack_decision_variables(np.inf, np.inf * np.ones((N + 1, x_dim)),
                                planar_quad.max_thrust_per_prop * np.ones((N, u_dim))))

    def equality_constraints(z):
        final_time, states, controls = unpack_decision_variables(z)
        dt = final_time / N
        constraint_list = [states[i + 1] - planar_quad.discrete_step(states[i], controls[i], dt) for i in range(N)]
        constraint_list.append(states[0] - x_0)
        constraint_list.append(states[-1] - x_f)
        return np.concatenate(constraint_list)

    def inequality_constraints(z):
        final_time, states, controls = unpack_decision_variables(z)
        return np.sum(np.square(states[:, [0, 2]] - OBSTACLE_CENTER), -1) - OBSTACLE_CLEARANCE**2

    result = scipy.optimize.minimize(cost,
                                     z_guess,
                                     bounds=bounds,
                                     constraints=[{"type": "eq", "fun": equality_constraints},
                                                  {"type": "ineq", "fun": inequality_constraints}])
    return unpack_decision_variables(result.x)


def plot_obstacle(ax, planar_quad):
    obstacle = matplotlib.patches.Circle(OBSTACLE_CENTER, OBSTACLE_CLEARANCE - 1.5 * planar_quad.l, facecolor="k")
    ax.add_patch(obstacle)


class WindField:
    """Smooth random wind field over [-3, 14] x [3, 9]; zero outside of it.

    The wind values on the grid are a Gaussian process sample (squared exponential
    kernel) stored in `filename`, so that every machine sees the same wind.
    """

    def __init__(self, filename="wind_field.npz", scale=1.0):
        self.xs, self.ys = np.arange(-3, 15), np.arange(3, 10)
        self.X, self.Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        data = np.load(filename)
        self.Wx, self.Wy = scale * data["Wx"], scale * data["Wy"]
        self.interp = scipy.interpolate.RegularGridInterpolator(
            (self.xs, self.ys), np.stack([self.Wx, self.Wy], -1).reshape(self.X.shape + (2,)),
            bounds_error=False, fill_value=0)

    def apply(self, state, dt):
        x, v_x, y, v_y, phi, omega = state
        dv_x, dv_y = dt * self.interp(np.array([x, y]))[0]
        return np.array([x, v_x + dv_x, y, v_y + dv_y, phi, omega])

    def plot(self, ax):
        ax.quiver(self.X, self.Y, self.Wx, self.Wy, width=3e-3, alpha=0.2)


def simulate(planar_quad, initial_state, controller, N, dt, wind=None):
    """Simulates N steps, applying controller(k, state) clipped to the thrust limits.

    Returns the states (N + 1, x_dim) and the applied controls (N, u_dim).
    """
    states = [np.asarray(initial_state, dtype=float)]
    controls = []
    for k in range(N):
        control = np.clip(controller(k, states[k]),
                          planar_quad.min_thrust_per_prop, planar_quad.max_thrust_per_prop)
        next_state = planar_quad.discrete_step(states[k], control, dt)
        if wind is not None:
            next_state = wind.apply(next_state, dt)
        states.append(next_state)
        controls.append(control)
    return np.array(states), np.array(controls)
