import typing as T

import numpy as np


# ---------------------------------------------------------------------------
# Problem 1

def snap_to_grid(state: np.ndarray, resolution: float) -> np.ndarray:
    """ Snap continuous coordinates to a finite-resolution grid """
    return resolution * np.round(state / resolution)


class StochOccupancyGrid2D(object):
    """ A stochastic occupancy grid derived from ROS2 map data (same as asl_tb3_lib)

    probs holds the map data in the ROS convention: -1 for unknown cells, otherwise the
    occupancy probability in percent (0 to 100). probs is indexed as probs[ky, kx].
    """

    def __init__(self,
        resolution: float,
        size_xy: np.ndarray,
        origin_xy: np.ndarray,
        window_size: int,
        probs: T.Sequence[float],
        thresh: float = 0.5
    ) -> None:
        self.resolution = resolution
        self.size_xy = size_xy
        self.origin_xy = origin_xy
        self.probs = np.reshape(np.asarray(probs), (size_xy[1], size_xy[0]))
        self.window_size = window_size
        self.thresh = thresh

    def state2grid(self, state_xy: np.ndarray) -> np.ndarray:
        """ convert real state coordinates (x, y) to integer grid indices (kx, ky) """
        state_snapped_xy = snap_to_grid(state_xy, self.resolution)
        grid_xy = ((state_snapped_xy - self.origin_xy) / self.resolution).astype(int)
        return grid_xy

    def grid2state(self, grid_xy: np.ndarray) -> np.ndarray:
        """ convert integer grid indices (kx, ky) to real state coordinates (x, y) """
        return (grid_xy * self.resolution + self.origin_xy).astype(float)

    def is_free(self, state_xy: np.ndarray) -> bool:
        """ True if the window around state_xy is free with probability above 1 - thresh """
        grid_xy = self.state2grid(state_xy)

        half_size = int(round((self.window_size - 1) / 2))
        grid_xy_lower = np.maximum(0, grid_xy - half_size)
        grid_xy_upper = np.minimum(self.size_xy, grid_xy + half_size + 1)

        prob_window = self.probs[grid_xy_lower[1]:grid_xy_upper[1],
                                 grid_xy_lower[0]:grid_xy_upper[0]]
        p_total = np.prod(1. - np.maximum(prob_window / 100., 0.))

        return (1. - p_total) < self.thresh


# Simulated indoor environment

RESOLUTION = 0.1
SIZE_XY = np.array([100, 80])     # 10 m x 8 m
SENSING_RADIUS = 3.0

# (x_min, x_max, y_min, y_max) in meters
WALLS = [
    (0.0, 10.0, 0.0, 0.2), (0.0, 10.0, 7.8, 8.0), (0.0, 0.2, 0.0, 8.0), (9.8, 10.0, 0.0, 8.0),
    (4.0, 4.2, 0.0, 3.0), (4.0, 4.2, 4.2, 8.0),      # wall between the rooms, doorway at y = 3.0 to 4.2
    (6.0, 10.0, 5.0, 5.2),                           # wall in the right room, opening at x = 4.2 to 6.0
    (1.2, 2.2, 5.4, 6.4), (7.0, 8.0, 1.6, 2.6),      # boxes
]


def ground_truth():
    """ (ny, nx) boolean array, True where a cell is occupied """
    occ = np.zeros((SIZE_XY[1], SIZE_XY[0]), dtype=bool)
    for x0, x1, y0, y1 in WALLS:
        kx0, kx1 = int(round(x0 / RESOLUTION)), int(round(x1 / RESOLUTION))
        ky0, ky1 = int(round(y0 / RESOLUTION)), int(round(y1 / RESOLUTION))
        occ[ky0:ky1, kx0:kx1] = True
    return occ


def sense(probs, occ_gt, state_xy, radius=SENSING_RADIUS):
    """ Marks every cell visible from state_xy within radius as free (0) or occupied (100).
    A cell is visible if the straight line from the robot to it does not pass through an
    occupied cell first. Returns an updated copy of probs. """
    probs = probs.copy()
    ny, nx = occ_gt.shape
    kx, ky = np.meshgrid(np.arange(nx), np.arange(ny))
    cells = np.stack([kx.ravel(), ky.ravel()], axis=1)
    centers = cells * RESOLUTION
    d = np.linalg.norm(centers - state_xy, axis=1)
    cells, centers, d = cells[d <= radius], centers[d <= radius], d[d <= radius]

    n = int(np.ceil(radius / (0.5 * RESOLUTION)))
    t = np.linspace(0.0, 1.0, n + 1)[:-1]
    pts = state_xy[None, None, :] + t[None, :, None] * (centers - state_xy)[:, None, :]
    pk = np.round(pts / RESOLUTION).astype(int)
    pk[..., 0] = np.clip(pk[..., 0], 0, nx - 1)
    pk[..., 1] = np.clip(pk[..., 1], 0, ny - 1)
    hit = occ_gt[pk[..., 1], pk[..., 0]]
    same = (pk[..., 0] == cells[:, None, 0]) & (pk[..., 1] == cells[:, None, 1])
    blocked = (hit & ~same).any(axis=1)

    vis = cells[~blocked]
    probs[vis[:, 1], vis[:, 0]] = np.where(occ_gt[vis[:, 1], vis[:, 0]], 100, 0)
    return probs


def make_occupancy(probs):
    return StochOccupancyGrid2D(resolution=RESOLUTION, size_xy=SIZE_XY, origin_xy=np.zeros(2),
                                window_size=9, probs=probs)


# ---------------------------------------------------------------------------
# Problems 2 and 3

def plot_error_ellipse(ax, mean, cov, color="red"):
    """ Draws the 95% confidence ellipse of a 2D Gaussian with the given mean and covariance """
    from matplotlib.patches import Ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    width, height = 2 * 2.4477 * np.sqrt(np.maximum(eigenvalues[::-1], 0.0))
    ax.add_patch(Ellipse(mean, width, height, angle=angle, edgecolor=color, fc="none", lw=1.5))
