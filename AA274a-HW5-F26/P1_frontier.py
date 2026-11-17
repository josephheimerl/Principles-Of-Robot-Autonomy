import numpy as np
from scipy.signal import convolve2d


def window_counts(mask, window_size):
    """
    Number of True cells of mask in the window_size x window_size window centered on every cell.
    Cells outside the map are not counted.
    Inputs:
        mask: (ny, nx) boolean array
        window_size: odd integer
    Output:
        (ny, nx) array of counts

    HINT: this is a convolution with a window of ones (scipy.signal.convolve2d with mode="same").
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return counts


def explore(occupancy, window_size=11, min_unknown=0.25, min_free=0.35, occupied_thresh=50):
    """
    Frontier cells of an occupancy grid.
    A cell of occupancy.probs is unknown if its value is -1, occupied if it is >= occupied_thresh,
    and free otherwise. A cell is a frontier cell if, in the window_size x window_size window
    centered on it (cells outside the map are not counted, but the fractions are still of
    window_size^2),
        - at least min_unknown of the window_size^2 cells are unknown,
        - none of the cells are occupied, and
        - at least min_free of the window_size^2 cells are free.
    The cell itself may be free or unknown. Use the window_size argument, not occupancy.window_size.
    Output:
        frontier_states: (N, 2) array of (x, y) states of the frontier cells, in row-major order
        (increasing ky, then increasing kx, which is the order np.nonzero returns)

    HINT: work with grid indices and convert to states at the end with occupancy.grid2state.
          probs is indexed as probs[ky, kx], while grid2state expects (kx, ky).
    """
    probs = occupancy.probs
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
    return frontier_states


def closest_frontier(frontier_states, current_state):
    """
    Output:
        (state, distance): the frontier state closest to current_state in Euclidean distance,
        and that distance. If several states are equally close, return the first one.
        Returns (None, np.inf) if there are no frontier states.
    """
    ########## Code starts here ##########
    raise NotImplementedError
    ########## Code ends here ##########
