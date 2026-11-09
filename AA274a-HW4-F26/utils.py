import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, sobel


def detect_corners(image, num_corners=300, border=10):
    """ Harris corner detector. Returns a (K, 2) integer array of (row, col) corners, strongest
    first, at least `border` pixels from the image edges. """
    Ix, Iy = sobel(image, 1, mode="nearest"), sobel(image, 0, mode="nearest")
    Sxx, Syy, Sxy = (gaussian_filter(v, 1.5) for v in (Ix * Ix, Iy * Iy, Ix * Iy))
    R = Sxx * Syy - Sxy**2 - 0.05 * (Sxx + Syy)**2
    peaks = (R == maximum_filter(R, size=11)) & (R > 0.01 * R.max())
    peaks[:border] = peaks[-border:] = False
    peaks[:, :border] = peaks[:, -border:] = False
    rc = np.argwhere(peaks)
    order = np.argsort(-R[rc[:, 0], rc[:, 1]], kind="stable")
    return rc[order[:num_corners]]
