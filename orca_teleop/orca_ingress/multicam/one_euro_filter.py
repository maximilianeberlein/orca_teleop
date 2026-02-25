"""Vectorized One Euro Filter for real-time signal smoothing.

Reference: Casiez et al., "1 Euro Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012.

Based on the canonical implementation by jaantollander/OneEuroFilter
and the numpy-vectorized variant by HoBeom/OneEuroFilter-Numpy.
"""

import numpy as np


def _smoothing_factor(t_e, cutoff):
    r = 2.0 * np.pi * cutoff * t_e
    return r / (r + 1.0)


class OneEuroFilter:
    """Vectorized One Euro Filter that operates on numpy arrays of any shape.

    Lazily initialized on the first call. NaN-safe: NaN inputs preserve
    the previous filtered value and do not corrupt filter state.
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t=None):
        if self.x_prev is None:
            self.x_prev = x.copy().astype(np.float64)
            self.dx_prev = np.zeros_like(self.x_prev)
            self.t_prev = t
            return x.copy()

        if t is not None and self.t_prev is not None:
            t_e = t - self.t_prev
            if t_e <= 0:
                t_e = 1.0 / 30.0
        else:
            t_e = 1.0 / 30.0
        self.t_prev = t

        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        valid = np.isfinite(x)
        self.x_prev = np.where(valid, x_hat, self.x_prev)
        self.dx_prev = np.where(valid, dx_hat, self.dx_prev)

        return np.where(valid, x_hat, self.x_prev)
