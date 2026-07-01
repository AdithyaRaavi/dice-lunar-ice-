"""
volume.py
---------
Uncertainty-bounded subsurface-ice volume for the top 0-5 m.

V = sum_i ( f_i * A_i * d_i ), summed over detected cells, where
    f_i = ice volume-fraction in the cell,
    A_i = cell ground area (pixel_m^2),
    d_i = thickness of the ice-bearing layer in the 0-5 m bracket.
We propagate uncertainty in f, d (and implicitly dielectric/loss-tangent, which sets
the L-vs-S depth bracket) by Monte Carlo and report a DISTRIBUTION, not one number.
"""
import numpy as np


def monte_carlo_volume(prob, pixel_m=25.0, thr=0.5,
                       depth_lo=0.5, depth_hi=2.5,
                       frac_lo=0.3, frac_hi=0.7,
                       n_iter=4000, seed=0):
    """Return dict with median and 5th/95th percentile volume (m^3) and the samples."""
    rng = np.random.default_rng(seed)
    detected = prob >= thr
    p = prob[detected]
    n = int(detected.sum())
    A = pixel_m ** 2
    if n == 0:
        return {"median": 0.0, "p05": 0.0, "p95": 0.0, "n_cells": 0, "samples": np.zeros(n_iter)}

    samples = np.empty(n_iter)
    for k in range(n_iter):
        # ice fraction scales with confidence x a per-iteration efficiency draw
        eff = rng.uniform(frac_lo, frac_hi)
        f = np.clip(p * eff + 0.03 * rng.standard_normal(n), 0, 1)
        d = rng.uniform(depth_lo, depth_hi, size=n)        # metres within 0-5 m
        samples[k] = float(np.sum(f * A * d))

    return {
        "median": float(np.median(samples)),
        "p05": float(np.percentile(samples, 5)),
        "p95": float(np.percentile(samples, 95)),
        "n_cells": n,
        "samples": samples,
    }
