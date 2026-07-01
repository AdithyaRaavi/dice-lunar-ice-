"""
synth.py
--------
Generates a synthetic south-polar crater scene with KNOWN ground truth, so the
pipeline can be validated end-to-end before real data is available. This is a
stand-in for MIDAS-derived CPR/DOP rasters — NOT real DFSAR data. Three classes:

  * ice            : buried-ice patches  -> CPR_L > 1, DOP < 0.13, CPR_S <= CPR_L
                     (only L-band penetrates deep enough; volume scattering -> low DOP)
  * rough_rock     : roughness false-positives -> high CPR but CPR_S > CPR_L and
                     moderate DOP. These exist precisely to test that the pipeline
                     does NOT blindly trust high CPR (cf. Bhiravarasu et al. 2021,
                     who found S-band CPR > L-band CPR in anomalous rocky regions).
  * surface        : background regolith -> CPR < 1, high DOP.

A simple DEM (for slope constraints in traverse planning) is included.
"""
import numpy as np


def _blobs(size, centers, sigmas, amps, rng):
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    field = np.zeros((size, size))
    for (cy, cx), s, a in zip(centers, sigmas, amps):
        field += a * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2)))
    return field


def generate_scene(size=64, seed=7):
    rng = np.random.default_rng(seed)

    # ---- ground-truth ice patches ----
    ice_centers = [(18, 20), (40, 44), (30, 30)]
    ice_sigmas  = [4.5, 5.5, 3.0]
    ice_amps    = [1.0, 0.9, 0.7]
    ice_strength = _blobs(size, ice_centers, ice_sigmas, ice_amps, rng)
    ice_mask = ice_strength > 0.35

    # ---- rough-rock false positives (high CPR, but S>L, moderate DOP) ----
    rock_centers = [(50, 16), (14, 48)]
    rock_sigmas  = [4.0, 3.5]
    rock_amps    = [0.95, 0.85]
    rock_strength = _blobs(size, rock_centers, rock_sigmas, rock_amps, rng)
    rock_mask = (rock_strength > 0.4) & (~ice_mask)

    # ---- base fields ----
    cpr_l = 0.45 + 0.10 * rng.standard_normal((size, size))          # surface background
    dop   = 0.55 + 0.08 * rng.standard_normal((size, size))
    cpr_s = cpr_l.copy()

    # ice: CPR_L high, DOP low, CPR_S a bit lower than L (deeper L penetration)
    cpr_l = np.where(ice_mask, 1.25 + 0.55 * ice_strength + 0.08 * rng.standard_normal((size, size)), cpr_l)
    cpr_s = np.where(ice_mask, cpr_l - (0.20 + 0.15 * ice_strength), cpr_s)
    dop   = np.where(ice_mask, 0.07 + 0.03 * rng.standard_normal((size, size)), dop)

    # rough rock: CPR high in BOTH but S even higher than L; DOP moderate
    cpr_l = np.where(rock_mask, 1.20 + 0.40 * rock_strength + 0.08 * rng.standard_normal((size, size)), cpr_l)
    cpr_s = np.where(rock_mask, cpr_l + (0.25 + 0.20 * rock_strength), cpr_s)
    dop   = np.where(rock_mask, 0.28 + 0.05 * rng.standard_normal((size, size)), dop)

    cpr_l = np.clip(cpr_l, 0.05, None)
    cpr_s = np.clip(cpr_s, 0.05, None)
    dop   = np.clip(dop, 0.0, 1.0)

    # volume-scattering fraction proxy (third axis): high for ice, lower for rock/surface
    vol_frac = np.clip(0.25 + 0.60 * ice_strength - 0.15 * rock_strength
                       + 0.04 * rng.standard_normal((size, size)), 0, 1)

    # simple DEM: a crater bowl + roughness (metres), for slope constraints
    yy, xx = np.mgrid[0:size, 0:size].astype(float)
    r = np.sqrt((xx - size/2) ** 2 + (yy - size/2) ** 2)
    dem = 150.0 * (r / r.max()) ** 2 + 6.0 * rng.standard_normal((size, size))

    return {
        "cpr_l": cpr_l, "cpr_s": cpr_s, "dop": dop, "vol_frac": vol_frac,
        "dem": dem, "ice_mask": ice_mask, "rock_mask": rock_mask,
        "pixel_m": 25.0,   # DFSAR ~25 m/pixel
        "size": size,
    }
