"""
io_dfsar.py
-----------
Load ANY crater's data into the pipeline. Two entry points:

1. load_crater(path) — point it at a folder (or .npz) holding the polarimetric
   parameter rasters that MIDAS / PolSARPro export:
        cpr_l*  (L-band CPR)   [required]
        dop*    (degree of polarization) [required]
        cpr_s*  (S-band CPR)   [optional, enables the dual-frequency confound check]
        vol_frac* (volume-scattering fraction) [optional]
        dem*    (DEM for slope) [optional]
   Files may be GeoTIFF (.tif, needs `rasterio`) or NumPy (.npy). This is the
   realistic "drop in a crater and run" path.

2. params_from_quadpol(Shh, Shv, Svh, Svv) — if instead you have CALIBRATED quad-pol
   complex channels as rasters, compute CPR / DOP / volume-fraction yourself. Raw or
   uncalibrated DFSAR data must be calibrated first (e.g. with MIDAS) before this step.
"""
import os
import glob
import numpy as np

try:
    import rasterio
    _HAS_RASTERIO = True
except Exception:
    _HAS_RASTERIO = False

# substrings used to recognise each parameter file by name (case-insensitive)
_ALIASES = {
    "cpr_l":    ["cpr_l", "cprl", "cpr_lband", "cpr"],   # a lone "cpr*" is treated as L-band
    "cpr_s":    ["cpr_s", "cprs", "cpr_sband"],
    "dop":      ["dop", "degree_of_pol"],
    "vol_frac": ["vol_frac", "volume_fraction", "volfrac", "pv_frac"],
    "dem":      ["dem", "elevation", "height", "lola"],
}


def _read_raster(path):
    low = path.lower()
    if low.endswith(".npy"):
        return np.load(path).astype(float)
    if low.endswith((".tif", ".tiff", ".img", ".vrt")):
        if not _HAS_RASTERIO:
            raise RuntimeError(f"Reading {path} needs rasterio. `pip install rasterio`, "
                               f"or export the rasters as .npy.")
        with rasterio.open(path) as ds:
            return ds.read(1).astype(float)
    raise ValueError(f"Unsupported raster format: {path}")


def load_crater(path, pixel_m=25.0, name=None):
    """Return a scene dict the pipeline understands, loaded from a folder or .npz."""
    scene = {"pixel_m": float(pixel_m), "name": name or os.path.basename(str(path).rstrip("/"))}

    if os.path.isfile(path) and path.lower().endswith(".npz"):
        z = np.load(path)
        for k in ("cpr_l", "cpr_s", "dop", "vol_frac", "dem"):
            if k in z:
                scene[k] = z[k].astype(float)
    elif os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*")))
        for key, aliases in _ALIASES.items():
            if key in scene:
                continue
            for f in files:
                stem = os.path.splitext(os.path.basename(f).lower())[0]
                if any(a in stem for a in aliases):
                    try:
                        scene[key] = _read_raster(f)
                        break
                    except Exception:
                        pass
    else:
        raise FileNotFoundError(f"{path} is not a folder or .npz file")

    if "cpr_l" not in scene or "dop" not in scene:
        raise ValueError(
            "Need at least an L-band CPR raster and a DOP raster. "
            f"Found: {sorted(k for k in scene if isinstance(scene.get(k), np.ndarray))}")

    ref = scene["cpr_l"]
    scene["size"] = ref.shape[0]
    if "dem" not in scene:
        scene["dem"] = np.zeros_like(ref)        # flat DEM -> no slope constraint
    return scene


def params_from_quadpol(Shh, Shv, Svh, Svv, eps=1e-12):
    """Compute (CPR, DOP, volume_fraction) from CALIBRATED quad-pol complex channels.

    Uses the RH-transmit hybrid (compact-pol) child Stokes formulation (Raney et al.).
    Assumes reciprocity (Shv == Svh) and calibrated data. CONVENTION-SENSITIVE — cross-
    check against MIDAS on a known scene before trusting absolute values.
    """
    Shh = np.asarray(Shh, complex); Shv = np.asarray(Shv, complex)
    Svh = np.asarray(Svh, complex); Svv = np.asarray(Svv, complex)

    inv = 1.0 / np.sqrt(2.0)
    E_H = inv * (Shh - 1j * Shv)        # received H,V fields for RH-circular transmit
    E_V = inv * (Svh - 1j * Svv)

    g0 = np.abs(E_H) ** 2 + np.abs(E_V) ** 2 + eps
    g1 = np.abs(E_H) ** 2 - np.abs(E_V) ** 2
    g2 = 2.0 * np.real(E_H * np.conj(E_V))
    g3 = -2.0 * np.imag(E_H * np.conj(E_V))

    dop = np.clip(np.sqrt(g1 ** 2 + g2 ** 2 + g3 ** 2) / g0, 0.0, 1.0)
    cpr = np.clip((g0 - g3) / (g0 + g3 + eps), 0.0, None)

    m = np.clip(dop, eps, 1.0)
    sin2chi = np.clip(-g3 / (m * g0), -1.0, 1.0)
    Pv = g0 * (1 - m)
    Ps = m * g0 * (1 - sin2chi) / 2.0
    Pd = m * g0 * (1 + sin2chi) / 2.0
    vol_frac = np.clip(Pv / (Ps + Pd + Pv + eps), 0.0, 1.0)
    return cpr, dop, vol_frac


def save_scene_npz(path, scene):
    """Convenience: persist a scene's rasters to a .npz (for re-runs / sharing)."""
    keys = [k for k in ("cpr_l", "cpr_s", "dop", "vol_frac", "dem") if k in scene]
    np.savez_compressed(path, **{k: scene[k] for k in keys})
