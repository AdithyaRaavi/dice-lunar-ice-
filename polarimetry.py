"""
polarimetry.py
--------------
Reference implementations of the standard polarimetric parameters used for lunar
ice detection. In the real workflow these are computed by MIDAS (ISRO/SAC) or ESA
PolSARPro from the calibrated 2x2 complex scattering matrix / 3x3 covariance matrix
of each DFSAR pixel; the formulas below document exactly what those tools produce so
the rest of the pipeline (detection, mapping, planning) is transparent and auditable.

Conventions follow Raney et al. (m-chi decomposition) and the DFSAR literature
(Bhiravarasu et al. 2021; CPR>1 with DOP<0.13 as a subsurface-ice indicator).
"""
import numpy as np


def dop_from_stokes(S0, S1, S2, S3):
    """Degree of Polarization from the Stokes vector.
    DOP = sqrt(S1^2 + S2^2 + S3^2) / S0,  in [0, 1].
    Volume scattering (e.g. icy regolith) -> low DOP; smooth surface -> high DOP.
    """
    S0 = np.asarray(S0, float)
    num = np.sqrt(np.asarray(S1, float)**2 + np.asarray(S2, float)**2 + np.asarray(S3, float)**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(S0 > 0, num / S0, 0.0)
    return np.clip(m, 0.0, 1.0)


def cpr_from_circular(sigma_sc, sigma_oc):
    """Circular Polarization Ratio = same-sense / opposite-sense circular power.
    CPR > 1 can arise from the coherent backscatter opposition effect in low-loss
    ice OR from wavelength-scale roughness / dihedral scattering off rock — i.e. it
    is intentionally non-unique (that ambiguity is the problem DICE addresses).
    """
    sc = np.asarray(sigma_sc, float)
    oc = np.asarray(sigma_oc, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cpr = np.where(oc > 0, sc / oc, 0.0)
    return cpr


def mchi_decomposition(S0, S3, dop):
    """Raney m-chi decomposition into surface / double-bounce / volume powers.
    sin(2*chi) = -S3 / (m * S0); child powers (in intensity):
        Ps (surface)      = m*S0*(1 - sin2chi)/2
        Pd (double-bounce) = m*S0*(1 + sin2chi)/2
        Pv (volume)        = S0*(1 - m)
    A volume-dominant pixel (Pv large) is consistent with scattering inside an
    icy/blocky regolith rather than a clean surface or dihedral reflection.
    """
    S0 = np.asarray(S0, float)
    m = np.asarray(dop, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sin2chi = np.where(m * S0 > 0, -np.asarray(S3, float) / (m * S0), 0.0)
    sin2chi = np.clip(sin2chi, -1.0, 1.0)
    Ps = m * S0 * (1 - sin2chi) / 2.0
    Pd = m * S0 * (1 + sin2chi) / 2.0
    Pv = S0 * (1 - m)
    return Ps, Pd, Pv


def volume_fraction(Ps, Pd, Pv, eps=1e-9):
    """Fraction of total power in the volume channel — used as a third detection axis."""
    tot = Ps + Pd + Pv + eps
    return Pv / tot
