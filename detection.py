"""
detection.py
------------
Turns CPR / DOP / volume-fraction rasters into a calibrated subsurface-ice
probability, with HONEST dual-frequency handling.

Primary criterion (state of the art; ISRO 2025-2026): a pixel is an ice candidate
where CPR > 1 AND DOP < 0.13. We express this as a smooth probability rather than a
hard mask, and add a volume-scattering axis.

Dual-frequency is used as a CONFOUND CHECK, not a naive "delta CPR > 0 = ice" rule.
The DFSAR team observed S-band CPR HIGHER than L-band in rough/rocky regions
(Bhiravarasu et al. 2021); we therefore DOWN-WEIGHT candidates where S-band CPR
markedly exceeds L-band (roughness signature) and flag them, instead of pretending
the L-S difference is a clean ice discriminator.
"""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ice_probability(cpr_l, dop, vol_frac=None, cpr_s=None,
                    cpr_thresh=1.0, dop_thresh=0.13,
                    k_cpr=6.0, k_dop=40.0, confound_tau=0.15):
    """Return (probability, flags) dicts of rasters.

    probability : float array in [0,1], P(subsurface ice) per pixel.
    flags       : dict with boolean 'confound' (S>>L roughness) and 'core' (passes CPR+DOP).
    """
    cpr_l = np.asarray(cpr_l, float)
    dop = np.asarray(dop, float)

    p_cpr = _sigmoid(k_cpr * (cpr_l - cpr_thresh))      # high when CPR_L > 1
    p_dop = _sigmoid(k_dop * (dop_thresh - dop))        # high when DOP < 0.13
    p = p_cpr * p_dop

    if vol_frac is not None:                            # third axis: volume dominance
        p = p * _sigmoid(8.0 * (np.asarray(vol_frac, float) - 0.35))

    confound = np.zeros_like(p, dtype=bool)
    if cpr_s is not None:
        cpr_s = np.asarray(cpr_s, float)
        delta = cpr_l - cpr_s
        confound = (cpr_s - cpr_l) > confound_tau       # S notably higher => roughness
        # down-weight confounded pixels; mild up-weight where L>=S (consistent w/ depth)
        factor = np.where(confound, 0.35, np.clip(1.0 + 0.25 * np.tanh(2.0 * delta), 0.7, 1.25))
        p = np.clip(p * factor, 0.0, 1.0)

    core = (cpr_l > cpr_thresh) & (dop < dop_thresh)
    return np.clip(p, 0.0, 1.0), {"confound": confound, "core": core}


def evaluate(prob, truth_mask, thr=0.5):
    """Precision / recall / F1 of detection vs known ground truth."""
    pred = prob >= thr
    tp = int(np.sum(pred & truth_mask))
    fp = int(np.sum(pred & ~truth_mask))
    fn = int(np.sum(~pred & truth_mask))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
