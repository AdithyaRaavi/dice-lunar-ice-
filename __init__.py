"""
DICE — Dual-frequency Ice Confidence-to-Exploration
A reproducible pipeline that turns Chandrayaan-2 DFSAR polarimetric products into
a probabilistic subsurface-ice map and downstream exploration decisions.

Modules
-------
polarimetry : reference implementations of the standard PolSAR parameters
              (CPR, DOP, m-chi decomposition) computed from calibrated Stokes data.
synth       : synthetic-scene generator with known ground truth (for validation).
detection   : CPR+DOP detection with honest dual-frequency confound handling.
mapping     : variogram + Gaussian-Process ice-confidence field (with uncertainty).
traverse    : information-maximizing rover traverse (GP active sensing).
volume      : Monte-Carlo, uncertainty-bounded ice-volume estimate (top 0-5 m).
"""
__version__ = "0.1.0"
