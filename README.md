# DICE — Dual-frequency Ice Confidence-to-Exploration

A reproducible pipeline that turns **Chandrayaan-2 DFSAR** polarimetric products into a
**probabilistic subsurface-ice map** and the downstream exploration decisions ISRO's
problem statement asks for: a landing site, an optimized rover traverse, and an ice-volume
estimate for the top 0–5 m.

Built for the **Bharatiya Antariksh Hackathon (BAH) 2026** concept phase.

---

## ⚠️ What this is — and what it is not

- ✅ It **is** a working, end-to-end implementation of the method, validated on synthetic
  data with **known ground truth** (so you can see it actually detects, suppresses
  false positives, plans, and quantifies — not just describes).
- ✅ It **is** designed to run **unchanged** on real CPR/DOP rasters once you process
  DFSAR data with MIDAS or PolSARPro (see below).
- ❌ It is **not** a real-crater result. The bundled demo uses **synthetic** data
  (`dice/synth.py`). Do not present its figures as real detections.
- ❌ It does **not** claim novelty over the published DFSAR science. The detection
  criterion (CPR > 1 with DOP < 0.13) is ISRO's recent result; our contribution is the
  **honest dual-frequency handling** and the **CS/ML decision-and-uncertainty layer**
  (Gaussian-Process field, information-maximizing traverse, Monte-Carlo volume).

---

## The idea in one paragraph

Radar backscatter is **non-unique**: high CPR can mean buried ice *or* rough rock. DICE
does not chase a magic threshold. It (1) computes the published **CPR + DOP** detection as
a **probability**, (2) adds a **volume-scattering** axis, (3) uses **dual-frequency** as a
**roughness confound check** — down-weighting pixels where S-band CPR exceeds L-band, since
the DFSAR team observed exactly that over rocky terrain (Bhiravarasu et al. 2021) — and then
(4) carries the resulting **uncertainty** into a Gaussian-Process ice field, an
**information-maximizing** rover traverse, and a Monte-Carlo **volume** estimate with error
bars.

---

## Pipeline

| Module | What it does |
|---|---|
| `dice/polarimetry.py` | Reference formulas for CPR, DOP, and the m-χ decomposition (what MIDAS/PolSARPro produce). |
| `dice/synth.py` | Synthetic scene with known ice + rough-rock + surface (for validation). |
| `dice/detection.py` | CPR+DOP probability, volume axis, and honest dual-frequency confound check. |
| `dice/mapping.py` | Empirical variogram + Gaussian-Process confidence field (estimate **and** uncertainty). |
| `dice/traverse.py` | Information-maximizing rover traverse via GP active sensing, under slope limits. |
| `dice/volume.py` | Monte-Carlo, uncertainty-bounded ice volume in the top 0–5 m. |
| `dice/io_dfsar.py` | Load any crater from a folder / `.npz` of parameter rasters; quad-pol → CPR/DOP. |
| `dice/pipeline.py` | Orchestrates the run and writes the per-crater report folder. |
| `dice/__main__.py` | Command-line interface (`python -m dice demo` / `run`). |

---

## Quickstart

```bash
pip install -r requirements.txt

# 1) validated synthetic demo (data with known ground truth)
python -m dice demo

# 2) generate a SAMPLE crater (GeoTIFFs) and run the tool on it
python examples/make_sample_crater.py
python -m dice run --crater examples/sample_crater --name SampleCrater
```

Each run writes a per-crater folder under `results/<name>/`:
`report.png`, `ice_probability.png` (+ a `.tif`), `confidence_field.png`,
`volume_hist.png`, and `summary.json` (landing site, traverse, volume CI, and
detection metrics when ground truth exists).

### Example output (synthetic validation)
```
[demo_synthetic] detected 343 ice cells; volume 0.13 x10^6 m^3 [0.09-0.18]; ...
        validation: precision=1.00 recall=0.96 rock-confound-flagged=100%
```
(The seed is fixed in `dice/synth.py`, so your numbers will match.)

---

## Running on YOUR crater

You don't need to send the data anywhere. Two steps:

1. **Process the DFSAR product to parameter rasters** with **MIDAS** (ISRO/SAC) or
   **ESA PolSARPro**: export `cpr_l` and `dop` (required) and, if available, `cpr_s`,
   `vol_frac`, and a `dem`, as GeoTIFFs.
2. **Drop them in a folder and run** — file names just need to contain the tokens
   `cpr_l`, `cpr_s`, `dop`, `vol_frac`, `dem`:

```bash
python -m dice run --crater path/to/your_crater_folder --name Faustini --pixel-m 25
# optional: --landing X Y   --steps 12   --out results
```

See `examples/sample_crater/` for the exact expected layout (the bundled example uses
synthetic data). The same pipeline runs on any crater you point it at.

If instead you have **calibrated quad-pol complex channels**, compute the parameters
yourself with `io_dfsar.params_from_quadpol(Shh, Shv, Svh, Svv)` (convention-sensitive —
cross-check against MIDAS on a known scene). **Raw / uncalibrated SLC must be calibrated
in MIDAS first** — turning complex SLC into CPR/DOP is a full polarimetric-calibration
job, not done here.

---

## Honest limitations

- **Detection is probabilistic.** Radar ice detection is inherently non-unique; this
  reduces ambiguity, it does not eliminate it. Outputs are probabilities with uncertainty.
- **S-band coverage is the limiting factor.** Where S-band is unavailable or unreliable, the
  pipeline degrades gracefully to the L-band CPR+DOP criterion and flags lower confidence.
- **No ground truth on the Moon.** Validate against independent datasets (Diviner thermal,
  LEND/neutron hydrogen, published DFSAR results), not direct confirmation.
- **Depth/volume depend on local loss tangent**, which varies with regolith composition;
  the bracket carries that uncertainty.

---

## References

- Bhiravarasu et al. (2021), *Chandrayaan-2 DFSAR: Performance Characterization and Initial Results*, PSJ (arXiv:2104.14259).
- *Exploring water-ice deposits in lunar polar craters with Chandrayaan-2 DFSAR data*, Icarus (2025).
- ISRO (2025–2026), DFSAR subsurface-ice detection using CPR > 1 with DOP < 0.13.
- Raney et al., m-χ decomposition / hybrid polarimetry.

*Student project for BAH 2026. Verify all citations and results before formal submission.*
