"""
pipeline.py
-----------
Run the full DICE workflow on ANY crater scene and write a per-crater result folder:

    <out>/<crater>/
        report.png              -- one-glance summary panel
        ice_probability.png     -- P(subsurface ice) map (+ traverse)
        confidence_field.png    -- GP mean + uncertainty
        volume_hist.png         -- Monte-Carlo volume distribution
        ice_probability.tif     -- GeoTIFF (only if rasterio available)
        summary.json            -- landing site, traverse, volume CI, (metrics if truth)

`scene` is the dict returned by io_dfsar.load_crater() or synth.generate_scene().
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from . import detection, mapping, traverse, volume

BLUE, ORANGE, NAVY, GRAY = "#1497D6", "#F2691E", "#0C1330", "#5C6678"


def _auto_landing(prob, dem, pixel_m, slope_max_deg=15.0):
    """Pick a safe, flat spot near high ice-confidence: maximise smoothed confidence
    minus a slope penalty, over slope-feasible cells."""
    slope = traverse._slope_deg(dem, pixel_m)
    conf = gaussian_filter(prob, sigma=3)
    feasible = slope <= slope_max_deg
    smax = slope.max() if slope.max() > 0 else 1.0
    score = np.where(feasible, conf - 0.25 * (slope / smax), -1.0)
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    return (float(ix), float(iy))


def run_pipeline(scene, out_dir="results", landing=None, n_steps=12,
                 step_radius=8.0, slope_max_deg=20.0, verbose=True):
    name = scene.get("name", "crater")
    cdir = os.path.join(out_dir, name)
    os.makedirs(cdir, exist_ok=True)

    cpr_l = scene["cpr_l"]; dop = scene["dop"]
    cpr_s = scene.get("cpr_s"); vol_frac = scene.get("vol_frac")
    dem = scene.get("dem", np.zeros_like(cpr_l))
    pixel_m = scene.get("pixel_m", 25.0)

    # 1) detection
    prob, flags = detection.ice_probability(cpr_l, dop, vol_frac=vol_frac, cpr_s=cpr_s)

    # 2) heterogeneity field
    lags, gamma = mapping.empirical_variogram(prob)
    (GX, GY), gp_mean, gp_std, _ = mapping.fit_confidence_field(prob)

    # 3) landing + information-maximizing traverse
    if landing is None:
        landing = _auto_landing(prob, dem, pixel_m, slope_max_deg=min(slope_max_deg, 15.0))
    tr = traverse.info_max_traverse(prob, dem, pixel_m, start=landing, n_steps=n_steps,
                                    step_radius=step_radius, slope_max_deg=slope_max_deg)

    # 4) volume
    vol = volume.monte_carlo_volume(prob, pixel_m=pixel_m, thr=0.5)

    # optional validation against known truth (synthetic scenes)
    metrics = None
    if "ice_mask" in scene:
        metrics = detection.evaluate(prob, scene["ice_mask"], thr=0.5)
        if "rock_mask" in scene and scene["rock_mask"].sum():
            metrics["rock_confound_flagged"] = float(np.mean(flags["confound"][scene["rock_mask"]]))

    # ---- write maps ----
    _save_prob_map(os.path.join(cdir, "ice_probability.png"), prob, tr, name)
    _save_field(os.path.join(cdir, "confidence_field.png"), gp_mean, gp_std)
    _save_volume(os.path.join(cdir, "volume_hist.png"), vol)
    _save_report(os.path.join(cdir, "report.png"), prob, flags, tr, vol, metrics, name, cpr_s is not None)
    _maybe_geotiff(os.path.join(cdir, "ice_probability.tif"), prob)

    summary = {
        "crater": name,
        "pixel_m": pixel_m,
        "landing_site_xy": [round(landing[0], 1), round(landing[1], 1)],
        "traverse_waypoints": [[round(x, 1), round(y, 1)] for x, y in tr["path"]],
        "cumulative_info_gain": round(float(np.sum(tr["info_profile"])), 3),
        "detected_ice_cells": vol["n_cells"],
        "ice_volume_top5m_m3": {
            "median": round(vol["median"], 1),
            "p05": round(vol["p05"], 1),
            "p95": round(vol["p95"], 1),
        },
        "dual_frequency_used": cpr_s is not None,
        "validation_metrics": metrics,
        "note": "Synthetic scene" if "ice_mask" in scene else "Real/loaded scene",
    }
    with open(os.path.join(cdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"[{name}] detected {vol['n_cells']} ice cells; "
              f"volume {vol['median']/1e6:.2f} x10^6 m^3 "
              f"[{vol['p05']/1e6:.2f}-{vol['p95']/1e6:.2f}]; "
              f"landing {summary['landing_site_xy']}; "
              f"{len(tr['path'])} waypoints. -> {cdir}/")
        if metrics:
            print(f"        validation: precision={metrics['precision']:.2f} "
                  f"recall={metrics['recall']:.2f} "
                  f"rock-confound-flagged={metrics.get('rock_confound_flagged', 0)*100:.0f}%")
    return summary


# ----------------------------- figures -----------------------------
def _save_prob_map(path, prob, tr, name):
    fig, ax = plt.subplots(figsize=(5, 4.4), dpi=150)
    im = ax.imshow(prob, origin="lower", cmap="YlOrRd", vmin=0, vmax=1)
    px = [p[0] for p in tr["path"]]; py = [p[1] for p in tr["path"]]
    ax.plot(px, py, "-o", color="#0B2E59", ms=4, lw=1.6)
    ax.plot(px[0], py[0], "*", color=BLUE, ms=16)
    ax.set_title(f"{name}: P(subsurface ice) + traverse", fontsize=10, color=NAVY)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _save_field(path, mean, std):
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6), dpi=150)
    a = ax[0].imshow(mean, origin="lower", cmap="YlOrRd", vmin=0, vmax=1)
    ax[0].set_title("GP confidence (mean)"); plt.colorbar(a, ax=ax[0], fraction=0.046)
    b = ax[1].imshow(std, origin="lower", cmap="viridis")
    ax[1].set_title("GP uncertainty (std)"); plt.colorbar(b, ax=ax[1], fraction=0.046)
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _save_volume(path, vol):
    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=150)
    ax.hist(vol["samples"] / 1e6, bins=40, color="#F58A4B", edgecolor="white")
    ax.axvline(vol["median"] / 1e6, color=ORANGE, lw=2)
    ax.set_title("Ice volume, top 0-5 m"); ax.set_xlabel(r"$\times10^6$ m$^3$")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _save_report(path, prob, flags, tr, vol, metrics, name, dual):
    fig = plt.figure(figsize=(9.4, 4.0), dpi=200)
    fig.text(0.035, 0.95, f"DICE result \u2014 {name}", fontsize=12, color=NAVY, fontweight="bold")
    sub = "real/loaded scene" if metrics is None else "synthetic scene with known truth"
    fig.text(0.035, 0.905, f"dual-frequency: {'on' if dual else 'off (L-band only)'}   \u2022   {sub}",
             fontsize=8.3, color=GRAY, style="italic")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 1],
                          left=0.05, right=0.965, top=0.80, bottom=0.10, wspace=0.28, hspace=0.6)
    axA = fig.add_subplot(gs[:, 0])
    im = axA.imshow(prob, origin="lower", cmap="YlOrRd", vmin=0, vmax=1)
    px = [p[0] for p in tr["path"]]; py = [p[1] for p in tr["path"]]
    axA.plot(px, py, "-o", color="#0B2E59", ms=3.5, lw=1.6, zorder=4)
    axA.plot(px[0], py[0], "*", color=BLUE, ms=15, zorder=5)
    axA.text(px[0] + 1, py[0] - 4, "Landing", fontsize=7.5, color="#0B2E59", fontweight="bold")
    axA.set_title("P(subsurface ice) + info-max traverse", fontsize=9, color=NAVY, fontweight="bold")
    axA.set_xticks([]); axA.set_yticks([])
    cb = plt.colorbar(im, ax=axA, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6.5)

    axB = fig.add_subplot(gs[0, 1]); axB.axis("off"); axB.set_xlim(0, 1); axB.set_ylim(0, 1)
    if metrics:
        axB.set_title("Detection vs known truth", fontsize=8.5, color=NAVY, fontweight="bold")
        axB.text(0, 0.6, f"Precision  {metrics['precision']:.2f}", fontsize=10, color=NAVY, fontweight="bold")
        axB.text(0, 0.32, f"Recall       {metrics['recall']:.2f}", fontsize=10, color=NAVY, fontweight="bold")
        if "rock_confound_flagged" in metrics:
            axB.text(0, 0.05, f"Roughness flagged  {metrics['rock_confound_flagged']*100:.0f}%",
                     fontsize=8.5, color=ORANGE, fontweight="bold")
    else:
        axB.set_title("Detected", fontsize=8.5, color=NAVY, fontweight="bold")
        axB.text(0, 0.55, f"{vol['n_cells']} ice cells", fontsize=10, color=NAVY, fontweight="bold")
        axB.text(0, 0.25, f"{len(tr['path'])} waypoints", fontsize=9, color=GRAY, fontweight="bold")

    axC = fig.add_subplot(gs[1, 1])
    axC.hist(vol["samples"] / 1e6, bins=36, color="#F58A4B", edgecolor="white")
    axC.axvline(vol["median"] / 1e6, color=ORANGE, lw=2)
    axC.set_title("Ice volume 0-5 m (Monte Carlo)", fontsize=8.3, color=NAVY, fontweight="bold")
    axC.set_xlabel(r"$\times10^6$ m$^3$", fontsize=7.5); axC.set_yticks([]); axC.tick_params(labelsize=6.5)
    fig.savefig(path); plt.close(fig)


def _maybe_geotiff(path, arr):
    try:
        import rasterio
        from rasterio.transform import from_origin
        h, w = arr.shape
        with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                           dtype="float32", transform=from_origin(0, h, 1, 1)) as ds:
            ds.write(arr.astype("float32"), 1)
    except Exception:
        pass  # GeoTIFF is optional; PNGs + JSON are always written
