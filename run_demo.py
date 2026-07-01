"""
run_demo.py — end-to-end DICE pipeline on a synthetic scene with KNOWN ground truth.

This proves the pipeline works before real data is available. To run on REAL data,
replace `synth.generate_scene()` with MIDAS/PolSARPro-derived CPR_L, CPR_S, DOP and a
DEM (see README -> "Running on real PRADAN data"). Everything downstream is identical.

    python run_demo.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from dice import synth, detection, mapping, traverse, volume

# ---- brand palette (matches the BAH 2026 deck) ----
BLUE, ORANGE, NAVY, GRAY = "#1497D6", "#F2691E", "#0C1330", "#5C6678"


def main():
    sc = synth.generate_scene(size=64, seed=7)

    # 1) detection (CPR+DOP + volume axis + honest dual-frequency confound handling)
    prob, flags = detection.ice_probability(
        sc["cpr_l"], sc["dop"], vol_frac=sc["vol_frac"], cpr_s=sc["cpr_s"])
    metrics = detection.evaluate(prob, sc["ice_mask"], thr=0.5)

    # how many rough-rock false positives were correctly suppressed?
    rock = sc["rock_mask"]
    rock_flagged = float(np.mean(flags["confound"][rock])) if rock.sum() else 0.0

    # 2) heterogeneity field
    lags, gamma = mapping.empirical_variogram(prob)
    (GX, GY), gp_mean, gp_std, _ = mapping.fit_confidence_field(prob)

    # 3) information-maximizing traverse from a feasible landing point
    tr = traverse.info_max_traverse(prob, sc["dem"], sc["pixel_m"],
                                    start=(40, 52), n_steps=10,
                                    step_radius=8.0, slope_max_deg=20.0)

    # 4) uncertainty-bounded volume (top 0-5 m)
    vol = volume.monte_carlo_volume(prob, pixel_m=sc["pixel_m"], thr=0.5)

    # ---------- console summary ----------
    print("=" * 60)
    print("DICE pipeline — synthetic validation")
    print("=" * 60)
    print(f"Detection vs ground truth:  precision={metrics['precision']:.2f}  "
          f"recall={metrics['recall']:.2f}  F1={metrics['f1']:.2f}")
    print(f"Rough-rock false positives flagged as roughness confound: {rock_flagged*100:.0f}%")
    print(f"Detected ice cells: {vol['n_cells']}  ({sc['pixel_m']:.0f} m/pixel)")
    print(f"Ice volume (top 0-5 m): median={vol['median']/1e6:.2f} x10^6 m^3  "
          f"[90% CI {vol['p05']/1e6:.2f}-{vol['p95']/1e6:.2f}]")
    print(f"Traverse: {len(tr['path'])} waypoints; cumulative info gain "
          f"{np.sum(tr['info_profile']):.2f}")
    print("Note: SYNTHETIC data — figures validate the method, not a real crater.")

    # ---------- repo validation figure ----------
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].imshow(sc["ice_mask"], origin="lower", cmap="Greens")
    ax[0].imshow(np.ma.masked_where(~sc["rock_mask"], sc["rock_mask"]),
                 origin="lower", cmap=ListedColormap(["#999999"]), alpha=0.7)
    ax[0].set_title("Ground truth (green=ice, grey=rough rock)")
    im = ax[1].imshow(prob, origin="lower", cmap="YlOrRd", vmin=0, vmax=1)
    px = [p[0] for p in tr["path"]]; py = [p[1] for p in tr["path"]]
    ax[1].plot(px, py, "-o", color="#0B2E59", ms=4, lw=1.6)
    ax[1].plot(px[0], py[0], "*", color=BLUE, ms=16)
    ax[1].set_title(f"P(ice) + traverse  (P={metrics['precision']:.2f}, R={metrics['recall']:.2f})")
    plt.colorbar(im, ax=ax[1], fraction=0.046)
    ax[2].hist(vol["samples"] / 1e6, bins=40, color="#F58A4B", edgecolor="white")
    ax[2].axvline(vol["median"] / 1e6, color=ORANGE, lw=2)
    ax[2].set_title("Ice volume, top 0-5 m (x10^6 m^3)")
    ax[2].set_xlabel("volume")
    for a in ax[:2]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("DICE — validated on synthetic ground truth (SYNTHETIC data, not a real crater)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("figures/validation.png", dpi=150)
    plt.close(fig)

    # ---------- deck-styled figure ----------
    _deck_figure(sc, prob, flags, metrics, rock_flagged, vol, tr)
    print("Saved figures/validation.png and figures/deck_pipeline.png")


def _deck_figure(sc, prob, flags, metrics, rock_flagged, vol, tr):
    fig = plt.figure(figsize=(9.4, 4.0), dpi=200)
    fig.text(0.035, 0.95, "DICE pipeline \u2014 validated on synthetic ground truth", fontsize=12,
             color=NAVY, fontweight="bold")
    fig.text(0.035, 0.905, "real code output on data with known truth; runs unchanged on PRADAN DFSAR products",
             fontsize=8.3, color=GRAY, style="italic")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 1],
                          left=0.05, right=0.965, top=0.80, bottom=0.10, wspace=0.28, hspace=0.6)

    axA = fig.add_subplot(gs[:, 0])
    im = axA.imshow(prob, origin="lower", cmap="YlOrRd", vmin=0, vmax=1)
    # outline rough-rock zones that were correctly suppressed
    axA.contour(sc["rock_mask"].astype(float), levels=[0.5], colors=["#3A6FB0"], linewidths=1.2)
    px = [p[0] for p in tr["path"]]; py = [p[1] for p in tr["path"]]
    axA.plot(px, py, "-o", color="#0B2E59", ms=3.5, lw=1.6, zorder=4)
    axA.plot(px[0], py[0], "*", color=BLUE, ms=15, zorder=5)
    axA.text(px[0] + 1, py[0] - 4, "Landing", fontsize=7.5, color="#0B2E59", fontweight="bold")
    axA.set_title("P(subsurface ice) + info-max traverse", fontsize=9, color=NAVY, fontweight="bold")
    axA.set_xticks([]); axA.set_yticks([])
    cb = plt.colorbar(im, ax=axA, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6.5)
    axA.text(0.02, -0.07, "blue outline = rough-rock zone suppressed by dual-frequency check",
             transform=axA.transAxes, fontsize=6.6, color="#3A6FB0", style="italic")

    axB = fig.add_subplot(gs[0, 1]); axB.axis("off"); axB.set_xlim(0, 1); axB.set_ylim(0, 1)
    axB.set_title("Detection vs known truth", fontsize=8.5, color=NAVY, fontweight="bold")
    axB.text(0.0, 0.62, f"Precision  {metrics['precision']:.2f}", fontsize=10, color=NAVY, fontweight="bold")
    axB.text(0.0, 0.34, f"Recall       {metrics['recall']:.2f}", fontsize=10, color=NAVY, fontweight="bold")
    axB.text(0.0, 0.06, f"Roughness flagged  {rock_flagged*100:.0f}%", fontsize=8.5, color=ORANGE, fontweight="bold")

    axC = fig.add_subplot(gs[1, 1])
    axC.hist(vol["samples"] / 1e6, bins=36, color="#F58A4B", edgecolor="white")
    axC.axvline(vol["median"] / 1e6, color=ORANGE, lw=2)
    axC.set_title("Ice volume 0\u20135 m (Monte Carlo)", fontsize=8.3, color=NAVY, fontweight="bold")
    axC.set_xlabel("\u00d710\u2076 m\u00b3", fontsize=7.5); axC.set_yticks([])
    axC.tick_params(labelsize=6.5)
    fig.savefig("figures/deck_pipeline.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
