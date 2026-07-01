"""
traverse.py
-----------
Information-maximizing rover traverse (the CS/decision contribution).

Instead of a shortest path to the single highest-confidence pixel, the rover visits
the locations that most reduce uncertainty about the whole ice field. At each step it
picks, among reachable + slope-feasible candidates, the one with the highest GP
predictive standard deviation (most informative), "observes" it, refits the GP, and
repeats. This maps heterogeneity efficiently rather than chasing one guess.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


def _slope_deg(dem, pixel_m):
    gy, gx = np.gradient(dem, pixel_m)
    return np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))


def info_max_traverse(prob, dem, pixel_m, start, n_steps=10,
                      step_radius=8.0, slope_max_deg=20.0, seed=0):
    """Greedy active-sensing traverse.

    Returns dict: path [(x,y)...], info_profile (GP variance reduced per step),
    slope_deg raster, and the visited measurement values.
    """
    rng = np.random.default_rng(seed)
    size = prob.shape[0]
    slope = _slope_deg(dem, pixel_m)

    yy, xx = np.mgrid[0:size, 0:size]
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    vals = prob.ravel()

    # seed GP with a sparse prior sample so it has a baseline field
    idx = rng.choice(len(vals), size=200, replace=False)
    Xtr = list(coords[idx]); ytr = list(vals[idx])

    kernel = (ConstantKernel(1.0) * RBF(length_scale=6.0, length_scale_bounds=(2.0, 25.0))
              + WhiteKernel(noise_level=0.02))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed)
    gp.fit(np.array(Xtr), np.array(ytr))

    path = [tuple(map(float, start))]
    info_profile = []
    cur = np.array(start, float)

    for _ in range(n_steps):
        d = np.sqrt(((coords - cur) ** 2).sum(1))
        reachable = (d > 0) & (d <= step_radius)
        feasible = reachable & (slope.ravel() <= slope_max_deg)
        cand = coords[feasible]
        if len(cand) == 0:
            break
        _, std = gp.predict(cand, return_std=True)
        total_unc_before = float(std.sum())
        nxt = cand[int(np.argmax(std))]

        # "observe" the chosen location (uses the true field value in this demo)
        j = int(np.argmin(((coords - nxt) ** 2).sum(1)))
        Xtr.append(coords[j]); ytr.append(vals[j])
        gp.fit(np.array(Xtr), np.array(ytr))

        _, std_after = gp.predict(cand, return_std=True)
        info_profile.append(total_unc_before - float(std_after.sum()))
        path.append(tuple(map(float, nxt)))
        cur = nxt

    return {"path": path, "info_profile": info_profile, "slope_deg": slope}
