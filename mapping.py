"""
mapping.py
----------
Quantifies spatial heterogeneity of the ice-confidence field and represents it as a
Gaussian-Process random field, giving an estimate AND an uncertainty at every
location. The uncertainty is what the rover traverse later exploits.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


def empirical_variogram(prob, n_bins=12, max_lag=None, n_samples=1500, seed=0):
    """Binned empirical semivariogram of the confidence raster.
    Returns (lags, gamma). The lag where gamma plateaus ~ spatial correlation length
    (the characteristic ice-patch scale)."""
    rng = np.random.default_rng(seed)
    size = prob.shape[0]
    yy, xx = np.mgrid[0:size, 0:size]
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    vals = prob.ravel()
    idx = rng.choice(len(vals), size=min(n_samples, len(vals)), replace=False)
    c, v = coords[idx], vals[idx]
    if max_lag is None:
        max_lag = size / 2.0
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    sv = 0.5 * (v[:, None] - v[None, :]) ** 2
    bins = np.linspace(0, max_lag, n_bins + 1)
    lags, gamma = [], []
    for i in range(n_bins):
        m = (d >= bins[i]) & (d < bins[i + 1])
        if m.sum() > 0:
            lags.append(0.5 * (bins[i] + bins[i + 1]))
            gamma.append(sv[m].mean())
    return np.array(lags), np.array(gamma)


def fit_confidence_field(prob, n_train=350, downsample=2, seed=0):
    """Fit a GP to the confidence raster and predict mean+std on a coarse grid.
    Returns (grid_xy, mean_grid, std_grid, gp)."""
    rng = np.random.default_rng(seed)
    size = prob.shape[0]
    yy, xx = np.mgrid[0:size, 0:size]
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    vals = prob.ravel()

    idx = rng.choice(len(vals), size=min(n_train, len(vals)), replace=False)
    Xtr, ytr = coords[idx], vals[idx]

    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * RBF(length_scale=6.0, length_scale_bounds=(2.0, 25.0))
              + WhiteKernel(noise_level=0.02, noise_level_bounds=(1e-4, 0.5)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                  n_restarts_optimizer=1, random_state=seed)
    gp.fit(Xtr, ytr)

    gx = np.arange(0, size, downsample)
    GX, GY = np.meshgrid(gx, gx)
    grid = np.column_stack([GX.ravel(), GY.ravel()]).astype(float)
    mean, std = gp.predict(grid, return_std=True)
    shape = GX.shape
    return (GX, GY), mean.reshape(shape), std.reshape(shape), gp
