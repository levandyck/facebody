"""Axis helpers, legends, and bootstrap and within-subject CI functions shared by figures."""

import numpy as np
from scipy.stats import t as t_dist
import matplotlib.lines as mlines

from facebody.myutils.utils import clean_axes

def _load_nc_by_roi(subjects: list):
    """Per-voxel noise ceilings per ROI."""
    from facebody.encoding.summarize import load_nc_by_roi
    return load_nc_by_roi(subjects)

def _voxelwise(model_name: str, layers: list, subjects: list, rois: list, analysis: str):
    """Filtering cortex-wide fit to ROIs."""
    from facebody.encoding.summarize import voxelwise_to_roi
    rois_flat = [r for g in rois for r in g] if rois and isinstance(rois[0], list) else list(rois)
    if not subjects:
        return {}
    return {layer: voxelwise_to_roi(model_name, layer, subjects, rois_flat, analysis)
            for layer in layers}

def _depth_axis(ax, ylim: tuple, ylabel: str=None):
    """The 0-100 % layer-depth x-axis shared by every prevalence panel."""
    ax.set_xlim(-5, 105)
    ax.set_xticks(np.linspace(0, 100, 5))
    ax.set_xticklabels([f"{v:.0f}" for v in np.linspace(0, 100, 5)])
    ax.set_xlabel("Layer depth (%)")
    ax.set_ylim(*ylim)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    clean_axes(ax)

def _swatch_legend(ax, colors: list, labels: list, title: str=None, lw: float=2.5,
                   marker: str=None, align: str="left"):
    """Swatch legend placed outside the axes, on the right: lines, or marker squares."""
    handles = [mlines.Line2D([], [], marker=marker, linestyle="none", color=c)
               if marker else mlines.Line2D([], [], color=c, lw=lw)
               for c in colors]
    leg = ax.legend(
        handles,
        labels,
        title=title,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )
    if align:
        leg._legend_box.align = align
    return leg


# --------------------------------- Encoding --------------------------------- #
def _roi_x_positions(roi_groups: list, gap: float):
    """Bar x positions, one unit apart, with `gap` inserted between ROI groups."""
    x_pos, cursor = [], 0.0
    for g, group in enumerate(roi_groups):
        for _ in group:
            x_pos.append(cursor)
            cursor += 1.0
        if g < len(roi_groups) - 1:
            cursor += gap
    return np.array(x_pos)

def _nc_band(ax, nc_vals: np.ndarray, x_span, zorder: int=None):
    """Group-mean noise ceiling with its bootstrap CI, as a grey band."""
    nc_mean, nc_ci = compute_mean_ci(np.asarray(nc_vals))
    kw = {} if zorder is None else {"zorder": zorder}
    ax.fill_between(x_span, nc_mean - nc_ci, nc_mean + nc_ci,
                    color="gray", alpha=0.2, lw=0, **kw)
    ax.hlines(nc_mean, x_span[0], x_span[-1], color="gray", ls="--", lw=1, **kw)


# ------------------------------ Ventral pathway ----------------------------- #
def _mean_err(arr: np.ndarray, error: str="sem"):
    """Mean and SEM (or SD) across subjects."""
    n = np.isfinite(arr).sum(axis=0)
    mean_vec = np.full(arr.shape[1], np.nan)
    err_vec = np.full(arr.shape[1], np.nan)

    ok = n >= 2
    if ok.any():
        mean_vec[ok] = np.nanmean(arr[:, ok], axis=0)
        sd_vec = np.nanstd(arr[:, ok], axis=0, ddof=1)
        err_vec[ok] = sd_vec / np.sqrt(n[ok]) if error == "sem" else sd_vec
    return mean_vec, err_vec


# ----------------------------------- Utils ---------------------------------- #
def compute_mean_ci(data: np.ndarray, n_iter: int=10000, seed: int=0):
    """Mean and half-width of a between-subject percentile bootstrap CI."""
    if len(data) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = len(data)
    boot_means = np.array([
        np.mean(data[rng.choice(n, size=n, replace=True)], axis=0)
        for _ in range(n_iter)
    ])
    lower = np.percentile(boot_means, 2.5, axis=0)
    upper = np.percentile(boot_means, 97.5, axis=0)
    ci = (upper - lower) / 2

    return np.mean(data, axis=0), ci

def compute_mean_ci_hierarch_models_layers_units(models_data: dict, n_iter: int=10000,
                                                 rng: np.random.Generator=None,
                                                 seed: int=0):
    """Hierarchical bootstrap: models → layers → units."""
    if rng is None:
        rng = np.random.default_rng(seed)

    model_names = [m for m in models_data if models_data[m]]
    if not model_names:
        return np.nan, np.nan, np.nan

    # Precompute a flat list of (layer_data_arrays) per model for speed
    all_data_flat = np.concatenate([
        arr
        for layers in models_data.values()
        for arr in layers.values()
        if len(arr) > 0
    ])
    grand_mean = float(np.nanmean(all_data_flat))

    boot_means = []
    n_models = len(model_names)

    for _ in range(n_iter):
        # Level 1: Resample models with replacement
        sampled_models = rng.choice(n_models, size=n_models, replace=True)
        iteration_vals = []

        for mi in sampled_models:
            model = model_names[mi]
            layer_dict = models_data[model]
            layer_names = [l for l in layer_dict if len(layer_dict[l]) > 0]
            if not layer_names:
                continue

            # Level 2: Resample layers with replacement
            n_layers = len(layer_names)
            sampled_layers = rng.choice(n_layers, size=n_layers, replace=True)

            for li in sampled_layers:
                units = layer_dict[layer_names[li]]

                # Level 3: Resample units with replacement
                boot_units = rng.choice(units, size=len(units), replace=True)
                iteration_vals.append(boot_units)

        if iteration_vals:
            boot_means.append(np.mean(np.concatenate(iteration_vals)))

    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    return grand_mean, ci_lo, ci_hi

def compute_mean_ci_within(X: np.ndarray, alpha: float=0.05):
    """Condition means and Cousineau-Morey within-subject CI half-widths."""
    if X.size == 0 or np.all(np.isnan(X)):
        C = X.shape[1] if X.ndim == 2 else 0
        return (np.full(C, np.nan), np.full(C, np.nan))

    S, C = X.shape
    means = np.nanmean(X, axis=0)

    # Remove between-subject offset (Cousineau normalization)
    grand = np.nanmean(X)
    subj_means = np.nanmean(X, axis=1, keepdims=True)
    X_norm = X - subj_means + grand

    # Standard error from normalized data
    sd = np.nanstd(X_norm, axis=0, ddof=1)
    se = sd / np.sqrt(S)

    # Morey correction
    cf = np.sqrt(C / (C - 1)) if C > 1 else 1.0
    tcrit = t_dist.ppf(1 - alpha / 2, df=max(S - 1, 1))
    ci_half = tcrit * se * cf
    return means, ci_half
