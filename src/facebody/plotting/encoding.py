"""Bar and line figures for encoding results, per ROI and across layers."""

from pathlib import Path
import numpy as np
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from facebody.config import FIG_ROOT
from facebody.style import SEL_INFO, MIXED_COLOR, SHARED_COLOR
from .common import (
    _load_nc_by_roi, _nc_band, _roi_x_positions, _swatch_legend, _voxelwise,
    compute_mean_ci, compute_mean_ci_within
)
from facebody.myutils.utils import clean_axes

def plot_encoding_sep(model_name: str, layer: str,
                      subjects: list, rois: list,
                      r2_adj: bool=True, ylim: tuple=(0, None),
                      gap: float=0.5, fig_title: str=None,
                      out_path: Path=None):
    """Bar plot: Mean R2 of unit types per ROI."""
    if rois and not isinstance(rois[0], list):
        rois = [rois]
    rois_flat = [roi for group in rois for roi in group]

    sel_types = list(SEL_INFO.keys())
    sel_keys = ["f", "b", "m", "ns"]
    sel_key_map = dict(zip(sel_types, sel_keys))

    # Load R2 and noise ceilings
    res = _voxelwise(model_name, [layer], subjects, rois_flat, "sep")[layer]
    fmri = _load_nc_by_roi(subjects)

    # Collect per-subject ROI means
    scores = {roi: {sel: [] for sel in sel_types} for roi in rois_flat}
    nc_by_roi = {roi: [] for roi in rois_flat}
    n_vox = {roi: [] for roi in rois_flat}

    for subj in subjects:
        subj_res = res.get(subj, {})
        subj_fmri = fmri.get(subj, {})

        for roi in rois_flat:
            comp = subj_res.get(roi)
            fmri_roi = subj_fmri.get(roi)
            if not comp or fmri_roi is None:
                continue
            r2_vals = comp["r2"]
            n = len(r2_vals[sel_keys[0]])
            if n == 0:
                continue

            nc_mean = np.nanmean(fmri_roi["ncsnr"])
            nc_by_roi[roi].append(nc_mean)
            n_vox[roi].append(n)
            for sel in sel_types:
                r2_mean = np.nanmean(r2_vals[sel_key_map[sel]])
                scores[roi][sel].append(r2_mean / nc_mean if r2_adj else r2_mean)

    print(f"{model_name} | {layer} | n={len(subjects)} subjects | mean voxels: "
          + ", ".join(f"{r}={int(np.mean(n_vox[r]))}" for r in rois_flat if n_vox[r]))

    # Group means with a between-subject percentile bootstrap CI (half-width)
    ci_data = {}
    for roi in rois_flat:
        if not scores[roi][sel_types[0]]:
            ci_data[roi] = (np.full(len(sel_types), np.nan),) * 2
            continue
        data = np.stack([scores[roi][sel] for sel in sel_types], axis=1)
        valid = ~np.isnan(data).any(axis=1)
        ci_data[roi] = compute_mean_ci(data[valid])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    offsets = (np.arange(len(sel_types)) - (len(sel_types) - 1) / 2) * width

    x = _roi_x_positions(rois, gap)

    # Noise ceiling CI
    if not r2_adj:
        total = width * len(sel_types)
        for i, roi in enumerate(rois_flat):
            if nc_by_roi[roi]:
                _nc_band(ax, np.array(nc_by_roi[roi]),
                         [x[i] - total / 2, x[i] + total / 2])

    # Grouped bars and subject means
    for idx, sel in enumerate(sel_types):
        pos = x + offsets[idx]
        r2_means = [ci_data[r][0][idx] for r in rois_flat]
        r2_errs = [ci_data[r][1][idx] for r in rois_flat]

        ax.bar(
            pos, r2_means, width,
            color=SEL_INFO[sel]["color"],
            yerr=r2_errs, capsize=0,
            error_kw={"elinewidth": 1.5, "ecolor": "dimgray"},
            label=SEL_INFO[sel]["label"],
        )

        for j, roi in enumerate(rois_flat):
            vals = scores[roi][sel]
            if vals:
                ax.scatter([pos[j]] * len(vals), vals, color="gray", alpha=0.5, s=5)

    ax.set_ylim(ylim)
    ax.set_title(fig_title)
    ax.set_ylabel(r"Explained variance ($\mathrm{R^2_{norm}}$)" if r2_adj else r"Explained variance ($\mathrm{R^2}$)")
    ax.set_xticks(x)
    ax.set_xticklabels(rois_flat, rotation=30, ha='right')
    clean_axes(ax)

    _swatch_legend(ax,
                   [SEL_INFO[sel]["color"] for sel in sel_types],
                   [SEL_INFO[sel]["label"] for sel in sel_types],
                   title="Unit type", marker="s")

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()

def plot_encoding_varpart(model_name: str, layer: str,
                          subjects: list, rois: list,
                          r2_adj: bool=True, show_delta: bool=True,
                          ylim: tuple=(0, None), gap: float=0.5,
                          fig_title: str=None, out_path: Path=None):
    """Bar plot: Mean R2 components of unit types per ROI."""
    if rois and not isinstance(rois[0], list):
        rois = [rois]
    rois_flat = [roi for group in rois for roi in group]

    keys = ["u_f", "u_b", "s_fb"] + (["delta_m"] if show_delta else [])
    labels = ["Face unique", "Body unique", "Face-Body shared"] + (["Mixed unique"] if show_delta else [])
    colors = [SEL_INFO["face"]["color"],
                SEL_INFO["body"]["color"],
                SHARED_COLOR,
    ] + ([MIXED_COLOR] if show_delta else [])

    # Load R2 and noise ceilings
    res_fb = _voxelwise(model_name, [layer], subjects, rois_flat, "fb_varpart")[layer]
    res_delta = (_voxelwise(model_name, [layer], subjects, rois_flat, "delta_m")[layer]
                 if show_delta else {})
    fmri = _load_nc_by_roi(subjects)

    # Collect per-subject ROI means
    scores = {roi: {k: {s: np.nan for s in subjects} for k in keys} for roi in rois_flat}

    for subj in subjects:
        fb_map = res_fb.get(subj, {})
        subj_fmri = fmri.get(subj, {})

        for roi in rois_flat:
            comp = fb_map.get(roi)
            fmri_roi = subj_fmri.get(roi)
            if not comp or fmri_roi is None:
                continue

            nc_mean = np.nanmean(fmri_roi["ncsnr"])

            for k in keys:
                if k == "delta_m":
                    r2_vals = res_delta.get(subj, {}).get(roi, {}).get("delta_m", None)
                else:
                    r2_vals = comp.get(k, None)

                if r2_vals is None:
                    continue

                r2_mean = np.nanmean(r2_vals)
                scores[roi][k][subj] = (r2_mean / nc_mean) if r2_adj else r2_mean

    # Within-subject means + 95% CI
    ci_data = {}
    for roi in rois_flat:
        data = np.vstack([[scores[roi][k][s] for s in subjects] for k in keys]).T
        data = data.astype(float)
        valid = ~np.isnan(data).any(axis=1)
        if valid.any():
            means, ci = compute_mean_ci_within(data[valid])
        else:
            means = np.full(len(keys), np.nan)
            ci = np.full(len(keys), np.nan)
        ci_data[roi] = (np.asarray(means, dtype=float), np.asarray(ci, dtype=float))

    x = _roi_x_positions(rois, gap)

    # Bar layout
    n_keys = len(keys)
    width = 0.8 / n_keys
    offsets = (np.arange(n_keys) - (n_keys - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(10, 5))

    # Noise ceiling CI
    if not r2_adj:
        total_w = width * n_keys
        for i, roi in enumerate(rois_flat):
            nc_vals = [
                np.nanmean(fmri[s][roi]["ncsnr"])
                for s in fmri
                if isinstance(fmri.get(s), dict)
                and roi in fmri[s]
                and fmri[s][roi] is not None
            ]
            nc_vals = np.asarray(nc_vals, dtype=float)
            nc_vals = nc_vals[np.isfinite(nc_vals)]
            if nc_vals.size:
                _nc_band(ax, nc_vals,
                         [x[i] - total_w / 2, x[i] + total_w / 2])

    # Grouped bars and subject means
    for idx, (lab, col) in enumerate(zip(labels, colors)):
        pos = x + offsets[idx]
        means = [ci_data[roi][0][idx] for roi in rois_flat]
        errs = [ci_data[roi][1][idx] for roi in rois_flat]

        ax.bar(
            pos, means, width,
            color=col, label=lab,
            yerr=errs, capsize=0,
            error_kw={"elinewidth": 1.5, "ecolor": "dimgray"},
        )

        k = keys[idx]
        for j, roi in enumerate(rois_flat):
            vals = [scores[roi][k][s] for s in subjects]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                ax.scatter([pos[j]] * len(vals), vals, color="gray", alpha=0.5, s=5)

    ax.set_ylim(ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(rois_flat, rotation=30, ha="right")
    ax.set_ylabel(r"Explained variance ($\mathrm{R^2_{norm}}$)" if r2_adj else r"Explained variance ($\mathrm{R^2}$)")
    clean_axes(ax)
    fig.suptitle(fig_title)

    _swatch_legend(ax, colors, labels, title="Variance portion", marker="s")

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()

def _roi_grid(rois: list, layers: list, sharey: bool, n_cols: int=3):
    """Three-column grid of ROI panels, one x tick per layer."""
    n_rows = int(np.ceil((len(rois) + 1) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False, sharey=sharey,
    )
    return fig, axes, np.arange(len(layers)), n_cols, n_rows

def _layer_axis(ax, roi: str, layers: list, x: np.ndarray, idx: int, n_cols: int,
                ylim: tuple, r2_adj: bool):
    """One ROI panel: layer names on x, the R2 label only on the leftmost column."""
    ax.set_title(roi)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=30, ha="right")
    if idx % n_cols == 0:
        ax.set_ylabel(r"Explained variance ($\mathrm{R^2_{norm}}$)" if r2_adj else r"Explained variance ($\mathrm{R^2}$)")
    ax.set_ylim(ylim)
    clean_axes(ax)

def _grid_legend(fig, axes, n: int, n_rows: int, n_cols: int, colors: list, labels: list,
                 title: str, r2_adj: bool):
    """Drop unused panels, then one figure-level legend beside grid."""
    for j in range(n, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols][j % n_cols])

    handles = [mlines.Line2D([], [], color=c, lw=2, label=l)
               for c, l in zip(colors, labels)]
    if not r2_adj:
        handles.append(mpatches.Patch(facecolor="gray", alpha=0.2, label="Noise ceiling"))

    leg = fig.legend(
        handles,
        [h.get_label() for h in handles],
        title=title,
        loc="center left",
        bbox_to_anchor=(1, 0.6),
        frameon=False,
    )
    leg._legend_box.align = "left"
    return leg

def plot_encoding_sep_layers(model_name: str, layers: list,
                             subjects: list, rois: list,
                             r2_adj: bool=True, ylim: tuple=(0, None),
                             fig_title: str=None, out_path: Path=None):
    """Line plot: Mean R2 of unit type across layers per ROI."""
    sel_types = list(SEL_INFO.keys())
    sel_keys = ["f", "b", "m", "ns"]
    sel_key_map = dict(zip(sel_types, sel_keys))

    # Load R2 and noise ceilings
    res = _voxelwise(model_name, layers, subjects, rois, "sep")
    fmri = _load_nc_by_roi(subjects)

    # Aggregate per-subject R2 per ROI and layer
    scores = {roi: {lay: {sel: [] for sel in sel_types} for lay in layers} for roi in rois}
    for lay in layers:
        layer_res = res.get(lay, {})
        for subj, roi_map in layer_res.items():
            subj_fmri = fmri.get(subj, {})
            for roi in rois:
                metrics = roi_map.get(roi)
                fmri_roi = subj_fmri.get(roi)
                if not metrics or fmri_roi is None:
                    continue

                nc_mean = np.nanmean(fmri_roi["ncsnr"])
                r2_map = metrics.get("r2", {})

                for sel in sel_types:
                    arr = r2_map.get(sel_key_map[sel])
                    if arr is None:
                        continue
                    val = np.nanmean(arr)
                    scores[roi][lay][sel].append(val / nc_mean if r2_adj else val)

    # Within-subject means + 95% CIs
    ci_data = {roi: {} for roi in rois}
    for roi in rois:
        for lay in layers:
            data = np.stack([scores[roi][lay][sel] for sel in sel_types], axis=1)
            valid = ~np.isnan(data).any(axis=1)
            if valid.any():
                means, ci_half = compute_mean_ci_within(data[valid])
            else:
                # NaN, so the panel shows a gap rather than a point at zero.
                means = np.full(len(sel_types), np.nan)
                ci_half = np.full(len(sel_types), np.nan)
            ci_data[roi][lay] = (means, ci_half)

    n = len(rois)
    fig, axes, x, n_cols, n_rows = _roi_grid(rois, layers, sharey=False)

    for idx, roi in enumerate(rois):
        ax = axes[idx // n_cols][idx % n_cols]

        # Noise ceiling CI
        if not r2_adj:
            nc_vals = np.array([
                np.nanmean(subj_fmri[roi]["ncsnr"])
                for subj_fmri in fmri.values()
                if subj_fmri.get(roi) is not None
            ], dtype=float)
            nc_vals = nc_vals[np.isfinite(nc_vals)]
            if nc_vals.size:
                _nc_band(ax, nc_vals, x, zorder=1)

        # Mean R2 + 95% CI
        for sel_idx, sel in enumerate(sel_types):
            means = [ci_data[roi][lay][0][sel_idx] for lay in layers]
            cis = [ci_data[roi][lay][1][sel_idx] for lay in layers]

            info = SEL_INFO[sel]
            ax.plot(x, means, label=info["label"], color=info["color"], lw=2)
            ax.fill_between(
                x,
                np.array(means) - np.array(cis),
                np.array(means) + np.array(cis),
                color=info["color"],
                alpha=0.3,
                lw=0,
                zorder=2,
            )

        _layer_axis(ax, roi, layers, x, idx, n_cols, ylim, r2_adj)
    fig.suptitle(fig_title)

    leg = _grid_legend(
        fig, axes, n, n_rows, n_cols,
        [SEL_INFO[s]["color"] for s in sel_types],
        [SEL_INFO[s]["label"] for s in sel_types],
        "Unit type", r2_adj,
    )

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.show()

def plot_encoding_varpart_layers(model_name: str, layers: list,
                                 subjects: list, rois: list,
                                 r2_adj: bool=True, show_delta: bool=True,
                                 ylim: tuple=(0, None), fig_title: str=None,
                                 out_path: Path=None):
    """Line plot: Mean R2 components across layers per ROI."""
    comp_keys = ["u_f", "u_b", "s_fb"] + (["delta_m"] if show_delta else [])
    comp_labels = ["Face unique", "Body unique", "Face-Body shared"] + (["Mixed unique"] if show_delta else [])
    comp_colors = [
        SEL_INFO["face"]["color"],
        SEL_INFO["body"]["color"],
        SHARED_COLOR
    ] + ([MIXED_COLOR] if show_delta else [])

    # Load R2 and noise ceilings
    res_fb = _voxelwise(model_name, layers, subjects, rois, "fb_varpart")
    res_delta = _voxelwise(model_name, layers, subjects, rois, "delta_m") if show_delta else {}
    fmri = _load_nc_by_roi(subjects)

    # Aggregate per-subject R2 per ROI, layer, and component
    scores = {roi: {lay: {k: [] for k in comp_keys} for lay in layers} for roi in rois}
    for lay in layers:
        fb_layer = res_fb.get(lay, {}) or {}
        delta_layer = (res_delta.get(lay, {}) if show_delta else {}) or {}

        subj_keys = set(fb_layer.keys()) | set(delta_layer.keys())
        for subj in subj_keys:
            subj_fmri = fmri.get(subj, {}) or {}

            for roi in rois:
                fmri_roi = subj_fmri.get(roi)
                nc_mean = (
                    np.nanmean(fmri_roi["ncsnr"])
                    if (isinstance(fmri_roi, dict) and "ncsnr" in fmri_roi) else np.nan
                )

                comp_roi = (fb_layer.get(subj, {}) or {}).get(roi, {}) or {}
                delta_roi = (delta_layer.get(subj, {}) or {}).get(roi, {}) or {}

                for key in comp_keys:
                    r2_vals = delta_roi.get("delta_m") if key == "delta_m" else comp_roi.get(key)
                    if r2_vals is None:
                        r2_mean = np.nan
                    elif r2_adj:
                        # Without a usable noise ceiling the subject drops out, rather
                        # than contributing a raw R2 alongside normalized ones.
                        r2_mean = (np.nanmean(r2_vals) / nc_mean
                                   if np.isfinite(nc_mean) and nc_mean != 0 else np.nan)
                    else:
                        r2_mean = np.nanmean(r2_vals)
                    scores[roi][lay][key].append(r2_mean)

    # Within-subject means + 95% CIs
    ci_data = {roi: {} for roi in rois}
    for roi in rois:
        for lay in layers:
            data = np.stack([np.asarray(scores[roi][lay][k], dtype=float) for k in comp_keys], axis=1)
            valid = ~np.isnan(data).any(axis=1)
            if valid.any():
                means, ci_half = compute_mean_ci_within(data[valid])
            else:
                # NaN, so the panel shows a gap rather than a point at zero.
                means = np.full(len(comp_keys), np.nan)
                ci_half = np.full(len(comp_keys), np.nan)
            ci_data[roi][lay] = (means, ci_half)

    n = len(rois)
    fig, axes, x, n_cols, n_rows = _roi_grid(rois, layers, sharey=True)

    for idx, roi in enumerate(rois):
        ax = axes[idx // n_cols][idx % n_cols]

        # Noise ceiling CI
        if not r2_adj:
            nc_vals = np.array([
                np.nanmean(subj_fmri[roi]["ncsnr"])
                for subj_fmri in fmri.values()
                if subj_fmri.get(roi) is not None
            ], dtype=float)
            nc_vals = nc_vals[np.isfinite(nc_vals)]
            if nc_vals.size:
                _nc_band(ax, nc_vals, x, zorder=1)

        # Mean R2 + 95% CI
        for c_idx, (lab, col) in enumerate(zip(comp_labels, comp_colors)):
            means = [ci_data[roi][lay][0][c_idx] for lay in layers]
            cis = [ci_data[roi][lay][1][c_idx] for lay in layers]

            ax.plot(x, means, label=lab, color=col, lw=2, zorder=3)
            ax.fill_between(
                x,
                np.asarray(means) - np.asarray(cis),
                np.asarray(means) + np.asarray(cis),
                color=col, alpha=0.3, lw=0, zorder=2
            )

        _layer_axis(ax, roi, layers, x, idx, n_cols, ylim, r2_adj)
    fig.suptitle(fig_title)

    leg = _grid_legend(
        fig, axes, n, n_rows, n_cols,
        comp_colors, comp_labels,
        "Variance portion", r2_adj,
    )

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.show()
