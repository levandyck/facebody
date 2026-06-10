from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.cm as cm

from facebody.config import PROJECT_ROOT, FIG_ROOT
from facebody.guided_gradcam import compute_guided_gradcam
from myutils.utils import load_pickle, clean_axes

plt.rcParams.update({
    "axes.linewidth": 1,
    "lines.linewidth": 2.5,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.title_fontsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 13,
    "figure.dpi": 300
})

SEL_INFO = {
    "face": {"label": "Face-selective", "color": "#dc267fff"},
    "body": {"label": "Body-selective", "color": "#ffb000ff"},
    "mixed": {"label": "Mixed-selective", "color": "#648fffff"},
    "nonselective":{"label": "Non-selective", "color": "#ccccccff"},
}

# ------------------------------- MAIN FIGURES ------------------------------- #
# -------------------------------- Selectivity ------------------------------- #
def plot_perc_layers(model_dict: dict, fig_title: str=None, ylim: tuple=(0, 8),
                     n_grid: int=10, out_path: Path=None):
    """Line plot: Percentage of selective units across layers."""
    sel_types = list(SEL_INFO.keys())[:3]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    get_layers = lambda model, i: model_dict[model]

    # Shared normalized layer depth grid
    x_grid = np.linspace(0, 1, n_grid)

    # Store interpolated curves for each unit type
    all_curves = {sel: [] for sel in sel_types}

    for m, model in enumerate(model_dict.keys()):
        layers = get_layers(model, m)
        res = load_pickle(PROJECT_ROOT / "selectivity" / model / "floc_res.pkl")

        x_model = np.linspace(0, 1, len(layers))

        totals = np.array(
            [len(res[lay]["stats"]["face"]["dvals"]) for lay in layers],
            dtype=float
        )

        for sel in sel_types:
            counts = np.array(
                [len(res[lay]["unit_ids"].get(sel, [])) for lay in layers],
                dtype=float
            )
            vec = np.divide(
                100.0 * counts,
                totals,
                out=np.full_like(totals, np.nan),
                where=totals != 0
            )

            ok = ~np.isnan(vec)
            if ok.sum() < 2:
                interp_vec = np.full_like(x_grid, np.nan, dtype=float)
            else:
                interp_vec = np.interp(
                    x_grid,
                    x_model[ok],
                    vec[ok],
                )

            all_curves[sel].append(interp_vec)

    # Aggregate and plot
    for sel in sel_types:
        arr = np.vstack(all_curves[sel])
        n = arr.shape[0]
        mean_vec = np.nanmean(arr, axis=0)
        sem_vec = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n)

        ax.plot(
            x_grid*100,
            mean_vec,
            color=SEL_INFO[sel]["color"],
            lw=2.5,
            label=SEL_INFO[sel]["label"],
        )
        ax.fill_between(
            x_grid*100,
            mean_vec - sem_vec,
            mean_vec + sem_vec,
            color=SEL_INFO[sel]["color"],
            alpha=0.2,
            linewidth=0,
        )

        ok = ~np.isnan(mean_vec)
        rho, p = spearmanr(x_grid[ok], mean_vec[ok]) if ok.sum() > 1 else (np.nan, np.nan)
        print(f"{sel}: Model mean Spearman r={rho:.2f}, p={p:.3f}")

    ax.set_xlim(-5, 105)
    ax.set_xticks(np.linspace(0, 100, 5))
    ax.set_xticklabels([f"{v:.0f}" for v in np.linspace(0, 100, 5)])
    ax.set_xlabel("Layer depth (%)")
    ax.set_ylim(*ylim)
    ax.set_ylabel("Selective units (%)")
    clean_axes(ax)

    sel_handles = [
        mlines.Line2D([], [], color=SEL_INFO[sel]["color"], lw=2.5)
        for sel in sel_types
    ]
    sel_labels = [SEL_INFO[sel]["label"] for sel in sel_types]
    leg = ax.legend(
        sel_handles,
        sel_labels,
        title="Unit type",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )
    leg._legend_box.align = "left"

    ax.set_title(fig_title)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_dprime_dist(model_dict: dict, fig_title: str=None, out_path: Path=None,
                     ylims: tuple=(-2, 2.5), n_boot: int=1000):
    """Violin plot: Face and body selectivity per unit type."""
    sel_types = list(SEL_INFO.keys())

    pooled = {
        sel: {"face": [], "body": []}
        for sel in sel_types
    }

    hier_data = {
        sel: {"face": {}, "body": {}}
        for sel in sel_types
    }

    for model in model_dict.keys():
        res = load_pickle(PROJECT_ROOT / "selectivity" / model / "floc_res.pkl")
        dvals = load_pickle(PROJECT_ROOT / "selectivity" / model / "validation_dprime.pkl")

        for lay in model_dict[model]:
            dvals_f = np.asarray(dvals[lay]["face_d"], dtype=float)
            dvals_b = np.asarray(dvals[lay]["body_d"], dtype=float)
            all_units = np.arange(dvals_f.shape[0])

            groups = {
                sel: np.asarray(res[lay]["unit_ids"].get(sel, []), dtype=int)
                for sel in sel_types
                if sel != "nonselective"
            }

            used = [ids for ids in groups.values() if ids.size > 0]
            used = np.concatenate(used) if used else np.array([], dtype=int)
            groups["nonselective"] = np.setdiff1d(all_units, used)

            for sel, ids in groups.items():
                if ids.size == 0:
                    continue

                vals = {
                    "face": dvals_f[ids],
                    "body": dvals_b[ids],
                }

                for key, dvals_arr in (("face", dvals_f), ("body", dvals_b)):
                    arr = dvals_arr[ids]
                    arr = arr[~np.isnan(arr)]
                    if arr.size == 0:
                        continue

                    pooled[sel][key].append(arr)
                    hier_data[sel][key].setdefault(model, {})[lay] = arr

    rng = np.random.default_rng(0)

    fig, (ax_face, ax_body) = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)

    def _plot_panel(ax, key, title, rng):
        pos, data, sels = [], [], []

        for i, sel in enumerate(sel_types):
            # Violin data
            if pooled[sel][key]:
                vals = np.concatenate(pooled[sel][key])
                vals = vals[~np.isnan(vals)]
                if vals.size > 0:
                    pos.append(i)
                    data.append(vals)
                    sels.append(sel)

            # Hierarchical bootstrap CI
            grand_mean, ci_lo, ci_hi = compute_mean_ci_hierarch_models_layers_units(
                hier_data[sel][key], n_iter=n_boot, rng=rng,
            )
            if not np.isfinite(grand_mean):
                continue

            ax.errorbar(
                i, grand_mean,
                yerr=[[grand_mean - ci_lo], [ci_hi - grand_mean]],
                fmt="o",
                color="darkgray",
                markerfacecolor="darkgray",
                markeredgecolor="darkgray",
                capsize=3,
                markersize=4,
                zorder=5,
            )

        if data:
            vp = ax.violinplot(
                data,
                positions=pos,
                widths=0.8,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body, sel in zip(vp["bodies"], sels):
                body.set_facecolor(SEL_INFO[sel]["color"])
                body.set_edgecolor(SEL_INFO[sel]["color"])
                body.set_linewidth(0)
                body.set_alpha(1)

        ax.set_title(title)
        ax.set_xticks([])
        ax.axhline(0, color="gray", ls="--", lw=1, zorder=0)
        ax.set_xlim(-0.6, len(sel_types) - 0.4)
        ax.set_ylim(*ylims)
        ax.grid(False)
        clean_axes(ax)

    rng = np.random.default_rng(0)
    _plot_panel(ax_face, "face", "Face d'", rng)
    _plot_panel(ax_body, "body", "Body d'", rng)

    ax_face.set_ylabel("Selectivity (d')")

    handles = [
        mlines.Line2D([], [], color=SEL_INFO[sel]["color"], lw=3)
        for sel in sel_types
    ]
    labels = [SEL_INFO[sel]["label"] for sel in sel_types]

    ax_body.legend(
        handles,
        labels,
        title="Unit type",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )

    plt.suptitle(fig_title)
    plt.tight_layout()

    if out_path:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_guided_gradcam(model_name: str, target_layer: str, cam_layer: str=None,
                        controlled: bool=True, device: str="cuda",
                        img_path: str=None, n_imgs: int=3, norm_types: bool=True,
                        seed: int=None, fig_title: str=None, out_path: Path=None):
    """Heatmaps: Guided Grad-CAM saliency map per unit type overlaid on example images."""
    sel_types = list(SEL_INFO.keys())

    res = compute_guided_gradcam(
        model_name=model_name,
        target_layer_pretty=target_layer,
        cam_layer_pretty=cam_layer,
        device=device,
        sel_types=list(SEL_INFO.keys()),
        controlled=controlled,
        imgs_path=img_path,
        n_imgs=n_imgs,
        norm_types=norm_types,
        seed=seed,
        post_blur_sigma=3.0,
    )

    n_rows, n_cols = len(sel_types), n_imgs
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2 * n_cols, 2.2 * n_rows),
        gridspec_kw={"wspace": 0.02, "hspace": 0.2},
        squeeze=False,
    )
    axes = np.array(axes, dtype=object)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        if n_rows == 1:
            axes = axes.reshape(1, n_cols)
        else:
            axes = axes.reshape(n_rows, 1)

    cmap = cm.get_cmap("turbo")

    for i, res in enumerate(res):
        img = res["pil_img"]
        maps = res["heatmaps"]

        for s, sel in enumerate(sel_types):
            ax = axes[s, i]

            im0 = ax.imshow(img)
            extent = im0.get_extent()

            hm = maps[sel]
            ax.imshow(
                hm,
                cmap=cmap,
                vmin=np.percentile(hm, 10),
                vmax=np.percentile(hm, 99),
                alpha=0.6,
                interpolation="nearest",
                extent=extent,
            )
            ax.axis("off")

            if i == 0:
                row_label = SEL_INFO[sel]["label"]
                ax.text(
                    -0.08, 0.5,
                    row_label,
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=ax.transAxes,
                    clip_on=False,
                )

    fig.suptitle(fig_title)
    if out_path:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()


# --------------------------------- Encoding --------------------------------- #
def plot_encoding_sep(model_name: str, layer: str, rois: list,
                      r2_adj: bool=False, gap: float=0.5,
                      fig_title: str=None, out_path: Path=None):
    """Bar plot: Mean R² of unit types per ROI."""
    res = load_pickle(PROJECT_ROOT / "encoding" / model_name / "sep.pkl")[layer]
    fmri = load_pickle(PROJECT_ROOT / "nsd" / "fmri_activs.pkl")

    sel_types = list(SEL_INFO.keys())
    sel_keys = ["f", "b", "m", "ns"]
    sel_key_map = dict(zip(sel_types, sel_keys))

    if rois and not isinstance(rois[0], list):
        rois = [rois]

    flat_rois = [roi for group in rois for roi in group]

    # Collect per-subject scores
    scores = {roi: {sel: [] for sel in sel_types} for roi in flat_rois}
    for subj, roi_map in res.items():
        subj_fmri = fmri.get(subj, {})
        for roi in flat_rois:
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
                scores[roi][sel].append(val / nc_mean if r2_adj else val)

    # Fixed-effects within-subject means + 95% CI half-widths
    ci_data = {}
    for roi in flat_rois:
        data = np.stack([scores[roi][sel] for sel in sel_types], axis=1)
        valid = ~np.isnan(data).any(axis=1)
        ci_data[roi] = compute_mean_ci(data[valid])

    # Build x positions
    x_positions = []
    cursor = 0.0
    for g, group in enumerate(rois):
        for roi in group:
            x_positions.append(cursor)
            cursor += 1.0
        if g < len(rois) - 1:
            cursor += gap
    x = np.array(x_positions)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.2
    offsets = (np.arange(len(sel_types)) - (len(sel_types) - 1) / 2) * width

    # Noise ceiling ribbon
    if not r2_adj:
        total = width * len(sel_types)
        for i, roi in enumerate(flat_rois):
            nc_vals = [
                np.nanmean(roi_dict["ncsnr"])
                for s in res
                if (roi_dict := fmri.get(s, {}).get(roi)) is not None
            ]
            if nc_vals:
                lo, hi = np.percentile(nc_vals, [2.5, 97.5])
                ax.fill_between([x[i] - total / 2, x[i] + total / 2], lo, hi,
                                color="gray", alpha=0.3, lw=0)

    # Grouped bars and subject means
    for idx, sel in enumerate(sel_types):
        pos = x + offsets[idx]
        means = [ci_data[r][0][idx] for r in flat_rois]
        errs = [ci_data[r][1][idx] for r in flat_rois]

        info = SEL_INFO[sel]
        ax.bar(
            pos, means, width,
            color=info["color"],
            yerr=errs, capsize=0,
            error_kw={"elinewidth": 1.5, "ecolor": "dimgray"},
            label=info["label"],
        )

        for j, roi in enumerate(flat_rois):
            vals = scores[roi][sel]
            if vals:
                ax.scatter([pos[j]] * len(vals), vals, color="gray", alpha=0.5, s=5)

    ax.set_ylim(0)
    ax.set_title(fig_title)
    ax.set_ylabel("Explained variance (R² adj.)" if r2_adj else "Explained variance (R²)")
    ax.set_xticks(x)
    ax.set_xticklabels(flat_rois, rotation=30, ha='right')
    clean_axes(ax)

    # Legend
    sel_handles = [
        mlines.Line2D([], [], marker="s", linestyle="none", color=SEL_INFO[sel]["color"])
        for sel in sel_types
    ]
    sel_labels = [SEL_INFO[sel]["label"] for sel in sel_types]
    leg = ax.legend(
        sel_handles, sel_labels,
        title="Unit type",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )
    leg._legend_box.align = "left"

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()

def plot_encoding_varpart(model_name: str, layer: str, rois: list,
                          r2_adj: bool=False, show_delta: bool=True,
                          gap: float=0.5, fig_title: str=None,
                          out_path: Path=None):
    """Bar plot: Mean R² components of unit types per ROI."""
    res_fb = load_pickle(PROJECT_ROOT / "encoding" / model_name / "fb_varpart.pkl")[layer]
    subjects = list(res_fb.keys())
    res_delta = (load_pickle(PROJECT_ROOT / "encoding" / model_name / "delta_m.pkl")[layer]
                 if show_delta else {})
    fmri = load_pickle(PROJECT_ROOT / "nsd" / "fmri_activs.pkl")

    keys = ["u_f", "u_b", "s_fb"] + (["delta_m"] if show_delta else [])
    labels = ["Face", "Body", "Face ∩ Body"] + (["Δ Mixed"] if show_delta else [])
    colors = [SEL_INFO["face"]["color"],
              SEL_INFO["body"]["color"],
              "#F9CEB0",
    ] + (["#648FFF"] if show_delta else [])

    if rois and not isinstance(rois[0], list):
        rois = [rois]

    flat_rois = [roi for group in rois for roi in group]

    scores = {roi: {k: {s: np.nan for s in subjects} for k in keys} for roi in flat_rois}

    for subj in subjects:
        fb_map = res_fb.get(subj, {})
        subj_fmri = fmri.get(subj, {})

        for roi in flat_rois:
            comp = fb_map.get(roi)
            fmri_roi = subj_fmri.get(roi)
            if not comp or fmri_roi is None:
                continue

            nc_mean = np.nanmean(fmri_roi["ncsnr"])

            for k in keys:
                if k == "delta_m":
                    arr = res_delta.get(subj, {}).get(roi, {}).get("delta_m", None)
                else:
                    arr = comp.get(k, None)

                if arr is None:
                    continue

                v = np.nanmean(arr)
                scores[roi][k][subj] = (v / nc_mean) if r2_adj else v

    # Fixed-effects within-subject means + 95% CI half-widths
    mean_ci = {}
    for roi in flat_rois:
        mat = np.vstack([[scores[roi][k][s] for s in subjects] for k in keys]).T
        mat = mat.astype(float)
        valid = ~np.isnan(mat).any(axis=1)
        if valid.any():
            means, ci = compute_mean_ci_within(mat[valid])
        else:
            means = np.full(len(keys), np.nan)
            ci = np.full(len(keys), np.nan)
        mean_ci[roi] = (np.asarray(means, dtype=float), np.asarray(ci, dtype=float))

    # Build x positions with inter-group gaps
    x_positions = []
    cursor = 0.0
    for g, group in enumerate(rois):
        for roi in group:
            x_positions.append(cursor)
            cursor += 1.0
        if g < len(rois) - 1:
            cursor += gap
    x = np.array(x_positions)

    # Bar layout
    n_keys = len(keys)
    width = 0.8 / n_keys
    offsets = (np.arange(n_keys) - (n_keys - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(10, 4))

    # Noise ceiling
    if not r2_adj:
        total_w = 0.8
        for i, roi in enumerate(flat_rois):
            subj_ncs = [
                np.nanmean(fmri[s][roi]["ncsnr"])
                for s in fmri
                if isinstance(fmri.get(s), dict)
                and roi in fmri[s]
                and fmri[s][roi] is not None
            ]
            if subj_ncs:
                lo, hi = np.percentile(subj_ncs, [2.5, 97.5])
                ax.fill_between(
                    [x[i] - total_w / 2, x[i] + total_w / 2],
                    lo, hi,
                    color="gray", alpha=0.3, lw=0, zorder=0
                )

    # Grouped bars and subject means
    for j, (lab, col) in enumerate(zip(labels, colors)):
        pos = x + offsets[j]
        means = [mean_ci[roi][0][j] for roi in flat_rois]
        errs = [mean_ci[roi][1][j] for roi in flat_rois]

        ax.bar(
            pos, means, width,
            color=col, label=lab,
            yerr=errs, capsize=0,
            error_kw={"elinewidth": 1.5, "ecolor": "dimgray"},
        )

        k = keys[j]
        for ri, roi in enumerate(flat_rois):
            vals = [scores[roi][k][s] for s in subjects]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                ax.scatter([pos[ri]] * len(vals), vals, color="gray", alpha=0.5, s=5)

    ax.set_ylim(0, None)
    ax.set_xticks(x)
    ax.set_xticklabels(flat_rois, rotation=30, ha="right")
    ax.set_ylabel("Explained variance (R² adj.)" if r2_adj else "R²")
    clean_axes(ax)
    fig.suptitle(fig_title)

    var_handles = [
        mlines.Line2D([], [], marker="s", linestyle="none", color=c)
        for c in colors
    ]
    leg = ax.legend(
        var_handles, labels,
        title="Variance portion",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )
    leg._legend_box.align = "left"

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()

def plot_varpart_integrated(model_name: str, layers: list,
                            face_rois: tuple=("OFA", "FFA", "aTL-faces"),
                            body_rois: tuple=("EBA", "FBA", "mTL-bodies"),
                            r2_adj: bool=False, include_delta: bool=True,
                            show_group_mean: bool=True, fig_title: str=None,
                            out_path: Path=None):
    """Line plot: Integrated variance across cortical hierarchy for both region types."""
    if isinstance(layers, str):
        layers = [layers]

    res_fb = load_pickle(PROJECT_ROOT / "encoding" / model_name / "fb_varpart.pkl")
    res_delta = load_pickle(PROJECT_ROOT / "encoding" / model_name / "delta_m.pkl") if include_delta else {}
    fmri = load_pickle(PROJECT_ROOT / "nsd" / "fmri_activs.pkl")

    face_color = SEL_INFO["face"]["color"]
    body_color = SEL_INFO["body"]["color"]

    # Build subject list
    subj_sets = [set((res_fb.get(lay, {}) or {}).keys()) for lay in layers]
    subj_all = sorted(set().union(*subj_sets)) if subj_sets else []

    n_f, n_b = len(face_rois), len(body_rois)
    face_x = np.arange(n_f)
    body_x = np.arange(n_b)

    def _overlap_one(subj, layer, roi):
        fb = (res_fb.get(layer, {}) or {}).get(subj, {}).get(roi)
        fmri_roi = fmri.get(subj, {}).get(roi)
        if fb is None or fmri_roi is None:
            return np.nan

        uf = np.maximum(0, np.nanmean(fb["u_f"]))
        ub = np.maximum(0, np.nanmean(fb["u_b"]))
        sfb = np.maximum(0, np.nanmean(fb["s_fb"]))

        dm = 0.0
        if include_delta:
            dm = np.maximum(0, np.nanmean(
                (res_delta.get(layer, {}) or {}).get(subj, {}).get(roi, {}).get("delta_m", 0.0)
            ))

        if r2_adj:
            noise = np.nanmean(fmri_roi["ncsnr"])
            if (noise is None) or (not np.isfinite(noise)) or noise == 0:
                return np.nan
            uf, ub, sfb, dm = uf / noise, ub / noise, sfb / noise, dm / noise

        tot = uf + ub + sfb + dm
        if tot > 0 and np.isfinite(tot):
            return (sfb + dm) / tot
        return np.nan

    def _vector_for_rois(subj, rois):
        """Per-ROI overlaps for a subject, averaged across layers."""
        out = np.full(len(rois), np.nan, dtype=float)
        for i, roi in enumerate(rois):
            vals = np.array([_overlap_one(subj, lay, roi) for lay in layers], dtype=float)
            out[i] = np.nanmean(vals) if np.any(np.isfinite(vals)) else np.nan
        return out

    def check_monotonicity(face_mat, body_mat, keep_subj):
        """Check if each participant shows monotonic increase, including incomplete data."""
        
        def check_mono_with_missing(vals):
            """Check monotonicity for values that may have NaNs."""
            finite_mask = np.isfinite(vals)
            if np.sum(finite_mask) < 2:  # Need at least 2 points
                return None, "insufficient"
            
            finite_vals = vals[finite_mask]
            diffs = np.diff(finite_vals)
            is_monotonic = np.all(diffs >= 0)
            
            if np.all(finite_mask):
                return is_monotonic, "complete"
            else:
                return is_monotonic, "incomplete"
        
        # Face hierarchy
        face_complete_mono, face_incomplete_mono = [], []
        face_insufficient = 0
        
        for i, subj in enumerate(keep_subj):
            vals = face_mat[i]
            is_mono, status = check_mono_with_missing(vals)
            
            if status == "insufficient":
                face_insufficient += 1
            elif status == "complete":
                face_complete_mono.append(is_mono)
                if not is_mono:
                    print(f"Non-monotonic face (complete): {subj}, values: {vals}")
            else:  # incomplete
                face_incomplete_mono.append(is_mono)
                if not is_mono:
                    print(f"Non-monotonic face (incomplete): {subj}, values: {vals}")
        
        # Body hierarchy
        body_complete_mono, body_incomplete_mono = [], []
        body_insufficient = 0
        
        for i, subj in enumerate(keep_subj):
            vals = body_mat[i]
            is_mono, status = check_mono_with_missing(vals)
            
            if status == "insufficient":
                body_insufficient += 1
            elif status == "complete":
                body_complete_mono.append(is_mono)
                if not is_mono:
                    print(f"Non-monotonic body (complete): {subj}, values: {vals}")
            else:  # incomplete
                body_incomplete_mono.append(is_mono)
                if not is_mono:
                    print(f"Non-monotonic body (incomplete): {subj}, values: {vals}")
        
        # Report
        print("\n=== Face Hierarchy Monotonicity ===")
        print(f"Complete data: {sum(face_complete_mono)}/{len(face_complete_mono)} monotonic")
        if face_incomplete_mono:
            print(f"Incomplete data: {sum(face_incomplete_mono)}/{len(face_incomplete_mono)} monotonic")
        if face_insufficient:
            print(f"Insufficient data: {face_insufficient}")
        
        print("\n=== Body Hierarchy Monotonicity ===")
        print(f"Complete data: {sum(body_complete_mono)}/{len(body_complete_mono)} monotonic")
        if body_incomplete_mono:
            print(f"Incomplete data: {sum(body_incomplete_mono)}/{len(body_incomplete_mono)} monotonic")
        if body_insufficient:
            print(f"Insufficient data: {body_insufficient}")
        
        return {
            'face_complete': face_complete_mono,
            'face_incomplete': face_incomplete_mono,
            'body_complete': body_complete_mono,
            'body_incomplete': body_incomplete_mono
        }

    # Collect per-subject vectors
    face_mat, body_mat, keep_subj = [], [], []
    for subj in subj_all:
        fv = _vector_for_rois(subj, face_rois)
        bv = _vector_for_rois(subj, body_rois)
        if np.any(np.isfinite(fv)) and np.any(np.isfinite(bv)):
            face_mat.append(fv)
            body_mat.append(bv)
            keep_subj.append(subj)

    face_mat = np.asarray(face_mat, dtype=float) if face_mat else np.empty((0, n_f))
    body_mat = np.asarray(body_mat, dtype=float) if body_mat else np.empty((0, n_b))

    mono_results = check_monotonicity(face_mat, body_mat, keep_subj)

    # Prepare data for face hierarchy
    face_data = []
    for i, subj in enumerate(keep_subj):
        for level_idx, roi in enumerate(face_rois):
            if np.isfinite(face_mat[i, level_idx]):
                face_data.append({
                    'participant': subj,
                    'level': level_idx + 1,  # 1, 2, 3
                    'integrated_variance': face_mat[i, level_idx]
                })

    # Prepare data for body hierarchy
    body_data = []
    for i, subj in enumerate(keep_subj):
        for level_idx, roi in enumerate(body_rois):
            if np.isfinite(body_mat[i, level_idx]):
                body_data.append({
                    'participant': subj,
                    'level': level_idx + 1,  # 1, 2, 3
                    'integrated_variance': body_mat[i, level_idx]
                })
    
    # Fit LMMs
    if face_data:
        df_face = pd.DataFrame(face_data)
        try:
            lmm_face = smf.mixedlm("integrated_variance ~ level", 
                                   data=df_face, 
                                   groups=df_face["participant"]).fit()
            print("\n=== Face-selective hierarchy LMM ===")
            print(f"β = {lmm_face.params['level']:.3f}, SE = {lmm_face.bse['level']:.3f}")
            print(f"t = {lmm_face.tvalues['level']:.2f}, p = {lmm_face.pvalues['level']:.4f}")
            print(lmm_face.summary())
        except Exception as e:
            print(f"Face LMM failed: {e}")

    if body_data:
        df_body = pd.DataFrame(body_data)
        try:
            lmm_body = smf.mixedlm("integrated_variance ~ level", 
                                   data=df_body, 
                                   groups=df_body["participant"]).fit()
            print("\n=== Body-selective hierarchy LMM ===")
            print(f"β = {lmm_body.params['level']:.3f}, SE = {lmm_body.bse['level']:.3f}")
            print(f"t = {lmm_body.tvalues['level']:.2f}, p = {lmm_body.pvalues['level']:.4f}")
            print(lmm_body.summary())
        except Exception as e:
            print(f"Body LMM failed: {e}")

    fig, (ax_f, ax_b) = plt.subplots(1, 2, sharey=True, figsize=(6, 4), constrained_layout=True)

    for i, _s in enumerate(keep_subj):
        ax_f.plot(face_x, face_mat[i], marker="o", lw=1.5, alpha=0.7, color=face_color)
        ax_b.plot(body_x, body_mat[i], marker="o", lw=1.5, alpha=0.7, color=body_color)

    # Group means
    if show_group_mean and face_mat.size and body_mat.size:
        ax_f.plot(face_x, np.nanmean(face_mat, axis=0), marker="o", lw=3, color=face_color)
        ax_b.plot(body_x, np.nanmean(body_mat, axis=0), marker="o", lw=3, color=body_color)
        ax_f.legend(frameon=False, loc="upper left")
        ax_b.legend(frameon=False, loc="upper left")

    ax_f.set_xticks(face_x)
    ax_b.set_xticks(body_x)
    ax_f.set_xticklabels(face_rois, rotation=30, ha="right")
    ax_b.set_xticklabels(body_rois, rotation=30, ha="right")

    ax_f.set_title("Face-selective regions")
    ax_b.set_title("Body-selective regions")
    ax_f.set_ylabel("Overlap (% of total variance)")
    ax_f.set_ylim(0.3, 0.47)

    for ax in (ax_f, ax_b):
        clean_axes(ax)

    fig.suptitle(fig_title)
    if out_path:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()


# --------------------------------- Lesioning -------------------------------- #
def plot_lesioning_drop(model_name: str, tasks: tuple=("face", "person", "action"),
                        norm_drop: bool=True, fig_title: str=None, out_path: Path=None):
    """Bar plot: Lesioning drops for different classification tasks."""
    datasets = {
        "face": "Faces",
        "person": "Persons",
        "action": "Actions",
    }
    lims = {
        "face": [0, 30],
        "person": [0, 10],
        "action": [0, 10],
    }

    norm_key = "rel" if norm_drop else "abs"

    n_tasks = len(tasks)
    fig, axes = plt.subplots(1, n_tasks, figsize=(2.5 * n_tasks, 3), squeeze=True)
    if n_tasks == 1:
        axes = [axes]

    sel_types = list(SEL_INFO.keys())
    colors = [SEL_INFO[s]["color"] for s in sel_types]
    x = np.arange(len(sel_types))

    for ax, task in zip(axes, tasks):
        res = load_pickle(PROJECT_ROOT / "lesioning" / model_name / task / "lesion_global.pkl")
        per_sel = res["summary"]["per_sel"]

        means = np.array([per_sel[s][norm_key]["mean_drop"] for s in sel_types])
        ci_lows = np.array([per_sel[s][norm_key]["ci_low"] for s in sel_types])
        ci_highs = np.array([per_sel[s][norm_key]["ci_high"] for s in sel_types])

        # Compute error bars: distance from mean to CI bounds
        lower_errs = means - ci_lows
        upper_errs = ci_highs - means
        errs = np.array([lower_errs, upper_errs])

        ax.bar(
            x,
            means * 100,
            yerr=errs * 100,
            capsize=2,
            color=colors,
            error_kw={"elinewidth": 1.5, "ecolor": "dimgray"},
        )
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_ylim(lims[task])
        ax.set_xticks([])
        ax.set_ylabel("Norm. accuracy drop (%)" if norm_drop else "Accuracy drop (%)")
        ax.set_title(datasets[task])
        clean_axes(ax)

    fig.suptitle(fig_title)
    plt.tight_layout()
    if out_path:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()


# --------------------------- SUPPLEMENTARY FIGURES -------------------------- #
def plot_perc_layers_mixed(model_dict: dict, fig_title: str=None, ylim: tuple=(0, 5),
                           n_grid: int=10, out_path: Path=None):
    """Line plot: Percentage of selective units across layers (mixed selectivity)."""
    mixed_sel_types = ["face&body", "face&scene", "body&scene"]
    mixed_sel_info = {
        "face&body":  {"color": "#648FFF", "label": "Face & Body"},
        "face&scene": {"color": "#AE3ECC", "label": "Face & Scene"},
        "body&scene": {"color": "#00C43E", "label": "Body & Scene"},
    }

    fig, ax = plt.subplots(figsize=(7, 3.5))

    get_layers = lambda model, i: model_dict[model]

    # Shared normalized layer depth grid
    x_grid = np.linspace(0, 1, n_grid)

    # Store interpolated curves for each mixed sel type
    all_curves = {sel: [] for sel in mixed_sel_types}

    for m, model in enumerate(model_dict.keys()):
        layers = get_layers(model, m)
        res = load_pickle(PROJECT_ROOT / "selectivity" / model / "floc_res_mixed.pkl")

        x_model = np.linspace(0, 1, len(layers))
        totals = np.array(
            [len(res[lay]["stats"]["face"]["dvals"]) for lay in layers],
            dtype=float,
        )

        for sel in mixed_sel_types:
            counts = np.array(
                [len(res[lay]["unit_ids"].get(sel, [])) for lay in layers],
                dtype=float,
            )
            vec = np.divide(
                100.0 * counts,
                totals,
                out=np.full_like(totals, np.nan),
                where=totals != 0,
            )

            ok = ~np.isnan(vec)
            if ok.sum() < 2:
                interp_vec = np.full_like(x_grid, np.nan, dtype=float)
            else:
                interp_vec = np.interp(x_grid, x_model[ok], vec[ok])

            all_curves[sel].append(interp_vec)

    # Aggregate and plot
    for sel in mixed_sel_types:
        arr = np.vstack(all_curves[sel])
        n = arr.shape[0]
        mean_vec = np.nanmean(arr, axis=0)
        sem_vec = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n)

        ax.plot(
            x_grid * 100,
            mean_vec,
            color=mixed_sel_info[sel]["color"],
            lw=2.5,
            label=mixed_sel_info[sel]["label"],
        )
        ax.fill_between(
            x_grid * 100,
            mean_vec - sem_vec,
            mean_vec + sem_vec,
            color=mixed_sel_info[sel]["color"],
            alpha=0.2,
            linewidth=0,
        )

        ok = ~np.isnan(mean_vec)
        rho, p = spearmanr(x_grid[ok], mean_vec[ok]) if ok.sum() > 1 else (np.nan, np.nan)
        max_val = np.nanmax(mean_vec)
        print(f"{sel}: r={rho:.2f}, p={p:.3f}, max={max_val:.2f}%")

    ax.set_xlim(-5, 105)
    ax.set_xticks(np.linspace(0, 100, 5))
    ax.set_xticklabels([f"{v:.0f}" for v in np.linspace(0, 100, 5)])
    ax.set_xlabel("Layer depth (%)")
    ax.set_ylim(ylim)
    ax.set_ylabel("Selective units (%)")
    clean_axes(ax)

    sel_handles = [
        mlines.Line2D([], [], color=mixed_sel_info[sel]["color"], lw=2.5)
        for sel in mixed_sel_types
    ]
    sel_labels = [mixed_sel_info[sel]["label"] for sel in mixed_sel_types]
    leg = ax.legend(
        sel_handles,
        sel_labels,
        title="Unit type",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )
    leg._legend_box.align = "left"

    ax.set_title(fig_title)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_dprime_layers(model_dict: dict, img_db: str="validation",
                       fig_title: str=None, ylim: tuple=(-0.5, 1),
                       out_path: Path=None):
    """Line plot: Mean d' for each unit type across layers (mean ± SEM across models)."""
    sel_types = list(SEL_INFO.keys())[:3]
    type_colors = [SEL_INFO[sel]["color"] for sel in sel_types]

    if img_db not in ("floc", "validation"):
        raise ValueError(f"img_db must be 'floc' or 'validation', got {img_db!r}")

    # Shared normalized layer depth grid
    n_grid = 10
    x_grid = np.linspace(0, 1, n_grid)

    # Load validation d' if necessary
    val_data = {}
    if img_db == "validation":
        for model in model_dict.keys():
            val_data[model] = load_pickle(PROJECT_ROOT / "selectivity" / model / "validation_dprime.pkl")

    # Store interpolated mean-d' curves per model per sel type
    all_curves = {sel: [] for sel in sel_types}

    for model in model_dict.keys():
        layers = model_dict[model]
        res = load_pickle(PROJECT_ROOT / "selectivity" / model / "floc_res.pkl")

        x_model = np.linspace(0, 1, len(layers))

        # Compute per-layer mean d' for each sel type
        means_per_sel = {sel: [] for sel in sel_types}
        for lay in layers:
            sel_ids = res[lay]["unit_ids"]

            if img_db == "floc":
                floc_stats = res[lay]["stats"]
                d_all = {
                    sel: np.array(floc_stats[sel].get("dvals", []))
                    for sel in sel_types
                }
                for sel in sel_types:
                    ids = sel_ids.get(sel, [])
                    if len(ids) > 0 and d_all[sel].size > 0:
                        m_val, _ = compute_mean_ci(d_all[sel][ids])
                    else:
                        m_val = np.nan
                    means_per_sel[sel].append(m_val)

            else: # Validation
                vp = val_data.get(model, {})
                dvals_lay = vp.get(lay, {})
                d_all = {
                    "face":  np.asarray(dvals_lay.get("face_d",  [])),
                    "body":  np.asarray(dvals_lay.get("body_d",  [])),
                    "mixed": np.asarray(dvals_lay.get("mixed_d", [])),
                }
                for sel in sel_types:
                    ids = np.asarray(sel_ids.get(sel, []), dtype=int)
                    if ids.size > 0:
                        m_val, _ = compute_mean_ci(d_all[sel][ids])
                    else:
                        m_val = np.nan
                    means_per_sel[sel].append(m_val)

        # Interpolate each sel type onto shared x_grid
        for sel in sel_types:
            vec = np.array(means_per_sel[sel])
            ok = ~np.isnan(vec)
            if ok.sum() < 2:
                interp_vec = np.full_like(x_grid, np.nan, dtype=float)
            else:
                interp_vec = np.interp(x_grid, x_model[ok], vec[ok])
            all_curves[sel].append(interp_vec)

    # Aggregate and plot
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axhline(0, color="gray", ls="--", lw=1, zorder=0)

    for sel, color in zip(sel_types, type_colors):
        arr = np.vstack(all_curves[sel])
        n = arr.shape[0]
        mean_vec = np.nanmean(arr, axis=0)
        sem_vec  = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n)

        ax.plot(x_grid * 100, mean_vec, color=color, lw=2.5)
        ax.fill_between(
            x_grid * 100,
            mean_vec - sem_vec,
            mean_vec + sem_vec,
            color=color,
            alpha=0.2,
            linewidth=0,
        )

        ok = ~np.isnan(mean_vec)
        rho, p = spearmanr(x_grid[ok], mean_vec[ok]) if ok.sum() > 1 else (np.nan, np.nan)
        print(f"{sel}: r={rho:.2f}, p={p:.3f}")

    ax.set_xlim(-5, 105)
    ax.set_xticks(np.linspace(0, 100, 5))
    ax.set_xticklabels([f"{v:.0f}" for v in np.linspace(0, 100, 5)])
    ax.set_xlabel("Layer depth (%)")
    ax.set_ylim(ylim)
    ax.set_ylabel("Selectivity (d')")
    clean_axes(ax)

    sel_handles = [
        mlines.Line2D([], [], color=SEL_INFO[sel]["color"], lw=2.5)
        for sel in sel_types
    ]
    sel_labels = [SEL_INFO[sel]["label"] for sel in sel_types]
    leg = ax.legend(
        sel_handles, sel_labels,
        title="Unit type",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )
    leg._legend_box.align = "left"

    ax.set_title(fig_title)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_encoding_sep_layers(model_name: str, layers: list, rois: list,
                             r2_adj: bool=False, fig_title: str=None,
                             out_path: Path=None):
    """Line plot: Mean R² of unit type across layers per ROI."""
    res = load_pickle(PROJECT_ROOT / "encoding" / model_name / "sep.pkl")
    fmri = load_pickle(PROJECT_ROOT / "nsd" / "fmri_activs.pkl")

    sel_types = list(SEL_INFO.keys())
    sel_keys = ["f", "b", "m", "ns"]
    sel_key_map = dict(zip(sel_types, sel_keys))

    # Aggregate per-subject R² per ROI and layer
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

    # Within-subject means and 95% CIs for each ROI and layer
    ci_data = {roi: {} for roi in rois}
    for roi in rois:
        for lay in layers:
            data = np.stack([scores[roi][lay][sel] for sel in sel_types], axis=1)
            valid = ~np.isnan(data).any(axis=1)
            if valid.any():
                means, ci_half = compute_mean_ci_within(data[valid])
            else:
                means = np.zeros(len(sel_types))
                ci_half = np.zeros(len(sel_types))
            ci_data[roi][lay] = (means, ci_half)

    # Plot grid of ROIs
    n = len(rois)
    n_rows = 2
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    x = np.arange(len(layers))

    for idx, roi in enumerate(rois):
        ax = axes[idx // n_cols][idx % n_cols]

        # Noise-ceiling ribbon across layers
        if not r2_adj:
            nc_vals = [
                np.nanmean(roi_dict["ncsnr"])
                for s in layer_res
                if (roi_dict := fmri.get(s, {}).get(roi)) is not None
            ]
            if nc_vals:
                lo, hi = np.percentile(nc_vals, [2.5, 97.5])
                ax.fill_between(x, lo, hi, color="gray", alpha=0.3, lw=0, zorder=1)

        # Plot mean R² and 95% CI ribbons per selection type
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

        ax.set_title(roi)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=30, ha="right")
        if idx % n_cols == 0:
            ax.set_ylabel("Explained variance (R² adj.)" if r2_adj else "Explained variance (R²)")
        ax.set_ylim(bottom=0)
        clean_axes(ax)
    fig.suptitle(fig_title)

    for j in range(n, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols][j % n_cols])

    # Legend
    sel_handles = [
        mlines.Line2D([], [], color=SEL_INFO[s]["color"], lw=2, label=SEL_INFO[s]["label"])
        for s in sel_types
    ]
    handles = sel_handles[:]
    if not r2_adj:
        handles.append(mpatches.Patch(facecolor="gray", alpha=0.3, label="Noise ceiling"))

    leg = fig.legend(
        handles,
        [h.get_label() for h in handles],
        title="Unit type",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False,
    )
    leg._legend_box.align = "left"

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()

def plot_encoding_varpart_layers(model_name: str, layers: list, rois: list,
                                 r2_adj: bool=False, show_delta: bool=True,
                                 fig_title: str=None, out_path: Path=None):
    """Line plot: Mean R² components (variance partitioning) across layers per ROI."""
    res_fb = load_pickle(PROJECT_ROOT / "encoding" / model_name / "fb_varpart.pkl")
    res_delta = load_pickle(PROJECT_ROOT / "encoding" / model_name / "delta_m.pkl") if show_delta else {}
    fmri = load_pickle(PROJECT_ROOT / "nsd" / "fmri_activs.pkl")

    comp_keys = ["u_f", "u_b", "s_fb"] + (["delta_m"] if show_delta else [])
    comp_labels = ["Face", "Body", "Face ∩ Body"] + (["Δ Mixed"] if show_delta else [])
    comp_colors = [
        SEL_INFO["face"]["color"],
        SEL_INFO["body"]["color"],
        "#F9CEB0"
    ] + (["#648fff"] if show_delta else [])

    # Aggregate per-subject values per ROI/layer/component
    scores = {roi: {lay: {k: [] for k in comp_keys} for lay in layers} for roi in rois}
    for lay in layers:
        fb_layer = res_fb.get(lay, {}) or {}
        delta_layer = (res_delta.get(lay, {}) if show_delta else {}) or {}

        subj_keys = set(fb_layer.keys()) | set(delta_layer.keys())
        for subj in subj_keys:
            subj_fmri = fmri.get(subj, {}) or {}

            for roi in rois:
                fmri_roi = subj_fmri.get(roi)
                noise = (
                    np.nanmean(fmri_roi["ncsnr"])
                    if (isinstance(fmri_roi, dict) and "ncsnr" in fmri_roi) else np.nan
                )

                comp_roi = (fb_layer.get(subj, {}) or {}).get(roi, {}) or {}
                delta_roi = (delta_layer.get(subj, {}) or {}).get(roi, {}) or {}

                for key in comp_keys:
                    arr = delta_roi.get("delta_m") if key == "delta_m" else comp_roi.get(key)
                    if arr is None:
                        val = np.nan
                    else:
                        base = np.nanmean(arr)
                        if r2_adj and np.isfinite(noise) and noise != 0:
                            val = base / noise
                        else:
                            val = base
                    scores[roi][lay][key].append(val)

    # Fixed-effects within-subject means + 95% CI half-widths
    ci_data = {roi: {} for roi in rois}
    for roi in rois:
        for lay in layers:
            mat = np.stack([np.asarray(scores[roi][lay][k], dtype=float) for k in comp_keys], axis=1)
            valid = ~np.isnan(mat).any(axis=1)
            if valid.any():
                means, ci_half = compute_mean_ci_within(mat[valid])
            else:
                means = np.zeros(len(comp_keys))
                ci_half = np.zeros(len(comp_keys))
            ci_data[roi][lay] = (means, ci_half)

    # Line plot
    n = len(rois)
    n_rows = 2
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    x = np.arange(len(layers))

    for idx, roi in enumerate(rois):
        ax = axes[idx // n_cols][idx % n_cols]

        # Noise-ceiling ribbon across layers
        if not r2_adj:
            subj_ncs = [
                np.nanmean(fmri[s][roi]["ncsnr"])
                for s in fmri
                if isinstance(fmri.get(s), dict)
                and roi in fmri[s]
                and fmri[s][roi] is not None
                and "ncsnr" in fmri[s][roi]
            ]
            if subj_ncs and np.isfinite(subj_ncs).any():
                lo, hi = np.nanpercentile(subj_ncs, [2.5, 97.5])
                ax.fill_between(
                    x,
                    np.full_like(x, lo, dtype=float),
                    np.full_like(x, hi, dtype=float),
                    color="gray", alpha=0.3, lw=0, zorder=1
                )

        # Mean + CI ribbons for each component
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

        ax.set_title(roi)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=30, ha="right")
        if idx % n_cols == 0:
            ax.set_ylabel("Explained variance (R² adj.)" if r2_adj else "Explained variance (R²)")
        ax.set_ylim(bottom=0)
        clean_axes(ax)
    fig.suptitle(fig_title)

    for j in range(n, n_rows * n_cols):
        fig.delaxes(axes[j // n_cols][j % n_cols])

    # Legend
    handles = [mlines.Line2D([], [], color=c, lw=2) for c in comp_colors]
    leg = fig.legend(
        handles, comp_labels,
        title="Variance portion",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False
    )
    leg._legend_box.align = "left"

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()


# ----------------------------------- Utils ---------------------------------- #
def compute_mean_ci(data: np.ndarray, n_iter: int=10000):
    """Compute mean and 95% CI via bootstrapping."""
    if len(data) == 0:
        return np.nan, np.nan

    n = len(data)
    boot_means = np.array([
        np.mean(data[np.random.choice(n, size=n, replace=True)], axis=0)
        for _ in range(n_iter)
    ])
    lower = np.percentile(boot_means, 2.5, axis=0)
    upper = np.percentile(boot_means, 97.5, axis=0)
    ci = (upper - lower) / 2

    return np.mean(data, axis=0), ci

def compute_mean_ci_hierarch_models_layers_units(
    models_data: dict,
    n_iter: int=10000,
    rng: np.random.Generator=None,
):
    """Hierarchical bootstrapping: models → layers → units."""
    if rng is None:
        rng = np.random.default_rng()

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
    """Within-subject CI on condition means."""
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
