import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as t_dist
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
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

MODEL_STYLES = ["-", "--"]

SEL_INFO = {
    "face": {"label": "Face-selective", "color": "#dc267fff"},
    "body": {"label": "Body-selective", "color": "#ffb000ff"},
    "mixed": {"label": "Mixed-selective", "color": "#648fffff"},
    "nonselective":{"label": "Non-selective", "color": "#ccccccff"},
}

# ------------------------------- MAIN FIGURES ------------------------------- #
# -------------------------------- Selectivity ------------------------------- #
def plot_perc_layers(model_names, model_names_legend, layers,
                     fig_title=None, out_path=None):
    """Line plot: Percentage of selective units across layers."""
    sel_types = list(SEL_INFO.keys())[:3]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(layers))

    for m, model in enumerate(model_names):
        res = load_pickle(PROJECT_ROOT / "selectivity" / model / "floc_res.pkl")

        # Total units per layer
        totals = np.array([len(res[lay]["stats"]["face"]["dvals"]) for lay in layers], dtype=float)

        # Percent selective per layer
        data = {}
        for sel in sel_types:
            counts = np.array([len(res[lay]["unit_ids"].get(sel, [])) for lay in layers], dtype=float)
            data[sel] = np.divide(100.0 * counts, totals, out=np.full_like(totals, np.nan), where=totals != 0)

        # Plot each selective unit type
        for sel in sel_types:
            ax.plot(
                x, data[sel],
                color=SEL_INFO[sel]["color"],
                ls=MODEL_STYLES[m],
                alpha=0.8
            )

        # Compute and print stats
        print(f"{model}: Spearman r of proportion and layer depth")
        for sel in sel_types:
            vec = np.array(data[sel], dtype=float)
            ok = ~np.isnan(vec)
            rho, p = spearmanr(x[ok], vec[ok])
            max_val = np.nanmax(vec)
            mean_val = np.nanmean(vec)

            # Mean for fc layers only
            fc_mask = np.array(['fc' in lay for lay in layers])
            mean_fc = np.nanmean(vec[fc_mask]) if fc_mask.any() else np.nan

            print(f"{sel}: r={rho:.2f}, p={p:.3f}, max={max_val:.2f}%, mean={mean_val:.2f}%, mean_fc={mean_fc:.2f}%")

    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.set_ylim(0, 12)
    ax.set_ylabel("Selective units (%)")
    clean_axes(ax)

    # Legend
    # Models
    model_handles = [
        mlines.Line2D([], [], color="gray", ls=MODEL_STYLES[m])
        for m, _ in enumerate(model_names_legend)
    ]
    leg1 = ax.legend(
        model_handles, model_names_legend,
        title="Model", # Dataset
        loc="center left",
        bbox_to_anchor=(1, 0.7),
        frameon=False)
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)

    # Unit types
    sel_handles = [
        mlines.Line2D([], [], color=SEL_INFO[sel]["color"])
        for sel in sel_types
    ]
    sel_labels = [SEL_INFO[sel]["label"] for sel in sel_types]
    leg2 = ax.legend(
        sel_handles, sel_labels,
        title="Unit type",
        loc="center left",
        bbox_to_anchor=(1, 0.2),
        frameon=False)
    leg2._legend_box.align = "left"

    ax.set_title(fig_title)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()

def plot_dprime_dist(model_names, model_names_legend, layers,
                     fig_title=None, out_path=None):
    """Violin plot: Face and body selectivity per unit type."""
    sel_types = list(SEL_INFO.keys())
    sel_labels = [SEL_INFO[sel]["label"] for sel in sel_types]

    stats_data = {
        model: {
            sel: {"face": {}, "body": {}}
            for sel in sel_types + ["nonselective"]
        }
        for model in model_names_legend
    }

    rows = []
    for model, model_legend in zip(model_names, model_names_legend):
        res = load_pickle(PROJECT_ROOT / "selectivity" / model / "floc_res.pkl")
        dvals = load_pickle(PROJECT_ROOT / "selectivity" / model / "validation_dprime.pkl")

        for lay in layers:
            dvals_f = np.array(dvals[lay]["face_d"])
            dvals_b = np.array(dvals[lay]["body_d"])
            all_u = dvals_f.shape[0]

            groups = {
                sel: np.asarray(res[lay]["unit_ids"].get(sel, np.array([], dtype=int)), dtype=int)
                for sel in sel_types
            }
            sel_u = np.concatenate(list(groups.values()))
            groups["nonselective"] = np.setdiff1d(np.arange(all_u), sel_u)

            for sel, ids in groups.items():
                if ids.size == 0:
                    continue
                unit_label = SEL_INFO[sel]["label"]

                # Store by layer for hierarchical bootstrap
                stats_data[model_legend][sel]["face"][lay] = dvals_f[ids]
                stats_data[model_legend][sel]["body"][lay] = dvals_b[ids]

                # For violin plot
                rows.extend([(unit_label, "Face d'", float(v), model_legend) for v in dvals_f[ids]])
                rows.extend([(unit_label, "Body d'", float(v), model_legend) for v in dvals_b[ids]])

    df = pd.DataFrame(rows, columns=["unit_type", "selectivity", "dprime", "model"])

    # Require two models for split violin
    models_present = df["model"].unique().tolist()
    if len(models_present) != 2:
        raise ValueError("Requires exactly two distinct models for split violin.")

    model1, model2 = model_names_legend
    if set(models_present) != {model1, model2}:
        raise ValueError("model_names_legend must match the two models present in df.")
    n_types = len(sel_types)

    # Split df
    df_face = df[df["selectivity"] == "Face d'"].copy()
    df_body = df[df["selectivity"] == "Body d'"].copy()

    fig, (ax_face, ax_body) = plt.subplots(1, 2, figsize=(6, 2.5), sharey=False)
    y_min, y_max = -1.5, 2.0

    width = 0.8
    half_w = width / 2.0
    offsets = {model1: -half_w / 2.0, model2: +half_w / 2.0}

    for ax, subset_df, title in zip((ax_face, ax_body), (df_face, df_body), ("Face d'", "Body d'")):
        for s, lab in enumerate(sel_labels):
            base_color = SEL_INFO[sel_types[s]]["color"]

            vals1 = subset_df[
                (subset_df["unit_type"] == lab) & (subset_df["model"] == model1)
            ]["dprime"].values
            vals2 = subset_df[
                (subset_df["unit_type"] == lab) & (subset_df["model"] == model2)
            ]["dprime"].values

            # Clip rectangles
            clip_left = Rectangle(
                (s - half_w, y_min - 1e-3),
                half_w,
                (y_max - y_min) + 2e-3,
                transform=ax.transData
            )
            clip_right = Rectangle(
                (s, y_min - 1e-3),
                half_w,
                (y_max - y_min) + 2e-3,
                transform=ax.transData
            )

            # Left half: filled
            if vals1.size > 0:
                parts1 = ax.violinplot(
                    vals1, positions=[s], widths=width,
                    showmeans=False, showmedians=False, showextrema=False
                )
                for body in parts1["bodies"]:
                    body.set_facecolor(base_color)
                    body.set_edgecolor(None)
                    body.set_alpha(1.0) # 0.8
                    body.set_clip_path(clip_left)

            # Right half: hatched
            if vals2.size > 0:
                parts2 = ax.violinplot(
                    vals2, positions=[s], widths=width,
                    showmeans=False, showmedians=False, showextrema=False
                )
                for body in parts2["bodies"]:
                    body.set_facecolor("none")
                    body.set_edgecolor(base_color)
                    body.set_hatch("///")
                    body.set_alpha(1.0) # 0.8
                    body.set_clip_path(clip_right)

            # Overlay means & CIs
            for mdl, vals in ((model1, vals1), (model2, vals2)):
                if vals.size == 0:
                    continue

                # Get stats
                sel_type = sel_types[s]
                if title == "Face d'":
                    layer_dict = stats_data[mdl][sel_type]["face"]
                else:
                    layer_dict = stats_data[mdl][sel_type]["body"]

                m_val, ci = compute_mean_ci_hierarch(layer_dict, n_iter=5000)

                ax.errorbar(
                    s + offsets[mdl], m_val,
                    yerr=ci,
                    fmt="o",
                    color="darkgray",
                    capsize=3,
                    markersize=4,
                    zorder=5,
                )

        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.set_xlim(-0.6, n_types - 0.4)
        ax.axhline(0, color="gray", ls="--", lw=1, zorder=0)
        ax.set_ylabel("Selectivity (d')" if title == "Face d'" else "")
        ax.set_ylim(y_min, y_max)
        ax.grid(False)
        clean_axes(ax)

    # Legend for models
    model_handles = [
        mlines.Line2D([], [], marker="s", ls="none", color="gray"),
        mlines.Line2D([], [], marker="s", ls="none", color="lightgray")
    ]
    ax_body.legend(
        model_handles, model_names_legend,
        title="Model",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        frameon=False
    )

    plt.suptitle(fig_title)
    plt.tight_layout()
    if out_path:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()

def plot_guided_gradcam(model_name, target_layer, cam_layer=None,
                        controlled=True, device="cuda",
                        img_path=None, n_imgs=3, norm_types=True, seed=None,
                        fig_title=None, out_path=None):
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
def plot_encoding_sep(model_name, layer, rois, r2_adj=False,
                      fig_title=None, out_path=None):
    """Bar plot: Mean R² of unit types per ROI."""
    res = load_pickle(PROJECT_ROOT / "encoding" / model_name / "sep.pkl")[layer]
    fmri = load_pickle(PROJECT_ROOT / "nsd" / "fmri_activs.pkl")

    sel_types = list(SEL_INFO.keys())
    sel_keys = ["f", "b", "m", "ns"]
    sel_key_map = dict(zip(sel_types, sel_keys))

    # Collect per-subject scores
    scores = {roi: {sel: [] for sel in sel_types} for roi in rois}
    for subj, roi_map in res.items():
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
                scores[roi][sel].append(val / nc_mean if r2_adj else val)

    # Fixed-effects within-subject means + 95% CI half-widths
    ci_data = {}
    for roi in rois:
        data = np.stack([scores[roi][sel] for sel in sel_types], axis=1)
        valid = ~np.isnan(data).any(axis=1)
        ci_data[roi] = compute_mean_ci(data[valid])

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rois))
    width = 0.2
    offsets = (np.arange(len(sel_types)) - (len(sel_types) - 1) / 2) * width

    # Noise ceiling ribbon
    if not r2_adj:
        total = width * len(sel_types)
        for i, roi in enumerate(rois):
            nc_vals = [
                np.nanmean(roi_dict["ncsnr"])
                for s in res
                if (roi_dict := fmri.get(s, {}).get(roi)) is not None
            ]
            if nc_vals:
                lo, hi = np.percentile(nc_vals, [2.5, 97.5])
                ax.fill_between([i - total / 2, i + total / 2], lo, hi,
                                color="gray", alpha=0.3, lw=0)

    # Grouped bars and subject means
    for idx, sel in enumerate(sel_types):
        pos = x + offsets[idx]
        means = [ci_data[r][0][idx] for r in rois]
        errs = [ci_data[r][1][idx] for r in rois]

        info = SEL_INFO[sel]
        ax.bar(
            pos, means, width,
            color=info["color"],
            yerr=errs, capsize=0,
            error_kw={"elinewidth": 1.5, "ecolor": "dimgray"},
            label=info["label"],
        )

        for j, roi in enumerate(rois):
            vals = scores[roi][sel]
            if vals:
                ax.scatter([pos[j]] * len(vals), vals, color="gray", alpha=0.5, s=5)

    ax.set_ylim(0)
    ax.set_title(fig_title)
    ax.set_ylabel("Explained variance (R² adj.)" if r2_adj else "Explained variance (R²)")
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=30, ha='right')
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

def plot_encoding_varpart(model_name, layer, rois, r2_adj=False, show_delta=True,
                          fig_title=None, out_path=None):
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
              "#F9CEB0"
    ] + (["#648FFF"] if show_delta else [])

    scores = {roi: {k: {s: np.nan for s in subjects} for k in keys} for roi in rois}

    for subj in subjects:
        fb_map = res_fb.get(subj, {})
        subj_fmri = fmri.get(subj, {})

        for roi in rois:
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
    for roi in rois:
        mat = np.vstack([[scores[roi][k][s] for s in subjects] for k in keys]).T
        mat = mat.astype(float)
        valid = ~np.isnan(mat).any(axis=1)
        if valid.any():
            means, ci = compute_mean_ci_within(mat[valid])
        else:
            means = np.full(len(keys), np.nan)
            ci = np.full(len(keys), np.nan)
        means = np.asarray(means, dtype=float)
        ci = np.asarray(ci, dtype=float)
        mean_ci[roi] = (means, ci)

    # Plot bars
    n_keys = len(keys)
    x = np.arange(len(rois))
    width = 0.8 / n_keys
    spacing = width
    offsets = (np.arange(n_keys) - (n_keys-1)/2) * spacing

    fig, ax = plt.subplots(figsize=(8, 4))

    # Noise ceiling
    if not r2_adj:
        total_w = 0.8
        for i, roi in enumerate(rois):
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
                    [i-total_w/2, i+total_w/2],
                    lo, hi,
                    color="gray", alpha=0.3, lw=0, zorder=0
                )

    # Grouped bars and subject means
    for j, (lab, col) in enumerate(zip(labels, colors)):
        pos = x + offsets[j]
        means = [mean_ci[roi][0][j] for roi in rois]
        errs = [mean_ci[roi][1][j] for roi in rois]

        ax.bar(
            pos, means, width,
            color=col, label=lab,
            yerr=errs, capsize=0,
            error_kw={"elinewidth": 1.5, "ecolor": "dimgray"},
        )

        # Per-subject dots
        k = keys[j]
        for ri, roi in enumerate(rois):
            # vals = [v for v in scores[roi][k] if np.isfinite(v)]
            vals = [scores[roi][k][s] for s in subjects]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                ax.scatter([pos[ri]] * len(vals), vals, color="gray", alpha=0.5, s=5)

    ax.set_ylim(0, None)
    ax.set_xticks(x)
    ax.set_xticklabels(rois, rotation=30, ha="right")
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
        frameon=False
    )
    leg._legend_box.align = "left"

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()

def plot_varpart_integrated(model_name, layers,
                            face_rois=("OFA", "FFA", "aTL-faces"),
                            body_rois=("EBA", "FBA", "mTL-bodies"),
                            r2_adj=False, include_delta=True,
                            show_group_mean=True, fig_title=None,
                            out_path=None):
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
def plot_lesioning_drop(model_name, tasks=("face", "person", "action"), norm_drop=True,
                        fig_title=None, out_path=None):
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


# -------------------------------- Integration ------------------------------- #
def plot_integration_coefs(model_name, layer, coefs=["beta_F", "beta_B", "beta_FB"],
                           n_boot=10000, random_state=0,
                           fig_title=None, out_path=None):
    """Violin plots with bootstrap CIs for a single layer."""
    sel_types = list(SEL_INFO.keys())
    colors = {s: SEL_INFO[s]["color"] for s in sel_types}

    coef_labels = {
        "beta_F": "Face",
        "beta_B": "Body",
        "beta_FB": "Interaction",
    }

    res = load_pickle(PROJECT_ROOT / "fb_integration" / model_name / "fb_integration.pkl")

    # Collect data per type/coefficient for the specified layer
    data = {coef: {} for coef in coefs}
    present_types = set()

    layer_dict = res.get(layer, {})
    for coef in coefs:
        for s in sel_types:
            df = layer_dict.get(s)
            if df is None or df.empty or (coef not in df.columns):
                continue

            vals = df[coef].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size:
                data[coef][s] = vals
                present_types.add(s)

    present_types = [s for s in sel_types if s in present_types]

    if not present_types:
        raise ValueError(f"No coefficient data found for layer {layer}.")

    n_coefs = len(coefs)
    fig, axes = plt.subplots(1, n_coefs, figsize=(3 * n_coefs, 2.5), sharey=True)
    if n_coefs == 1:
        axes = [axes]

    rng = np.random.default_rng(random_state)

    for ax_idx, (ax, coef) in enumerate(zip(axes, coefs)):

        # Bootstrap per sel_type
        violin_data = []
        means = []
        medians = []
        cis_lower = []
        cis_upper = []
        x_pos = []

        for si, sel in enumerate(present_types):
            vals = data[coef].get(sel)
            if vals is None or vals.size == 0:
                continue

            # Bootstrap the mean for this selectivity type
            boot_means = np.zeros(n_boot)
            for b in range(n_boot):
                boot_sample = rng.choice(vals, size=len(vals), replace=True)
                boot_means[b] = np.mean(boot_sample)

            boot_means = boot_means[np.isfinite(boot_means)]
            violin_data.append(boot_means)
            means.append(np.mean(boot_means))
            medians.append(np.median(boot_means))
            cis_lower.append(np.percentile(boot_means, 2.5))
            cis_upper.append(np.percentile(boot_means, 97.5))

            x_pos.append(si)

        if not violin_data:
            continue

        # Violin plot
        parts = ax.violinplot(
            violin_data,
            positions=x_pos,
            widths=0.5,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        # Color violins by sel_type
        for pc, sel in zip(parts["bodies"], present_types):
            pc.set_facecolor(colors[sel])
            pc.set_edgecolor("none")
            pc.set_alpha(1)

        # Error bars + median dots
        ax.errorbar(
            x_pos,
            means,
            [[m - lo for m, lo in zip(means, cis_lower)],
             [hi - m for m, hi in zip(means, cis_upper)]],
            fmt="none",
            capsize=2,
            elinewidth=1.5,
            ecolor="dimgray",
            zorder=3,
        )
        ax.scatter(x_pos, medians, s=12, color="dimgray", zorder=4)
        ax.set_ylim(-0.5, 1.7)
        ax.axhline(0, color="gray", ls="--", lw=0.8, zorder=0)
        ax.set_title(coef_labels[coef], fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Coefficient (β)")
        ax.set_xticks([])
        clean_axes(ax)

    if fig_title:
        fig.suptitle(fig_title)
    plt.tight_layout()

    if out_path:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()


# --------------------------- SUPPLEMENTARY FIGURES -------------------------- #
def plot_perc_layers_mixed(model_names, model_names_legend, layers, ylim=(0, 5),
                           fig_title=None, out_path=None):
    """Line plot: Percentage of selective units across layers (mixed selectivity)."""
    # Define mixed selectivity types and colors
    mixed_sel_types = ["face&body", "face&scene", "body&scene"]
    mixed_sel_info = {
        "face&body": {"color": "#648FFF", "label": "Face & Body"},
        "face&scene": {"color": "#AE3ECC", "label": "Face & Scene"},
        "body&scene": {"color": "#00C43E", "label": "Body & Scene"}
    }

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(layers))

    for m, model in enumerate(model_names):
        res = load_pickle(PROJECT_ROOT / "selectivity_mixed" / model / "floc_res.pkl")

        # Total units per layer
        totals = np.array([len(res[lay]["stats"]["face"]["dvals"]) for lay in layers], dtype=float)

        # Percent selective per layer
        data = {}
        for sel in mixed_sel_types:
            counts = np.array([len(res[lay]["unit_ids"].get(sel, [])) for lay in layers], dtype=float)
            data[sel] = np.divide(100.0 * counts, totals, out=np.full_like(totals, np.nan), where=totals != 0)

        # Plot each mixed selective unit type
        for sel in mixed_sel_types:
            ax.plot(
                x, data[sel],
                color=mixed_sel_info[sel]["color"],
                ls=MODEL_STYLES[m],
                alpha=0.8
            )

        # Compute and print Spearman correlation and max percentage
        print(f"{model}: Spearman r of proportion and layer depth")
        for sel in mixed_sel_types:
            vec = np.array(data[sel], dtype=float)
            ok = ~np.isnan(vec)
            rho, p = spearmanr(x[ok], vec[ok])
            max_val = np.nanmax(vec)
            print(f"{sel}: r={rho:.2f}, p={p:.3f}, max={max_val:.2f}%")

    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.set_ylim(ylim)
    ax.set_ylabel("Selective units (%)")
    clean_axes(ax)

    # Legend
    # Models
    model_handles = [
        mlines.Line2D([], [], color="gray", ls=MODEL_STYLES[m])
        for m, _ in enumerate(model_names_legend)
    ]
    leg1 = ax.legend(
        model_handles, model_names_legend,
        title="Model",
        loc="center left",
        bbox_to_anchor=(1, 0.7),
        frameon=False)
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)

    # Unit types
    sel_handles = [
        mlines.Line2D([], [], color=mixed_sel_info[sel]["color"])
        for sel in mixed_sel_types
    ]
    sel_labels = [mixed_sel_info[sel]["label"] for sel in mixed_sel_types]
    leg2 = ax.legend(
        sel_handles, sel_labels,
        title="Unit type",
        loc="center left",
        bbox_to_anchor=(1, 0.2),
        frameon=False)
    leg2._legend_box.align = "left"

    ax.set_title(fig_title)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()

def plot_dprime_layers(model_names, layers, img_db="validation",
                       fig_title=None, subplot_titles=None, out_path=None):
    """Line plot: Mean d' for each unit type across layers."""
    sel_types = list(SEL_INFO.keys())[:3]
    type_colors = [SEL_INFO[sel]["color"] for sel in sel_types]
    ds_linestyles = {"ecoset": "-", "imagenet": "--"}

    if img_db not in ("floc", "validation"):
        raise ValueError(f"img_db must be 'floc' or 'validation', got {img_db!r}")

    # Load validation d' if necessary
    val_data = {}
    if img_db == "validation":
        for model in model_names:
            val_data[model] = load_pickle(PROJECT_ROOT / "selectivity" / model / "validation_dprime.pkl")

    nplots = 1 if not subplot_titles else len(subplot_titles)
    fig, axes = plt.subplots(1, nplots, figsize=(6*nplots, 5), sharey=(nplots>1))
    if nplots == 1:
        axes = [axes]

    # Split models across subplots
    groups = np.array_split(model_names, 2) if nplots == 2 else [model_names]

    # Plot each group
    x = np.arange(len(layers))
    titles = (subplot_titles or [fig_title])

    for ax, group, title in zip(axes, groups, titles):
        for model in group:
            res = load_pickle(PROJECT_ROOT / "selectivity" / model / "floc_res.pkl")

            # Decide linestyle from model name
            low = model.lower()
            if "ecoset" in low:
                ds = "ecoset"
            elif "imagenet" in low:
                ds = "imagenet"
            else:
                ds = None
            ls = ds_linestyles.get(ds, "-")

            # Mean and CI per layer per sel_type
            stats = {sel: [] for sel in sel_types}
            for lay in layers:
                sel_ids = res[lay]["unit_ids"]

                if img_db == "floc":
                    # floc_stats = res[lay]["dvals"]
                    floc_stats = res[lay]["stats"]
                    d_all = {
                        # sel: np.array(floc_stats.get(sel, []))
                        sel: np.array(floc_stats[sel].get("dvals", []))
                        for sel in sel_types
                    }

                    for sel in sel_types:
                        ids = sel_ids.get(sel, [])
                        if len(ids)>0 and d_all[sel].size>0:
                            m_val, ci = compute_mean_ci(d_all[sel][ids])
                        else:
                            m_val, ci = np.nan, np.nan
                        stats[sel].append((m_val, ci))

                else:
                    vp = val_data.get(model, {})
                    dvals_lay = vp.get(lay, {})

                    d_all = {
                        "face": np.asarray(dvals_lay.get("face_d", [])),
                        "body": np.asarray(dvals_lay.get("body_d", [])),
                        "mixed": np.asarray(dvals_lay.get("mixed_d", [])),
                    }

                    for sel in sel_types:
                        ids = np.asarray(sel_ids.get(sel, []), dtype=int)
                        if ids.size > 0:
                            m_val, ci = compute_mean_ci(d_all[sel][ids])
                        else:
                            m_val, ci = np.nan, np.nan
                        stats[sel].append((m_val, ci))

            low = model.lower()
            if "ecoset" in low: ds = "ecoset"
            elif "imagenet" in low: ds = "imagenet"
            else: ds = None
            ls = ds_linestyles.get(ds, "-")

            # Plot each unit type
            for sel, color in zip(sel_types, type_colors):
                means = np.array([m for m,_ in stats[sel]])
                cis = np.array([c for _,c in stats[sel]])

                ax.plot(x, means, color=color, linestyle=ls, alpha=0.8)
                ax.fill_between(
                    x,
                    means - cis,
                    means + cis,
                    color=color,
                    alpha=0.2,
                    lw=0
                )

            # Spearman: depth vs d'
            print(f"{model}: Spearman r of d' and layer depth")
            for sel in sel_types:
                vec = np.array([m for m,_ in stats[sel]])
                ok = ~np.isnan(vec)
                if ok.sum()>1:
                    rho, p = spearmanr(x[ok], vec[ok])
                    print(f"{sel}: r={rho:.2f}, p={p:.3f}")

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=30, ha="right")
        ax.set_ylim(0, 1.5)
        if ax is axes[0]:
            ax.set_ylabel("Selectivity (d')")
        clean_axes(ax)

    # Legend
    # Datasets
    ds_handles = [
        mlines.Line2D([], [], color="gray", ls="-"),
        mlines.Line2D([], [], color="gray", ls="--")
    ]
    ds_labels = ["Ecoset", "ImageNet"]
    leg1 = ax.legend(ds_handles, ds_labels,
                     title="Dataset",
                     loc="upper left",
                     bbox_to_anchor=(1, 0.8),
                     frameon=False)
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)

    # Unit types
    sel_handles = [
        mlines.Line2D([], [], color=SEL_INFO[sel]["color"])
        for sel in sel_types
    ]
    sel_labels = [SEL_INFO[sel]["label"] for sel in sel_types]
    leg2 = ax.legend(
        sel_handles, sel_labels,
        title="Unit type",
        loc="center left",
        bbox_to_anchor=(1, 0.3),
        frameon=False)
    leg2._legend_box.align = "left"

    plt.suptitle(fig_title)
    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()

def plot_encoding_sep_layers(model_name, layers, rois, r2_adj=False,
                             fig_title=None, out_path=None):
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

def plot_encoding_varpart_layers(model_name, layers, rois, r2_adj=False, show_delta=True,
                                 fig_title=None, out_path=None):
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

def plot_integration_coef_layers(model_name, layers,
                                 coefs=["obs_net_full", "pred_net_headbody", "beta_FB"],
                                 filt_enhanced=True, n_boot=10000, random_state=0,
                                 fig_title=None, out_path=None):
    """Line plots with bootstrap CIs per layer"""
    sel_types = list(SEL_INFO.keys())
    labels = {s: SEL_INFO[s]["label"] for s in sel_types}
    colors = {s: SEL_INFO[s]["color"] for s in sel_types}

    coef_labels = {
        "obs_net_full": "Observed (Whole)",
        "pred_net_headbody": "Predicted (Face + Body)",
        "beta_FB": "Interaction (Enhanced)",
    }

    res = load_pickle(PROJECT_ROOT / "fb_integration" / model_name / "fb_integration.pkl")

    # Collect data per layer/type/coefficient
    data = {coef: {lay: {} for lay in layers} for coef in coefs}
    present_types = set()

    for coef in coefs:
        for lay in layers:
            layer_dict = res.get(lay, {})
            for s in sel_types:
                df = layer_dict.get(s)
                if df is None or df.empty or (coef not in df.columns):
                    continue

                if coef == "beta_FB" and filt_enhanced:
                    # Only include units that enhance response to whole persons
                    mask = df["obs_net_full"] > 0
                    vals = df.loc[mask, coef].to_numpy()
                else:
                    vals = df[coef].to_numpy()

                vals = vals[np.isfinite(vals)]
                if vals.size:
                    data[coef][lay][s] = vals
                    present_types.add(s)

    if not present_types:
        raise ValueError(f"No coefficient values found for the requested layers.")

    present_types = [s for s in sel_types if s in present_types]

    n_coefs = len(coefs)
    fig, axes = plt.subplots(1, n_coefs, figsize=(3.5 * n_coefs, 3), sharey=False)
    if n_coefs == 1:
        axes = [axes]
    elif n_coefs >= 2:
        axes[1].sharey(axes[0])

    n_layers = len(layers)
    base_pos = np.arange(n_layers)
    rng = np.random.default_rng(random_state)

    for ax_idx, (ax, coef) in enumerate(zip(axes, coefs)):
        # Bootstrap CIs per layer
        means = {s: np.full(n_layers, np.nan) for s in present_types}
        ci_lower = {s: np.full(n_layers, np.nan) for s in present_types}
        ci_upper = {s: np.full(n_layers, np.nan) for s in present_types}
        ns = {s: np.zeros(n_layers, dtype=int) for s in present_types}

        for li, lay in enumerate(layers):
            for s in present_types:
                vals = data[coef][lay].get(s)
                if vals is None or vals.size == 0:
                    continue

                ns[s][li] = vals.size

                boot_means = np.zeros(n_boot)
                for b in range(n_boot):
                    boot_sample = rng.choice(vals, size=len(vals), replace=True)
                    boot_means[b] = np.mean(boot_sample)

                means[s][li] = np.mean(boot_means)
                ci_lower[s][li] = np.percentile(boot_means, 2.5)
                ci_upper[s][li] = np.percentile(boot_means, 97.5)

        for s in present_types:
            xs = base_pos
            ys = means[s].copy()
            ys_lower = ci_lower[s].copy()
            ys_upper = ci_upper[s].copy()

            valid = np.isfinite(ys)
            if not valid.any():
                continue

            ax.plot(xs[valid], ys[valid],
                    linewidth=2,
                    color=colors[s], alpha=0.8,
                    label=labels[s] if ax_idx == 0 else None,
                    zorder=3)
            ax.fill_between(
                xs[valid], 
                ys_lower[valid], 
                ys_upper[valid],
                color=colors[s], 
                alpha=0.3, 
                linewidth=0,
                zorder=2
            )

        ax.axhline(0, color="dimgray", ls="--", lw=0.8, zorder=0)
        # ax.set_xlim(base_pos[0] - 0.75, base_pos[-1] + 0.75)
        ax.set_xticks(base_pos)
        ax.set_xticklabels(layers, rotation=30, ha="right")
        ax.set_title(coef_labels.get(coef, coef))
        if ax_idx == 0:
            ax.set_ylabel("Norm. response")
        elif ax_idx == 1:
            ax.set_ylabel("")
        clean_axes(ax)

    # After all plotting, align zero baseline
    if n_coefs >= 3:
        y0_min, y0_max = axes[0].get_ylim()
        zero_frac = -y0_min / (y0_max - y0_min)

        for ax_idx in range(2, n_coefs):
            y_min, y_max = axes[ax_idx].get_ylim()

            # Determine minimum required ranges
            range_below = abs(min(y_min, 0))
            range_above = max(y_max, 0)

            # Calculate what total range each requirement implies
            total_range_from_below = range_below / zero_frac
            total_range_from_above = range_above / (1 - zero_frac)

            # Take the larger to ensure all data fits
            total_range = max(total_range_from_below, total_range_from_above)
    
            new_y_min = -zero_frac * total_range
            new_y_max = (1 - zero_frac) * total_range

            axes[ax_idx].set_ylim(new_y_min, new_y_max)

    if fig_title:
        fig.suptitle(fig_title)
    plt.tight_layout()

    if out_path:
        plt.savefig(FIG_ROOT / out_path, dpi=300)
    plt.show()


# ----------------------------------- Utils ---------------------------------- #
def compute_mean_ci(data: np.ndarray, n_iter: int=10000):
    """Compute mean and 95% CI."""
    if len(data) == 0:
        return np.nan, np.nan

    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_iter)]
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    ci = (upper - lower) / 2

    return np.mean(data), ci

def compute_mean_ci_hierarch(data_by_layer: dict, n_iter: int=10000):
    """Compute mean and 95% CI accounting for layer structure."""
    all_data = np.concatenate([v for v in data_by_layer.values() if len(v) > 0])
    if len(all_data) == 0:
        return np.nan, np.nan

    layers = list(data_by_layer.keys())
    if len(layers) == 0:
        return np.nan

    means = []
    for _ in range(n_iter):
        # Resample layers with replacement
        sampled_layers = np.random.choice(layers, size=len(layers), replace=True)

        # For each sampled layer, resample units within that layer
        resampled_values = []
        for layer in sampled_layers:
            layer_data = data_by_layer[layer]
            if len(layer_data) > 0:
                # Resample units within this layer
                resampled_units = np.random.choice(layer_data, size=len(layer_data), replace=True)
                resampled_values.extend(resampled_units)

        if len(resampled_values) > 0:
            means.append(np.mean(resampled_values))

    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    ci = (upper - lower) / 2

    return np.mean(all_data), ci

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
