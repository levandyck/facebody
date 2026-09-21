"""Figures and statistics for DNN unit selectivity: Prevalence across layers, d' distributions, and training-diet contrasts."""

from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from facebody.config import FIG_ROOT, PROJECT_ROOT
from facebody.style import SEL_INFO, CTRL_INFO, UNIT_INFO, MIXED_COLOR
from facebody.fmri.gradient import gradient_stats
from .common import (
    _depth_axis, _swatch_legend, compute_mean_ci,
    compute_mean_ci_hierarch_models_layers_units
)
from facebody.myutils.utils import clean_axes, load_pickle

# ------------------------------- MAIN FIGURES ------------------------------- #
# -------------------------------- Selectivity ------------------------------- #
def _sel_counts(model: str, layers: list, sel_types: list,
                res_name: str="floc_res"):
    """Per-layer selective-unit counts and unit totals for one model."""
    res = load_pickle(PROJECT_ROOT / "selectivity" / model / f"{res_name}.pkl")

    totals = np.array(
        [len(res[lay]["stats"]["face"]["dvals"]) for lay in layers],
        dtype=float
    )
    counts = {
        sel: np.array(
            [len(res[lay]["unit_ids"].get(sel, [])) for lay in layers],
            dtype=float
        )
        for sel in sel_types
    }
    return counts, totals

def _perc_layers_curves(model_dict: dict, sel_types: tuple=("face", "body", "mixed"),
                        n_grid: int=10, res_name: str="floc_res"):
    """Percentage of selective units per model, interpolated onto a common depth grid."""
    x_grid = np.linspace(0, 1, n_grid)
    models = list(model_dict.keys())
    curves = {sel: [] for sel in sel_types}

    for model in models:
        layers = model_dict[model]
        counts, totals = _sel_counts(model, layers, sel_types, res_name)
        x_model = np.linspace(0, 1, len(layers))

        for sel in sel_types:
            vec = np.divide(
                100.0 * counts[sel],
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

            curves[sel].append(interp_vec)

    return {sel: np.vstack(v) for sel, v in curves.items()}, x_grid, models

def plot_perc_layers(model_dict: dict, sel_types: tuple=["face", "body", "mixed"],
                     fig_title: str=None, ylim: tuple=(0, 8),
                     n_grid: int=10, out_path: Path=None):
    """Line plot: Percentage of selective units across layers."""
    if sel_types is None:
        sel_types = list(SEL_INFO.keys())[:3]
    sel_types = list(sel_types)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))

    all_curves, x_grid, _ = _perc_layers_curves(model_dict, sel_types, n_grid)

    # Aggregate and plot
    for sel in sel_types:
        arr = all_curves[sel]
        n = arr.shape[0]
        mean_vec = np.nanmean(arr, axis=0)
        sem_vec = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n)

        ax.plot(
            x_grid*100,
            mean_vec,
            color=UNIT_INFO[sel]["color"],
            lw=2.5,
            label=UNIT_INFO[sel]["label"],
        )
        ax.fill_between(
            x_grid*100,
            mean_vec - sem_vec,
            mean_vec + sem_vec,
            color=UNIT_INFO[sel]["color"],
            alpha=0.2,
            linewidth=0,
        )

    _depth_axis(ax, ylim, "Selective units (%)")
    _swatch_legend(ax,
                   [UNIT_INFO[sel]["color"] for sel in sel_types],
                   [UNIT_INFO[sel]["label"] for sel in sel_types],
                   title="Unit type")

    ax.set_title(fig_title)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()

def stats_perc_layers(model_dict: dict, n_grid: int=10,
                      sel_types: tuple=("face", "body", "mixed"),
                      correct: str="fdr_bh", verbose: bool=True):
    """Change in the percentage of selective units across layer depth, tested across models."""
    curves, x_grid, models = _perc_layers_curves(model_dict, sel_types, n_grid)
    gradient_stats(curves, x_grid * 100, models, tuple(sel_types),
                   label="% of units", correct=correct, verbose=verbose)

def _unit_groups(res_lay: dict, n_units: int):
    """Mutually exclusive unit ids per unit type, with the non-selective remainder."""
    sel_types = [sel for sel in SEL_INFO if sel != "nonselective"]
    groups = {
        sel: np.asarray(res_lay["unit_ids"].get(sel, []), dtype=int)
        for sel in sel_types
    }

    used = [ids for ids in groups.values() if ids.size > 0]
    used = np.concatenate(used) if used else np.array([], dtype=int)
    groups["nonselective"] = np.setdiff1d(np.arange(n_units), used)

    return groups

def _violin_panel(ax, key: str, title: str, rng, pooled: dict, hier: dict,
                  sel_types: list, ylims: tuple, n_boot: int):
    """One violin per unit type, with the hierarchical-bootstrap mean and CI on top."""
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
            hier[sel][key], n_iter=n_boot, rng=rng,
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
            groups = _unit_groups(res[lay], dvals_f.shape[0])

            for sel, ids in groups.items():
                if ids.size == 0:
                    continue

                for key, dvals_arr in (("face", dvals_f), ("body", dvals_b)):
                    arr = dvals_arr[ids]
                    arr = arr[~np.isnan(arr)]
                    if arr.size == 0:
                        continue

                    pooled[sel][key].append(arr)
                    hier_data[sel][key].setdefault(model, {})[lay] = arr

    fig, (ax_face, ax_body) = plt.subplots(1, 2, figsize=(8, 2.8), sharey=True)

    rng = np.random.default_rng(0)
    _violin_panel(ax_face, "face", "Face d'", rng,
                  pooled, hier_data, sel_types, ylims, n_boot)
    _violin_panel(ax_body, "body", "Body d'", rng,
                  pooled, hier_data, sel_types, ylims, n_boot)

    ax_face.set_ylabel("Selectivity (d')")

    _swatch_legend(ax_body,
                   [SEL_INFO[sel]["color"] for sel in sel_types],
                   [SEL_INFO[sel]["label"] for sel in sel_types],
                   title="Unit type", lw=3, align=None)

    plt.suptitle(fig_title)
    plt.tight_layout()

    if out_path:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()

def stats_dprime(model_dict: dict, n_boot: int=10000):
    """Group mean d' and 95% CI per unit type."""
    sel_types = list(SEL_INFO.keys())
    hier = {sel: {"face": {}, "body": {}} for sel in sel_types}

    for model in model_dict:
        res = load_pickle(PROJECT_ROOT / "selectivity" / model / "floc_res.pkl")
        dvals = load_pickle(PROJECT_ROOT / "selectivity" / model / "validation_dprime.pkl")

        for lay in model_dict[model]:
            dvals_f = np.asarray(dvals[lay]["face_d"], dtype=float)
            dvals_b = np.asarray(dvals[lay]["body_d"], dtype=float)
            groups = _unit_groups(res[lay], dvals_f.shape[0])

            for sel, ids in groups.items():
                if ids.size == 0:
                    continue
                for key, arr_all in (("face", dvals_f), ("body", dvals_b)):
                    arr = arr_all[ids]
                    arr = arr[~np.isnan(arr)]
                    if arr.size:
                        hier[sel][key].setdefault(model, {})[lay] = arr

    rng = np.random.default_rng(0)
    print(f"Validation d' (preferred category vs. objects), "
          f"hierarchical bootstrap, {n_boot} iterations\n")
    print(f"{'unit type':<16}{'face d prime':>26}{'body d prime':>26}")
    for sel in sel_types:
        cells = []
        for key in ("face", "body"):
            m, lo, hi = compute_mean_ci_hierarch_models_layers_units(
                hier[sel][key], n_iter=n_boot, rng=rng)
            cells.append(f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                         if np.isfinite(m) else "--")
        print(f"{SEL_INFO[sel]['label']:<16}{cells[0]:>26}{cells[1]:>26}")
    return hier

def _dprime_pairs(model_dict: dict):
    """Paired (face d', body d') per unit, kept unit-aligned, nested model -> layer."""
    pairs = {sel: {} for sel in SEL_INFO}

    for model in model_dict:
        res = load_pickle(PROJECT_ROOT / "selectivity" / model / "floc_res.pkl")
        dvals = load_pickle(PROJECT_ROOT / "selectivity" / model / "validation_dprime.pkl")

        for lay in model_dict[model]:
            dvals_f = np.asarray(dvals[lay]["face_d"], dtype=float)
            dvals_b = np.asarray(dvals[lay]["body_d"], dtype=float)
            groups = _unit_groups(res[lay], dvals_f.shape[0])

            for sel, ids in groups.items():
                if ids.size == 0:
                    continue
                fi, bi = dvals_f[ids], dvals_b[ids]
                ok = np.isfinite(fi) & np.isfinite(bi)
                if not ok.any():
                    continue
                pairs[sel].setdefault(model, {})[lay] = np.column_stack([fi[ok], bi[ok]])

    return pairs

def stats_dprime_dd(model_dict: dict, n_iter: int=1000, seed: int=0,
                    correct: str="fdr_bh", cell_tests: bool=True,
                    extra_contrasts: bool=True, verbose: bool=True):
    """Double dissociation of face and body selectivity on the validation set."""
    from statsmodels.stats.multitest import multipletests

    sel_types = list(SEL_INFO.keys())
    pairs = _dprime_pairs(model_dict)
    rng = np.random.default_rng(seed)

    models = [m for m in model_dict if any(pairs[sel].get(m) for sel in sel_types)]
    layers = {m: sorted({lay for sel in sel_types for lay in pairs[sel].get(m, {})})
              for m in models}

    def _means(draw, boot):
        """Mean [face d', body d'] per unit type over one set of (model, layer) draws."""
        acc = {sel: [] for sel in sel_types}
        for model, lay in draw:
            for sel in sel_types:
                arr = pairs[sel].get(model, {}).get(lay)
                if arr is None or arr.shape[0] == 0:
                    continue
                if boot:
                    arr = arr[rng.integers(0, arr.shape[0], arr.shape[0])]
                acc[sel].append(arr)
        return {sel: (np.concatenate(acc[sel]).mean(axis=0) if acc[sel]
                      else np.array([np.nan, np.nan]))
                for sel in sel_types}

    # Contrasts as (name, fn) specs, so the set stays extensible.
    CONTRASTS = [
        ("face units: face d' - body d'",
         lambda m: m["face"][0] - m["face"][1]),
        ("body units: body d' - face d'",
         lambda m: m["body"][1] - m["body"][0]),
        ("interaction: (face - body) d', face units - body units",
         lambda m: (m["face"][0] - m["face"][1]) - (m["body"][0] - m["body"][1])),
    ]
    if extra_contrasts:
        CONTRASTS += [
            ("mixed units: face d' - body d'",
             lambda m: m["mixed"][0] - m["mixed"][1]),
            ("non-selective: face d' - body d'",
             lambda m: m["nonselective"][0] - m["nonselective"][1]),
        ]
    names = [n for n, _ in CONTRASTS]
    n_c = len(CONTRASTS)

    def _contrasts(m):
        return np.array([f(m) for _, f in CONTRASTS])

    full = [(m, lay) for m in models for lay in layers[m]]
    obs_means = _means(full, boot=False)
    obs = _contrasts(obs_means)

    boot_c = np.empty((n_iter, n_c), dtype=float)
    boot_m = np.empty((n_iter, len(sel_types), 2), dtype=float)

    for i in range(n_iter):
        draw = []
        for mi in rng.integers(0, len(models), len(models)):
            model = models[mi]
            lays = layers[model]
            if not lays:
                continue
            draw += [(model, lays[li]) for li in rng.integers(0, len(lays), len(lays))]
        m = _means(draw, boot=True)
        boot_c[i] = _contrasts(m)
        boot_m[i] = np.vstack([m[sel] for sel in sel_types])

    # One two-sided percentile-inversion p, reused for the per-cell tests.
    def _boot_p(v):
        """Two-sided bootstrap p: twice the smaller tail mass around zero."""
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan
        return 2 * min((v <= 0).mean(), (v >= 0).mean())

    lo, hi = np.percentile(boot_c, [2.5, 97.5], axis=0)
    m_lo, m_hi = np.percentile(boot_m, [2.5, 97.5], axis=0)

    p_c = np.array([_boot_p(boot_c[:, j]) for j in range(n_c)])
    p_cell = np.array([[_boot_p(boot_m[:, si, ki]) for ki in range(2)]
                       for si in range(len(sel_types))])

    # One FDR family over the contrasts and, if requested, the per-cell tests.
    p_raw = np.concatenate([p_c, p_cell.ravel()]) if cell_tests else p_c
    p_adj = p_raw.copy()
    if correct:
        ok = np.isfinite(p_raw)
        p_adj[ok] = multipletests(p_raw[ok], alpha=0.05, method=correct)[1]
    p_adj_c, p_raw_c = p_adj[:n_c], p_raw[:n_c]
    if cell_tests:
        p_adj_cell = p_adj[n_c:].reshape(p_cell.shape)
    else:
        p_adj_cell = np.full_like(p_cell, np.nan)

    out = {}
    for j, name in enumerate(names):
        out[name] = {"diff": float(obs[j]), "ci_low": float(lo[j]), "ci_high": float(hi[j]),
                     "p": float(p_adj_c[j]), "p_unc": float(p_raw_c[j])}

    cells = {}
    for si, sel in enumerate(sel_types):
        for ki, key in enumerate(("face_d", "body_d")):
            cells[(sel, key)] = {"mean": float(obs_means[sel][ki]),
                                 "ci_low": float(m_lo[si, ki]),
                                 "ci_high": float(m_hi[si, ki]),
                                 "p": float(p_adj_cell[si, ki]),
                                 "p_unc": float(p_cell[si, ki])}

    if verbose:
        floor = 2.0 / n_iter
        fmt_p = lambda p: ("n/a" if not np.isfinite(p) else
                           f"<{floor:.4f}" if p <= floor else f"{p:.3f}")
        print(f"\nValidation d' double dissociation | hierarchical bootstrap "
              f"(models -> layers -> units), {n_iter} iterations | "
              f"{correct} x {int(np.isfinite(p_raw).sum())}")
        print("\n  Group means [95% CI] (p vs 0):")
        print(f"    {'unit type':<16}{'face d prime':>34}{'body d prime':>34}")
        for sel in sel_types:
            cs = [cells[(sel, k)] for k in ("face_d", "body_d")]
            txt = [f"{c['mean']:+.2f} [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}] "
                   f"p={fmt_p(c['p'])}" for c in cs]
            print(f"    {SEL_INFO[sel]['label']:<16}{txt[0]:>34}{txt[1]:>34}")
        print("\n  Contrasts:")
        for name in names:
            r = out[name]
            print(f"    {name:<56} d(d')={r['diff']:+.2f} "
                  f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]  "
                  f"p={fmt_p(r['p'])} (raw {fmt_p(r['p_unc'])})")

    return {"contrasts": out, "cells": cells}


# --------------------------- SUPPLEMENTARY FIGURES -------------------------- #
_TRAINSET_LINESTYLES = {
    "ecoset": "-",
    "imagenet": "--",
    "untrained": ":",
}
_LINESTYLE_CYCLE = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]
_TRAINSET_LABELS = {
    "ecoset": "Ecoset",
    "imagenet": "ImageNet",
    "untrained": "Untrained",
    "faces": "VGGFace2",
    "vggface2": "VGGFace2",
}

def _model_line_styles(models: list, alpha_range: tuple=(1.0, 0.2)):
    """Per-model (linestyle, alpha) for the thin individual-model lines."""
    parts = []
    for model in models:
        backbone, sep, train_set = model.rpartition("_")
        if not sep:
            backbone, train_set = model, ""
        parts.append((backbone, train_set))

    # Order of first appearance, so the legend follows the order of `model_dict`.
    backbones = list(dict.fromkeys(b for b, _ in parts))
    train_sets = list(dict.fromkeys(t for _, t in parts))

    lo, hi = max(alpha_range), min(alpha_range)
    alphas = ([lo] if len(backbones) == 1
              else list(np.linspace(lo, hi, len(backbones))))
    backbone_alphas = dict(zip(backbones, alphas))

    trainset_linestyles, free = {}, [
        ls for ls in _LINESTYLE_CYCLE if ls not in _TRAINSET_LINESTYLES.values()
    ]
    for train_set in train_sets:
        if train_set in _TRAINSET_LINESTYLES:
            trainset_linestyles[train_set] = _TRAINSET_LINESTYLES[train_set]
        else:
            trainset_linestyles[train_set] = free.pop(0) if free else "-"

    styles = {
        model: (trainset_linestyles[t], backbone_alphas[b])
        for model, (b, t) in zip(models, parts)
    }
    return styles, backbones, backbone_alphas, train_sets, trainset_linestyles

def _style_legend(fig, handles: list, labels: list, title: str, loc: str,
                  bbox: tuple):
    """Legend outside the panels."""
    leg = fig.legend(handles, labels, title=title, loc=loc, bbox_to_anchor=bbox,
                     bbox_transform=fig.transFigure, frameon=False)
    leg._legend_box.align = "left"
    return leg

def _panel_grid(n_panels: int, n_cols: int=2, panel_size: tuple=(4, 3),
                sharey: bool=True):
    """Panels filled row by row in `n_cols` columns; spare axes hidden."""
    n_cols = min(n_cols, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows),
        sharey=sharey, squeeze=False,
    )
    axes = axes.ravel()
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    return fig, axes, n_cols

def plot_perc_layers_by_model(model_dict: dict,
                              sel_types: list=["face", "body", "mixed", "scene"],
                              ylim: tuple=(0, None), n_grid: int=10,
                              fig_title: str=None, out_path: Path=None):
    """Line plot: percentage of selective units across layers, one panel per unit type."""
    if sel_types is None:
        sel_types = list(SEL_INFO.keys())[:3] + list(CTRL_INFO.keys())
    sel_types = list(sel_types)

    curves, x_grid, models = _perc_layers_curves(model_dict, sel_types, n_grid)
    styles, backbones, backbone_alphas, train_sets, trainset_linestyles = (
        _model_line_styles(models)
    )

    fig, axes, n_cols = _panel_grid(len(sel_types), sharey=False)

    for ax, sel in zip(axes, sel_types):
        color = UNIT_INFO[sel]["color"]
        arr = curves[sel]
        n = arr.shape[0]
        mean_vec = np.nanmean(arr, axis=0)
        sem_vec = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n)

        # Individual models first
        for i, model in enumerate(models):
            linestyle, alpha = styles[model]
            ax.plot(
                x_grid*100,
                arr[i],
                color=color,
                lw=1.0,
                ls=linestyle,
                alpha=alpha,
                zorder=2,
            )

            vec = arr[i]
            ok = ~np.isnan(vec)
            rho, p = spearmanr(x_grid[ok], vec[ok]) if ok.sum() > 1 else (np.nan, np.nan)
            print(f"{model} / {sel}: mean={np.nanmean(vec):.2f} %, "
                  f"depth Spearman r={rho:.2f}, p={p:.3f}")

        ax.plot(
            x_grid*100,
            mean_vec,
            color=color,
            lw=2.5,
            zorder=3,
        )
        ax.fill_between(
            x_grid*100,
            mean_vec - sem_vec,
            mean_vec + sem_vec,
            color=color,
            alpha=0.2,
            linewidth=0,
            zorder=1,
        )

        ok = ~np.isnan(mean_vec)
        rho, p = spearmanr(x_grid[ok], mean_vec[ok]) if ok.sum() > 1 else (np.nan, np.nan)
        print(f"{UNIT_INFO[sel]['label']} (n={n} models): "
              f"mean={np.nanmean(mean_vec):.2f} %, "
              f"depth Spearman r={rho:.2f}, p={p:.3f}")

    _depth_axis(axes[0], ylim, "Selective units (%)")
    _depth_axis(axes[2], ylim, "Selective units (%)")
    for ax, sel in zip(axes, sel_types):
        if ax is not axes[0]:
            _depth_axis(ax, ylim)
        ax.set_title(UNIT_INFO[sel]["label"])

    _style_legend(
        fig,
        [mlines.Line2D([], [], color="black", lw=1.5, alpha=backbone_alphas[b])
         for b in backbones],
        backbones,
        title="Architecture",
        loc="lower left",
        bbox=(1.01, 0.52),
    )
    _style_legend(
        fig,
        [mlines.Line2D([], [], color="black", lw=1.5, ls=trainset_linestyles[t])
         for t in train_sets],
        [_TRAINSET_LABELS.get(t, t) for t in train_sets],
        title="Training set",
        loc="upper left",
        bbox=(1.01, 0.48),
    )

    if fig_title:
        fig.suptitle(fig_title)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_perc_layers_mixed(model_dict: dict, fig_title: str=None, ylim: tuple=(0, 5),
                           n_grid: int=10, out_path: Path=None):
    """Line plot: Percentage of selective units across layers (mixed selectivity)."""
    mixed_sel_types = ["face&body", "face&scene", "body&scene"]
    mixed_sel_info = {
        "face&body":  {"color": MIXED_COLOR, "label": "Face & Body"},
        "face&scene": {"color": "#AE3ECC", "label": "Face & Scene"},
        "body&scene": {"color": "#00C43E", "label": "Body & Scene"},
    }

    fig, ax = plt.subplots(figsize=(7, 3.5))

    all_curves, x_grid, _ = _perc_layers_curves(
        model_dict, mixed_sel_types, n_grid, res_name="floc_res_mixed"
    )

    # Aggregate and plot
    for sel in mixed_sel_types:
        arr = all_curves[sel]
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

    _depth_axis(ax, ylim, "Selective units (%)")
    _swatch_legend(ax,
                   [mixed_sel_info[sel]["color"] for sel in mixed_sel_types],
                   [mixed_sel_info[sel]["label"] for sel in mixed_sel_types],
                   title="Unit type")

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

    _depth_axis(ax, ylim, "Selectivity (d')")

    _swatch_legend(ax,
                   [SEL_INFO[sel]["color"] for sel in sel_types],
                   [SEL_INFO[sel]["label"] for sel in sel_types],
                   title="Unit type")

    ax.set_title(fig_title)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_resp_dist(model_dict: dict, cats: tuple=("face", "body", "object"),
                   fig_title: str=None, out_path: Path=None,
                   ylims: tuple=(-1.5, 1.5), n_boot: int=1000, img_db: str="validation"):
    """Violin plot: Mean response to each category per unit type."""
    sel_types = list(SEL_INFO.keys())
    cats = tuple(cats)

    cat_labels = {
        "face": "Faces", "body": "Bodies", "object": "Objects",
        "scene": "Scenes", "scrambled": "Scrambled",
    }

    pooled = {
        sel: {cat: [] for cat in cats}
        for sel in sel_types
    }

    hier_data = {
        sel: {cat: {} for cat in cats}
        for sel in sel_types
    }

    for model in model_dict.keys():
        res = load_pickle(PROJECT_ROOT / "selectivity" / model / "floc_res.pkl")
        resp = load_pickle(PROJECT_ROOT / "selectivity" / model / f"{img_db}_resp.pkl")

        for lay in model_dict[model]:
            zvals = {
                cat: np.asarray(resp[lay][f"{cat}_z"], dtype=float)
                for cat in cats
            }
            groups = _unit_groups(res[lay], zvals[cats[0]].shape[0])

            for sel, ids in groups.items():
                if ids.size == 0:
                    continue

                for cat in cats:
                    arr = zvals[cat][ids]
                    arr = arr[~np.isnan(arr)]
                    if arr.size == 0:
                        continue

                    pooled[sel][cat].append(arr)
                    hier_data[sel][cat].setdefault(model, {})[lay] = arr

    fig, axes = plt.subplots(
        1, len(cats), figsize=(4 * len(cats), 2.8), sharey=True, squeeze=False
    )
    axes = axes[0]

    rng = np.random.default_rng(0)
    for ax, cat in zip(axes, cats):
        _violin_panel(ax, cat, cat_labels.get(cat, cat.capitalize()), rng,
                      pooled, hier_data, sel_types, ylims, n_boot)

    axes[0].set_ylabel("Mean response (z)")

    _swatch_legend(axes[-1],
                   [SEL_INFO[sel]["color"] for sel in sel_types],
                   [SEL_INFO[sel]["label"] for sel in sel_types],
                   title="Unit type", lw=3, align=None)

    plt.suptitle(fig_title)
    plt.tight_layout()

    if out_path:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()


# --------------------------------- APPENDIX --------------------------------- #
_DIET_BASELINE_COLOR = "#999999"
_DIET_COLOR_CYCLE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]

def _diet_line_colors(diets: list, base: str, diet_colors: dict=None):
    """Baseline diet gray, the rest from `_DIET_COLOR_CYCLE`; `diet_colors` overrides."""
    colors, i = {}, 0
    for diet in diets:
        if diet == base:
            colors[diet] = _DIET_BASELINE_COLOR
        else:
            colors[diet] = _DIET_COLOR_CYCLE[i % len(_DIET_COLOR_CYCLE)]
            i += 1
    if diet_colors:
        colors.update(diet_colors)
    return colors

def _diet_mean_sem(arr: np.ndarray):
    """Across-architecture mean and SEM. SEM is zero-width at n = 1, not NaN."""
    n = arr.shape[0]
    if n == 1:
        return arr[0], np.zeros(arr.shape[1])
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n)

def _diet_matched_models(diet_dicts: dict, baseline: str=None):
    """
    Validate a diet contrast: architectures matched by position, layers identical.

    Returns (diets, baseline label, contrast labels, n_arch).
    """
    diets = list(diet_dicts)
    if not diets:
        raise ValueError("diet_dicts is empty")

    ref = diets[0]
    n_arch = len(diet_dicts[ref])

    for diet in diets:
        if len(diet_dicts[diet]) != n_arch:
            raise ValueError(
                f"'{diet}': {len(diet_dicts[diet])} models vs {n_arch} in '{ref}'"
            )
        for m_ref, m_diet in zip(diet_dicts[ref], diet_dicts[diet]):
            if list(diet_dicts[diet][m_diet]) != list(diet_dicts[ref][m_ref]):
                raise ValueError(f"'{diet}': layers of {m_diet} differ from {m_ref}")

    base = diets[0] if baseline is None else baseline
    if base not in diet_dicts:
        raise ValueError(f"baseline '{base}' not in diet_dicts")

    contrasts = [diet for diet in diets if diet != base]
    if not contrasts:
        raise ValueError("need at least one diet besides the baseline")

    return diets, base, contrasts, n_arch

def plot_perc_layers_by_diet(diet_dicts: dict, show: str="both",
                             baseline: str="ImageNet",
                             sel_types: list=None, n_grid: int=10,
                             res_name: str="floc_res", ylim: tuple=None,
                             diet_colors: dict=None, fig_title: str=None,
                             out_path: Path=None):
    """
    Line plot: selective-unit prevalence across layers, one panel per unit type and
    one curve per training diet (mean across architectures +- SEM ribbon).
    """
    if show not in ("raw", "diff", "both"):
        raise ValueError(f"show must be 'raw', 'diff' or 'both', got '{show}'")

    if sel_types is None:
        sel_types = list(SEL_INFO.keys())[:3]
    sel_types = list(sel_types)

    if not diet_dicts:
        raise ValueError("diet_dicts is empty")
    for diet, models in diet_dicts.items():
        if not models:
            raise ValueError(f"'{diet}': no models")

    if show == "raw":
        # No contrast is taken, so diets need not be matched across architectures.
        diets = list(diet_dicts)
        base = diets[0] if baseline is None else baseline
        if base not in diet_dicts:
            raise ValueError(f"baseline '{base}' not in diet_dicts")
        contrasts = [diet for diet in diets if diet != base]
    else:
        # A per-architecture difference requires matched models and equal layers.
        diets, base, contrasts, _ = _diet_matched_models(diet_dicts, baseline)

    # One interpolated curve per architecture, per diet, per unit type.
    curves, models_by_diet = {}, {}
    for diet in diets:
        curves[diet], x_grid, models_by_diet[diet] = _perc_layers_curves(
            diet_dicts[diet], sel_types, n_grid, res_name
        )

    colors = _diet_line_colors(diets, base, diet_colors)
    rows = ["raw", "diff"] if show == "both" else [show]

    fig, axes = plt.subplots(
        len(rows), len(sel_types),
        figsize=(4 * len(sel_types), 3 * len(rows)),
        squeeze=False,
    )

    for r, mode in enumerate(rows):
        panel_diets = diets if mode == "raw" else contrasts

        for c, sel in enumerate(sel_types):
            ax = axes[r][c]

            if mode == "diff":
                ax.axhline(0, color="gray", ls="--", lw=1, zorder=0)

            for diet in panel_diets:
                arr = curves[diet][sel]
                if mode == "diff":
                    arr = arr - curves[base][sel]
                mean_vec, sem_vec = _diet_mean_sem(arr)

                # The reference sits under the manipulations it is there to anchor.
                z = 2 if diet == base else 3
                ax.plot(x_grid * 100, mean_vec, color=colors[diet], lw=2.5, zorder=z)
                ax.fill_between(
                    x_grid * 100,
                    mean_vec - sem_vec,
                    mean_vec + sem_vec,
                    color=colors[diet], alpha=0.2, linewidth=0, zorder=z - 2,
                )

                # Per architecture first, so sign consistency is visible.
                unit = "%" if mode == "raw" else " points"
                fmt = ".2f" if mode == "raw" else "+.2f"
                for model, vec in zip(models_by_diet[diet], arr):
                    print(f"  {diet} | {model} / {sel}: "
                          f"mean={np.nanmean(vec):{fmt}}{unit}, "
                          f"deepest={vec[-1]:{fmt}}{unit}")

                ok = ~np.isnan(mean_vec)
                rho, p = (spearmanr(x_grid[ok], mean_vec[ok])
                          if ok.sum() > 1 else (np.nan, np.nan))
                tag = f"{diet}" if mode == "raw" else f"{diet} vs {base}"
                print(f"{tag} / {sel} (n={arr.shape[0]}): "
                      f"mean={np.nanmean(mean_vec):{fmt}}{unit}, "
                      f"deepest={mean_vec[-1]:{fmt}}{unit}, "
                      f"depth Spearman r={rho:.2f}, p={p:.3f}")

            ylabel = None
            if c == 0:
                ylabel = ("Selective units (%)" if mode == "raw"
                          else f"\u0394 Selective units (%)\nvs. {base}")
            _depth_axis(ax, ylim if ylim is not None else (None, None), ylabel)

            # Unit type named once per column, in its own colour.
            if r == 0:
                ax.set_title(UNIT_INFO[sel]["label"])

    if fig_title:
        fig.suptitle(fig_title)
    plt.tight_layout()

    # Figure legend, right of every panel, after tight_layout (which ignores it).
    legend_diets = diets if "raw" in rows else contrasts
    _style_legend(
        fig,
        [mlines.Line2D([], [], color=colors[diet], lw=2.5) for diet in legend_diets],
        [f"{diet}" if diet == base and "raw" in rows else diet
         for diet in legend_diets],
        title="Training dataset",
        loc="center left",
        bbox=(1.01, 0.5),
    )

    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()
