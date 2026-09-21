"""Figures along the ventral pathway: Selective voxel prevalence and integration curves."""

from pathlib import Path
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from facebody.config import FIG_ROOT, PROJECT_ROOT
from facebody.style import SEL_INFO, MIXED_COLOR, SHARED_COLOR
from facebody.fmri.gradient import (
    N_GRID, gradient_bins, gradient_stats, load_gradient, perc_ventral_stream
)
from .common import _mean_err
from facebody.myutils.utils import clean_axes, load_pickle

def _ventral_lineplot(curves: dict, x_grid: np.ndarray, labels: list, colors: list,
                      ylabel: str, leg_title: str, fig_title: str=None,
                      ylim: tuple=None, error: str="sem", out_path: Path=None,
                      ax=None, legend: bool=True):
    """One line per key, mean +- error across subjects."""
    panel = ax is not None
    if not panel:
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for (key, arr), color in zip(curves.items(), colors):
        mean_vec, err_vec = _mean_err(arr, error)
        ax.plot(x_grid, mean_vec, color=color, lw=2.5)
        ax.fill_between(x_grid, mean_vec - err_vec, mean_vec + err_vec,
                        color=color, alpha=0.2, linewidth=0)

    ax.set_xlim(-5, 105)
    ax.set_xticks(np.linspace(0, 100, 5))
    ax.set_xlabel("Ventral pathway (%)")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    clean_axes(ax)

    if legend:
        handles = [mlines.Line2D([], [], color=c, lw=2.5) for c in colors]
        leg = ax.legend(handles, labels, title=leg_title, loc="center left",
                        bbox_to_anchor=(1, 0.5), frameon=False)
        leg._legend_box.align = "left"

    ax.set_title(fig_title)
    if panel:
        return
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()

def _encoding_ventral_curves(subjects: list, model_name: str, layer: str, keys: tuple,
                             analysis: str="sep", n_grid: int=N_GRID, r2_adj: bool=False,
                             clip: bool=False, proj_dir: Path=PROJECT_ROOT):
    """Bin voxelwise encoding results along gradient."""
    from facebody.encoding.summarize import load_voxelwise

    all_curves = {k: [] for k in keys}
    for subj in subjects:
        res, meta = load_voxelwise(model_name, layer, subj, proj_dir)
        src = res["r2"] if analysis == "sep" else {**res["delta_m"], **res["varpart"]}

        idx, x_grid = gradient_bins(load_gradient(subj, proj_dir, "fit")["d_norm"], n_grid)
        on = idx >= 0
        den = (np.bincount(idx[on], weights=meta["nc"][on], minlength=n_grid) if r2_adj
               else np.bincount(idx[on], minlength=n_grid).astype(float))

        for k in keys:
            vals = np.asarray(src[k], dtype=float)
            num = np.bincount(idx[on], weights=np.clip(vals, 0, None)[on] if clip
                              else vals[on], minlength=n_grid)
            all_curves[k].append(np.divide(num, den, out=np.full(n_grid, np.nan),
                                           where=den > 0))

    return {k: np.vstack(v) for k, v in all_curves.items()}, x_grid

def _integration_ratios(parts: dict):
    """Integration ratios."""
    def ratio(num, den):
        return np.divide(100.0 * parts[num], parts[den],
                         out=np.full_like(parts[den], np.nan), where=parts[den] > 0)

    return {"U_face": ratio("u_f", "r2_fb"),
            "U_body": ratio("u_b", "r2_fb"),
            "I_shared": ratio("s_fb", "r2_fb"),
            "I_mixed": ratio("delta_m", "r2_fbm")}

def _encoding_subjects(model_name: str, layer: str, subjects: list=None):
    """Subjects to use: the ones given, or everyone already fit for this layer."""
    from facebody.encoding.summarize import available_subjects

    subjects = available_subjects(model_name, layer) if subjects is None else subjects
    assert subjects, f"no voxelwise results for {model_name}/{layer}"
    return list(subjects)

def plot_perc_ventral(subjects: list, fig_title: str=None, ylim: tuple=(0, 18),
                      n_grid: int=N_GRID, error: str="sem",
                      responsive: bool=True, proj_dir: Path=PROJECT_ROOT,
                      out_path: Path=None):
    """Line plot: Percentage of selective voxels along ventral pathway."""
    sel_types = list(SEL_INFO.keys())[:3]
    unit = "% of responsive voxels" if responsive else "% of voxels"
    curves, x_grid = perc_ventral_stream(subjects, n_grid, responsive, proj_dir)
    gradient_stats(curves, x_grid, subjects, tuple(sel_types), label=unit)

    _ventral_lineplot(curves, x_grid,
                      labels=[SEL_INFO[s]["label"] for s in sel_types],
                      colors=[SEL_INFO[s]["color"] for s in sel_types],
                      ylabel="Selective voxels (%)",
                      leg_title="Voxel type",
                      fig_title=fig_title, ylim=ylim, error=error, out_path=out_path)

def stats_perc_ventral():
    """Print the percentage of responsive voxels of each type, per subject."""
    out = load_pickle(PROJECT_ROOT / "selectivity" / "cortex" / "floc.pkl")
    perc = {s: [] for s in ("face", "body", "mixed")}
    for subj, o in out.items():
        vm = o["res"]["vox_masks"]
        resp = vm["responsive"]
        for s in perc:
            perc[s].append(100.0 * (vm[s] & resp).sum() / resp.sum())

    for s, v in perc.items():
        v = np.asarray(v)
        print(f"{s:6s} {v.mean():5.2f} +/- {v.std(ddof=1):4.2f} %  "
          f"(range {v.min():.2f}-{v.max():.2f}, n={v.size})")

def plot_sep_ventral(model_name: str, layer: str, subjects: list=None,
                     fig_title: str=None, ylim: tuple=None, n_grid: int=N_GRID,
                     r2_adj: bool=True, error: str="sem",
                     proj_dir: Path=PROJECT_ROOT, out_path: Path=None):
    """Line plot: R2 of each unit type along ventral pathway."""
    sel_types = list(SEL_INFO.keys())
    keys = ("f", "b", "m", "ns")

    subjects = _encoding_subjects(model_name, layer, subjects)
    curves, x_grid = _encoding_ventral_curves(subjects, model_name, layer, keys, "sep",
                                              n_grid, r2_adj, False, proj_dir)
    gradient_stats(curves, x_grid, subjects, keys, label="R²")

    _ventral_lineplot(curves, x_grid,
                      labels=[SEL_INFO[s]["label"] for s in sel_types],
                      colors=[SEL_INFO[s]["color"] for s in sel_types],
                      ylabel="Explained variance (R² adj.)" if r2_adj
                             else "Explained variance (R²)",
                      leg_title="Unit type", fig_title=fig_title, ylim=ylim,
                      error=error, out_path=out_path)

def plot_varpart_ventral(model_name: str, layer: str, subjects: list=None,
                         fig_title: str=None, ylim: tuple=None, n_grid: int=N_GRID,
                         r2_adj: bool=False, show_delta: bool=True, error: str="sem",
                         proj_dir: Path=PROJECT_ROOT,
                         out_path: Path=None):
    """Line plot: Variance partitioning along ventral pathway."""
    keys = ("u_f", "u_b", "s_fb") + (("delta_m",) if show_delta else ())
    labels = ["Face", "Body", "Face ∩ Body"] + (["Δ Mixed"] if show_delta else [])
    colors = [SEL_INFO["face"]["color"], SEL_INFO["body"]["color"], SHARED_COLOR] \
             + ([MIXED_COLOR] if show_delta else [])

    subjects = _encoding_subjects(model_name, layer, subjects)
    curves, x_grid = _encoding_ventral_curves(subjects, model_name, layer, keys,
                                              "varpart", n_grid, r2_adj, False, proj_dir)
    gradient_stats(curves, x_grid, subjects, keys, label="R²")

    _ventral_lineplot(curves, x_grid,
                      labels=labels, colors=colors,
                      ylabel="Explained variance (R² adj.)" if r2_adj
                             else "Explained variance (R²)",
                      leg_title="Variance portion", fig_title=fig_title, ylim=ylim,
                      error=error, out_path=out_path)

def plot_integration_ventral(model_name: str, layer: str, subjects: list=None,
                             fig_title: str=None, ylim: tuple=None, n_grid: int=N_GRID,
                             error: str="sem", proj_dir: Path=PROJECT_ROOT,
                             ylabel: str="Proportion of total explained variance (%)",
                             out_path: Path=None):
    """
    Line plot: Unique face/body variance, shared face/body variance and unique mixed variance
    along the ventral pathway, each as a proportion of the fit it comes from.
    """
    keys = ("u_f", "u_b", "s_fb", "r2_fb", "delta_m", "r2_fbm")

    subjects = _encoding_subjects(model_name, layer, subjects)
    parts, x_grid = _encoding_ventral_curves(subjects, model_name, layer, keys,
                                             "varpart", n_grid, False, False, proj_dir)

    curves = _integration_ratios(parts)
    gradient_stats(curves, x_grid, subjects, tuple(curves), label="% of variance explained")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    specs = [
        (
            ("U_face", "U_body"),
            (SEL_INFO["face"]["color"], SEL_INFO["body"]["color"]),
            ("Face unique", "Body unique"),
        ),
        (
            ("I_shared",),
            (SHARED_COLOR,),
            ("Face-Body shared",),
        ),
        (
            ("I_mixed",),
            (SEL_INFO["mixed"]["color"],),
            ("Mixed unique", ),
        ),
    ]
    for i, (ax, (ks, colors, labels)) in enumerate(zip(axes, specs)):
        _ventral_lineplot({k: curves[k] for k in ks}, x_grid,
                          labels=list(labels),
                          colors=list(colors), ylabel=ylabel if i == 0 else "",
                          leg_title=None,
                          ylim=ylim, error=error, ax=ax, legend=False)
        if labels:
            handles = [mlines.Line2D([], [], color=c, lw=2.5) for c in colors]
            ax.legend(handles, list(labels), loc="upper left", frameon=False)

    fig.suptitle(fig_title)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()

def stats_integration_ventral(model_name: str, subjects: list, layer: str="fc7"):
    """Print peaks."""
    keys = ("u_f", "u_b", "s_fb", "r2_fb", "delta_m", "r2_fbm")
    parts, x = _encoding_ventral_curves(subjects, model_name, layer, keys, "varpart")
    for name, arr in _integration_ratios(parts).items():
        m, e = _mean_err(arr, "sem")
        print(f"{name:9s} posterior {m[0]:5.2f} +/- {e[0]:.2f} | anterior {m[-1]:5.2f} +/- {e[-1]:.2f}")
