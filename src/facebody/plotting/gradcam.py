"""Guided Grad-CAM overlays and top NSD images per unit type."""

from pathlib import Path
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import h5py

from facebody.config import DATA_ROOT, FIG_ROOT, PROJECT_ROOT
from facebody.style import SEL_INFO
from facebody.dnn.guided_gradcam import compute_guided_gradcam, normalize_maps

# ------------------------------ Guided GradCAM ------------------------------ #
GGC_STYLE = dict(
    desaturate=0.85,
    lighten=0.35,
    alpha_max=0.9,
    alpha_gamma=1,
    clip_pct=99,
)

def _sel_cmap(sel: str):
    """White to the unit type's own color."""
    color = "#4d4d4d" if sel in ("non", "nonselective") else SEL_INFO[sel]["color"][:7]
    return LinearSegmentedColormap.from_list(sel, ["#ffffff", color])

def _pair_cmap(sel_pos: str, sel_neg: str):
    """Diverging colormap running from the second type's color through white to the first's."""
    return LinearSegmentedColormap.from_list(
        f"{sel_pos}_{sel_neg}",
        [SEL_INFO[sel_neg]["color"][:7], "#ffffff", SEL_INFO[sel_pos]["color"][:7]],
    )

def _robust_vmax(arrs: list):
    """Symmetric color limit shared by a row of maps, robust to a few extreme pixels."""
    v = float(np.percentile(np.abs(np.concatenate([a.ravel() for a in arrs])), GGC_STYLE["clip_pct"]))
    return v if v > 0 else 1.0

def _ggc_overlay(ax, pil_img, hm, cmap, vmax, diverging: bool, layer: str="both"):
    """Saliency map over desaturated image."""
    if layer not in ("both", "background", "heatmap"):
        raise ValueError(f"layer must be 'both', 'background' or 'heatmap', got {layer!r}")

    rgb = np.asarray(pil_img, dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    extent = (-0.5, w - 0.5, h - 0.5, -0.5)

    if layer in ("both", "background"):
        gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        bg = (1 - GGC_STYLE["desaturate"]) * rgb + GGC_STYLE["desaturate"] * gray[..., None]
        bg = np.clip((1 - GGC_STYLE["lighten"]) * bg + GGC_STYLE["lighten"], 0, 1)
        ax.imshow(bg, interpolation="bilinear", extent=extent)

    if layer in ("both", "heatmap"):
        mag = np.abs(hm) / vmax if diverging else hm / vmax
        alpha = GGC_STYLE["alpha_max"] * np.clip(mag, 0, 1) ** GGC_STYLE["alpha_gamma"]
        ax.imshow(hm, cmap=cmap, vmin=-vmax if diverging else 0, vmax=vmax,
                  alpha=alpha, interpolation="bilinear", extent=extent)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")

def plot_guided_gradcam(model_name: str, target_layer: str, cam_layer: str=None,
                        controlled: bool=True, device: str="cuda",
                        img_path: str=None, n_imgs: int=2,
                        seed: int=0, layer: str="both",
                        fig_title: str=None, out_path: Path=None):
    """Heatmaps: Guided Grad-CAM saliency map per unit type overlaid on example images."""
    sel_types = list(SEL_INFO.keys())

    res = compute_guided_gradcam(
        model_name=model_name,
        target_layer_pretty=target_layer,
        cam_layer_pretty=cam_layer,
        device=device,
        sel_types=sel_types,
        controlled=controlled,
        imgs_path=img_path,
        n_imgs=n_imgs,
        norm_types=False,
        seed=seed,
        post_blur_sigma=3.0,
    )
    maps = [normalize_maps(r["heatmaps"]) for r in res]
    vmax = {sel: _robust_vmax([m[sel] for m in maps]) for sel in sel_types}

    fig, axes = plt.subplots(
        len(sel_types), n_imgs,
        figsize=(2 * n_imgs, 2.2 * len(sel_types)),
        gridspec_kw={"wspace": 0.02, "hspace": 0.06},
        squeeze=False,
    )

    for i, (r, m) in enumerate(zip(res, maps)):
        for s, sel in enumerate(sel_types):
            ax = axes[s, i]
            _ggc_overlay(
                ax, r["pil_img"], m[sel],
                _sel_cmap(sel), vmax[sel],
                diverging=False, layer=layer,
            )

            if i == 0:
                ax.text(-0.04, 0.5, SEL_INFO[sel]["label"], rotation=90, va="center", ha="center",
                        color=SEL_INFO[sel]["color"][:7], transform=ax.transAxes, clip_on=False)

    fig.suptitle(fig_title)
    if out_path:
        fig.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_guided_gradcam_diff(model_name: str, target_layer: str, cam_layer: str=None,
                             controlled: bool=True, device: str="cuda",
                             img_path: str=None, n_imgs: int=3, seed: int=0,
                             pairs: list=None, layer: str="both",
                             fig_title: str=None, out_path: Path=None):
    """Heatmaps: pairwise differences between Guided Grad-CAM maps of the selective unit types."""
    pairs = [tuple(p) for p in ([("face", "body"), ("face", "mixed"), ("body", "mixed")] if pairs is None else pairs)]
    sel_types = sorted({s for pair in pairs for s in pair}, key=list(SEL_INFO).index)

    res = compute_guided_gradcam(
        model_name=model_name,
        target_layer_pretty=target_layer,
        cam_layer_pretty=cam_layer,
        device=device,
        sel_types=sel_types,
        controlled=controlled,
        imgs_path=img_path,
        n_imgs=n_imgs,
        norm_types=False,
        seed=seed,
        post_blur_sigma=3.0,
    )

    # Difference maps per image: [image][pair]
    diffs = []
    for r in res:
        maps = normalize_maps(r["heatmaps"])
        diffs.append({p: maps[p[0]] - maps[p[1]] for p in pairs})
    vmax = {p: _robust_vmax([d[p] for d in diffs]) for p in pairs}

    fig = plt.figure(figsize=(2.05 * n_imgs + 1.3, 2.15 * len(pairs) + 0.6))
    gs = fig.add_gridspec(len(pairs), n_imgs + 1,
                          width_ratios=[1] * n_imgs + [0.09], wspace=0.03, hspace=0.06)

    for s, pair in enumerate(pairs):
        cmap = _pair_cmap(*pair)

        for i, d in enumerate(diffs):
            ax = fig.add_subplot(gs[s, i])
            _ggc_overlay(
                ax, res[i]["pil_img"],
                d[pair], cmap, vmax[pair],
                diverging=True, layer=layer,
            )

            if i == 0:
                ax.text(-0.04, 0.5, f"{pair[0]}\n-\n{pair[1]}",
                        va="center", ha="right", linespacing=1.4,
                        transform=ax.transAxes, clip_on=False)

        # Per-row colorbar, tick labels colored to match the two unit types
        cax = fig.add_subplot(gs[s, -1])
        sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=-vmax[pair], vmax=vmax[pair]))
        cbar = fig.colorbar(sm, cax=cax, ticks=[-vmax[pair], 0, vmax[pair]])
        cbar.ax.set_yticklabels([pair[1], "0", pair[0]], fontsize=9)
        for tick, sel in zip(cbar.ax.get_yticklabels(), [pair[1], None, pair[0]]):
            tick.set_color("#444444" if sel is None else SEL_INFO[sel]["color"][:7])
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0)

    fig.suptitle(fig_title)
    if out_path:
        fig.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()


# -------------------------------- Top images -------------------------------- #
def plot_top_nsd_imgs(model_dict: dict, n_top: int=10,
                      sel_types: list=["face", "body", "mixed", "nonselective"],
                      nsd_h5_path: Path=None, out_path: Path=None):
    """
    Plot top-n NSD images per selective unit type, ranked by
    mean response of last layer averaged across models.
    """
    if nsd_h5_path is None:
        nsd_h5_path = DATA_ROOT / "datasets" / "nsd" / "nsd_stimuli.hdf5"

    # Mean response per image and unit type, averaged across models
    resp = {sel: [] for sel in sel_types}
    for model, layers in model_dict.items():
        resp_path = (
            PROJECT_ROOT / "nsd" / "models" / "mean_resp_sel" /
            f"resp_{model}_{layers[-1]}.npy"
        )

        res = np.load(resp_path, allow_pickle=True).item()
        nsd_ids = np.asarray(res["nsd_ids"], dtype=np.int64)
        for sel in sel_types:
            r = res["mean_resp"][sel]
            resp[sel].append((r - r.mean()) / r.std())
    resp = {sel: np.mean(r, axis=0) for sel, r in resp.items()}

    # Top-n images per unit type
    top_ids = {sel: nsd_ids[np.argsort(r)[::-1][:n_top]] for sel, r in resp.items()}
    wanted = np.unique(np.concatenate(list(top_ids.values())))
    with h5py.File(nsd_h5_path, "r") as f:
        imgs = dict(zip(wanted.tolist(), f["/imgBrick"][wanted]))

    fig, axes = plt.subplots(
        len(sel_types), n_top,
        figsize=(n_top * 1.5, len(sel_types) * 1.5),
        gridspec_kw={"wspace": 0, "hspace": 0.1},
        squeeze=False,
    )

    for axes_row, sel in zip(axes, sel_types):
        for ax, img_id in zip(axes_row, top_ids[sel]):
            ax.imshow(imgs[int(img_id)], aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes_row[0].set_ylabel(
            SEL_INFO[sel]["label"],
            color=SEL_INFO[sel]["color"], fontsize=9, fontweight="bold",
        )

    if out_path is not None:
        fig.savefig(FIG_ROOT / out_path, dpi=300, bbox_inches="tight")
    plt.show()
