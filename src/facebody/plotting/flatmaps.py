"""Cortical flatmaps of selective voxels and of the encoding results, via pycortex."""

from pathlib import Path
from functools import lru_cache
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
import cortex
from cortex.quickflat.utils import (
    _convert_svg_kwargs, _parse_defaults, _get_extents, _get_height,
)

from facebody.config import PROJECT_ROOT, FIG_ROOT
from facebody.style import SEL_INFO, NON_COLOR, SHARED_COLOR_FLATMAP as SHARED_COLOR
from facebody.myutils.nsd import SubjectLoader

XFM_NAME = "auto-align"
SUBJ_GRID_SHAPE = (3, 3)
PANEL_SIZE = (3, 4)
CBAR_BAND = 1.6
CBAR_SIZE = (1.5, 0.1)
KEY_SIZE = 0.85

FACE_COLOR = SEL_INFO["face"]["color"][:7]
BODY_COLOR = SEL_INFO["body"]["color"][:7]
MIXED_COLOR = SEL_INFO["mixed"]["color"][:7]

SEL_ROIS = {
    "face": {"color": FACE_COLOR, "label": "Face"},
    "body": {"color": BODY_COLOR, "label": "Body"},
    "mixed": {"color": MIXED_COLOR, "label": "Mixed"},
    "non": {"color": NON_COLOR, "label": "Non"},
}

FACE_ROIS = ("OFA", "FFA", "aTL-faces", "mTL-faces")
BODY_ROIS = ("EBA", "FBA", "mTL-bodies")
CONTEXT_ROIS = ("OPA", "PPA", "EVC")
ROI_COLORS = {
    **{r: NON_COLOR for r in CONTEXT_ROIS},
    **{r: FACE_COLOR for r in FACE_ROIS},
    **{r: BODY_COLOR for r in BODY_ROIS},
}

BLOCKS = {
    "f": ("r2", "f", "Face", FACE_COLOR),
    "b": ("r2", "b", "Body", BODY_COLOR),
    "m": ("r2", "m", "Mixed", MIXED_COLOR),
    "ns": ("r2", "ns", "Non", NON_COLOR),
    "fb": ("r2", "fb", "Face + Body", SHARED_COLOR),
    "fbm": ("r2", "fbm", "Face + Body + Mixed", MIXED_COLOR),
    "u_f": ("varpart", "u_f", "Face unique", FACE_COLOR),
    "u_b": ("varpart", "u_b", "Body unique", BODY_COLOR),
    "s_fb": ("varpart", "s_fb", "Face-Body shared", SHARED_COLOR),
    "u_m": ("delta_m", "delta_m", "Mixed unique", MIXED_COLOR),
}
SEP_BLOCKS = ("f", "b", "m", "ns")
VARPART_PANELS = (("u_f", "u_b"), ("s_fb",), ("u_m",))

_title = lambda block: BLOCKS[block][2]
_color = lambda block: BLOCKS[block][3]


# ---------------------------- Volume -> surface ----------------------------- #
@lru_cache(maxsize=None)
def _brain_mask(mask_path: str):
    """Flattened brain mask and the volume shape it came from."""
    img = nib.load(mask_path)
    return np.asarray(img.dataobj).astype(bool).ravel(), img.shape

@lru_cache(maxsize=None)
def _mapper(subj: str, xfmname: str=XFM_NAME):
    """Cached nearest-neighbour volume -> surface mapper, plus its coverage mask."""
    m = cortex.utils.get_mapper(subj, xfmname, "nearest")
    covered = np.concatenate([np.asarray(mask.sum(1)).ravel() for mask in m.masks]) > 0
    return m, covered

def _to_vertices(subj: str, sl: SubjectLoader, vec: np.ndarray,
                 xfmname: str=XFM_NAME, fill: float=np.nan):
    """Brain-mask-length vector -> vertex-length vector."""
    mask, shape = _brain_mask(str(sl.roi_mask_dir / "brainmask.nii.gz"))
    vol = np.full(mask.size, fill, dtype=np.float32)
    vol[mask] = vec
    vol = cortex.Volume(vol.reshape(shape).T, subject=subj, xfmname=xfmname)

    mapper, covered = _mapper(subj, xfmname)
    v = mapper(vol)
    verts = np.concatenate([np.ravel(v.left), np.ravel(v.right)]).astype(np.float32)
    verts[~covered] = fill
    return verts

def _streams_vector(sl: SubjectLoader, meta: dict, values: np.ndarray,
                    keep: np.ndarray=None):
    """Visual-cortex vector -> brain-mask-length vector."""
    vec = np.full(int(sl.load_brain_mask()[0].sum()), np.nan, dtype=np.float32)
    vals = np.asarray(values, dtype=np.float32).copy()
    if keep is not None:
        vals[~np.asarray(keep, bool)] = np.nan
    vec[meta["vox_ids"]] = vals
    return vec

def _scalar_view(subj: str, sl: SubjectLoader, vec: np.ndarray, cmap, lo: float,
                 hi: float, xfmname: str=XFM_NAME):
    """Brain-length values -> Vertex colormapped between lo and hi (NaN transparent)."""
    return cortex.Vertex(_to_vertices(subj, sl, vec, xfmname), subject=subj,
                         vmin=lo, vmax=hi, cmap=cmap)

def _rgb_view(subj: str, sl: SubjectLoader, rgb: np.ndarray, alpha: np.ndarray,
              xfmname: str=XFM_NAME):
    """Brain-length RGB (3, n) and opacity (n), all 0-1, -> VertexRGB."""
    chans = tuple(np.ascontiguousarray((255 * np.clip(np.nan_to_num(
        _to_vertices(subj, sl, c, xfmname, fill=0.0)), 0.0, 1.0)).astype(np.uint8))
        for c in (*rgb, alpha))
    return cortex.VertexRGB(*chans[:3], subject=subj, alpha=chans[3])

def _label_view(subj: str, sl: SubjectLoader, labels: np.ndarray, colors: tuple,
                xfmname: str=XFM_NAME, weights: np.ndarray=None):
    """Brain-length labels (-1 = not shown) -> VertexRGB, weights set each voxel's opacity."""
    rgb = np.zeros((3, labels.size), dtype=np.float32)
    alpha = (np.ones(labels.size, np.float32) if weights is None
             else np.clip(np.nan_to_num(np.asarray(weights, np.float32)), 0.0, 1.0))
    for i, color in enumerate(colors):
        rgb[:, labels == i] = np.asarray(to_rgb(color), np.float32)[:, None]
    return _rgb_view(subj, sl, rgb, alpha * (labels >= 0), xfmname)


# -------------------------------- Color scales ------------------------------- #
def _fade_cmap(color: str):
    """Colormap from transparent to color, so low values keep showing the curvature."""
    r, g, b = to_rgb(color)
    return LinearSegmentedColormap.from_list("fade", [(r, g, b, 0.0), (r, g, b, 1.0)])

def _norm(vals: np.ndarray, lo: float, hi: float):
    """Values -> 0-1 between lo and hi, clipped, NaN -> 0."""
    w = (np.asarray(vals, float) - lo) / max(hi - lo, 1e-12)
    return np.clip(np.nan_to_num(w, nan=0.0), 0.0, 1.0).astype(np.float32)

def _blend_2d(w_v: np.ndarray, w_h: np.ndarray, c_v: str, c_h: str,
              c_both: str=SHARED_COLOR):
    """Two 0-1 weights -> RGB (3, n) and opacity (n): clear where both are 0, c_both
    where both are 1."""
    w_v, w_h = np.clip(w_v, 0.0, 1.0), np.clip(w_h, 0.0, 1.0)
    corners = ((w_v * (1 - w_h), c_v), ((1 - w_v) * w_h, c_h), (w_v * w_h, c_both))
    rgb = sum(np.asarray(to_rgb(c), np.float32)[:, None] * w[None] for w, c in corners)
    alpha = 1 - (1 - w_v) * (1 - w_h)
    return rgb / np.maximum(alpha, 1e-12)[None], alpha.astype(np.float32)

def _scale(vecs: list, vmin: float=None, vmax: float=None, pct: float=99):
    """Value range, by default 0 to the pct-th percentile of the pooled vectors."""
    lo = 0.0 if vmin is None else float(vmin)
    if vmax is not None:
        return lo, float(vmax)
    vals = np.concatenate([np.asarray(v, float).ravel() for v in vecs])
    hi = float(np.nanpercentile(vals, pct)) if np.any(np.isfinite(vals)) else 1.0
    return lo, max(hi, lo + 1e-6)

def _per_panel(value, n: int):
    """Scalar or None -> one entry per panel; a sequence is passed through."""
    return list(value) if isinstance(value, (list, tuple)) else [value] * n

def _suptitle(subjects, layer: str, unit: str=None):
    """Subject and layer for one subject, layer alone for a subject grid."""
    head = f"{subjects[0]} {layer}" if len(subjects) == 1 else layer
    return head if unit is None else f"{head} {unit}"

def _r2_unit(normalize: bool):
    """Axis label for a noise-ceiling-normalized or raw R2."""
    return "Explained variance (R² adj.)" if normalize else "Explained variance (R²)"


# ------------------------- Outlines, zoom, and legend ------------------------ #
@lru_cache(maxsize=None)
def _overlay(subj: str):
    """Parsed overlays.svg, kept per subject rather than re-read per ROI."""
    return cortex.db.get_overlay(subj)

@lru_cache(maxsize=None)
def _roi_texture(subj: str, rois: tuple, height: int, color: str, linewidth: float,
                 labelsize: int, with_labels: bool):
    """Outlines for rois, all in one color."""
    kws = _parse_defaults("rois_paths")
    kws.update(_convert_svg_kwargs(dict(linewidth=linewidth, linecolor=color,
                                        labelcolor="white", labelsize=labelsize)))
    return _overlay(subj).get_texture("rois", height, labels=with_labels,
                                      shape_list=list(rois), **kws)

def _add_roi_outlines(ax, subj: str, rois: dict=None, linewidth: float=4,
                      labelsize: int=12, with_labels: bool=True):
    """Draw ROIs in their own colors, one transparent layer per color."""
    groups = {}
    for roi, color in (ROI_COLORS if rois is None else rois).items():
        groups.setdefault(color, []).append(roi)

    for color, group in groups.items():
        im = _roi_texture(subj, tuple(group), _get_height(ax), color, linewidth,
                          labelsize, with_labels)
        ax.imshow(im, aspect="equal", interpolation="bicubic", extent=_get_extents(ax),
                  label="rois", zorder=1000)

def zoom_to_box(ax=None, box: float=0.5):
    """Crop flatmap to a centered square (scalar) or an explicit (cx, cy, w, h) box."""
    ax = ax or plt.gca()
    box = 0.5 if box is True else box
    x0, x1, y0, y1 = _get_extents(ax)
    w, h = x1 - x0, y1 - y0
    cx, cy, bw, bh = (0.5, 0.5, box, box * w / h) if np.isscalar(box) else box
    ax.axis([x0 + (cx - bw / 2) * w, x0 + (cx + bw / 2) * w,
             y0 + (cy - bh / 2) * h, y0 + (cy + bh / 2) * h])

def _fill_handles(types: tuple):
    """Voxel-type legend entries."""
    return [mpatches.Patch(facecolor=SEL_ROIS[t]["color"], label=SEL_ROIS[t]["label"])
            for t in SEL_ROIS if t in types]

def _outline_handles(outlines: dict=None):
    """ROI-boundary legend entries."""
    named = {FACE_COLOR: "Face ROIs", BODY_COLOR: "Body ROIs"}
    color_by_label = {}
    for color in (ROI_COLORS if outlines is None else outlines).values():
        color_by_label.setdefault(named.get(color, "Other ROIs"), color)

    order = ["Face ROIs", "Body ROIs", "Other ROIs"]
    labels = sorted(color_by_label, key=lambda l: order.index(l))
    return [mlines.Line2D([], [], color=color_by_label[l], lw=2, label=l) for l in labels]

def _add_legends(target, types: tuple, outline_rois: dict=None, y: float=-0.10):
    """Voxel-type and ROI-outline legends, stacked below the target."""
    groups = [h for h in (_fill_handles(types), _outline_handles(outline_rois)) if h]
    for i, handles in enumerate(groups):
        leg = target.legend(handles=handles, loc="lower center",
                            bbox_to_anchor=(0.5, y - 0.08 * i), ncol=len(handles),
                            frameon=False, fontsize=10)
        if i < len(groups) - 1 and not isinstance(target, plt.Figure):
            target.add_artist(leg)


# ---------------------------------- Colorbars -------------------------------- #
def _band(fig):
    """Height of the colorbar strip below the panels, as a figure fraction."""
    return CBAR_BAND / fig.get_figheight()

def _add_cbar(fig, rect: tuple, cmap, lo: float, hi: float, label: str,
              fontsize: int=10):
    """Horizontal colorbar labelled at both ends."""
    bar = fig.colorbar(ScalarMappable(Normalize(lo, hi), cmap), cax=fig.add_axes(rect),
                       orientation="horizontal")
    if bar.solids is not None:
        bar.solids.set_alpha(None)
    bar.set_ticks([lo, hi], labels=[f"{lo:.0f}", f"{hi:.2f}"])
    bar.set_label(label, fontsize=fontsize + 2)
    bar.ax.tick_params(labelsize=fontsize)
    return bar

def _cbar_rect(fig, x_center: float, y: float=None):
    """Rect for a colorbar of fixed size (CBAR_SIZE, in inches) centered on x_center."""
    w, h = CBAR_SIZE
    fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
    y = 0.55 * CBAR_BAND / fig_h if y is None else y
    return (x_center - w / 2 / fig_w, y, w / fig_w, h / fig_h)

def _add_cbar_under(fig, ax, cmap, lo: float, hi: float, label: str):
    """Colorbar centered under one panel."""
    pos = ax.get_position()
    return _add_cbar(fig, _cbar_rect(fig, pos.x0 + pos.width / 2), cmap, lo, hi, label)

def _add_2d_key(fig, ax, lo: float, hi: float, colors: tuple, labels: tuple,
                side: float=KEY_SIZE, n: int=256, fontsize: int=10):
    """Square 2D color key centered under one panel, first block on the vertical axis."""
    w_h, w_v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    rgb, alpha = _blend_2d(w_v.ravel(), w_h.ravel(), *colors)
    rgb = (1 - alpha + alpha * rgb).T.reshape(n, n, 3)

    pos = ax.get_position()
    fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
    cax = fig.add_axes((pos.x0 + pos.width / 2 - side / 2 / fig_w,
                        0.40 * CBAR_BAND / fig_h, side / fig_w, side / fig_h))
    cax.imshow(rgb, origin="lower", extent=(lo, hi, lo, hi), aspect="auto",
               interpolation="bilinear")
    cax.set_xticks([lo, hi], labels=[f"{lo:.0f}", f"{hi:.2f}"])
    cax.set_yticks([lo, hi], labels=[f"{lo:.0f}", f"{hi:.2f}"])
    cax.set_xlabel(labels[1], fontsize=fontsize + 2)
    cax.set_ylabel(labels[0], fontsize=fontsize + 2)
    cax.tick_params(labelsize=fontsize)
    return cax


# ----------------------------------- Drawing --------------------------------- #
def _save(fig, out_path: Path, tag: str="saved"):
    """Write a figure under FIG_ROOT, creating the directory if needed."""
    p = FIG_ROOT / out_path
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"{tag} -> {p}")
    return p

def _draw(view, subj: str, ax=None, outline_rois: dict=None, outline_labels: bool=True,
          linewidth: float=5, labelsize: int=20, zoom: float=0.5,
          title: str=None, recache: bool=False):
    """Render one flatmap, cropped to the central square."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    cortex.quickflat.make_figure(view, recache=recache, dpi=300, fig=ax,
                                 with_rois=False, with_colorbar=False, with_curvature=True)
    _add_roi_outlines(ax, subj, outline_rois, linewidth, labelsize, outline_labels)
    if zoom:
        zoom_to_box(ax, zoom)
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    return ax

def _row_label(ax, text: str, fontsize: int=12):
    """Subject name to the left of grid row."""
    ax.text(-0.05, 0.5, text, transform=ax.transAxes, rotation=90, va="center",
            ha="right", fontsize=fontsize)

def _subject_grid(plot_fn, subjects, legend_types: tuple, figsize: tuple=(9, 14),
                  cbars=None, out_path: Path=None, **kwargs):
    """Run a single-panel flatmap function over subjects, one shared legend below."""
    n_rows, n_cols = SUBJ_GRID_SHAPE
    assert len(subjects) <= n_rows * n_cols, (
        f"{len(subjects)} subjects don't fit a {n_rows}x{n_cols} grid")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for ax, subj in zip(axes.ravel(), subjects):
        plot_fn(subj, ax=ax, legend=False, title=subj, **kwargs)
    for ax in axes.ravel()[len(subjects):]:
        ax.axis("off")

    fig.tight_layout()
    _add_legends(fig, legend_types, kwargs.get("outline_rois"), y=-0.02)
    if cbars is not None:
        cbars(fig)
    if out_path is not None:
        _save(fig, out_path, "grid")
    return fig, axes


# ------------------------- Selective voxels (fLoc ROIs) ---------------------- #
def _selective_view(subj: str, types: tuple, proj_dir: Path, xfmname: str):
    """Colors each voxel by the selectivity type it belongs to."""
    sl = SubjectLoader(subj, proj_dir)
    vox_meta = sl.load_vox_meta()
    missing = [t for t in types if t not in vox_meta.columns]
    assert not missing, (f"{subj}: {missing} not in vox_meta.csv (has "
                         f"{list(vox_meta.columns)}) -- run run_fmri_selectivity.py "
                         f"first")

    labels = np.full(len(vox_meta), -1, dtype=np.int16)
    for i, t in enumerate(types):
        labels[vox_meta[t].to_numpy() != 0] = i
    return _label_view(subj, sl, labels, tuple(SEL_ROIS[t]["color"] for t in types),
                       xfmname)

def plot_roi_flatmap(subj: str, types: tuple=("face", "body", "mixed", "non"),
                     proj_dir: Path=PROJECT_ROOT, xfmname: str=XFM_NAME, ax=None,
                     legend: bool=True, title: str=None, out_path: Path=None,
                     **draw_kw):
    """Flatmap with selective voxels for one subject."""
    view = _selective_view(subj, types, proj_dir, xfmname)
    ax = _draw(view, subj, ax, title=subj if title is None else title, **draw_kw)
    if legend:
        _add_legends(ax, types, draw_kw.get("outline_rois"))
    if out_path is not None:
        _save(ax.figure, out_path, subj)
    return ax

def plot_roi_flatmaps_subjects(subjects, types: tuple=("face", "body", "mixed", "non"),
                               figsize: tuple=(7.5, 8), out_path: Path=None, **kwargs):
    """Flatmaps with selective voxels for all subjects."""
    return _subject_grid(plot_roi_flatmap, subjects, legend_types=types, types=types,
                         figsize=figsize, out_path=out_path, **kwargs)


# --------------------------- Selective voxels in 3D -------------------------- #
def save_roi_inflated_webgl(subj: str, types: tuple=("face", "body", "mixed"),
                            proj_dir: Path=PROJECT_ROOT, xfmname: str=XFM_NAME,
                            surface_types: tuple=("inflated",),
                            overlays: tuple=("rois",), recache: bool=True,
                            title: str=None, out_path: Path=None):
    """Standalone WebGL viewer (index.html + assets) of one subject's selective voxels."""
    view = _selective_view(subj, types, proj_dir, xfmname)
    out_path = Path("webgl") / subj if out_path is None else out_path
    outdir = FIG_ROOT / out_path
    cortex.webgl.make_static(outpath=str(outdir), data=view, recache=recache, pixelwise=True,
                             types=surface_types, overlays_visible=list(overlays),
                             labels_visible=[], title=subj if title is None else title)
    print(f"saved -> {outdir}/index.html")
    return outdir


# ------------------------------ Encoding results ----------------------------- #
def _load(model_name: str, layer: str, subj: str, proj_dir: Path, results: tuple=None):
    """Voxelwise results for one subject, or the pre-loaded tuple if given."""
    from facebody.encoding.summarize import load_voxelwise
    return results or load_voxelwise(model_name, layer, subj, proj_dir)

def _block_values(res: dict, meta: dict, block: str, normalize: bool, nc_min: float):
    """Visual cortex R2 for one block, optionally noise ceiling normalized."""
    assert block in BLOCKS, f"unknown block {block!r}; choose from {list(BLOCKS)}"
    source, key, _, _ = BLOCKS[block]
    vals = np.asarray(res[source][key], float)
    if normalize:
        nc = np.asarray(meta["nc"], float)
        vals = np.divide(vals, np.maximum(nc, nc_min), where=nc > nc_min,
                         out=np.full_like(vals, np.nan))
    return vals

def _keep_mask(res: dict, subj: str, layer: str, mask_sig: bool, alpha_fwe: float):
    """Voxels the layer predicts above chance, FWE-corrected."""
    if not mask_sig:
        return None
    from facebody.encoding.permute import sig_mask
    assert "perm" in res, (f"{subj} {layer}: no permutation test stored -- add the layer "
                           f"to VoxelwiseEncodingConfig.perm_layers, or pass mask_sig=False")
    return sig_mask(res["perm"], alpha_fwe)

def _block_vectors(subj: str, model_name: str, layer: str, blocks: tuple, proj_dir: Path,
                   normalize: bool, nc_min: float, mask_sig: bool, alpha_fwe: float,
                   results: tuple=None):
    """Subject loader and one brain-length R2 vector per block."""
    sl = SubjectLoader(subj, proj_dir)
    res, meta = _load(model_name, layer, subj, proj_dir, results)
    keep = _keep_mask(res, subj, layer, mask_sig, alpha_fwe)
    vecs = {b: _streams_vector(sl, meta,
                               _block_values(res, meta, b, normalize, nc_min), keep)
            for b in blocks}
    return sl, vecs


# ------------------------ Encoding: winner take all -------------------------- #
def _wta_best(subj: str, model_name: str, layer: str, blocks: tuple, proj_dir: Path,
              normalize: bool, nc_min: float, mask_sig: bool, alpha_fwe: float,
              results: tuple=None):
    """Winning block index and its R2 per brain-mask voxel (-1 / NaN where not shown)."""
    sl, vecs = _block_vectors(subj, model_name, layer, blocks, proj_dir, normalize,
                              nc_min, mask_sig, alpha_fwe, results)
    vals = np.stack([vecs[b] for b in blocks])

    shown = np.any(np.isfinite(vals), axis=0)
    winner = np.full(vals.shape[1], -1, dtype=np.int16)
    best = np.full(vals.shape[1], np.nan, dtype=np.float32)
    winner[shown] = np.nanargmax(vals[:, shown], axis=0)
    best[shown] = np.nanmax(vals[:, shown], axis=0)
    return sl, winner, best

def _wta_panel(subj: str, sl: SubjectLoader, winner: np.ndarray, best: np.ndarray,
               blocks: tuple, lo: float, hi: float, xfmname: str, ax=None, **draw_kw):
    """Draw one winner-take-all flatmap, each voxel in its winning block's color."""
    view = _label_view(subj, sl, winner, tuple(_color(b) for b in blocks), xfmname,
                       weights=_norm(best, lo, hi))
    return _draw(view, subj, ax, **draw_kw)

def _add_wta_cbars(fig, blocks: tuple, lo: float, hi: float, unit: str, y: float=None,
                   width: float=0.15, height: float=0.012, gap: float=0.075):
    """One transparent -> color bar per unit type, side by side on a shared scale."""
    n = len(blocks)
    x0 = 0.5 - (n * width + (n - 1) * gap) / 2
    y = 0.55 * CBAR_BAND / fig.get_figheight() if y is None else y
    for i, b in enumerate(blocks):
        _add_cbar(fig, (x0 + i * (width + gap), y, width, height),
                  _fade_cmap(_color(b)), lo, hi, _title(b))
    fig.text(0.5, y - 0.8 / fig.get_figheight(), unit, ha="center", fontsize=12)

def plot_encoding_sep_wta_flatmap(subj: str, model_name: str, layer: str,
                                  blocks: tuple=SEP_BLOCKS, proj_dir: Path=PROJECT_ROOT,
                                  normalize: bool=True, nc_min: float=0.0,
                                  mask_sig: bool=True, alpha_fwe: float=0.05,
                                  vmin: float=None, vmax: float=None, pct: float=90,
                                  xfmname: str=XFM_NAME, ax=None, legend: bool=True,
                                  title: str=None, results: tuple=None,
                                  out_path: Path=None, **draw_kw):
    """Flatmap with the winning unit type per voxel, opacity scaled by its R2."""
    sl, winner, best = _wta_best(subj, model_name, layer, blocks, proj_dir, normalize,
                                 nc_min, mask_sig, alpha_fwe, results)
    lo, hi = _scale([best], vmin, vmax, pct)
    ax = _wta_panel(subj, sl, winner, best, blocks, lo, hi, xfmname, ax,
                    title=f"{subj}  {layer}" if title is None else title, **draw_kw)
    if legend:
        _add_wta_cbars(ax.figure, blocks, lo, hi, _r2_unit(normalize), y=-0.14)
    if out_path is not None:
        _save(ax.figure, out_path, subj)
    return ax

def plot_encoding_sep_wta_flatmaps_subjects(subjects, model_name: str, layer: str,
                                            blocks: tuple=SEP_BLOCKS,
                                            proj_dir: Path=PROJECT_ROOT,
                                            normalize: bool=True, nc_min: float=0.0,
                                            mask_sig: bool=True, alpha_fwe: float=0.05,
                                            vmin: float=None, vmax: float=None,
                                            pct: float=95, xfmname: str=XFM_NAME,
                                            figsize: tuple=(7.5, 8),
                                            out_path: Path=None, **draw_kw):
    """Winner-take-all flatmaps for all subjects on a shared scale."""
    n_rows, n_cols = SUBJ_GRID_SHAPE
    assert len(subjects) <= n_rows * n_cols, (
        f"{len(subjects)} subjects don't fit a {n_rows}x{n_cols} grid")

    wta = {s: _wta_best(s, model_name, layer, blocks, proj_dir, normalize, nc_min,
                        mask_sig, alpha_fwe) for s in subjects}
    lo, hi = _scale([best for _, _, best in wta.values()], vmin, vmax, pct)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for ax, subj in zip(axes.ravel(), subjects):
        sl, winner, best = wta[subj]
        _wta_panel(subj, sl, winner, best, blocks, lo, hi, xfmname, ax, title=subj,
                   **draw_kw)
    for ax in axes.ravel()[len(subjects):]:
        ax.axis("off")

    fig.tight_layout()
    _add_wta_cbars(fig, blocks, lo, hi, _r2_unit(normalize), y=-0.02)
    if out_path is not None:
        _save(fig, out_path, "grid")
    return fig, axes


# --------------------- Encoding: R2 panels per unit type --------------------- #
def _panel_figure(subjects, model_name: str, layer: str, panels: tuple, view_fn,
                  scales_fn, cbar_fn, proj_dir: Path, normalize: bool, nc_min: float,
                  mask_sig: bool, alpha_fwe: float, xfmname: str, results: tuple,
                  figsize: tuple, suptitle: str, out_path: Path, **draw_kw):
    """One column per panel, one row per subject, colorbars below the last row."""
    assert results is None or len(subjects) == 1, "results= is for a single subject"
    blocks = tuple(dict.fromkeys(b for p in panels for b in p))
    data = {s: _block_vectors(s, model_name, layer, blocks, proj_dir, normalize, nc_min,
                              mask_sig, alpha_fwe, results) for s in subjects}
    scales = scales_fn([vecs for _, vecs in data.values()])

    n_rows, n_cols = len(subjects), len(panels)
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=figsize or (
        PANEL_SIZE[0] * n_cols, PANEL_SIZE[1] * n_rows + CBAR_BAND))
    for r, subj in enumerate(subjects):
        sl, vecs = data[subj]
        for ax, panel, scale in zip(axes[r], panels, scales):
            _draw(view_fn(subj, sl, vecs, panel, scale, xfmname), subj, ax,
                  title=" / ".join(_title(b) for b in panel) if r == 0 else None,
                  **draw_kw)
        if n_rows > 1:
            _row_label(axes[r, 0], subj)

    fig.tight_layout(rect=(0, _band(fig), 1, 1 - 0.75 / fig.get_figheight()))
    cbar_fn(fig, axes[-1], panels, scales)
    fig.suptitle(suptitle, fontsize=14, y=1 - 0.25 / fig.get_figheight())
    if out_path is not None:
        _save(fig, out_path, subjects[0] if n_rows == 1 else "grid")
    return fig, axes

def _sep_view(subj: str, sl: SubjectLoader, vecs: dict, panel: tuple, scale: tuple,
              xfmname: str):
    """One unit type on the inferno colormap."""
    return _scalar_view(subj, sl, vecs[panel[0]], "inferno", *scale, xfmname)

def _sep_figure(subjects, model_name: str, layer: str, blocks: tuple=SEP_BLOCKS,
                proj_dir: Path=PROJECT_ROOT, normalize: bool=True, nc_min: float=0.0,
                mask_sig: bool=True, alpha_fwe: float=0.05, vmin: float=None,
                vmax: float=None, pct: float=95, xfmname: str=XFM_NAME,
                results: tuple=None, figsize: tuple=None, out_path: Path=None,
                **draw_kw):
    """R2 flatmaps per unit type, inferno on one scale shared by all panels."""
    unit = _r2_unit(normalize)

    def scales_fn(all_vecs):
        return [_scale([v for vecs in all_vecs for v in vecs.values()],
                       vmin, vmax, pct)] * len(blocks)

    def cbar_fn(fig, last_row, panels, scales):
        _add_cbar(fig, _cbar_rect(fig, 0.5), "inferno", *scales[0], unit)

    return _panel_figure(subjects, model_name, layer, tuple((b,) for b in blocks),
                         _sep_view, scales_fn, cbar_fn, proj_dir, normalize, nc_min,
                         mask_sig, alpha_fwe, xfmname, results, figsize,
                         _suptitle(subjects, layer), out_path, **draw_kw)

def plot_encoding_sep_flatmaps(subj: str, model_name: str, layer: str, **kwargs):
    """One row of R² flatmaps, one per unit type, on a shared inferno scale."""
    return _sep_figure([subj], model_name, layer, **kwargs)

def plot_encoding_sep_flatmaps_subjects(subjects, model_name: str, layer: str,
                                        **kwargs):
    """R2 flatmaps per unit type for all subjects, one row per subject."""
    return _sep_figure(list(subjects), model_name, layer, **kwargs)


# ------------------- Encoding: R2 panels per variance portion ---------------- #
def _varpart_view(subj: str, sl: SubjectLoader, vecs: dict, panel: tuple, scale: tuple,
                  xfmname: str):
    """One variance portion fading in to its color, or two blended in a 2D colormap."""
    if len(panel) == 1:
        return _scalar_view(subj, sl, vecs[panel[0]], _fade_cmap(_color(panel[0])),
                            *scale, xfmname)
    rgb, alpha = _blend_2d(*[_norm(vecs[b], *scale) for b in panel],
                           *[_color(b) for b in panel])
    return _rgb_view(subj, sl, rgb, alpha, xfmname)

def _varpart_figure(subjects, model_name: str, layer: str, panels: tuple=VARPART_PANELS,
                    proj_dir: Path=PROJECT_ROOT, normalize: bool=True, nc_min: float=0.0,
                    mask_sig: bool=True, alpha_fwe: float=0.05, vmin=None, vmax=None,
                    pct: float=95, xfmname: str=XFM_NAME, results: tuple=None,
                    figsize: tuple=None, out_path: Path=None, **draw_kw):
    """R2 flatmaps per variance portion, each panel on its own scale."""
    assert all(len(p) in (1, 2) for p in panels), "each panel takes one or two blocks"
    unit = _r2_unit(normalize)

    def scales_fn(all_vecs):
        return [_scale([vecs[b] for vecs in all_vecs for b in p], lo, hi, pct)
                for p, lo, hi in zip(panels, _per_panel(vmin, len(panels)),
                                     _per_panel(vmax, len(panels)))]

    def cbar_fn(fig, last_row, panels, scales):
        for ax, panel, scale in zip(last_row, panels, scales):
            if len(panel) == 2:
                _add_2d_key(fig, ax, *scale, tuple(_color(b) for b in panel),
                            tuple(_title(b) for b in panel))
            else:
                _add_cbar_under(fig, ax, _fade_cmap(_color(panel[0])), *scale,
                                _title(panel[0]))

    return _panel_figure(subjects, model_name, layer, panels, _varpart_view, scales_fn,
                         cbar_fn, proj_dir, normalize, nc_min, mask_sig, alpha_fwe,
                         xfmname, results, figsize, _suptitle(subjects, layer, unit),
                         out_path, **draw_kw)

def plot_encoding_varpart_flatmaps(subj: str, model_name: str, layer: str, **kwargs):
    """One row of R2 flatmaps, one per variance portion, each on its own scale."""
    return _varpart_figure([subj], model_name, layer, **kwargs)

def plot_encoding_varpart_flatmaps_subjects(subjects, model_name: str, layer: str,
                                            **kwargs):
    """Variance-portion flatmaps for all subjects, one row per subject."""
    return _varpart_figure(list(subjects), model_name, layer, **kwargs)
