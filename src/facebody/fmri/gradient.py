"""Build posterior-to-anterior geodesic axis through ventral stream and measure selectivity along it."""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import spearmanr, wilcoxon, binomtest
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from facebody.config import PROJECT_ROOT
from .floc import SEL_TYPES, LABEL_COLS
from .streams import ROIS_COMPLETE
from facebody.myutils.nsd import SubjectLoader

VENTRAL_STREAMS = ("early", "midventral", "ventral")
ROI_SEED = "V1"
HEMIS = ("lh", "rh")

N_GRID = 10
NORM_PCT = 99.0

# ----------------------------------- Masks ---------------------------------- #
def vox_meta_mask(sl: SubjectLoader, columns):
    """Brain-mask-length boolean mask, true where any of `columns` is set."""
    vox_meta = sl.load_vox_meta()
    m = np.zeros(len(vox_meta), bool)
    for col in np.atleast_1d(columns):
        if col in vox_meta.columns:
            m |= vox_meta[col].to_numpy() != 0
        else:
            print(f"  note: '{col}' not a vox_meta column, skipped")
    return m

def ventral_stream_mask(sl: SubjectLoader):
    """Get ventral stream mask."""
    from .floc import build_stream_mask

    streams = build_stream_mask(sl, VENTRAL_STREAMS, verbose=False)[0]
    path = streams | vox_meta_mask(sl, ROIS_COMPLETE)
    seed = vox_meta_mask(sl, ROI_SEED) & path
    assert seed.any(), (f"{sl.subject}: no '{ROI_SEED}' voxels inside the ventral "
                        f"mask -- check vox_meta.csv")
    return path, seed

def hemi_masks(sl: SubjectLoader):
    """Get hemisphere masks."""
    brain_mask, _ = sl.load_brain_mask()
    affine, shape = sl.load_volume_info()
    idx3 = np.array(np.unravel_index(np.flatnonzero(brain_mask), shape)).T
    x = nib.affines.apply_affine(affine, idx3)[:, 0]
    return {"lh": x < np.median(x), "rh": x >= np.median(x)}


# -------------------------------- Define axis ------------------------------- #
def define_axis(sl: SubjectLoader, mask: np.ndarray, seed: np.ndarray,
                verbose: bool=True):
    """Geodesic distance in mm from seed through mask."""
    brain_mask, _ = sl.load_brain_mask()
    affine, shape = sl.load_volume_info()
    flat_ids = np.flatnonzero(brain_mask)[np.flatnonzero(mask)]

    idx3 = np.array(np.unravel_index(flat_ids, shape)).T
    node_of = np.full(shape, -1, np.int64)
    node_of[idx3[:, 0], idx3[:, 1], idx3[:, 2]] = np.arange(len(idx3))
    world = nib.affines.apply_affine(affine, idx3)

    rows, cols, wts = [], [], []

    connectivity_offsets = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
        for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)],
    )
    for off in connectivity_offsets:
        nb = idx3 + off
        inb = ((nb >= 0) & (nb < np.asarray(shape))).all(1)
        j_all = np.full(len(idx3), -1, np.int64)
        j_all[inb] = node_of[nb[inb, 0], nb[inb, 1], nb[inb, 2]]
        i = np.flatnonzero(j_all >= 0)
        rows.append(i)
        cols.append(j_all[i])
        wts.append(np.linalg.norm(world[i] - world[j_all[i]], axis=1))

    g = sparse.coo_matrix((np.concatenate(wts),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(len(idx3),) * 2).tocsr()
    g = g.maximum(g.T)

    seed_local = np.flatnonzero(seed[np.flatnonzero(mask)])
    assert seed_local.size, f"{sl.subject}: no seed voxels in this mask"

    n_comp, lab = connected_components(g, directed=False)
    if n_comp > 1:
        keep_lab = np.bincount(lab[seed_local], minlength=n_comp).argmax()
        drop = lab != keep_lab
        if verbose:
            print(f"    {n_comp} components; dropping "
                  f"{int(drop.sum())}/{len(idx3)} voxels off the main one")
    else:
        drop = np.zeros(len(idx3), bool)

    d_local = dijkstra(g, directed=False, indices=seed_local, min_only=True)
    d_local[drop | ~np.isfinite(d_local)] = np.nan

    d = np.full(brain_mask.sum(), np.nan, np.float32)
    d[np.flatnonzero(mask)] = d_local
    return d


# ------------------------------ Build gradient ------------------------------ #
def gradient_path(subj: str, proj_dir: Path=PROJECT_ROOT):
    return Path(proj_dir) / "nsd" / "gradient" / f"{subj}_gradient.npz"

def build_gradient(subj: str, sl: SubjectLoader=None, proj_dir: Path=PROJECT_ROOT,
                   norm_pct: float=NORM_PCT, save: bool=True, verbose: bool=True):
    """Build gradient per hemisphere and pool into one normalized gradient."""
    sl = sl or SubjectLoader(subj, proj_dir)
    path, seed = ventral_stream_mask(sl)
    hemis = hemi_masks(sl)

    d_mm = np.full(path.size, np.nan, np.float32)
    d_norm = np.full(path.size, np.nan, np.float32)
    hemi = np.full(path.size, -1, np.int8)
    scale_mm = np.full(len(HEMIS), np.nan)

    if verbose:
        print(f"{subj}:")
    for h, name in enumerate(HEMIS):
        d = define_axis(sl, path & hemis[name], seed & hemis[name], verbose)
        on = np.isfinite(d)
        scale_mm[h] = float(np.nanpercentile(d[on], norm_pct))

        d_mm[on] = d[on]
        d_norm[on] = np.clip(d[on] / scale_mm[h], 0, 1)
        hemi[on] = h
        if verbose:
            print(f"  {name}: {int(on.sum())} voxels on the axis, "
                  f"P{norm_pct:.0f} = {scale_mm[h]:.0f} mm")

    if save:
        gradient_path(subj, proj_dir).parent.mkdir(parents=True, exist_ok=True)
        np.savez(gradient_path(subj, proj_dir), d_norm=d_norm, d_mm=d_mm, hemi=hemi,
                 scale_mm=scale_mm, hemis=np.array(HEMIS, dtype=object))
    return {"d_norm": d_norm, "d_mm": d_mm, "hemi": hemi, "scale_mm": scale_mm}

def load_gradient(subj: str, proj_dir: Path=PROJECT_ROOT, order: str="brain"):
    """Load gradient in brain-mask order, or in fit-mask order of encoding."""
    assert order in ("brain", "fit"), f"order must be 'brain' or 'fit', got '{order}'"

    z = np.load(gradient_path(subj, proj_dir), allow_pickle=True)
    out = {k: z[k] for k in ("d_norm", "d_mm", "hemi")}
    out["scale_mm"] = z["scale_mm"]

    if order == "fit":
        vox_ids = np.load(SubjectLoader(subj, proj_dir).resp_dir / "streams_meta.npz",
                          allow_pickle=True)["vox_ids"]
        out.update({k: out[k][vox_ids] for k in ("d_norm", "d_mm", "hemi")})
    return out


# ---------------------------- Selectivity profile --------------------------- #
def gradient_bins(d_norm: np.ndarray, n_grid: int=N_GRID):
    """Normalized position along gradient."""
    idx = np.full(d_norm.shape, -1, int)
    ok = np.isfinite(d_norm)
    idx[ok] = np.clip((d_norm[ok] * n_grid).astype(int), 0, n_grid - 1)
    return idx, (np.arange(n_grid) + 0.5) / n_grid * 100

def perc_ventral_stream(subjects: list, n_grid: int=N_GRID,
                        responsive: bool=True, proj_dir: Path=PROJECT_ROOT):
    """Percentage of selective voxels per bin."""
    cols = list(LABEL_COLS) + (["responsive"] if responsive else [])

    all_curves = {sel: [] for sel in SEL_TYPES}
    for subj in subjects:
        sl = SubjectLoader(subj, proj_dir)
        path = sl.resp_dir / "vox_meta.csv"

        missing = [c for c in cols if c not in pd.read_csv(path, nrows=0).columns]
        assert not missing, (f"{subj}: vox_meta.csv is missing {missing} -- rerun "
                             f"run_fmri_selectivity.py (and run_fmri_glm.py first if "
                             f"'responsive' is missing, for the visual>baseline map)")

        vox_meta = pd.read_csv(path, usecols=cols)
        d = load_gradient(subj, proj_dir)["d_norm"]

        searched = vox_meta[list(LABEL_COLS)].to_numpy().any(axis=1)
        if responsive:
            searched &= vox_meta["responsive"].to_numpy() != 0
        idx, x_grid = gradient_bins(np.where(searched, d, np.nan), n_grid)
        totals = np.bincount(idx[idx >= 0], minlength=n_grid).astype(float)

        for sel in SEL_TYPES:
            lab = (vox_meta[sel].to_numpy() != 0) & (idx >= 0)
            counts = np.bincount(idx[lab], minlength=n_grid).astype(float)
            all_curves[sel].append(np.divide(100.0 * counts, totals,
                                             out=np.full(n_grid, np.nan),
                                             where=totals != 0))

    return {sel: np.vstack(v) for sel, v in all_curves.items()}, x_grid


# -------------------------------- Statistics -------------------------------- #
def gradient_stats(curves: dict, x_grid: np.ndarray, subjects: list,
                   sel_types: tuple=SEL_TYPES, label: str="% of voxels",
                   correct: str="fdr_bh", verbose: bool=True):
    """Change of each binned curve along the hierarchy, tested across subjects/models."""
    rows, out = [], {}
    for sel in sel_types:
        rs = []
        for i, subj in enumerate(subjects):
            y = curves[sel][i]
            ok = np.isfinite(y)
            r = float(spearmanr(x_grid[ok], y[ok]).statistic) if ok.sum() >= 3 else np.nan
            rs.append(r)
            rows.append({"subject": subj, "sel": sel, "spearman_r": r})

        rs = np.asarray(rs, float)
        ok = np.isfinite(rs)
        n_ok = int(ok.sum())
        n_pos = int((rs[ok] > 0).sum())

        if n_ok > 1 and np.any(rs[ok] != 0):
            wt = wilcoxon(rs[ok])
            w, p = float(wt.statistic), float(wt.pvalue)
        else:
            w, p = np.nan, np.nan

        p_sign = (float(binomtest(max(n_pos, n_ok - n_pos), n_ok, 0.5,
                                  alternative="two-sided").pvalue) if n_ok else np.nan)

        out[sel] = {"r": rs,
                    "median_r": float(np.nanmedian(rs)) if n_ok else np.nan,
                    "r_min": float(np.nanmin(rs)) if n_ok else np.nan,
                    "r_max": float(np.nanmax(rs)) if n_ok else np.nan,
                    "w": w, "p": p,
                    "n": n_ok, "n_pos": n_pos, "p_sign": p_sign}

    if correct:
        from statsmodels.stats.multitest import multipletests

        keys = [k for k in sel_types if np.isfinite(out[k]["p"])]
        if keys:
            p_adj = multipletests([out[k]["p"] for k in keys],
                                  alpha=0.05, method=correct)[1]
            for k, pa in zip(keys, p_adj):
                out[k]["p_raw"] = out[k]["p"]
                out[k]["p"] = float(pa)

    df = pd.DataFrame(rows)
    if verbose:
        print(f"Spearman(position, {label}), n = {len(subjects)}"
              + (f", {correct} across {len(sel_types)}" if correct else ", uncorrected"))
        print("   " + df.pivot(index="subject", columns="sel", values="spearman_r")
                        .round(3).to_string().replace("\n", "\n   "))
        for sel in sel_types:
            s = out[sel]
            print(f"   {sel:6s} r = {s['median_r']:.2f}, p = {s['p']:.3f}"
                  f"   | median of {s['n']}; range {s['r_min']:.2f} to {s['r_max']:.2f}; "
                  f"positive in {s['n_pos']}/{s['n']}; raw p = {s.get('p_raw', s['p']):.3f}")
    return out, df


# ------------------------------- Run profile -------------------------------- #
def run_fmri_profile(subjects: list, proj_dir: Path=PROJECT_ROOT, n_grid: int=N_GRID,
                     responsive: bool=True, save: bool=True):
    """Profile selectivity along gradient, test it, and save curves + stats."""
    curves, x_grid = perc_ventral_stream(subjects, n_grid, responsive, proj_dir)
    stats, per_subject = gradient_stats(
        curves, x_grid, subjects,
        label="% of responsive voxels" if responsive else "% of voxels")

    out = {"x_grid": x_grid, "curves": curves, "stats": stats,
           "per_subject": per_subject, "subjects": list(subjects), "n_grid": n_grid,
           "responsive": responsive}
    if save:
        from facebody.myutils.utils import save_pickle
        save_pickle(out, Path(proj_dir) / "nsd" / "gradient" / "profile.pkl")
