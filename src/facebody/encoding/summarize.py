"""Read voxelwise HDF5 results back and filter them to ROIs."""

import h5py
import numpy as np

from facebody.config import PROJECT_ROOT
from facebody.fmri.floc import load_sel_masks
from facebody.fmri.streams import load_streams_meta

SEL_KEYS = ("f", "b", "m", "ns")
VP_KEYS = ("r2_f", "r2_b", "r2_fb", "u_f", "u_b", "s_fb")
OVERLAP_ROIS = (("FFA", "FBA"),)
SEL_ROIS = ("face", "body", "mixed", "non")

# ------------------------------- Load and filter ---------------------------- #
def load_voxelwise(model_name: str, layer: str, subj: str, proj_dir=PROJECT_ROOT):
    """Voxelwise results for one subject and layer."""
    path = proj_dir / "encoding" / model_name / "voxelwise" / f"{layer}.h5"
    with h5py.File(path, "r") as f:
        g = f[subj]
        blocks = list(g.attrs["blocks"])
        res = {
            "r2": {b: g["r2"][i] for i, b in enumerate(blocks)},
            "alpha": {b: g["alpha"][i] for i, b in enumerate(blocks)},
            "varpart": {k: g["varpart"][i] for i, k in enumerate(g["varpart"].attrs["components"])},
            "delta_m": {k: g["delta_m"][i] for i, k in enumerate(g["delta_m"].attrs["components"])},
        }
        if "perm" in g:
            res["perm"] = {k: g["perm"][k][:] for k in g["perm"]}
        res["attrs"] = dict(g.attrs)

        written = res["attrs"].get("model_name")
        if written is not None and str(written) != model_name:
            raise ValueError(
                f"{path}/{subj} was written by model '{written}', not '{model_name}'. "
                f"The activation cache was probably shared between models -- see "
                f"NSDRepository.activs_cache_dir.")

    return res, load_streams_meta(subj, proj_dir)

def available_subjects(model_name: str, layer: str, proj_dir=PROJECT_ROOT):
    """Subjects already written for this layer."""
    path = proj_dir / "encoding" / model_name / "voxelwise" / f"{layer}.h5"
    if not path.exists():
        return []
    with h5py.File(path, "r") as f:
        return sorted(f.keys())

def roi_masks(meta: dict, exclusive: bool=True, overlap: tuple=OVERLAP_ROIS,
              subj: str=None, proj_dir=PROJECT_ROOT, sel_rois: tuple=SEL_ROIS):
    """
    ROI masks over the fit voxels, from the stored extraction.

    With exclusive=True each ROI has every other ROI removed from it,
    so "FFA" means FFA minus the others and the FFA/FBA overlap is its own ROI.
    Selectivity masks (face/body/mixed/non) are added unchanged.
    """
    raw = {str(name): meta["roi_mask"][:, i] for i, name in enumerate(meta["roi_names"])}
    sel = load_sel_masks(subj, proj_dir, "fit", vox_types=sel_rois) if subj else {}
    if not exclusive:
        return {**raw, **sel}

    out = {}
    for roi, m in raw.items():
        others = np.zeros_like(m)
        for other, om in raw.items():
            if other != roi:
                others |= om
        out[roi] = m & ~others
    for a, b in overlap:
        if a in raw and b in raw:
            out[f"{a}&{b}"] = raw[a] & raw[b]
    return {**out, **sel}


# --------------------------------- To ROIs ---------------------------------- #
def voxelwise_to_roi(model_name: str, layer: str, subjects: list, rois: list,
                     analysis: str="sep", proj_dir=PROJECT_ROOT, sig_only: bool=False):
    """Filter voxel-wise map to ROIs."""
    keys = {"sep": SEL_KEYS, "fb_varpart": VP_KEYS,
            "delta_m": ("r2_fb", "r2_fbm", "delta_m")}[analysis]

    out = {}
    for subj in subjects:
        res, meta = load_voxelwise(model_name, layer, subj, proj_dir)
        masks = roi_masks(meta, subj=subj, proj_dir=proj_dir)
        src = res["r2"] if analysis == "sep" else res[
            "varpart" if analysis == "fb_varpart" else "delta_m"]

        keep = np.ones(len(meta["nc"]), bool)
        if sig_only and "perm" in res:
            from .permute import sig_mask
            keep = sig_mask(res["perm"])

        out[subj] = {}
        for roi in rois:
            m = masks.get(roi)
            if m is None or not (m & keep).any():
                out[subj][roi] = None
                continue
            vals = {k: np.asarray(src[k])[m & keep] for k in keys}
            out[subj][roi] = {"r2": vals} if analysis == "sep" else vals
    return out

def load_nc_by_roi(subjects: list=None, proj_dir=PROJECT_ROOT):
    """Per-voxel noise ceilings per subject and ROI, as {subj: {roi: {"ncsnr": nc}}}."""
    subjects = [f"subj0{i}" for i in range(1, 9)] if subjects is None else subjects

    out = {}
    for subj in subjects:
        try:
            meta = load_streams_meta(subj, proj_dir)
        except FileNotFoundError:
            continue
        out[subj] = {roi: {"ncsnr": meta["nc"][m]}
                     for roi, m in roi_masks(meta, subj=subj, proj_dir=proj_dir).items()
                     if m.any()}
    return out
