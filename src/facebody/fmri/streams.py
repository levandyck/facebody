"""Extract responses for every voxel in visual cortex, from NSD's streams atlas plus ROIs kept whole."""

from pathlib import Path
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from facebody.config import PROJECT_ROOT
from .floc import STREAMS, build_stream_mask
from facebody.myutils.nsd import SubjectLoader, save_resp_h5_lowmem

ROIS = (
    "OFA", "FFA", "aTL-faces",
    "EBA", "FBA", "mTL-bodies",
    "FFA&FBA", "OPA", "PPA",
)
ROIS_COMPLETE = ("PPA", "aTL-faces", "mTL-bodies")

# ------------------------------ Load and inspect ---------------------------- #
def load_streams_meta(subj: str, proj_dir: Path=PROJECT_ROOT):
    """Voxel meta data for visual cortex mask."""
    sl = SubjectLoader(subj, proj_dir)
    meta = np.load(sl.resp_dir / "streams_meta.npz", allow_pickle=True)
    return {k: meta[k] for k in meta.files}

def load_streams_resp(subj: str, proj_dir: Path=PROJECT_ROOT):
    """Load stream responses per subject."""
    sl = SubjectLoader(subj, proj_dir)
    with h5py.File(sl.resp_dir / "resp_streams.hdf5", "r") as f:
        return _resp_dataset(f, "streams")[:].astype(np.float32) / 300

def _resp_dataset(f, name: str):
    """Dataset inside resp_*.hdf5."""
    if f"resp_{name}" in f:
        return f[f"resp_{name}"]
    keys = list(f.keys())
    assert len(keys) == 1, f"resp_{name}.hdf5 holds {keys}, cannot pick one"
    return f[keys[0]]

def _load_resp_cols(sl: SubjectLoader, vox_ids: np.ndarray, name: str="brain",
                    step: int=500):
    """Load column subset."""
    n_brain = int(sl.load_brain_mask()[0].sum())
    with h5py.File(sl.resp_dir / f"resp_{name}.hdf5", "r") as f:
        dset = _resp_dataset(f, name)
        assert dset.shape[1] == n_brain, (
            f"{sl.subject}: resp_{name}.hdf5 has {dset.shape[1]} columns but the brain "
            f"mask has {n_brain} voxels. vox_ids index brain-mask space, so this file is "
            f"in a different voxel space -- pass the matching resp_name.")
        out = np.empty((dset.shape[0], len(vox_ids)), dtype=np.float32)
        for i in tqdm(range(0, dset.shape[0], step), desc=f"Loading {name} for {sl.subject}"):
            out[i:i+step] = dset[i:i+step, :][:, vox_ids].astype(np.float32) / 300
    return out


# --------------------------------- ROI masks -------------------------------- #
def roi_columns(vox_meta: pd.DataFrame, rois: tuple, subj: str=None):
    """Raw ROI membership, one boolean column per ROI."""
    missing = [r for r in rois if r not in vox_meta.columns]
    if missing:
        print(f"{subj}: not in vox_meta, skipped -> {missing} (add them to define_rois)")
    return {r: vox_meta[r].to_numpy() != 0 for r in rois if r in vox_meta.columns}


# --------------------------- Visual cortex responses ------------------------ #
def extract_streams_resp(subjects: list, rois: tuple,
                         rois_complete: tuple=ROIS_COMPLETE,
                         proj_dir: Path=PROJECT_ROOT, streams: tuple=STREAMS,
                         resp_name: str="brain", vox_meta_name: str="vox_meta",
                         verbose: bool=True):
    """Extract responses for every voxel in visual cortex."""
    for subj in subjects:
        sl = SubjectLoader(subj, proj_dir)
        stream_mask, _ = build_stream_mask(sl, streams, verbose=verbose)
        vox_meta = pd.read_csv(sl.resp_dir / f"{vox_meta_name}.csv")
        masks = roi_columns(vox_meta, rois, subj)

        fit_mask = stream_mask.copy()
        for roi in rois_complete:
            if roi in masks:
                fit_mask |= masks[roi]
        vox_ids = np.where(fit_mask)[0]

        # Label each voxel by stream; -1 for voxels the union added outside the atlas
        stream_id = np.full(fit_mask.sum(), -1, dtype=np.int16)
        for i, s in enumerate(streams):
            s_mask, _ = build_stream_mask(sl, (s,), verbose=False)
            stream_id[s_mask[fit_mask]] = i

        resp = _load_resp_cols(sl, vox_ids, resp_name)
        ncsnr = nib.load(sl.resp_dir / "ncsnr.nii.gz").get_fdata().flatten()[
            sl.load_brain_mask()[0]][vox_ids]

        save_resp_h5_lowmem(sl, resp, "streams")
        np.savez_compressed(
            sl.resp_dir / "streams_meta.npz",
            vox_ids=vox_ids.astype(np.int32),
            nc=sl.load_nc()[vox_ids].astype(np.float32),  # noise ceiling, in R2 units
            ncsnr=ncsnr.astype(np.float32),
            stream_id=stream_id,
            roi_mask=np.stack([m[fit_mask] for m in masks.values()], axis=1),
            roi_names=np.array(list(masks), dtype=object),
            streams=np.array(streams, dtype=object),
        )

        if verbose:
            print(f"{subj}: {fit_mask.sum()} voxels "
                  f"({int(stream_mask.sum())} in streams, "
                  f"{int((fit_mask & ~stream_mask).sum())} added by {list(rois_complete)})")
            for roi, m in masks.items():
                kept = int((m & fit_mask).sum())
                lost = int(m.sum()) - kept
                flag = f"  <-- {lost} voxels outside the mask" if lost else ""
                print(f"  {roi}: {kept} voxels{flag}")
            print()
