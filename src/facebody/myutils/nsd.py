import os
from pathlib import Path
import ast
import json
import pickle as pkl
import h5py
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import requests
from tqdm import tqdm

from .utils import get_imagenet_transform
from facebody.config import DATA_ROOT

N_TRIALS_SESSION = 750
H5_SCALE = 300.0

# ----------------------------------- FMRI ----------------------------------- #
# ----------------------------- Download NSD data ---------------------------- #
def download_sessions(subjects: list):
    """Download sessions with fMRI single trial responses."""
    for subj in subjects:
        sl = SubjectLoader(subj)
        base_url = (
            "https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/"
            f"{subj}/func1pt8mm/betas_fithrf_GLMdenoise_RR/"
        )

        for i in tqdm(range(sl.n_sessions), desc=f"Downloading sessions for {sl.subject}"):
            fname = sl.raw_dir / f"betas_session{i+1:02d}.hdf5"
            url = base_url + f"betas_session{i+1:02d}.hdf5"

            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)


# ------------------------ Preprocess image meta data ------------------------ #
def get_img_meta_subj(sl, img_meta_all: pd.DataFrame):
    """Get image meta data for subject."""
    trial_ids = img_meta_all[f"{sl.subject_name}_rep2"].values
    subj_mask = (trial_ids > 0) & (trial_ids <= sl.n_trials)
    return img_meta_all[subj_mask]

def prepro_img_meta(data_dir: Path=DATA_ROOT):
    """Preprocess image meta data."""
    subjects_all = [f"subj{s:02}" for s in range(1, 9)]

    nsd_info_dir = data_dir / "datasets" / "nsd" / "nsd_info"
    img_meta_all = pd.read_csv(nsd_info_dir / "nsd_stim_info_merged_anns.csv")

    # Get shared images
    shared_nsd_ids = set(img_meta_all["nsdId"])
    for subj in subjects_all:
        sl = SubjectLoader(subj)
        img_meta_subj = get_img_meta_subj(sl, img_meta_all)
        shared_nsd_ids &= set(img_meta_subj["nsdId"]) # Intersect with current subject's nsdIds

    # Preprocess img meta data
    for subj in subjects_all:
        sl = SubjectLoader(subj)
        img_meta_subj = get_img_meta_subj(sl, img_meta_all).copy()
        img_meta_subj["shared515"] = img_meta_subj["nsdId"].isin(shared_nsd_ids)

        cols_rep = [f"{sl.subject_name}_rep{i}" for i in range(3)]
        img_meta_subj["trialIds"] = img_meta_subj[cols_rep].apply(lambda row: [x-1 for x in row], axis=1)
        img_meta_subj["trialIds"] = img_meta_subj["trialIds"].apply(json.dumps)

        cols_keep = [
            "nsdId", "trialIds", "shared515",
            "category_ids", "category_labels", "supercategory_labels",
        ]
        img_meta_subj = img_meta_subj.loc[:, cols_keep]
        img_meta_subj.to_csv(sl.resp_dir / "img_meta.csv", index=False)


# ------------------------ Preprocess voxel responses ------------------------ #
def zscore_trials(sl, sessions: list):
    """Z-score each voxel across trials within session and concatenate along trials."""
    brain_mask = sl.load_brain_mask()[0]

    resp_z = []
    for sess_name in tqdm(sessions, desc=f"Z-scoring responses within sessions for {sl.subject}"):
        with h5py.File(sl.raw_dir / sess_name, "r") as f:
            resp = np.array(f["betas"], dtype=np.int16).T.astype(np.float64) / H5_SCALE
        resp = resp.reshape(-1, N_TRIALS_SESSION)[brain_mask]

        mean_all = np.nanmean(resp, axis=1, keepdims=True)
        std_all = np.nanstd(resp, axis=1, ddof=1, keepdims=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            resp_z_session = (resp - mean_all) / std_all
            resp_z_session = np.nan_to_num(resp_z_session)

        resp_z.append(resp_z_session)

    return np.concatenate(resp_z, axis=1).T

def compute_ncsnr(sl, resp_z: np.ndarray, img_meta_long: pd.DataFrame):
    """Compute voxel-wise NCSNR according to NSD."""
    # Group trials by nsdId and filter those with 3 trials
    trial_groups = img_meta_long.groupby("nsdId")["trialIds"].apply(list)
    filt_groups = trial_groups[trial_groups.apply(len) == 3]
    if filt_groups.empty:
        return np.zeros(resp_z.shape[1])

    # Compute within-image variance
    var_within = np.vstack([
        np.nanvar(resp_z[trial_ids, :], axis=0, ddof=1)
        for trial_ids in filt_groups
    ])

    # Compute voxel-wise NCSNR
    var_noise = np.nanmean(var_within, axis=0)
    var_signal = np.maximum(0, 1 - var_noise)

    with np.errstate(divide="ignore", invalid="ignore"):
        ncsnr = np.sqrt(var_signal) / np.sqrt(var_noise)
        ncsnr = np.nan_to_num(ncsnr)

    vol = VolumeConverter(sl.subject)
    vol.save_nii(vol.to_volume(ncsnr, "brain"), vol.resp_dir / "ncsnr.nii.gz")

def average_reps(sl, resp_z: np.ndarray, img_meta: pd.DataFrame):
    """Average single-trial z-scored responses across repetitions for each unique nsdId."""
    trial_ids_list = [np.array(trial_ids) for trial_ids in img_meta["trialIds"]]
    return np.array([
        resp_z[trial_ids].mean(axis=0)
        for trial_ids in tqdm(trial_ids_list, desc=f"Averaging repetitions for {sl.subject}")
    ])

def prepro_resp(subjects: list):
    """Preprocess single-trial voxel responses."""
    for subj in subjects:
        sl = SubjectLoader(subj)
        sessions = sorted([file for file in os.listdir(sl.raw_dir) if file.endswith(".hdf5")])

        # Prepare image meta data
        img_meta = pd.read_csv(sl.resp_dir / "img_meta.csv")
        img_meta["nsdId"] = img_meta["nsdId"].astype(int)
        img_meta["trialIds"] = img_meta["trialIds"].apply(json.loads)
        img_meta["supercategory_labels"] = img_meta["supercategory_labels"].apply(ast.literal_eval)

        img_meta_long = (
            img_meta
            .explode("trialIds")
            .sort_values("trialIds")
            .reset_index(drop=True)
        )

        resp_z = zscore_trials(sl, sessions)
        compute_ncsnr(sl, resp_z, img_meta_long)

        resp_z_avg = average_reps(sl, resp_z, img_meta)
        save_resp_h5_lowmem(sl, resp_z_avg, "brain")
        del resp_z_avg

        print(f"Preprocessed responses for {sl.subject}.")


# -------------------------------- Define ROIs ------------------------------- #
def narrow_rois(subjects: list, proj_dir: Path, rois: list=None,
                t_thresh: float=2, overwrite: bool=False):
    """Narrow down ROIs from t>0 to t>threshold."""
    floc_tval_alises = {"characters": "word", "scenes": "places"}
    rois = rois or ["faces", "bodies", "places", "characters"]
    for subject in subjects:
        roi_dir = proj_dir / "nsd" / "subjects" / subject / "roi"
        for roi in rois:
            out_path = roi_dir / f"floc-{roi}_t{t_thresh:g}.nii.gz"
            if out_path.exists() and not overwrite:
                print(f"{subject} {roi}: exists, skipped")
                continue

            mask_path = roi_dir / f"floc-{roi}.nii.gz"
            tval_path = roi_dir / f"floc_{roi}tval.nii.gz"
            if not tval_path.exists():
                alias = floc_tval_alises.get(roi)
                tval_path = roi_dir / f"floc_{alias}tval.nii.gz" if alias else tval_path
            for p in (mask_path, tval_path):
                assert p.exists(), (f"{subject}: {p.name} not found. Available: "
                                    f"{sorted(q.name for q in roi_dir.glob('floc*'))}")

            vol_mask = nib.load(mask_path)
            vol_tval = nib.load(tval_path)
            assert vol_mask.shape == vol_tval.shape, \
                f"{subject} {roi}: mask {vol_mask.shape} vs t-map {vol_tval.shape}"

            data = np.asanyarray(vol_mask.dataobj).copy()
            tval = np.asarray(vol_tval.get_fdata())

            keep = np.nan_to_num(tval, nan=-np.inf) > t_thresh
            n_before = int((data > 0).sum())
            data[(data > 0) & ~keep] = 0
            n_after = int((data > 0).sum())

            nib.save(nib.Nifti1Image(data, vol_mask.affine, vol_mask.header), out_path)
            print(f"{subject} {roi}: {n_before} -> {n_after} voxels "
                  f"({100 * n_after / max(n_before, 1):.0f}%) -> {out_path.name}")

def define_rois(subjects: list, proj_dir: Path):
    """Define ROIs based on fLOC (t>2)."""
    roi_labels_dir = proj_dir / "nsd" / "nsd_info" / "roi_labels"

    roi_groups = {
        "OFA": [["OFA"], ["floc-faces_t2"]],
        "FFA": [["FFA-1", "FFA-2"], ["floc-faces_t2"]],
        "aTL-faces": [["aTL-faces"], ["floc-faces_t2"]],
        "mTL-faces": [["mTL-faces"], ["floc-faces_t2"]],
        "EBA": [["EBA"], ["floc-bodies_t2"]],
        "FBA": [["FBA-1", "FBA-2"], ["floc-bodies_t2"]],
        "mTL-bodies": [["mTL-bodies"], ["floc-bodies_t2"]],
        "V1": [["V1v", "V1d"], ["prf-visualrois"]],
        "OPA": [["OPA"], ["floc-places_t2"]],
        "PPA": [["PPA"], ["floc-places_t2"]],
        "RSC": [["RSC"], ["floc-places_t2"]],
        "EVC": [["early"], ["streams"]],
        }

    mask_files = {roi_group: value[1][0] + ".nii.gz" for roi_group, value in roi_groups.items()}
    label_files = {roi_group: label[1][0] + ".mgz.ctab" for roi_group, label in roi_groups.items()}

    # Create voxel meta data
    for subj in subjects:
        sl = SubjectLoader(subj, proj_dir)
        brain_mask = sl.load_brain_mask()[0]
        n_voxels_brain = brain_mask.sum()

        vox_meta = pd.DataFrame(index=np.arange(n_voxels_brain))
        for roi_group, roi_list in roi_groups.items():
            mask = np.array(nib.load(sl.roi_mask_dir / mask_files[roi_group]).get_fdata()).flatten()[brain_mask]
            labels = pd.read_csv(roi_labels_dir / label_files[roi_group], sep=" ", header=None, names=["index", "label"])

            roi_ids = [int(labels.loc[labels["label"] == roi, "index"].iloc[0]) for roi in roi_list[0]]

            vox_meta[roi_group] = 0
            vox_ids = np.where(np.isin(mask, roi_ids))[0]
            vox_meta.loc[vox_ids, roi_group] = 1
            print(f"Nr. of voxels in {roi_group}: {len(vox_ids)}")

        vox_meta.to_csv(sl.resp_dir / "vox_meta.csv", index=False)
        print(f"Defined ROIs for {sl.subject}.")

def get_roi_masks(roi_masks):
    roi_df = pd.DataFrame(roi_masks)

    def get_combination(row):
        selected = [roi for roi in roi_df.columns if row[roi]]
        return "&".join(sorted(selected)) if selected else None

    roi_df["comb"] = roi_df.apply(get_combination, axis=1)

    new_roi_masks = {}
    for comb, _group in roi_df.groupby("comb"):
        if comb is not None:
            new_roi_masks[comb] = (roi_df["comb"] == comb)
    return new_roi_masks

def extract_roi_responses(subjects: list, proj_dir: Path,
                          rois_exclusive: tuple, rois_overlap: tuple,
                          nc_thresh: float=0.15, min_vox: int=10):
    """
    For each subject, extract responses from exclusive and overlap ROIs.
    Each exclusive ROI contains only voxels that are not in any of the other ROIs.
    Each overlap ROI contains only voxels that are in both ROIs.
    """
    resps = {}
    for subj in subjects:
        sl = SubjectLoader(subj)
        resp_brain = load_resp_h5_lowmem(sl, "brain_facebody")
        vox_meta = pd.read_csv(sl.resp_dir / "vox_meta_facebody.csv")
 
        present = [r for r in rois_exclusive if r in vox_meta.columns]
        missing = [r for r in rois_exclusive if r not in vox_meta.columns]
        if missing:
            print(f"{subj}: not in vox_meta, skipped -> {missing} "
                  f"(add them to define_rois)")
        masks = {r: vox_meta[r].to_numpy() != 0 for r in present}
 
        roi_masks = {}
        for roi, m in masks.items():
            others = np.zeros_like(m)
            for other, om in masks.items():
                if other != roi:
                    others |= om
            roi_masks[roi] = m & ~others
        for a, b in rois_overlap:
            if a in masks and b in masks:
                roi_masks[f"{a}&{b}"] = masks[a] & masks[b]
 
        ncsnr = sl.load_nc()
 
        def extract_from_mask(mask):
            vox_ids = np.where(mask)[0]
            if vox_ids.size == 0:
                return None
            ncsnr_roi = ncsnr[vox_ids]
            valid_ids = np.where(ncsnr_roi > nc_thresh)[0]
            if valid_ids.size <= min_vox:
                return None
            return {"responses": resp_brain[:, vox_ids][:, valid_ids],
                    "ncsnr": ncsnr_roi[valid_ids]}
 
        subject_rois = {roi: extract_from_mask(m) for roi, m in roi_masks.items()}
        resps[subj] = subject_rois
 
        print(f"Subject {subj}:")
        for roi, data in subject_rois.items():
            if data is None:
                print(f"  {roi}: dropped (<= {min_vox} voxels with ncsnr > {nc_thresh})")
            else:
                print(f"  {roi}: {data['responses'].shape[1]} voxels  "
                      f"(ncsnr range = {data['ncsnr'].min():.2f}-{data['ncsnr'].max():.2f})")
        print()
 
    fname = proj_dir / "nsd" / "fmri_activs.pkl"
    fname.parent.mkdir(parents=True, exist_ok=True)
    with fname.open("wb") as f:
        pkl.dump(resps, f)


# ------------------------------ Save image ids ------------------------------ #
def save_subj_img_ids(subjects, proj_dir: Path):
    """Save image ids for each subject."""
    img_ids_subjs = {}
    for subj in subjects:
        sl = SubjectLoader(subj)
        img_meta = pd.read_csv(sl.resp_dir / "img_meta.csv")
        img_meta["supercategory_labels"] = img_meta["supercategory_labels"].apply(ast.literal_eval)
        subj_ids = img_meta["nsdId"].unique()
        img_ids_subjs[subj] = subj_ids

    fname = proj_dir / "nsd" / "img_ids.pkl"
    fname.parent.mkdir(parents=True, exist_ok=True)
    with open(fname, "wb") as f:
        pkl.dump(img_ids_subjs, f)


# ------------------------------- SubjectLoader ------------------------------ #
class SubjectLoader():
    """Load subject data."""
    def __init__(self, subject: str, proj_dir: Path, data_dir: Path=DATA_ROOT):
        self.subject = subject
        self.subject_name = f"subject{int(self.subject[4:]):d}"
        self.data_dir = data_dir
        self.proj_dir = proj_dir

        self.subject_dir = proj_dir / "nsd" / "subjects" / subject
        self.raw_dir = self.subject_dir / "raw"
        self.resp_dir = self.subject_dir / "resp"
        self.roi_mask_dir = self.subject_dir / "roi"
        for d in (self.raw_dir, self.resp_dir, self.roi_mask_dir):
            os.makedirs(d, exist_ok=True)

        n_sessions_all = {
            "subj01": 40, "subj02": 40, "subj03": 32, "subj04": 30,
            "subj05": 40, "subj06": 32, "subj07": 40, "subj08": 30,
        }
        self.n_sessions = n_sessions_all[self.subject]
        self.n_trials = self.n_sessions * N_TRIALS_SESSION

        self.img_meta_loaded = False
        self.img_meta = None

    def load_img_meta(self):
        if not self.img_meta_loaded:
            self.img_meta = pd.read_csv(self.resp_dir / "img_meta.csv")
            self.img_meta_loaded = True
        return self.img_meta

    def load_ncsnr(self):
        brain_mask = self.load_brain_mask()[0]
        ncsnr = nib.load(self.resp_dir / "ncsnr.nii.gz").get_fdata().flatten()[brain_mask]
        ncsnr = (ncsnr**2) / ((ncsnr**2) + (1/3))
        return ncsnr

    def load_resp(self, roi: str):
        return np.load(self.resp_dir / f"resp_{roi}_t2.npy")

    def load_imgs(self, ids: np.ndarray=None):
        img_meta = pd.read_csv(self.resp_dir / "img_meta.csv")
        nsd_ids = np.unique(img_meta["nsdId"])

        with h5py.File(self.data_dir / "datasets" / "nsd" / "nsd_stimuli.hdf5", "r") as f:
            img_brick = f["/imgBrick"]
            imgs = img_brick[nsd_ids]

        if ids is not None and len(ids) > 0:
            imgs = imgs[ids]
        return imgs.T

    def load_vox_meta(self):
        return pd.read_csv(self.resp_dir / "vox_meta.csv")

    def load_roi_mask(self, roi: str, ids_ref: str):
        vox_meta = self.load_vox_meta()
        roi_mask = vox_meta[roi] != 0

        if ids_ref == "brain":
            roi_ids = np.where(roi_mask)[0]
        elif ids_ref == "volume":
            brain_ids = self.load_brain_mask()[1]
            roi_ids = brain_ids[np.where(roi_mask)[0]]
        else:
            print("Select 'brain' or 'volume' as reference.")

        return roi_mask, roi_ids

    def load_brain_mask(self):
        brain_mask = nib.load(self.roi_mask_dir / "brainmask.nii.gz").get_fdata().astype(bool).flatten()
        brain_ids = np.where(brain_mask)[0]
        return brain_mask, brain_ids

    def load_volume_info(self):
        brain_mask_vol = nib.load(self.roi_mask_dir / "brainmask.nii.gz")
        return brain_mask_vol.affine, brain_mask_vol.get_fdata().shape

    def load_nc(self):
        brain_mask = self.load_brain_mask()[0]
        ncsnr = nib.load(self.resp_dir / "ncsnr.nii.gz").get_fdata().flatten()[brain_mask]
        return (ncsnr**2) / ((ncsnr**2) + (1 / 3))


# ------------------------------ VolumeConverter ----------------------------- #
class VolumeConverter(SubjectLoader):
    """Convert matrix to volume."""
    def __init__(self, subject):
        SubjectLoader.__init__(self, subject)
        self.affine, self.volume_shape = self.load_volume_info()
        self.vox_meta = self.load_vox_meta()

    def to_volume(self, data, roi):
        brain_ids = self.load_brain_mask()[1]

        data = np.atleast_2d(data)
        volumes = []
        for d in data:
            volume = np.zeros(self.volume_shape).flatten()
            if roi == "brain":
                volume[brain_ids] = d
            else:
                roi_ids = self.load_roi_mask(roi, "brain")[1]
                volume[brain_ids[roi_ids]] = d
            volumes.append(volume.reshape(self.volume_shape))

        return np.stack(volumes, axis=-1) if data.shape[0] > 1 else volumes[0]

    def save_nii(self, volume, filename):
        img = nib.Nifti1Image(volume, self.affine)
        img.header.get_xyzt_units()
        img.to_filename(filename)


# ----------------------------------- Utils ---------------------------------- #
def save_resp_h5_lowmem(sl, resp: np.ndarray, name: str="brain"):
    resp = (resp*300).astype(dtype=np.int16) # Low memory transformation
    with h5py.File(sl.resp_dir / f"resp_{name}.hdf5", "w") as f:
        f.create_dataset(f"resp_{name}", data=resp)

def load_resp_h5_lowmem(sl, name: str="brain"):
    with h5py.File(sl.resp_dir / f"resp_{name}.hdf5", "r") as f:
        resp = f[f"resp_{name}"][:]
    return resp.astype(dtype=np.int64) / 300 # Undo low memory transformation


# ---------------------------------- IMAGES ---------------------------------- #
class NSDImageLoader(Dataset):
    """Map-style dataset over images stored in HDF5 file."""
    def __init__(self, h5_path: Path, ids: list, size: tuple=(224, 224)):
        self.h5_path = str(h5_path)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.size = size
        self._h5 = None
        self._imgs = None
        self.tf = get_imagenet_transform()

    def _lazy_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._imgs = self._h5["/imgBrick"]

    def __getitem__(self, i: int):
        self._lazy_open()
        idx = int(self.ids[i])
        x = self._imgs[idx]
        x = Image.fromarray(x)
        return self.tf(x), 0

    def __len__(self):
        return len(self.ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_imgs"] = None
        return state

    def close(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5, self._imgs = None, None

    def __del__(self):
        self.close()

def _nsd_img_worker_init(_):
    """Initialize worker for H5Images DataLoader."""
    info = torch.utils.data.get_worker_info()
    if info is not None and isinstance(info.dataset, NSDImageLoader):
        info.dataset.close()
