"""Download NSD's fLoc runs, fit localizer GLM, and localize selective voxels."""

from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import norm

from facebody.config import PROJECT_ROOT
from facebody.myutils.nsd import SubjectLoader
from facebody.myutils.utils import save_pickle

BUCKET = "https://natural-scenes-dataset.s3.amazonaws.com"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

FLOC_RUNS = (3, 4, 7, 8, 11, 12)

# No spatial smoothing
SMOOTHING_FWHM = None
HIGH_PASS = 1 / 128

# fLoc domains
DOMAINS = ("faces", "bodies", "objects", "places", "characters")

# Contrasts with shared baseline (objects) and direct face-body comparison
CONTRASTS = {
    "face>object": "faces - objects",
    "body>object": "bodies - objects",
    "place>object": "places - objects",
    "character>object": "characters - objects",
    "face>body": "faces - bodies",
    "visual>baseline": "faces + bodies + objects + places + characters",
}
VISUAL = "visual>baseline"

# fLoc category to domain
CATEGORY_TO_DOMAIN = {
    "adult": "faces", "child": "faces",
    "body": "bodies", "limb": "bodies",
    "car": "objects", "instrument": "objects",
    "corridor": "places", "house": "places",
    "word": "characters", "number": "characters",
}
DOMAIN_ALIASES = {
    "face": "faces", "faces": "faces",
    "body": "bodies", "bodies": "bodies",
    "object": "objects", "objects": "objects",
    "place": "places", "places": "places", "scene": "places", "scenes": "places",
    "character": "characters", "characters": "characters",
    "word": "characters", "words": "characters", "text": "characters",
}
BLANK_LABELS = {"blank", "baseline", "fixation", "rest", "null", "none", "n/a", "nan", ""}

# One-sided thresholds on the FDR-corrected localizer contrasts
# Kliger & Yovel (2024) used uncorrected p_pos = 0.0001 and p_neg = 0.01
P_POS = 0.05
P_NEG = 0.05
FDR_METHOD = "fdr_bh"

STREAMS = (
    "early",
    "midventral", "ventral",
    "midlateral", "lateral",
    "midparietal", "parietal",
)
SEL_TYPES = ("face", "body", "mixed")
LABEL_COLS = (*SEL_TYPES, "non", "other")

# ---------------------------- Paths and headers ----------------------------- #
def run_tr(path):
    """Read sampling interval from header of each run."""
    z = nib.load(path).header.get_zooms()
    assert len(z) > 3, f"{Path(path).name}: not 4-D"
    return float(z[3])

def floc_dir(subj: str, proj_dir: Path=PROJECT_ROOT):
    return Path(proj_dir) / "nsd" / "subjects" / subj / "floc"

def map_dir(subj: str, proj_dir: Path=PROJECT_ROOT):
    return Path(proj_dir) / "selectivity" / "floc" / subj


# ------------------------ Download preprocessed runs ------------------------ #
def _list_keys(prefix: str):
    """List all object keys under `prefix` in NSD bucket."""
    import requests

    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keys += [c.find("s3:Key", S3_NS).text for c in root.findall("s3:Contents", S3_NS)]
        trunc = root.find("s3:IsTruncated", S3_NS)
        if trunc is None or trunc.text != "true":
            return keys
        token = root.find("s3:NextContinuationToken", S3_NS).text

def _download(key: str, dest: Path, retries: int=3):
    """Stream one object to `dest`, skipping it if the size already matches."""
    import requests

    for attempt in range(1, retries + 1):
        try:
            with requests.get(f"{BUCKET}/{key}", stream=True, timeout=300) as r:
                r.raise_for_status()
                remote = int(r.headers.get("Content-Length", 0))
                if dest.exists() and remote and dest.stat().st_size == remote:
                    return dest
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                tmp.rename(dest)
                print(f"    got {dest.name} ({remote/1e6:.1f} MB)")
                return dest
        except Exception as e:
            print(f"    retry {attempt}/{retries} ({dest.name}): {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed to download {key}")

def download_floc_data(subjects: list, proj_dir: Path=PROJECT_ROOT,
                       runs: tuple=FLOC_RUNS):
    """For each subject, load what the fLoc GLM needs."""
    want = {f"timeseries_prffloc_run{r:02d}.nii.gz" for r in runs}
    for subj in subjects:
        n = int(subj.replace("subj", ""))
        out = floc_dir(subj, proj_dir)
        print(f"=== {subj} -> {out} ===")

        ts_keys = sorted(k for k in _list_keys(
            f"nsddata_timeseries/ppdata/{subj}/func1pt8mm/timeseries/")
            if Path(k).name in want)
        assert len(ts_keys) == len(want), (
            f"{subj}: found {len(ts_keys)} of the {len(want)} fLoc runs {sorted(runs)}")
        print(f"  {len(ts_keys)} fLoc runs")
        for k in ts_keys:
            _download(k, out / Path(k).name)

        ses_keys = _list_keys(f"nsddata_rawdata/sub-{n:02d}/ses-prffloc/")
        ev_keys = sorted(k for k in ses_keys if "task-floc" in k and k.endswith("events.tsv"))
        assert ev_keys, f"{subj}: no task-floc events.tsv in the rawdata"
        print(f"  {len(ev_keys)} events files")
        for k in ev_keys:
            _download(k, out / Path(k).name)

    return [floc_dir(s, proj_dir) for s in subjects]

def load_floc_runs(subj: str, proj_dir: Path=PROJECT_ROOT, runs: tuple=FLOC_RUNS):
    """Load preprocessed fLoc runs."""
    d = floc_dir(subj, proj_dir)
    bold = [d / f"timeseries_prffloc_run{r:02d}.nii.gz" for r in runs]
    missing = [p.name for p in bold if not p.exists()]
    assert not missing, f"{subj}: missing {missing} -- run download_floc_data first"

    events = sorted(d.glob("*task-floc*_events.tsv"))
    assert len(bold) == len(events), \
        f"{subj}: {len(bold)} runs selected but {len(events)} events files"
    return [str(p) for p in bold], [str(p) for p in events]


# ---------------------------------- Events ---------------------------------- #
def _domain_series(ev: pd.DataFrame, domain_col: str=None):
    """Events table -> canonical fLoc domain labels (NaN for blank/baseline rows)."""
    if domain_col is None:
        for c in ("domain", "trial_type", "category", "condition"):
            if c in ev.columns:
                domain_col = c
                break
    assert domain_col is not None, f"no usable label column (cols={list(ev.columns)})"

    raw = ev[domain_col].astype(str).str.strip().str.lower()
    out = raw.map(lambda s: DOMAIN_ALIASES.get(s, CATEGORY_TO_DOMAIN.get(s)))
    out[raw.isin(BLANK_LABELS)] = np.nan
    unknown = sorted(set(raw[out.isna() & ~raw.isin(BLANK_LABELS)]))
    assert not unknown, (f"unrecognised labels in '{domain_col}': {unknown} -- extend "
                         f"DOMAIN_ALIASES / CATEGORY_TO_DOMAIN")
    return out

def load_events(events_file, domain_col: str=None):
    """BIDS events.tsv -> nilearn design events (onset, duration, trial_type=domain)."""
    ev = pd.read_csv(events_file, sep="\t")
    dom = _domain_series(ev, domain_col)
    keep = dom.notna()
    return (pd.DataFrame({"onset": ev.loc[keep, "onset"].astype(float),
                          "duration": ev.loc[keep, "duration"].astype(float),
                          "trial_type": dom[keep].values})
            .sort_values("onset").reset_index(drop=True))


# ---------------------------------- Fit GLM --------------------------------- #
def fit_floc_glm(bold_runs: list, events_files: list, mask_img,
                 hrf_model: str="glover", high_pass: float=HIGH_PASS,
                 smoothing_fwhm: float=SMOOTHING_FWHM, domain_col: str=None):
    """Fit one first-level GLM across a subject's fLoc runs and return contrast maps."""
    from nilearn.glm.first_level import FirstLevelModel

    assert len(bold_runs) == len(events_files), "runs / events count mismatch"

    trs = {round(run_tr(f), 4) for f in bold_runs}
    assert len(trs) == 1, f"runs disagree on TR: {trs}"
    t_r = trs.pop()

    design = [load_events(f, domain_col) for f in events_files]
    for f, ev in zip(events_files, design):
        missing = [d for d in DOMAINS if d not in set(ev["trial_type"])]
        assert not missing, f"{Path(f).name}: domains {missing} absent from this run"

    glm = FirstLevelModel(t_r=t_r, hrf_model=hrf_model, mask_img=mask_img,
                          drift_model="cosine", high_pass=high_pass, noise_model="ar1",
                          standardize=False, signal_scaling=0,
                          smoothing_fwhm=smoothing_fwhm, minimize_memory=True)
    glm.fit(bold_runs, events=design)

    # Build one contrast vector per run from that run's own design matrix columns
    def _con(dm, expr):
        """Contrast vector for one run, from that run's own design-matrix columns."""
        cols = list(dm.columns)
        sides = [[t.strip() for t in side.split("+")] for side in expr.split("-")]
        c = np.zeros(dm.shape[1])
        for sign, terms in zip((1, -1), sides):
            for term in terms:
                c[cols.index(term)] += sign / len(terms)
        return c

    # z is what the localizer thresholds; t is kept for the flatmap figures
    maps = {"t": {}, "z": {}}
    for name, expr in CONTRASTS.items():
        cons = [_con(dm, expr) for dm in glm.design_matrices_]
        maps["t"][name] = glm.compute_contrast(cons, stat_type="t", output_type="stat")
        maps["z"][name] = glm.compute_contrast(cons, stat_type="t", output_type="z_score")
    print(f"  GLM: {len(bold_runs)} runs, TR {t_r} s, smoothing {smoothing_fwhm} mm")
    return maps

def save_maps(maps: dict, out_dir: Path):
    """Write each contrast map as floc_<contrast>_<t|z>.nii.gz."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, group in maps.items():
        for name, img in group.items():
            img.to_filename(out_dir / f"floc_{name.replace('>', '_gt_')}_{kind}.nii.gz")
    print(f"  maps -> {out_dir}")

def load_maps(subj: str, proj_dir: Path=PROJECT_ROOT, kind: str="z"):
    """Load saved contrast maps."""
    d = map_dir(subj, proj_dir)
    out = {}
    for name in CONTRASTS:
        p = d / f"floc_{name.replace('>', '_gt_')}_{kind}.nii.gz"
        assert p.exists(), f"{p} not found -- run run_floc_glm first"
        out[name] = nib.load(p)
    return out

def maps_to_arrays(sl: SubjectLoader, maps: dict):
    """Contrast maps -> per-brain-voxel arrays."""
    brain_mask, _ = sl.load_brain_mask()
    _, shape = sl.load_volume_info()
    arrs = {}
    for k, img in maps.items():
        assert img.shape[:3] == shape, \
            f"{k}: map shape {img.shape[:3]} != brainmask {shape}"
        arrs[k] = np.nan_to_num(np.asarray(img.get_fdata()).flatten()[brain_mask])
    return arrs


# --------------------------------- Run GLM ---------------------------------- #
def run_floc_glm(subjects: list, proj_dir: Path=PROJECT_ROOT,
                 runs: tuple=FLOC_RUNS, smoothing_fwhm: float=SMOOTHING_FWHM):
    """Fit fLoc GLM for each subject and save contrast maps."""
    for subj in subjects:
        print(f"\n=== {subj}: fLoc GLM ===")
        sl = SubjectLoader(subj, proj_dir)
        bold_runs, events_files = load_floc_runs(subj, proj_dir, runs)
        mask_img = nib.load(sl.roi_mask_dir / "brainmask.nii.gz")

        maps = fit_floc_glm(bold_runs, events_files, mask_img,
                            smoothing_fwhm=smoothing_fwhm)
        save_maps(maps, map_dir(subj, proj_dir))


# ------------------------------ Define streams ------------------------------ #
def build_stream_mask(sl: SubjectLoader, streams: tuple=STREAMS,
                      nc_thresh: float=0.0, verbose: bool=True):
    """Voxel mask from NSD's streams atlas."""
    brain_mask = sl.load_brain_mask()[0]
    stream_vol = np.asarray(
        nib.load(sl.roi_mask_dir / "streams.nii.gz").get_fdata()).flatten()[brain_mask]

    labels = pd.read_csv(sl.proj_dir / "nsd" / "nsd_info" / "roi_labels" / "streams.mgz.ctab",
                         sep=r"\s+", header=None, names=["index", "label"], engine="python")
    missing = [s for s in streams if s not in set(labels["label"])]
    assert not missing, f"streams {missing} not in streams.mgz.ctab"
    ids = [int(labels.loc[labels["label"] == s, "index"].iloc[0]) for s in streams]

    mask = np.isin(stream_vol, ids)
    if nc_thresh > 0:
        mask = mask & (sl.load_nc() > nc_thresh)

    if verbose:
        print(f"{sl.subject}: {int(mask.sum())} voxels "
              f"{ {s: int((stream_vol == i).sum()) for s, i in zip(streams, ids)} }"
              + (f" nc>{nc_thresh}" if nc_thresh else ""))
    return mask, np.where(mask)[0]


# ----------------------------- Localize voxels ------------------------------ #
def _neg_key(domain: str):
    """'places' -> 'place>object', the contrast name used in the maps."""
    return f"{domain[:-1] if domain.endswith('s') else domain}>object"

def contrast_pvals(zmaps: dict, mask: np.ndarray, fdr_method: str=FDR_METHOD):
    """One-sided p-values per contrast, FDR-corrected."""
    from statsmodels.stats.multitest import multipletests

    p, raw = {}, {}
    for name, z in {**zmaps, "body>face": -zmaps["face>body"]}.items():
        pr = np.ones(mask.size)
        pr[mask] = norm.sf(np.asarray(z)[mask])
        pc = pr.copy()
        if fdr_method:
            pc[mask] = multipletests(pr[mask], method=fdr_method)[1]
        p[name], raw[name] = pc, pr
    return p, raw

def localize_voxels(mask: np.ndarray, zmaps: dict, p_pos: float=P_POS,
                    p_neg: float=P_NEG, neg_domains: tuple=("places",),
                    fdr_method: str=FDR_METHOD, verbose: bool=True):
    """Kliger & Yovel (2024) classification from fLoc contrast z-maps."""
    p, raw = contrast_pvals(zmaps, mask, fdr_method)

    face_up, body_up = p["face>object"] < p_pos, p["body>object"] < p_pos
    face_gt_body, body_gt_face = p["face>body"] < p_pos, p["body>face"] < p_pos
    no_body, no_face = p["body>object"] >= p_neg, p["face>object"] >= p_neg
    no_other = np.ones_like(face_up)
    for domain in neg_domains:
        no_other &= p[_neg_key(domain)] >= p_neg

    masks = {
        "face": mask & face_up & face_gt_body & no_body & no_other,
        "body": mask & body_up & body_gt_face & no_face & no_other,
        "mixed": mask & face_up & body_up & no_other,
    }
    non = mask & ~face_up & ~body_up
    other = mask & ~(masks["face"] | masks["body"] | masks["mixed"] | non)

    vox_masks = {**masks, "non": non, "other": other}
    if VISUAL in p:
        vox_masks["responsive"] = mask & (p[VISUAL] < p_pos)

    cutoff = {k: (float(raw[k][p[k] < p_pos].max()) if (p[k] < p_pos).any() else 0.0)
              for k in ("face>object", "body>object")}

    res = {"vox_masks": vox_masks,
           "n_selective": {k: int(m.sum()) for k, m in masks.items()},
           "n_non": int(non.sum()), "n_other": int(other.sum()),
           "n_responsive": int(vox_masks["responsive"].sum()) if VISUAL in p else None,
           "n_total": int(mask.sum()),
           "p_pos": p_pos, "p_neg": p_neg, "fdr_method": fdr_method,
           "raw_p_cutoff": cutoff, "neg_domains": tuple(neg_domains)}

    if verbose:
        n, tot = res["n_selective"], max(res["n_total"], 1)
        print(f"  {f'FDR-{fdr_method}' if fdr_method else 'uncorrected'} "
              f"p<{p_pos:g} / p>{p_neg:g} (raw p < "
              + ", ".join(f"{v:.1e}" for v in cutoff.values()) + "): "
              + ", ".join(f"{k}={n[k]} ({100 * n[k] / tot:.1f}%)" for k in SEL_TYPES) +
              f", non={res['n_non']}, other={res['n_other']} / {tot}"
              + (f"\n    visually responsive: {res['n_responsive']} "
                 f"({100 * res['n_responsive'] / tot:.1f}%)" if VISUAL in p else
                 f"\n    note: no '{VISUAL}' map -- rerun run_fmri_glm.py to enable "
                 f"the responsive-voxel denominator"))
    return res


# --------------------------- Selective voxel ROIs --------------------------- #
def load_sel_masks(subj: str, proj_dir: Path=PROJECT_ROOT, order: str="fit",
                   responsive: bool=False, vox_types: tuple=LABEL_COLS):
    """
    Voxel-selectivity masks from voxel meta data, either in brain-mask order or in
    fit order of voxel-wise encoding results.
    """
    assert order in ("brain", "fit"), f"order must be 'brain' or 'fit', got '{order}'"

    sl = SubjectLoader(subj, proj_dir)
    path = sl.resp_dir / "vox_meta.csv"
    cols = list(vox_types) + (["responsive"] if responsive else [])
    missing = [c for c in cols if c not in pd.read_csv(path, nrows=0).columns]
    assert not missing, (f"{subj}: vox_meta.csv is missing {missing} -- rerun "
                         f"run_fmri_selectivity.py (and run_fmri_glm.py first if "
                         f"'responsive' is missing, for the visual>baseline map)")

    vox_meta = pd.read_csv(path, usecols=cols)
    masks = {c: vox_meta[c].to_numpy() != 0 for c in vox_types}
    if responsive:
        resp = vox_meta["responsive"].to_numpy() != 0
        masks = {k: m & resp for k, m in masks.items()}

    if order == "fit":
        vox_ids = np.load(sl.resp_dir / "streams_meta.npz",
                          allow_pickle=True)["vox_ids"]
        masks = {k: m[vox_ids] for k, m in masks.items()}
    return masks

def write_rois(sl: SubjectLoader, mask: np.ndarray, res: dict, verbose: bool=True):
    """Write voxel labels into subject's voxel metadata."""
    n_brain = int(sl.load_brain_mask()[0].sum())
    assert len(mask) == n_brain, f"mask length {len(mask)} != {n_brain} brain voxels"

    path = sl.resp_dir / "vox_meta.csv"
    if path.exists():
        vox_meta = pd.read_csv(path)
        assert len(vox_meta) == n_brain, \
            f"{path} has {len(vox_meta)} rows, expected {n_brain} -- wrong subject?"
    else:
        vox_meta = pd.DataFrame(index=np.arange(n_brain))

    cols = {k: np.asarray(v, bool) for k, v in res["vox_masks"].items()
            if k in LABEL_COLS or k == "responsive"}
    for name, m in cols.items():
        vox_meta[name] = m.astype(int)
    vox_meta.to_csv(path, index=False)

    if verbose:
        print(f"  vox_meta -> {path}\n    " +
              ", ".join(f"{n}={int(m.sum())}" for n, m in cols.items()) +
              f" / {int(mask.sum())} searched")
    return vox_meta


# --------------------------- Run fMRI selectivity --------------------------- #
def run_fmri_selectivity(subjects: list, proj_dir: Path=PROJECT_ROOT,
                         streams: tuple=STREAMS, nc_thresh: float=0.0,
                         p_pos: float=P_POS, p_neg: float=P_NEG,
                         neg_domains: tuple=("places",), fdr_method: str=FDR_METHOD):
    """Localize selective voxels for each subject from saved fLoc contrast maps."""
    out = {}
    for subj in subjects:
        print(f"\n=== {subj}: cortical selectivity ===")
        sl = SubjectLoader(subj, proj_dir)
        mask, vox_ids = build_stream_mask(sl, streams, nc_thresh)

        zmaps = maps_to_arrays(sl, load_maps(subj, proj_dir, "z"))
        res = localize_voxels(mask, zmaps, p_pos, p_neg, neg_domains, fdr_method)
        write_rois(sl, mask, res)

        out[subj] = {"mask": mask, "vox_ids": vox_ids, "res": res, "streams": streams}

    save_pickle(out, Path(proj_dir) / "selectivity" / "cortex" / "floc.pkl")

    totals = {k: sum(o["res"]["n_selective"][k] for o in out.values()) for k in SEL_TYPES}
    print(f"\nAcross {len(out)} subjects: "
          + ", ".join(f"{k}={v}" for k, v in totals.items()))
    return out
