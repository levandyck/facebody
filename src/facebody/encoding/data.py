from dataclasses import dataclass
from pathlib import Path

from facebody.config import DATA_ROOT, PROJECT_ROOT
from myutils.utils import load_pickle, save_pickle

# ---------------------------- Paths and File I/O ---------------------------- #
@dataclass(frozen=True)
class NSDRepository:
    data_root: Path = DATA_ROOT
    project_root: Path = PROJECT_ROOT

    def fmri_path(self):
        return self.project_root / "nsd" / "fmri_activs.pkl"

    def img_ids_path(self):
        return self.project_root / "nsd" / "img_ids.pkl"

    def nsd_stimuli_h5(self):
        return self.data_root / "datasets" / "nsd" / "nsd_stimuli.hdf5"

    def activs_cache_dir(self):
        return self.project_root / "features" / "activs_cache"

    def encoding_dir(self, model_name: str):
        return self.project_root / "encoding" / model_name

    def nmf_cache_path(self, model_name: str):
        return self.encoding_dir(model_name) / "nmf_models.pkl"

    def alpha_cache_path(self, model_name: str):
        return self.encoding_dir(model_name) / "ridge_alphas.pkl"

    def results_path(self, model_name: str, analysis: str):
        return self.encoding_dir(model_name) / f"{analysis}.pkl"

    def load(self, path: Path):
        return load_pickle(path)

    def save(self, obj, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(obj, path)

    def merge_save(self, obj: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = load_pickle(path)
            old.update(obj)
            save_pickle(old, path)
        else:
            save_pickle(obj, path)
