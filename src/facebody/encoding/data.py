"""Every path the encoding pipeline reads or writes."""

from dataclasses import dataclass
from pathlib import Path

from facebody.config import DATA_ROOT, PROJECT_ROOT
from facebody.myutils.utils import load_pickle, save_pickle

# ---------------------------- Paths and File I/O ---------------------------- #
@dataclass(frozen=True)
class NSDRepository:
    """Every path the encoding pipeline reads or writes."""
    data_root: Path = DATA_ROOT
    project_root: Path = PROJECT_ROOT

    def img_ids_path(self):
        return self.project_root / "nsd" / "img_ids.pkl"

    def nsd_stimuli_h5(self):
        return self.data_root / "datasets" / "nsd" / "nsd_stimuli.hdf5"

    def activs_cache_dir(self, model_name: str):
        return self.project_root / "features" / "activs_cache" / model_name

    def encoding_dir(self, model_name: str):
        return self.project_root / "encoding" / model_name

    def nmf_cache_path(self, model_name: str):
        return self.encoding_dir(model_name) / "nmf_models.pkl"

    def voxelwise_path(self, model_name: str, layer: str):
        return self.encoding_dir(model_name) / "voxelwise" / f"{layer}.h5"

    def load(self, path: Path):
        return load_pickle(path)

    def save(self, obj, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        save_pickle(obj, path)
