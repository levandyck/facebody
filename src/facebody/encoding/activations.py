"""Extract and cache DNN activations for NSD images of each subject."""

import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from facebody.myutils.models import ModelLoader
from facebody.myutils.feature_extractor import FeatureExtractor
from facebody.myutils.nsd import NSDImageLoader, _nsd_img_worker_init

# --------------------------- NSD unit activations --------------------------- #
class ActivationStore:
    """Cache activations for images per subject."""
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self._ram = {}

    def get(self, subj: str):
        """Cached activations for one subject; raises if neither in RAM nor on disk."""
        if subj in self._ram:
            return self._ram[subj]

        path = self.cache_dir / f"{subj}_activs.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self._ram[subj] = pkl.load(f)
            return self._ram[subj]

        raise RuntimeError(f"Activations for {subj} not found in RAM or disk: {path}")

    def put(self, subj: str, activs: dict, save_to_disk: bool=False):
        """Store one subject's activations in RAM, optionally also on disk."""
        self._ram[subj] = activs
        if save_to_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"{subj}_activs.pkl"
            with open(path, "wb") as f:
                pkl.dump(activs, f)

@dataclass(frozen=True)
class ActivationExtractConfig:
    """Batching and device settings for NSD activation pass."""
    layers: list
    batch_size: int = 128
    num_workers: int = 32
    prefetch_factor: int = 4
    persistent_workers: bool = True
    device: str = "cuda"

class NSDActivationExtractor:
    """Extract activations for images across subjects, then slice per subject."""
    def __init__(self, repo, model_name: str, cfg):
        self.repo = repo
        self.cfg = cfg
        self.model_loader = ModelLoader(model_name, repo.data_root, cfg.device)

    def extract_and_cache(self, subjects: list, img_ids_by_subj: dict,
                          act_store, pre_relu: bool=False,
                          save_to_disk: bool=False):
        """One forward pass over the union of images, then one slice per subject."""
        sorted_ids = np.unique(np.concatenate([img_ids_by_subj[s] for s in subjects]))

        ds = NSDImageLoader(self.repo.nsd_stimuli_h5(), sorted_ids, size=(224, 224))
        dl = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=self.cfg.persistent_workers,
            worker_init_fn=_nsd_img_worker_init,
        )

        fe = FeatureExtractor(self.model_loader, self.cfg.layers)
        with torch.no_grad():
            activs_all = fe.extract(dl, to_memory=True)[0]

        ds.close()

        pos_map = {int(img_id): i for i, img_id in enumerate(sorted_ids)}
        for subj in subjects:
            ids = img_ids_by_subj[subj]
            idx = [pos_map[int(x)] for x in ids]

            subj_activs: dict[str, np.ndarray] = {}
            for li, lay in enumerate(self.cfg.layers):
                act = activs_all[lay] if isinstance(activs_all, dict) else activs_all[li]
                if pre_relu:
                    act_subj = act[idx]
                else:
                    act_subj = np.maximum(act[idx], 0.0)
                subj_activs[lay] = act_subj

            act_store.put(subj, subj_activs, save_to_disk=save_to_disk)
