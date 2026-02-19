from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import NMF

from facebody.selectivity import filter_sel_activs

# ----------------------- NMF dimensionality reduction ----------------------- #
@dataclass(frozen=True)
class NMFConfig:
    n_components: int = 100
    max_iter: int = 500
    random_state: int = 0

class NMFStore:
    """Store NMF predictors."""
    def __init__(self, cfg: NMFConfig, cache: dict=None):
        self.cfg = cfg
        self.cache = {} if cache is None else dict(cache)

    def fit_if_needed(self, key: tuple, X: np.ndarray, force: bool=False):
        if (key in self.cache) and (not force):
            return

        if X.shape[1] == 0:
            self.cache[key] = None
            return

        n_comp = min(self.cfg.n_components, X.shape[0], X.shape[1])
        if n_comp < 1:
            self.cache[key] = None
            return

        nmf = NMF(
            n_components=int(n_comp),
            init="nndsvda",
            max_iter=int(self.cfg.max_iter),
            random_state=int(self.cfg.random_state),
        )
        nmf.fit(X)
        self.cache[key] = nmf

    def transform(self, key: tuple, X: np.ndarray, n_rows: int):
        if X.shape[1] == 0:
            return X
        model = self.cache.get(key)
        if model is None:
            return np.zeros((n_rows, 0), dtype=float)
        return model.transform(X)


# --------------------------- Unit type predictors --------------------------- #
@dataclass(frozen=True)
class BlockConfig:
    model_name: str
    controlled: bool

class BlockBuilder:
    """Builds predictor blocks (f/b/m/ns/fb/fbm) for a (subject, layer)."""
    UNIT_TYPES = ("face", "body", "mixed", "nonselective")
    def __init__(self, cfg: BlockConfig, nmf_store: NMFStore):
        self.cfg = cfg
        self.nmf_store = nmf_store

    def fit_nmf_subject_layer(self, subj: str, layer: str,
                              activs_by_layer: dict, force: bool=False):
        if self.nmf_store is None:
            return

        sel = filter_sel_activs(
            self.cfg.model_name, activs_by_layer, layer,
            controlled=self.cfg.controlled
        )
        for unit_type in self.UNIT_TYPES:
            key = (subj, layer, unit_type)
            X = sel.get(unit_type, np.zeros((activs_by_layer[layer].shape[0], 0)))
            self.nmf_store.fit_if_needed(key, X, force=force)

    def blocks_tuple(self, subj: str, layer: str, activs_by_layer: dict):
        sel = filter_sel_activs(
            self.cfg.model_name, activs_by_layer, layer,
            controlled=self.cfg.controlled
        )
        n = activs_by_layer[layer].shape[0]

        blocks = []
        for unit_type in self.UNIT_TYPES:
            X = sel.get(unit_type, np.zeros((n, 0)))
            if self.nmf_store is None:
                blocks.append(X)
            else:
                key = (subj, layer, unit_type)
                blocks.append(self.nmf_store.transform(key, X, n_rows=n))
        return tuple(blocks)

    def block_dict(self, subj: str, layer: str, activs_by_layer: dict):
        Xf, Xb, Xm, Xns = self.blocks_tuple(subj, layer, activs_by_layer)
        n = Xf.shape[0]
        empty = np.empty((n, 0))

        fb = np.hstack([Xf, Xb]) if (Xf.shape[1] + Xb.shape[1]) > 0 else empty
        fbm = np.hstack([Xf, Xb, Xm]) if (Xf.shape[1] + Xb.shape[1] + Xm.shape[1]) > 0 else empty

        return {"f": Xf, "b": Xb, "m": Xm, "ns": Xns, "fb": fb, "fbm": fbm}
