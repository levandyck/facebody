"""Turn each unit type's activations into NMF-reduced predictor block."""

from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import NMF

from facebody.dnn.floc import filter_sel_activs

# ------------------------- Dimensionality reduction ------------------------- #
@dataclass(frozen=True)
class NMFConfig:
    """NMF settings shared by every subject, layer, and unit type."""
    n_components: int = 100
    max_iter: int = 500
    random_state: int = 0
    fit_subsample: int = 0

def fit_nmf(cfg: NMFConfig, X: np.ndarray):
    """Fit NMF."""
    if X.shape[1] == 0:
        return None

    n_comp = min(cfg.n_components, X.shape[0], X.shape[1])
    if n_comp < 1:
        return None

    if cfg.fit_subsample and X.shape[0] > cfg.fit_subsample:
        rng = np.random.default_rng(cfg.random_state)
        X = X[np.sort(rng.choice(X.shape[0], cfg.fit_subsample, replace=False))]

    nmf = NMF(
        n_components=int(n_comp),
        init="nndsvda",
        max_iter=int(cfg.max_iter),
        random_state=int(cfg.random_state),
    )
    nmf.fit(X)
    return nmf

class NMFStore:
    """Store NMF predictors."""
    def __init__(self, cfg: NMFConfig, cache: dict=None):
        self.cfg = cfg
        self.cache = {} if cache is None else dict(cache)

    def transform(self, key: tuple, X: np.ndarray, n_rows: int):
        """Project X with the stored NMF for `key`; an empty block if there is none."""
        if X.shape[1] == 0:
            return X
        model = self.cache.get(key)
        if model is None:
            return np.zeros((n_rows, 0), dtype=float)
        return model.transform(X)


# --------------------------- Unit type predictors --------------------------- #
@dataclass(frozen=True)
class BlockConfig:
    """Which model the predictor blocks come from, and whether groups are size-matched."""
    model_name: str
    controlled: bool

class BlockBuilder:
    """Builds predictor blocks (f/b/m/ns/fb/fbm) for a (subject, layer)."""
    UNIT_TYPES = ("face", "body", "mixed", "nonselective")
    def __init__(self, cfg: BlockConfig, nmf_store: NMFStore):
        self.cfg = cfg
        self.nmf_store = nmf_store

    def blocks_tuple(self, subj: str, layer: str, activs_by_layer: dict):
        """NMF-projected predictors for face, body, mixed and non-selective units."""
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
