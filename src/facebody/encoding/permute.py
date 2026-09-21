"""Permutation null for encoding fit and max-statistic FWE threshold."""

import numpy as np
from tqdm.auto import tqdm

# ------------------------------ Permutation test ---------------------------- #
def permutation_null(ridge, X: np.ndarray, Y: np.ndarray, alpha: np.ndarray,
                     n_perm: int=1000, random_state: int=0, desc: str="perm"):
    """Null distribution of out-of-fold R2 under permuted image-response pairs."""
    rng = np.random.default_rng(random_state)
    r2_real = ridge.fit_fixed_alpha(X, Y, alpha)["full"]

    n_ge = np.zeros(Y.shape[1], dtype=int)
    null_max = np.empty(n_perm, dtype=np.float32)

    for i in tqdm(range(n_perm), desc=desc):
        perm = rng.permutation(Y.shape[0])
        r2_null = ridge.fit_fixed_alpha(X, Y[perm], alpha)["full"]
        n_ge += r2_null >= r2_real
        null_max[i] = r2_null.max()

    return {"r2": r2_real.astype(np.float32),
            "pperm": ((1 + n_ge) / (1 + n_perm)).astype(np.float32),
            "null_max": null_max}


# --------------------------------- Thresholds -------------------------------- #
def fwe_threshold(null_max: np.ndarray, alpha_fwe: float=0.05):
    """Max-statistic threshold on R2."""
    return float(np.percentile(np.asarray(null_max, float), 100 * (1 - alpha_fwe)))

def sig_mask(res: dict, alpha_fwe: float=0.05):
    """Voxels where DNN layer predicts responses above chance, FWE-corrected."""
    return np.asarray(res["r2"]) > fwe_threshold(res["null_max"], alpha_fwe)
