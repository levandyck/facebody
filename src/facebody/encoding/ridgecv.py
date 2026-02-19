import os
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

EPS = 1e-10

# --------------------- Cross-validated ridge regression --------------------- #
@dataclass(frozen=True)
class RidgeCVConfig:
    k_outer: int = 10
    k_inner: int = 5
    alpha_grid: tuple = tuple(10.0 ** np.arange(-1, 4))
    use_positive: bool = True
    random_state: int = 0

class RidgeCVEngine:
    def __init__(self, cfg: RidgeCVConfig):
        self.cfg = cfg

    @staticmethod
    def safe_alpha(alpha: float, default: float=1.0):
        return float(default if not np.isfinite(alpha) else alpha)

    def folds(self, n_samples: int):
        ids = np.arange(n_samples)
        return list(KFold(self.cfg.k_outer, shuffle=True, random_state=self.cfg.random_state).split(ids))

    def cv_preds(self, X: np.ndarray, y: np.ndarray,
                 folds: list, alpha: float, use_positive: bool=None):
        """Cross-validated predictions for X -> y."""
        use_positive = self.cfg.use_positive if use_positive is None else bool(use_positive)

        y_true = np.zeros_like(y)
        y_pred = np.zeros_like(y)

        if X.shape[1] == 0:
            for tr, te in folds:
                y_pred[te] = np.repeat(np.mean(y[tr], axis=0, keepdims=True), len(te), axis=0)
                y_true[te] = y[te]
            return y_true, y_pred

        ridge = Ridge(alpha=self.safe_alpha(alpha), positive=use_positive, fit_intercept=True)

        for tr, te in folds:
            X_tr, X_te = scale_train_test(X[tr], X[te])
            ridge.fit(X_tr.astype(np.float64), y[tr])
            y_pred[te] = ridge.predict(X_te.astype(np.float64))
            y_true[te] = y[te]

        return y_true, y_pred

    def cv_r2(self, X: np.ndarray, y: np.ndarray,
              folds: list, alpha: float,
              use_positive: bool=None,
    ):
        y_true, y_pred = self.cv_preds(X, y, folds, alpha, use_positive=use_positive)
        return r2_score(y_true, y_pred, multioutput="raw_values")

    def tune_alpha_nested(self, X: np.ndarray, y: np.ndarray,
                          outer_folds: list, use_positive: bool=None):
        """Nested CV tuning: for each outer-train split, pick best alpha via inner CV."""
        use_positive = self.cfg.use_positive if use_positive is None else bool(use_positive)

        if X.shape[1] == 0:
            return float("nan")

        inner = KFold(n_splits=self.cfg.k_inner, shuffle=True, random_state=self.cfg.random_state)
        alpha_grid = np.array(self.cfg.alpha_grid, dtype=float)

        chosen = []
        for tr_idx, _ in outer_folds:
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            means = np.empty(alpha_grid.size, dtype=float)

            for i, alpha in enumerate(alpha_grid):
                ridge = Ridge(alpha=float(alpha), positive=use_positive, fit_intercept=True)
                fold_scores = []
                for in_tr, in_va in inner.split(X_tr):
                    X_in_tr, X_in_va = scale_train_test(X_tr[in_tr], X_tr[in_va])
                    ridge.fit(X_in_tr.astype(np.float64), y_tr[in_tr])
                    y_hat = ridge.predict(X_in_va.astype(np.float64))
                    fold_scores.append(r2_score(y_tr[in_va], y_hat, multioutput="raw_values").mean())
                means[i] = float(np.mean(fold_scores))

            chosen.append(float(alpha_grid[int(np.argmax(means))]))

        chosen = np.array(chosen, dtype=float)
        chosen = chosen[np.isfinite(chosen) & (chosen > 0)]
        if chosen.size == 0:
            return float("nan")

        return float(np.exp(np.median(np.log(chosen))))


# ---------------------------------- Helpers --------------------------------- #
def set_single_thread_env():
    """Set environment variables for single-threaded BLAS/LAPACK (avoids oversubscription in multiprocessing)."""
    env = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)

def scale_train_test(X_tr: np.ndarray, X_te: np.ndarray):
    """Scale by train-only feature std (no centering)."""
    s = X_tr.std(axis=0, ddof=0)
    s[s < EPS] = 1.0
    return X_tr / s, X_te / s
