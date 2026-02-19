from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from .ridgecv import RidgeCVConfig, RidgeCVEngine, scale_train_test
from .types import VarPartResult, DeltaMResult

# ------------------------------- Separate fits ------------------------------ #
@dataclass(frozen=True)
class SeparateFitsAnalysis:
    def run_one(self, X: np.ndarray, y: np.ndarray, folds, alpha: float, use_positive: bool):
        engine = RidgeCVEngine(RidgeCVConfig(use_positive=use_positive))
        return engine.cv_r2(X, y, folds, alpha, use_positive=use_positive)


# --------------------------- Variance partitioning -------------------------- #
@dataclass(frozen=True)
class FaceBodyVarPartAnalysis:
    def run_one(self, Xf: np.ndarray, Xb: np.ndarray, y: np.ndarray, folds,
                alpha_fb: float, use_positive: bool):
        ridge = Ridge(alpha=float(alpha_fb), positive=bool(use_positive), fit_intercept=True)

        y_true = np.zeros_like(y)
        yhat_f = np.zeros_like(y)
        yhat_b = np.zeros_like(y)
        yhat_fb = np.zeros_like(y)

        nF = Xf.shape[1]

        for tr, te in folds:
            X_tr_fb = np.hstack([Xf[tr], Xb[tr]])
            X_te_fb = np.hstack([Xf[te], Xb[te]])
            X_tr_fb_z, X_te_fb_z = scale_train_test(X_tr_fb, X_te_fb)

            X_tr_f, X_tr_b = X_tr_fb_z[:, :nF], X_tr_fb_z[:, nF:]
            X_te_f, X_te_b = X_te_fb_z[:, :nF], X_te_fb_z[:, nF:]

            if nF > 0:
                ridge.fit(X_tr_f.astype(np.float64), y[tr])
                yhat_f[te] = ridge.predict(X_te_f.astype(np.float64))
            else:
                yhat_f[te] = np.repeat(np.mean(y[tr], axis=0, keepdims=True), len(te), axis=0)

            if X_tr_b.shape[1] > 0:
                ridge.fit(X_tr_b.astype(np.float64), y[tr])
                yhat_b[te] = ridge.predict(X_te_b.astype(np.float64))
            else:
                yhat_b[te] = np.repeat(np.mean(y[tr], axis=0, keepdims=True), len(te), axis=0)

            ridge.fit(X_tr_fb_z.astype(np.float64), y[tr])
            yhat_fb[te] = ridge.predict(X_te_fb_z.astype(np.float64))
            y_true[te] = y[te]

        r2_f = r2_score(y_true, yhat_f, multioutput="raw_values")
        r2_b = r2_score(y_true, yhat_b, multioutput="raw_values")
        r2_fb = r2_score(y_true, yhat_fb, multioutput="raw_values")

        return VarPartResult(
            r2_f=r2_f,
            r2_b=r2_b,
            r2_fb=r2_fb,
            u_f=r2_fb - r2_b,
            u_b=r2_fb - r2_f,
            s_fb=r2_f + r2_b - r2_fb,
        )

@dataclass(frozen=True)
class DeltaMAnalysis:
    def run_one(self, Xf: np.ndarray, Xb: np.ndarray, Xm: np.ndarray, y: np.ndarray, folds,
                alpha_fbm: float, use_positive: bool):
        nFB = Xf.shape[1] + Xb.shape[1]
        n_all = nFB + Xm.shape[1]

        if n_all == 0:
            z = np.zeros(y.shape[1])
            return DeltaMResult(r2_fb=z, r2_fbm=z, delta_m=z)

        ridge = Ridge(alpha=float(alpha_fbm), positive=bool(use_positive), fit_intercept=True)

        y_true = np.zeros_like(y)
        yhat_fb = np.zeros_like(y)
        yhat_fbm = np.zeros_like(y)

        for tr, te in folds:
            X_tr_all = np.hstack([Xf[tr], Xb[tr], Xm[tr]])
            X_te_all = np.hstack([Xf[te], Xb[te], Xm[te]])
            X_tr_all_z, X_te_all_z = scale_train_test(X_tr_all, X_te_all)

            X_tr_fb_z = X_tr_all_z[:, :nFB] if nFB else np.empty((len(tr), 0))
            X_te_fb_z = X_te_all_z[:, :nFB] if nFB else np.empty((len(te), 0))

            if nFB > 0:
                ridge.fit(X_tr_fb_z.astype(np.float64), y[tr])
                yhat_fb[te] = ridge.predict(X_te_fb_z.astype(np.float64))
            else:
                yhat_fb[te] = np.repeat(np.mean(y[tr], axis=0, keepdims=True), len(te), axis=0)

            ridge.fit(X_tr_all_z.astype(np.float64), y[tr])
            yhat_fbm[te] = ridge.predict(X_te_all_z.astype(np.float64))
            y_true[te] = y[te]

        r2_fb = r2_score(y_true, yhat_fb, multioutput="raw_values")
        r2_fbm = r2_score(y_true, yhat_fbm, multioutput="raw_values")
        return DeltaMResult(r2_fb=r2_fb, r2_fbm=r2_fbm, delta_m=r2_fbm - r2_fb)
