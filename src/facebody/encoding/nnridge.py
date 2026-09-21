"""Nested-CV non-negative ridge with a per-voxel alpha, fit to all voxels at once on GPU."""

from dataclasses import dataclass
import numpy as np
import torch
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

EPS = 1e-10

# ---------------------------------- Config ---------------------------------- #
@dataclass(frozen=True)
class VoxelRidgeConfig:
    """Nested-CV, solver, and precision settings for voxel-wise ridge."""
    k_outer: int = 10
    k_inner: int = 5
    alpha_grid: tuple = tuple(np.logspace(-2, 7, 10))
    max_sweeps: int = 200
    tol: float = 1e-7
    tol_tune: float = 1e-5
    vox_chunk: int = 32768
    device: str = "cuda"
    gram_dtype: str = "float32"
    solve_dtype: str = "float64"
    tune_dtype: str = "float32"
    random_state: int = 0

@dataclass
class VoxelFitResult:
    """Out-of-fold scores for one subject, layer, and block."""
    r2: dict
    alpha: np.ndarray


# -------------------------------- Statistics -------------------------------- #
@dataclass
class CellStats:
    """Per-cell sufficient statistics, so any train set is a sum over cells."""
    G: torch.Tensor
    C: torch.Tensor
    sx: torch.Tensor
    sy: torch.Tensor
    n: torch.Tensor

def cell_stats(X: torch.Tensor, Y: torch.Tensor, cell: np.ndarray, n_cell: int):
    """One pass over the data -> G and C per CV cell."""
    p, n_vox = X.shape[1], Y.shape[1]
    kw = dict(device=X.device, dtype=X.dtype)
    G = torch.zeros(n_cell, p, p, **kw)
    C = torch.zeros(n_cell, p, n_vox, **kw)
    sx = torch.zeros(n_cell, p, **kw)
    sy = torch.zeros(n_cell, n_vox, **kw)
    n = torch.zeros(n_cell, **kw)

    for k in range(n_cell):
        rows = torch.as_tensor(np.where(cell == k)[0], device=X.device)
        Xk, Yk = X[rows], Y[rows]
        G[k] = Xk.T @ Xk
        C[k] = Xk.T @ Yk
        sx[k] = Xk.sum(0)
        sy[k] = Yk.sum(0)
        n[k] = rows.numel()

    return CellStats(G=G, C=C, sx=sx, sy=sy, n=n)

def pool_cells(stats: CellStats, cells: list, dtype: torch.dtype):
    """Sum cells -> centred, train-std-scaled (G, C) plus stats needed to predict."""
    idx = torch.as_tensor(cells, device=stats.G.device)
    n = stats.n[idx].sum().to(dtype)
    G = stats.G[idx].sum(0).to(dtype)
    C = stats.C[idx].sum(0).to(dtype)
    xbar = stats.sx[idx].sum(0).to(dtype) / n
    ybar = stats.sy[idx].sum(0).to(dtype) / n

    G = G - n * torch.outer(xbar, xbar)
    C = C - n * xbar[:, None] * ybar[None, :]

    s = torch.sqrt(torch.clamp(torch.diagonal(G), min=0.0) / n)
    s = torch.where(s < EPS, torch.ones_like(s), s)

    inv = 1.0 / s
    return inv[:, None] * G * inv[None, :], inv[:, None] * C, s, xbar, ybar


# ---------------------------- Coordinate descent ---------------------------- #
def solve_nnridge(G: torch.Tensor, C: torch.Tensor, alpha, W: torch.Tensor,
                  max_sweeps: int=200, tol: float=1e-7):
    """Projected coordinate descent for min_{w>=0} w'(G + alpha*I)w - 2c'w, per column."""
    p = G.shape[0]
    if p == 0:
        return W

    tol = max(tol, 8 * float(torch.finfo(W.dtype).eps))

    GW = G @ W
    diag = torch.diagonal(G)
    W_prev = torch.empty_like(W)

    for _ in range(max_sweeps):
        W_prev.copy_(W)
        for j in range(p):
            wj = torch.clamp(
                W[j] + (C[j] - GW[j] - alpha * W[j]) / (diag[j] + alpha), min=0.0)
            delta = wj - W[j]
            W[j] = wj
            GW.addr_(G[:, j], delta)

        W_prev.sub_(W).abs_()
        if float(W_prev.max()) <= tol * max(1.0, float(W.abs().max())):
            break

    return W


# ----------------------------- Voxel-wise ridge ----------------------------- #
class VoxelwiseRidge:
    """Nested-CV non-negative ridge with per-voxel alpha, fit to all voxels at once."""
    def __init__(self, cfg: VoxelRidgeConfig):
        self.cfg = cfg
        grid = np.asarray(cfg.alpha_grid, dtype=float)
        self.alphas = grid[np.argsort(-grid)]
        self.device = torch.device(
            cfg.device if (torch.cuda.is_available() or "cuda" not in cfg.device) else "cpu"
        )
        self.gdtype = getattr(torch, cfg.gram_dtype)
        self.sdtype = getattr(torch, cfg.solve_dtype)
        self.tdtype = getattr(torch, cfg.tune_dtype)

    # ---------------------------------- Folds ---------------------------------- #
    def cells(self, n_samples: int):
        """Partition trials into k_outer x k_inner cells, so every train set is a sum of cells."""
        ids = np.arange(n_samples)
        outer = KFold(self.cfg.k_outer, shuffle=True, random_state=self.cfg.random_state)
        inner = KFold(self.cfg.k_inner, shuffle=True, random_state=self.cfg.random_state)

        cell = np.empty(n_samples, dtype=int)
        for o, (_, te) in enumerate(outer.split(ids)):
            for i, (_, sub) in enumerate(inner.split(te)):
                cell[te[sub]] = o * self.cfg.k_inner + i
        return cell

    def _cells_where(self, keep):
        """Cell ids whose (outer, inner) fold indices satisfy `keep`."""
        k_in = self.cfg.k_inner
        return [c for c in range(self.cfg.k_outer * k_in) if keep(c // k_in, c % k_in)]

    # -------------------------------- Prediction -------------------------------- #
    def _ss_res(self, X, Y, rows, W, s, xbar, ybar):
        """Residual sum of squares of the ridge prediction on rows."""
        if W.shape[0] == 0:
            return ((Y[rows] - ybar) ** 2).sum(0)
        Xs = (X[rows] - xbar) / s
        return ((Y[rows] - (Xs @ W + ybar)) ** 2).sum(0)

    def _alpha_col(self, n_c, dtype):
        """One alpha per column of the stacked (alpha x voxel) solve."""
        return torch.as_tensor(self.alphas, device=self.device,
                               dtype=dtype).repeat_interleave(n_c)

    def _solve_grid(self, G, C, n_c, W=None):
        """Solve every alpha in grid at once: returns flat (p, n_alpha * n_c)."""
        A, p = len(self.alphas), G.shape[0]
        if W is None or W.shape != (p, A * n_c) or W.dtype != G.dtype:
            W = torch.zeros(p, A * n_c, device=self.device, dtype=G.dtype)
        return solve_nnridge(G, C.repeat(1, A), self._alpha_col(n_c, G.dtype), W,
                             self.cfg.max_sweeps, self.cfg.tol_tune)

    # ----------------------------------- Fit ------------------------------------ #
    def fit(self, X: np.ndarray, Y: np.ndarray, sub_slices: dict=None,
            full_key: str="full", desc: str=None):
        """Nested-CV fit of X -> Y for every voxel in Y."""
        subs = dict(sub_slices) if sub_slices else {}
        subs.setdefault(full_key, slice(None))

        n, p = X.shape
        n_vox = Y.shape[1]
        n_cell = self.cfg.k_outer * self.cfg.k_inner
        cell = self.cells(n)
        rows_of = lambda cs: torch.as_tensor(np.where(np.isin(cell, cs))[0], device=self.device)

        Xd = torch.as_tensor(np.ascontiguousarray(X), device=self.device, dtype=self.gdtype)
        alphas = torch.as_tensor(self.alphas, device=self.device, dtype=self.sdtype)

        ss_res = {name: np.zeros(n_vox) for name in subs}
        alpha_folds = np.zeros((self.cfg.k_outer, n_vox))

        bar = tqdm(total=int(np.ceil(n_vox / self.cfg.vox_chunk)) * self.cfg.k_outer,
                   desc=desc, unit="fold", leave=False, disable=desc is None)
        bar.set_postfix(vox=n_vox, p=p)

        for v0 in range(0, n_vox, self.cfg.vox_chunk):
            v1 = min(v0 + self.cfg.vox_chunk, n_vox)
            Yd = torch.as_tensor(np.ascontiguousarray(Y[:, v0:v1]),
                                 device=self.device, dtype=self.gdtype)
            stats = cell_stats(Xd, Yd, cell, n_cell)
            n_c = v1 - v0
            W_warm = None  # carried across folds: same shape, and the optimum is unique

            for o in range(self.cfg.k_outer):
                # Tune: pooled inner-validation SS_res per alpha. SS_tot is identical
                # across alphas, so the best alpha is simply the one with the least error.
                ss_alpha = torch.zeros(len(self.alphas), n_c,
                                       device=self.device, dtype=self.sdtype)
                for i in range(self.cfg.k_inner):
                    G, C, s, xbar, ybar = pool_cells(
                        stats, self._cells_where(lambda a, b: a != o and b != i), self.sdtype)
                    rows = rows_of(self._cells_where(lambda a, b: a != o and b == i))

                    W_warm = self._solve_grid(G.to(self.tdtype), C.to(self.tdtype),
                                              n_c, W_warm)
                    W3 = W_warm.view(p, len(self.alphas), n_c)
                    for ai in range(len(self.alphas)):
                        ss_alpha[ai] += self._ss_res(
                            Xd, Yd, rows, W3[:, ai].to(self.gdtype), s.to(self.gdtype),
                            xbar.to(self.gdtype), ybar.to(self.gdtype)).to(self.sdtype)

                best = ss_alpha.argmin(0)
                alpha_star = alphas[best]
                alpha_folds[o, v0:v1] = alpha_star.cpu().numpy()

                # Refit on the outer-train set at each voxel's alpha, score held-out fold.
                # One solve per sub-model at the selected alpha -- no path to walk, since
                # the alphas are already chosen.
                G, C, s, xbar, ybar = pool_cells(
                    stats, self._cells_where(lambda a, _: a != o), self.sdtype)
                rows = rows_of(self._cells_where(lambda a, _: a == o))

                for name, sl in subs.items():
                    Gs, Cs = G[sl, sl], C[sl]
                    W_sel = solve_nnridge(
                        Gs, Cs, alpha_star,
                        torch.zeros(Gs.shape[0], n_c, device=self.device, dtype=self.sdtype),
                        self.cfg.max_sweeps, self.cfg.tol)
                    ss_res[name][v0:v1] += self._ss_res(
                        Xd[:, sl], Yd, rows, W_sel.to(self.gdtype), s[sl].to(self.gdtype),
                        xbar[sl].to(self.gdtype), ybar.to(self.gdtype)).cpu().numpy()

                bar.update(1)

            del stats, Yd

        bar.close()

        ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0, dtype=np.float64)
        r2 = {name: 1.0 - ss / np.maximum(ss_tot, EPS) for name, ss in ss_res.items()}
        alpha = np.exp(np.median(np.log(alpha_folds), axis=0))

        return VoxelFitResult(r2=r2, alpha=alpha)

    # -------------------------- Fit at a fixed alpha --------------------------- #
    def fit_fixed_alpha(self, X: np.ndarray, Y: np.ndarray, alpha: np.ndarray,
                        sub_slices: dict=None, full_key: str="full"):
        """Out-of-fold R2 at a given per-voxel alpha, skipping the inner CV."""
        subs = dict(sub_slices) if sub_slices else {}
        subs.setdefault(full_key, slice(None))

        n, p = X.shape
        n_vox = Y.shape[1]
        n_cell = self.cfg.k_outer * self.cfg.k_inner
        cell = self.cells(n)
        rows_of = lambda cs: torch.as_tensor(np.where(np.isin(cell, cs))[0], device=self.device)

        Xd = torch.as_tensor(np.ascontiguousarray(X), device=self.device, dtype=self.gdtype)
        ss_res = {name: np.zeros(n_vox) for name in subs}

        for v0 in range(0, n_vox, self.cfg.vox_chunk):
            v1 = min(v0 + self.cfg.vox_chunk, n_vox)
            Yd = torch.as_tensor(np.ascontiguousarray(Y[:, v0:v1]),
                                 device=self.device, dtype=self.gdtype)
            stats = cell_stats(Xd, Yd, cell, n_cell)
            a = torch.as_tensor(np.ascontiguousarray(alpha[v0:v1]),
                                device=self.device, dtype=self.sdtype)

            for o in range(self.cfg.k_outer):
                G, C, s, xbar, ybar = pool_cells(
                    stats, self._cells_where(lambda x, _: x != o), self.sdtype)
                rows = rows_of(self._cells_where(lambda x, _: x == o))

                for name, sl in subs.items():
                    Gs, Cs = G[sl, sl], C[sl]
                    W = torch.zeros(Gs.shape[0], v1 - v0,
                                    device=self.device, dtype=self.sdtype)
                    W = solve_nnridge(Gs, Cs, a, W, self.cfg.max_sweeps, self.cfg.tol)
                    ss_res[name][v0:v1] += self._ss_res(
                        Xd[:, sl], Yd, rows, W.to(self.gdtype), s[sl].to(self.gdtype),
                        xbar[sl].to(self.gdtype), ybar.to(self.gdtype)).cpu().numpy()

            del stats, Yd

        ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0, dtype=np.float64)
        return {name: 1.0 - ss / np.maximum(ss_tot, EPS) for name, ss in ss_res.items()}
