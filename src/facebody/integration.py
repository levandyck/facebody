from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from facebody.config import PROJECT_ROOT
from facebody.selectivity import filter_sel_activs
from facebody.plotting import SEL_INFO
from myutils.utils import save_pickle, load_pickle

# --------------------------- Integration analysis --------------------------- #
def _condition_masks(labels: np.ndarray, class_to_idx: dict):
    """Return boolean masks for None, Head, Body, Full."""
    idx_none = labels == class_to_idx["none"]
    idx_head = labels == class_to_idx["head"]
    idx_body = labels == class_to_idx["body"]
    idx_full = labels == class_to_idx["full"]

    if not (idx_none.any() and idx_head.any() and idx_body.any() and idx_full.any()):
        raise ValueError(
            "Missing one or more conditions (need None, Head, Body, Full). "
            f"Counts: none={idx_none.sum()}, head={idx_head.sum()}, "
            f"body={idx_body.sum()}, full={idx_full.sum()}"
        )

    return idx_none, idx_head, idx_body, idx_full

def _betas_from_condition_means(
    X: np.ndarray,
    idx_none: np.ndarray,
    idx_head: np.ndarray,
    idx_body: np.ndarray,
    idx_full: np.ndarray,
):
    """Compute contrast-style effects from condition means."""
    m_none = X[idx_none].mean(axis=0)
    m_head = X[idx_head].mean(axis=0)
    m_body = X[idx_body].mean(axis=0)
    m_full = X[idx_full].mean(axis=0)

    beta_F = m_head - m_none
    beta_B = m_body - m_none
    beta_FB = m_full - m_head - m_body + m_none

    return m_none, m_head, m_body, m_full, beta_F, beta_B, beta_FB

def compute_coefs(activs_layer: dict, labels: np.ndarray, class_to_idx: dict):
    """Compute normalized β_F, β_B, β_FB for all unit types in layer."""
    idx_none, idx_head, idx_body, idx_full = _condition_masks(labels, class_to_idx)
    out = {}

    for sel, X in activs_layer.items():
        if X is None or getattr(X, "size", 0) == 0:
            continue

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D activations, got shape {X.shape} for {sel}")

        m_none, m_head, m_body, m_full, beta_F, beta_B, beta_FB = _betas_from_condition_means(
            X, idx_none, idx_head, idx_body, idx_full
        )

        obs_net_full = m_full - m_none
        pred_net_headbody = beta_F + beta_B

        out[sel] = pd.DataFrame(
            {
                "unit": np.arange(X.shape[1], dtype=int),
                "m_none": m_none,
                "m_head": m_head,
                "m_body": m_body,
                "m_full": m_full,
                "beta_F": beta_F,
                "beta_B": beta_B,
                "beta_FB": beta_FB,
                "beta_FvsB": beta_F - beta_B,
                "obs_net_full": obs_net_full,
                "pred_net_headbody": pred_net_headbody,
            }
        )

    return out

def integration_analysis(model_name: str, layers: list, activs: dict,
                         labels: np.ndarray, class_to_idx: dict,
                         controlled: bool=False):
    """Run integration analysis per layer using condition means."""
    res_units: dict[str, dict[str, pd.DataFrame]] = {}
    all_betas = []

    for layer in layers:
        act_layer = filter_sel_activs(model_name, activs, layer, controlled=controlled)
        per_sel_dfs = compute_coefs(act_layer, labels, class_to_idx)
        res_units[layer] = per_sel_dfs

        for df in per_sel_dfs.values():
            all_betas.append(df["beta_F"].values)
            all_betas.append(df["beta_B"].values)
            all_betas.append(df["beta_FB"].values)

    all_betas = np.concatenate(all_betas)
    scale = float(all_betas.std(ddof=0))
    if scale == 0:
        scale = 1.0

    for per_sel_dfs in res_units.values():
        for df in per_sel_dfs.values():
            df["beta_F"] = df["beta_F"] / scale
            df["beta_B"] = df["beta_B"] / scale
            df["beta_FB"] = df["beta_FB"] / scale
            df["beta_FvsB"] = df["beta_FvsB"] / scale
            df["obs_net_full"] = df["obs_net_full"] / scale
            df["pred_net_headbody"] = df["pred_net_headbody"] / scale

    out_dir = Path(PROJECT_ROOT) / "fb_integration" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(res_units, out_dir / "fb_integration.pkl")


# ------------------------------- Bootstrapping ------------------------------ #
@dataclass(frozen=True)
class _CoefData:
    """Container for unit-level coefficient values used by bootstrap tests."""
    df_units: pd.DataFrame
    present_types: list[str]

class _CoefDataLoader:
    @staticmethod
    def load_units_single_layer(model_name: str, layer: str, coef: str):
        sel_types = list(SEL_INFO.keys())
        res = load_pickle(PROJECT_ROOT / "fb_integration" / model_name / "fb_integration.pkl")

        frames = []
        layer_dict = res.get(layer, {})
        for s in sel_types:
            df = layer_dict.get(s)
            if df is None or df.empty or coef not in df.columns:
                continue
            f = df[[coef]].copy()
            f["sel_type"] = s
            frames.append(f)

        if not frames:
            raise ValueError(f"No {coef} data found for layer {layer}.")

        df_units = pd.concat(frames, ignore_index=True)
        present_types = [s for s in sel_types if s in df_units["sel_type"].unique()]
        return _CoefData(df_units=df_units, present_types=present_types)

    @staticmethod
    def load_units_multi_layer(model_name: str, layers: list, coef: str):
        sel_types = list(SEL_INFO.keys())
        res = load_pickle(PROJECT_ROOT / "fb_integration" / model_name / "fb_integration.pkl")

        frames = []
        for lay in layers:
            layer_dict = res.get(lay, {})
            for s in sel_types:
                df = layer_dict.get(s)
                if df is None or df.empty or coef not in df.columns:
                    continue
                f = df[[coef]].copy()
                f["layer"] = lay
                f["sel_type"] = s
                frames.append(f)

        if not frames:
            raise ValueError(f"No {coef} data found.")

        df_units = pd.concat(frames, ignore_index=True)
        present_types = [s for s in sel_types if s in df_units["sel_type"].unique()]
        unique_layers = df_units["layer"].unique()
        return _CoefData(df_units=df_units, present_types=present_types), unique_layers

class _BootstrapEngine:
    @staticmethod
    def boot_means_within_type(
        df_units: pd.DataFrame,
        present_types: list,
        coef: str,
        n_boot: int,
        rng,
    ):
        """
        Bootstrap within layer:
          - resample units within each sel_type
          - pool all resampled units, then take mean per sel_type
        """
        vals_by_type = {
            sel: df_units.loc[df_units["sel_type"] == sel, coef].to_numpy()
            for sel in present_types
        }

        boot_means = {sel: np.zeros(n_boot) for sel in present_types}

        for b in range(n_boot):
            for sel in present_types:
                vals = vals_by_type[sel]
                if vals.size == 0:
                    boot_means[sel][b] = np.nan
                    continue
                boot_ids = rng.choice(vals.size, size=vals.size, replace=True)
                boot_means[sel][b] = float(np.mean(vals[boot_ids]))

        return boot_means

    @staticmethod
    def boot_means_hierarch(
        df_units: pd.DataFrame,
        unique_layers: np.ndarray,
        present_types: list,
        coef: str,
        n_boot: int,
        rng,
    ):
        """
        Hierarchical bootstrap across layers:
          - resample layers with replacement
          - within each resampled layer, resample units within each sel_type
          - pool all resampled units across bootstrapped layers, then take mean per sel_type
        """
        n_layers = len(unique_layers)
        boot_means = {sel: np.zeros(n_boot) for sel in present_types}

        vals = {}
        for lay in unique_layers:
            df_lay = df_units[df_units["layer"] == lay]
            for sel in present_types:
                arr = df_lay.loc[df_lay["sel_type"] == sel, coef].to_numpy()
                vals[(lay, sel)] = arr

        for b in range(n_boot):
            boot_layers = rng.choice(unique_layers, size=n_layers, replace=True)

            for sel in present_types:
                total = 0.0
                count = 0

                for lay in boot_layers:
                    arr = vals[(lay, sel)]
                    if arr.size == 0:
                        continue
                    boot_ids = rng.choice(arr.size, size=arr.size, replace=True)
                    samp = arr[boot_ids]
                    total += float(np.sum(samp))
                    count += int(samp.size)

                boot_means[sel][b] = (total / count) if count > 0 else np.nan

        return boot_means

class _CoefTestComputer:
    @staticmethod
    def observed_means_sds(df_units: pd.DataFrame, present_types: list, coef: str):
        obs_means = {}
        obs_sds = {}
        n_units = {}
        for sel in present_types:
            vals = df_units.loc[df_units["sel_type"] == sel, coef].to_numpy()
            vals = vals[np.isfinite(vals)]
            obs_means[sel] = float(np.mean(vals)) if vals.size else np.nan
            obs_sds[sel] = float(np.std(vals, ddof=1)) if vals.size else np.nan
            n_units[sel] = int((df_units["sel_type"] == sel).sum())
        return obs_means, obs_sds, n_units

    @staticmethod
    def one_sample_from_boot(boot_means: dict, obs_means: dict, obs_sds: dict, n_units: dict):
        one_sample_res = {}
        for sel, dist in boot_means.items():
            boot_dist = dist[np.isfinite(dist)]

            p_below = float(np.mean(boot_dist <= 0))
            pvals = float(2 * min(p_below, 1 - p_below))

            sd = float(obs_sds[sel])
            dvals = float(obs_means[sel] / sd) if (np.isfinite(sd) and sd > 0) else np.nan

            one_sample_res[sel] = {
                "mean": float(obs_means[sel]),
                "sd": float(obs_sds[sel]),
                "ci_low": float(np.percentile(boot_dist, 2.5)),
                "ci_high": float(np.percentile(boot_dist, 97.5)),
                "pvals": pvals,
                "dvals": dvals,
                "n_units": int(n_units[sel]),
            }
        return one_sample_res

    @staticmethod
    def pairwise_from_boot(boot_means: dict, obs_means: dict, obs_sds: dict, one_sample_res: dict):
        present_types = list(boot_means.keys())
        pairwise_res = {}

        for i, sel1 in enumerate(present_types):
            for sel2 in present_types[i + 1 :]:
                boot_diff = boot_means[sel1] - boot_means[sel2]
                boot_diff = boot_diff[np.isfinite(boot_diff)]

                obs_diff = float(obs_means[sel1] - obs_means[sel2])

                p_boot = float(np.mean(boot_diff >= 0))
                pvals = float(2 * min(p_boot, 1 - p_boot))

                n1 = one_sample_res[sel1]["n_units"]
                n2 = one_sample_res[sel2]["n_units"]
                sd1 = float(obs_sds[sel1])
                sd2 = float(obs_sds[sel2])
                pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
                dvals = float(obs_diff / pooled_sd) if pooled_sd > 0 else np.nan

                pairwise_res[(sel1, sel2)] = {
                    "mean_1": float(obs_means[sel1]),
                    "mean_2": float(obs_means[sel2]),
                    "mean_diff": obs_diff,
                    "ci_low": float(np.percentile(boot_diff, 2.5)),
                    "ci_high": float(np.percentile(boot_diff, 97.5)),
                    "pvals": pvals,
                    "dvals": dvals,
                }

        return pairwise_res

    @staticmethod
    def fdr_correct(one_sample_res: dict, pairwise_res: dict, fdr_alpha: float, fdr_method: str):
        all_pvals = []
        all_tests = []

        for sel in one_sample_res.keys():
            all_pvals.append(one_sample_res[sel]["pvals"])
            all_tests.append(("one_sample", sel))

        for pair in pairwise_res.keys():
            all_pvals.append(pairwise_res[pair]["pvals"])
            all_tests.append(("pairwise", pair))

        reject, p_corr = multipletests(all_pvals, alpha=fdr_alpha, method=fdr_method)[:2]

        for i, (test_type, key) in enumerate(all_tests):
            if test_type == "one_sample":
                one_sample_res[key]["pvals_corr"] = float(p_corr[i])
                one_sample_res[key]["reject"] = bool(reject[i])
            else:
                pairwise_res[key]["pvals_corr"] = float(p_corr[i])
                pairwise_res[key]["reject"] = bool(reject[i])

def test_coefs_within(
    model_name: str,
    layer: str,
    coef: str="beta_FB",
    n_boot: int=10000,
    random_state: int=None,
    fdr_method: str="fdr_by",
    fdr_alpha: float=0.05,
):
    """Test pairwise comparisons for given coefficient using bootstrap within layer."""
    data = _CoefDataLoader.load_units_single_layer(model_name, layer, coef)
    rng = np.random.default_rng(random_state)

    boot_means = _BootstrapEngine.boot_means_within_type(
        data.df_units, data.present_types, coef, n_boot, rng
    )

    obs_means, obs_sds, n_units = _CoefTestComputer.observed_means_sds(
        data.df_units, data.present_types, coef
    )
    one_sample_res = _CoefTestComputer.one_sample_from_boot(boot_means, obs_means, obs_sds, n_units)
    pairwise_res = _CoefTestComputer.pairwise_from_boot(boot_means, obs_means, obs_sds, one_sample_res)

    _CoefTestComputer.fdr_correct(one_sample_res, pairwise_res, fdr_alpha=fdr_alpha, fdr_method=fdr_method)

    summary = {
        "coef": coef,
        "layer": layer,
        "n_boot": n_boot,
        "one_sample": one_sample_res,
        "pairwise": pairwise_res,
    }
    _print_coef_tests(summary)

def test_coefs_hierarch(
    model_name: str,
    layers: list,
    coef: str="beta_FB",
    n_boot: int=10000,
    random_state: int=None,
    fdr_method: str="fdr_by",
    fdr_alpha: float=0.05,
):
    """Test pairwise comparisons for given coefficient using hierarchical bootstrap across layers."""
    data, unique_layers = _CoefDataLoader.load_units_multi_layer(model_name, layers, coef)
    rng = np.random.default_rng(random_state)

    boot_means = _BootstrapEngine.boot_means_hierarch(
        data.df_units, unique_layers, data.present_types, coef, n_boot, rng
    )

    obs_means, obs_sds, n_units = _CoefTestComputer.observed_means_sds(
        data.df_units, data.present_types, coef
    )
    one_sample_res = _CoefTestComputer.one_sample_from_boot(boot_means, obs_means, obs_sds, n_units)
    pairwise_res = _CoefTestComputer.pairwise_from_boot(boot_means, obs_means, obs_sds, one_sample_res)

    _CoefTestComputer.fdr_correct(one_sample_res, pairwise_res, fdr_alpha=fdr_alpha, fdr_method=fdr_method)

    summary = {
        "coef": coef,
        "n_layers": int(len(unique_layers)),
        "n_boot": n_boot,
        "layers": list(unique_layers),
        "one_sample": one_sample_res,
        "pairwise": pairwise_res,
    }
    _print_coef_tests(summary)

def _print_coef_tests(summary: dict):
    """Pretty print coefficient test results."""
    coef = summary["coef"]
    layer = summary.get("layer", "pooled")
    n_boot = summary["n_boot"]

    print(f"Bootstrap tests: {coef}")
    print(f"Layer: {layer}, {n_boot} bootstrap samples")

    print("\n--- Tests against zero ---")
    for sel, res in summary["one_sample"].items():
        sig = "*" if res["reject"] else "n.s."
        label = SEL_INFO[sel]["label"]
        print(f"\n{label}:")
        print(f"  M = {res['mean']:.2f}, SD = {res['sd']:.2f}")
        print(f"  CI: [{res['ci_low']:.2f}, {res['ci_high']:.2f}]")
        print(f"  p = {res['pvals_corr']:.3f} {sig}")
        print(f"  d = {res['dvals']:.2f}, n_units = {res['n_units']}")

    print("\n--- Pairwise comparisons ---")
    for (sel1, sel2), res in summary["pairwise"].items():
        sig = "*" if res["reject"] else "n.s."
        label1 = SEL_INFO[sel1]["label"]
        label2 = SEL_INFO[sel2]["label"]
        print(f"\n{label1} vs {label2}:")
        print(f"  ΔM = {res['mean_diff']:.2f}")
        print(f"  ({label1}: {res['mean_1']:.2f}, {label2}: {res['mean_2']:.2f})")
        print(f"  CI: [{res['ci_low']:.2f}, {res['ci_high']:.2f}]")
        print(f"  p = {res['pvals_corr']:.3f} {sig}")
        print(f"  d = {res['dvals']:.2f}")
