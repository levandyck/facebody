import itertools
from collections import defaultdict
import numpy as np
from scipy.stats import ttest_1samp, t as t_dist
from statsmodels.stats.multitest import multipletests

from facebody.plotting import SEL_INFO
from facebody.config import PROJECT_ROOT
from myutils.utils import load_pickle

# --------------------------- Statistical inference -------------------------- #
class StatsComputer:
    """Compute pairwise statistical comparisons for selectivity types."""
    SEL_TYPES = list(SEL_INFO.keys())
    SEL_KEYS = ["f", "b", "m", "ns"]

    @staticmethod
    def compute_stats_from_diff(diff: np.ndarray):
        diff = np.asarray(diff, dtype=float)
        diff = diff[~np.isnan(diff)]
        if diff.size < 2:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        res_t = ttest_1samp(diff, 0.0)
        t = float(res_t.statistic)
        df = float(res_t.df)
        p = float(res_t.pvalue)

        mean_diff = diff.mean()
        se = diff.std(ddof=1) / np.sqrt(diff.size)
        t_crit = t_dist.ppf(0.975, df)
        ci_lower = mean_diff - t_crit * se
        ci_upper = mean_diff + t_crit * se

        sd = diff.std(ddof=1)
        d = np.nan if (not np.isfinite(sd) or sd == 0) else (diff.mean() / sd)
        return t, df, p, d, ci_lower, ci_upper

    @staticmethod
    def get_random_effect_diff(subj_roi_means: dict, rois: list, s1: str, s2: str):
        subj_diffs = []
        for subj, roi_map in subj_roi_means.items():
            diffs_this = []
            for roi in rois:
                m1 = roi_map.get(roi, {}).get(s1, np.nan)
                m2 = roi_map.get(roi, {}).get(s2, np.nan)
                if np.isfinite(m1) and np.isfinite(m2):
                    diffs_this.append(m1 - m2)
            if diffs_this:
                subj_diffs.append(float(np.mean(diffs_this)))
        return np.array(subj_diffs, dtype=float)

    @staticmethod
    def get_fixed_effect_diff(voxel_values: dict, rois: list, s1: str, s2: str):
        v1_list = [np.concatenate(voxel_values[roi][s1]) for roi in rois if len(voxel_values[roi][s1]) > 0]
        v2_list = [np.concatenate(voxel_values[roi][s2]) for roi in rois if len(voxel_values[roi][s2]) > 0]
        v1 = np.concatenate(v1_list) if v1_list else np.array([])
        v2 = np.concatenate(v2_list) if v2_list else np.array([])
        return v1 - v2

    @staticmethod
    def print_roi_stats(roi: str, rows: list, effect_type: str):
        w_cmp = max(len("Comparison"), max(len(r[0]) for r in rows))
        hdr = (f"\nROI: {roi}   [{effect_type.upper()} EFFECTS]\n"
               f"{'Comparison':<{w_cmp}} |    t |  df | p-corr |   d | 95% CI | sig?")
        print(hdr)
        print("-" * len(hdr))
        for comp, t, df, p, pc, d, ci_lower, ci_upper, sig in rows:
            t_str = f"{t: .2f}" if np.isfinite(t) else "  --"
            df_str = f"{df: .0f}" if np.isfinite(df) else " --"
            pc_str = f"{pc:.3f}" if np.isfinite(pc) else "--"
            d_str = f"{d: .2f}" if np.isfinite(d) else "  --"
            ci_str = f"[{ci_lower:.2f}, {ci_upper:.2f}]" if np.isfinite(ci_lower) and np.isfinite(ci_upper) else "[--,--]"
            sigstr = "*" if sig else ""
            print(f"{comp:<{w_cmp}} | {t_str:>4} | {df_str:>3} | {pc_str:>6} | {d_str:>4} | {ci_str:>18} |  {sigstr}")

def stats_pairwise_comps_sep(model_name: str, layer: str, rois: list, effect_type: str="random"):
    if effect_type not in {"random", "fixed"}:
        raise ValueError("effect_type must be 'random' or 'fixed'")

    res = load_pickle(PROJECT_ROOT / "encoding" / model_name / "sep.pkl")[layer]
    comps = list(itertools.combinations(StatsComputer.SEL_TYPES, 2))

    subj_roi_means = defaultdict(lambda: defaultdict(dict))
    subj_means = defaultdict(lambda: defaultdict(list))
    voxel_values = defaultdict(lambda: defaultdict(list))

    for subj, roi_dict in res.items():
        for roi in rois:
            res_roi = roi_dict.get(roi)
            if not res_roi:
                continue
            r2s = res_roi.get("r2", {})
            for sel, key in zip(StatsComputer.SEL_TYPES, StatsComputer.SEL_KEYS):
                arr = r2s.get(key)
                if arr is None:
                    continue
                m = np.nanmean(arr)
                subj_roi_means[subj][roi][sel] = m
                subj_means[roi][sel].append(m)
                voxel_values[roi][sel].append(arr)

    for roi in rois:
        pvals = []
        rows_raw = []

        for s1, s2 in comps:
            if effect_type == "random":
                diff = np.array(subj_means[roi][s1], float) - np.array(subj_means[roi][s2], float)
            else:
                v1 = np.concatenate(voxel_values[roi][s1]) if voxel_values[roi][s1] else np.array([])
                v2 = np.concatenate(voxel_values[roi][s2]) if voxel_values[roi][s2] else np.array([])
                diff = v1 - v2

            t, df, p, d, ci_lower, ci_upper = StatsComputer.compute_stats_from_diff(diff)
            pvals.append(p)
            rows_raw.append((s1, s2, t, df, p, d, ci_lower, ci_upper))

        reject, p_corr, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")
        rows = [
            (f"{s1} vs {s2}", t, df, p, pc, d, ci_lower, ci_upper, sig)
            for (s1, s2, t, df, p, d, ci_lower, ci_upper), pc, sig in zip(rows_raw, p_corr, reject)
        ]
        StatsComputer.print_roi_stats(roi, rows, effect_type)

    pvals = []
    rows_raw = []
    for s1, s2 in comps:
        if effect_type == "random":
            diff = StatsComputer.get_random_effect_diff(subj_roi_means, rois, s1, s2)
        else:
            diff = StatsComputer.get_fixed_effect_diff(voxel_values, rois, s1, s2)

        t, df, p, d, ci_lower, ci_upper = StatsComputer.compute_stats_from_diff(diff)
        pvals.append(p)
        rows_raw.append((s1, s2, t, df, p, d, ci_lower, ci_upper))

    reject, p_corr, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")

    print(f"\nROI: ALL (pooled)   [{effect_type.upper()} EFFECTS]")
    print("Comparison |    t |  df | p-corr |   d | 95% CI | sig?")
    print("-----------------------------------------------------")
    for (s1, s2, t, df, p, d, ci_lower, ci_upper), pc, sig in zip(rows_raw, p_corr, reject):
        t_str = f"{t: .2f}" if np.isfinite(t) else "  --"
        df_str = f"{df: .0f}" if np.isfinite(df) else " --"
        pc_str = f"{pc:.3f}" if np.isfinite(pc) else "--"
        d_str = f"{d: .2f}" if np.isfinite(d) else "  --"
        ci_str = f"[{ci_lower:.2f}, {ci_upper:.2f}]" if np.isfinite(ci_lower) and np.isfinite(ci_upper) else "[--,--]"
        sigstr = "*" if sig else ""
        print(f"{s1} vs {s2} | {t_str:>4} | {df_str:>3} | {pc_str:>6} | {d_str:>4} | {ci_str:>18} |  {sigstr}")
