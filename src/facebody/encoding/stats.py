"""Pairwise comparisons of unit types and double dissociations."""

import itertools
from collections import defaultdict
import numpy as np
from scipy.stats import ttest_1samp, t as t_dist
from statsmodels.stats.multitest import multipletests

from facebody.style import SEL_INFO

# --------------------------- Statistical inference -------------------------- #
class StatsComputer:
    """Compute pairwise statistical comparisons for selectivity types."""
    SEL_TYPES = list(SEL_INFO.keys())
    SEL_KEYS = ["f", "b", "m", "ns"]

    @staticmethod
    def compute_stats_from_diff(diff: np.ndarray):
        """Paired t-test against zero on `diff`: t, df, p, Cohen's d and the 95% CI."""
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
        """One s1 - s2 difference per subject, averaged over the ROIs they have."""
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
        """s1 - s2 pooled over all voxels of all subjects, ignoring subject identity."""
        v1_list = [np.concatenate(voxel_values[roi][s1]) for roi in rois if len(voxel_values[roi][s1]) > 0]
        v2_list = [np.concatenate(voxel_values[roi][s2]) for roi in rois if len(voxel_values[roi][s2]) > 0]
        v1 = np.concatenate(v1_list) if v1_list else np.array([])
        v2 = np.concatenate(v2_list) if v2_list else np.array([])
        return v1 - v2

    @staticmethod
    def print_roi_stats(roi: str, rows: list, effect_type: str):
        """Print one ROI's comparison table."""
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

def stats_pairwise_comps_sep(model_name: str, layer: str, rois: list,
                             effect_type: str="random", subjects: list=None,
                             sig_only: bool=False, r2_adj: bool=True):
    """Pairwise comparisons of unit types per ROI, from voxelwise encoding models."""
    from .summarize import available_subjects, load_nc_by_roi, voxelwise_to_roi

    if effect_type not in {"random", "fixed"}:
        raise ValueError("effect_type must be 'random' or 'fixed'")

    subjects = available_subjects(model_name, layer) if subjects is None else subjects
    assert subjects, f"no voxelwise results for {model_name}/{layer}"
    res = voxelwise_to_roi(model_name, layer, subjects, rois, "sep", sig_only=sig_only)
    comps = list(itertools.combinations(StatsComputer.SEL_TYPES, 2))

    if r2_adj:
        nc_by_roi = load_nc_by_roi(subjects)

    subj_roi_means = defaultdict(lambda: defaultdict(dict))
    voxel_values = defaultdict(lambda: defaultdict(list))

    for subj, roi_dict in res.items():
        for roi in rois:
            res_roi = roi_dict.get(roi)
            if not res_roi:
                continue
            r2s = res_roi.get("r2", {})

            nc = 1.0
            if r2_adj:
                fmri_roi = nc_by_roi.get(subj, {}).get(roi)
                if fmri_roi is None:
                    continue
                nc = float(np.nanmean(fmri_roi["ncsnr"]))
                if not np.isfinite(nc) or nc <= 0:
                    continue

            for sel, key in zip(StatsComputer.SEL_TYPES, StatsComputer.SEL_KEYS):
                arr = r2s.get(key)
                if arr is None:
                    continue
                m = np.nanmean(arr) / nc
                subj_roi_means[subj][roi][sel] = m
                voxel_values[roi][sel].append(arr / nc)

    for roi in rois:
        pvals = []
        rows_raw = []

        for s1, s2 in comps:
            if effect_type == "random":
                # Paired by subject, not by list position: a subject missing one unit
                # type must not shift every later subject's partner.
                diff = StatsComputer.get_random_effect_diff(subj_roi_means, [roi], s1, s2)
            else:
                v1 = np.concatenate(voxel_values[roi][s1]) if voxel_values[roi][s1] else np.array([])
                v2 = np.concatenate(voxel_values[roi][s2]) if voxel_values[roi][s2] else np.array([])
                diff = v1 - v2

            t, df, p, d, ci_lower, ci_upper = StatsComputer.compute_stats_from_diff(diff)
            pvals.append(p)
            rows_raw.append((s1, s2, t, df, p, d, ci_lower, ci_upper))

        reject, p_corr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
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

    reject, p_corr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

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


# --------------------------- Double dissociations --------------------------- #
def _dd_cells(model_name: str, layer: str, subjects: list, analysis: str,
              keys: tuple, rois: tuple, r2_adj: bool=True):
    """Per-participant cell means for a double dissociation."""
    from .summarize import available_subjects, load_nc_by_roi, voxelwise_to_roi

    subjects = available_subjects(model_name, layer) if subjects is None else subjects
    assert subjects, f"no voxelwise results for {model_name}/{layer}"

    res = voxelwise_to_roi(model_name, layer, subjects, list(rois), analysis)
    nc_by_roi = load_nc_by_roi(subjects) if r2_adj else {}

    out = {roi: {k: [] for k in keys} for roi in rois}
    kept = []

    for subj in subjects:
        cell, ok = {}, True
        for roi in rois:
            comp = res.get(subj, {}).get(roi)
            if comp and analysis == "sep":
                comp = comp.get("r2")
            if not comp:
                ok = False
                break

            nc = 1.0
            if r2_adj:
                fmri_roi = nc_by_roi.get(subj, {}).get(roi)
                if fmri_roi is None:
                    ok = False
                    break
                nc = float(np.nanmean(fmri_roi["ncsnr"]))
                if not np.isfinite(nc) or nc <= 0:
                    ok = False
                    break

            vals = {}
            for k in keys:
                arr = comp.get(k)
                if arr is None:
                    ok = False
                    break
                vals[k] = float(np.nanmean(arr)) / nc
            if not ok:
                break
            cell[roi] = vals

        if not ok:
            continue
        kept.append(subj)
        for roi in rois:
            for k in keys:
                out[roi][k].append(cell[roi][k])

    cells = {roi: {k: np.asarray(v, dtype=float) for k, v in d.items()}
             for roi, d in out.items()}
    return cells, kept

def _dd_report(cells: dict, roi_a: str, roi_b: str, key_a: str, key_b: str,
               labels: tuple, header: str="", correct: str="fdr_bh", verbose: bool=True):
    """Three contrasts of a double dissociation from per-participant cell means."""
    a, b = cells[roi_a], cells[roi_b]
    diffs = {
        labels[0]: a[key_a] - a[key_b],
        labels[1]: b[key_b] - b[key_a],
        labels[2]: (a[key_a] - a[key_b]) - (b[key_a] - b[key_b]),
    }

    rows, pvals = [], []
    for name, diff in diffs.items():
        t, df, p, d, ci_lo, ci_hi = StatsComputer.compute_stats_from_diff(diff)
        rows.append((name, t, df, p, d, ci_lo, ci_hi))
        pvals.append(p)

    p_corr = (multipletests(pvals, alpha=0.05, method=correct)[1] if correct
              else np.asarray(pvals, dtype=float))

    out = {}
    for (name, t, df, p, d, ci_lo, ci_hi), pc in zip(rows, p_corr):
        out[name] = {"t": t, "df": df, "p": float(pc), "p_unc": p,
                     "d": d, "ci_low": ci_lo, "ci_high": ci_hi}

    means = {roi: {k: float(np.mean(v)) for k, v in cells[roi].items()}
             for roi in (roi_a, roi_b)}

    if verbose:
        fmt_p = lambda v: "< .001" if v < 0.001 else f"{v: .3f}"
        w = max(len(n) for n in labels)
        print(f"\n{header}")
        print("\n  Participant means:")
        print(f"    {'voxel type':<14}{key_a:>12}{key_b:>12}")
        for roi in (roi_a, roi_b):
            print(f"    {roi:<14}{means[roi][key_a]:>12.3f}{means[roi][key_b]:>12.3f}")
        print("\n  Contrasts:")
        for name in labels:
            r = out[name]
            print(f"    {name:<{w}} t({r['df']:.0f}) = {r['t']:6.2f}, "
                  f"p ={fmt_p(r['p'])} (raw{fmt_p(r['p_unc'])}), d = {r['d']:5.2f}, "
                  f"95% CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")

    return {"contrasts": out, "means": means}

def stats_varpart_dd(model_name: str, layer: str, subjects: list=None,
                     r2_adj: bool=True, correct: str="fdr_bh", verbose: bool=True):
    """Double dissociation of face unique and body unique variance."""
    cells, kept = _dd_cells(model_name, layer, subjects, "fb_varpart",
                            ("u_f", "u_b"), ("face", "body"), r2_adj)
    res = _dd_report(
        cells, "face", "body", "u_f", "u_b",
        ("face voxels: u_F - u_B",
         "body voxels: u_B - u_F",
         "interaction: (u_F - u_B), face voxels - body voxels"),
        header=(f"Unique variance double dissociation | {model_name} / {layer} | "
                f"{'R2_adj' if r2_adj else 'R2'} | n = {len(kept)} | {correct} x 3"),
        correct=correct, verbose=verbose)
    res["subjects"] = kept
    return res

def stats_sep_dd(model_name: str, layer: str, subjects: list=None,
                 r2_adj: bool=True, correct: str="fdr_bh", verbose: bool=True):
    """Double dissociation of face unique and body unique variance."""
    cells, kept = _dd_cells(model_name, layer, subjects, "sep",
                            ("f", "b"), ("face", "body"), r2_adj)
    res = _dd_report(
        cells, "face", "body", "f", "b",
        ("face voxels: face units - body units",
         "body voxels: body units - face units",
         "interaction: (face - body) units, face voxels - body voxels"),
        header=(f"Explained variance double dissociation | {model_name} / {layer} | "
                f"{'R2_adj' if r2_adj else 'R2'} | n = {len(kept)} | {correct} x 3"),
        correct=correct, verbose=verbose)
    res["subjects"] = kept
    return res
