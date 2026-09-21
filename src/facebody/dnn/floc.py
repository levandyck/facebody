"""Localize face-, body-, and mixed-selective units in DNN with fMRI-style functional localizer, and validate them on a held-out image set."""

import torch
from torch.utils.data import DataLoader
import numpy as np
import scipy.stats as st
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from itertools import combinations

from facebody.config import PROJECT_ROOT, DATA_ROOT
from facebody.myutils.models import ModelLoader
from facebody.myutils.feature_extractor import FeatureExtractor
from facebody.myutils.nsd import NSDImageLoader, _nsd_img_worker_init
from facebody.myutils.utils import load_pickle, save_pickle

EPS = 1e-10

# --------------------------------- DNN fLoc --------------------------------- #
class DNNfloc:
    """Identify selective DNN units using fMRI-inspired approach."""
    def __init__(self, model_name: str, layers: list,
                 class_to_idx: dict,
                 single_cats: list=None, mixed_tuples: list=None,
                 solo_cats: list=None,
                 baseline_cat: str="object",
                 scrambled_cat: str="scrambled",
                 fdr_method: str="fdr_by", 
                 alpha_pos: float=0.001,
                 alpha_neg: float=0.05):
        self.model_name = model_name
        self.layers = layers
        self.class_to_idx = class_to_idx
        self.baseline_cat = baseline_cat
        self.scrambled_cat = scrambled_cat
        self.fdr_method = fdr_method
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

        if baseline_cat not in class_to_idx:
            raise ValueError(f"Baseline category '{baseline_cat}' not found in class_to_idx")
        if scrambled_cat not in class_to_idx:
            raise ValueError(f"Noise category '{scrambled_cat}' not found in class_to_idx")

        self.baseline_idx = class_to_idx[baseline_cat]
        self.scrambled_idx = class_to_idx[scrambled_cat]
        self.single_cats = single_cats or [cat for cat in class_to_idx.keys() if cat != baseline_cat]
        self.mixed_tuples = mixed_tuples or list(combinations(self.single_cats, 2))
        self.solo_cats = list(solo_cats or [])
        self.all_cats = list(class_to_idx.keys())
        self._validate_categories()

        mixed_names = ["mixed"] if len(self.mixed_tuples) == 1 else ["&".join(t) for t in self.mixed_tuples]
        self.mixed_tuple_names = dict(zip(self.mixed_tuples, mixed_names))
        self.sel_types = self.single_cats + mixed_names

        print(f"Model: {self.model_name}")
        print(f"Pure selectivity: {self.single_cats}")
        print(f"Mixed selectivity: {['&'.join(t) for t in self.mixed_tuples]}")
        print(f"Solo selectivity: {self.solo_cats}")

    def _validate_categories(self):
        """Validate all single and mixed category definitions."""
        invalid_cats = [cat for cat in self.single_cats + self.solo_cats
                        if cat not in self.class_to_idx]
        if invalid_cats:
            raise ValueError(f"Categories {invalid_cats} not found in class_to_idx")

        overlap = set(self.solo_cats) & set(self.single_cats)
        if overlap:
            raise ValueError(f"Categories {sorted(overlap)} are in both single_cats and solo_cats")

        reserved = {self.baseline_cat, self.scrambled_cat} & set(self.solo_cats)
        if reserved:
            raise ValueError(f"Categories {sorted(reserved)} cannot be used as solo_cats")

        for cat_tuple in self.mixed_tuples:
            if len(cat_tuple) < 2:
                raise ValueError(f"Mixed tuple {cat_tuple} must have at least 2 categories")
            invalid_in_tuple = [cat for cat in cat_tuple if cat not in self.class_to_idx]
            if invalid_in_tuple:
                raise ValueError(f"Categories {invalid_in_tuple} in tuple {cat_tuple} not found in class_to_idx")

    @staticmethod
    def ttest_onesided(group1: np.ndarray, group2: np.ndarray,
                       fdr_method: str, fdr_alpha: float):
        """Perform Welch's one-sided t-test between two groups with FDR correction."""
        n1, n2 = len(group1), len(group2)
        m1, m2 = group1.mean(axis=0), group2.mean(axis=0)
        v1, v2 = group1.var(axis=0, ddof=1), group2.var(axis=0, ddof=1)

        valid = (v1 > EPS) & (v2 > EPS)
        tvals = np.zeros_like(m1)
        pvals = np.ones_like(m1)

        if valid.any():
            v1v = v1[valid]
            v2v = v2[valid]
            se = np.sqrt(v1v / n1 + v2v / n2)
            t = (m1[valid] - m2[valid]) / se
            df = ((v1v / n1 + v2v / n2) ** 2) / (((v1v / n1) ** 2) / (n1 - 1) + ((v2v / n2) ** 2) / (n2 - 1))
            p = st.t.sf(t, df)
            tvals[valid] = t
            pvals[valid] = p

        reject, pvals_corr = multipletests(pvals, alpha=fdr_alpha, method=fdr_method)[:2]
        mask = reject & (tvals > 0)
        return tvals, pvals_corr, mask

    @staticmethod
    def dprime(activs: np.ndarray, labels: np.ndarray, target: int):
        """Compute d' for target class vs. all others."""
        act1 = activs[labels == target]
        act2 = activs[labels != target]

        m1, m2 = act1.mean(axis=0), act2.mean(axis=0)
        v1, v2 = act1.var(axis=0, ddof=1), act2.var(axis=0, ddof=1)

        n1, n2 = act1.shape[0], act2.shape[0]
        df1, df2 = n1 - 1, n2 - 1
        denom = (df1 + df2) if (df1 + df2) > 0 else 1
        s2_pooled = (df1 * v1 + df2 * v2) / denom + EPS

        return (m1 - m2) / np.sqrt(s2_pooled)

    def _compute_pure_sel(self, grouped: dict, target_cat: str,
                          competing_cat: str=None):
        """
        Test for pure selectivity using baseline approach.

        All p-values are FDR-corrected across units: alpha_pos for the positive
        criteria, alpha_neg for the negative ones.

        Example for face-selective (competing_cat=body, baseline=object):
        1. faces > objects (p < alpha_pos)
        2. faces > scrambled (p < alpha_pos)
        3. faces > bodies (p < alpha_pos) (only if a competing cat is given)
        4. NOT bodies > objects (p >= alpha_neg)
        5. NOT scenes > objects (p >= alpha_neg)
        """
        target_idx = self.class_to_idx[target_cat]
        baseline_idx = self.baseline_idx
        scrambled_idx = self.scrambled_idx

        target_group = grouped[target_idx]
        baseline_group = grouped[baseline_idx]
        scrambled_group = grouped[scrambled_idx]

        # Positive criterion 1: target > baseline
        _, _, mask_target_vs_baseline = self.ttest_onesided(
            target_group, baseline_group,
            self.fdr_method, self.alpha_pos
        )

        # Positive criterion 2: target > scrambled
        _, _, mask_target_vs_scrambled = self.ttest_onesided(
            target_group, scrambled_group,
            self.fdr_method, self.alpha_pos
        )

        # Positive criterion 3: target > competing (only if a rival is specified)
        if competing_cat is None:
            mask_target_vs_competing = np.ones(target_group.shape[1], dtype=bool)
        else:
            competing_group = grouped[self.class_to_idx[competing_cat]]
            _, _, mask_target_vs_competing = self.ttest_onesided(
                target_group, competing_group,
                self.fdr_method, self.alpha_pos
            )

        # Negative criteria: ALL other categories NOT > baseline
        # Other categories = all cats except target, baseline, and scrambled
        other_cats = [
            cat for cat in self.all_cats 
            if cat not in [target_cat, self.baseline_cat, self.scrambled_cat]
        ]

        mask_list_neg = []
        for other_cat in other_cats:
            other_idx = self.class_to_idx[other_cat]
            other_group = grouped[other_idx]

            _, pvals_corr, _ = self.ttest_onesided(
                other_group, baseline_group,
                self.fdr_method, self.alpha_neg
            )

            # NOT significant = p >= alpha_neg
            mask_not_sel = (pvals_corr >= self.alpha_neg)
            mask_list_neg.append(mask_not_sel)

        # All negative criteria must pass
        mask_neg = np.stack(mask_list_neg, axis=0).all(axis=0)

        # All criteria must be met
        mask_all = (
            mask_target_vs_baseline & 
            mask_target_vs_scrambled &
            mask_target_vs_competing & 
            mask_neg
        )

        return {"mask": mask_all}

    def _compute_mixed_sel(self, grouped: dict, cat_tuple: tuple):
        """
        Test for mixed selectivity using baseline approach.

        All p-values are FDR-corrected across units: alpha_pos for the positive
        criteria, alpha_neg for the negative ones.

        Example for face&body-selective (baseline=object):
        1. faces > objects (p < alpha_pos)
        2. faces > scrambled (p < alpha_pos)
        3. bodies > objects (p < alpha_pos)
        4. bodies > scrambled (p < alpha_pos)
        5. NOT scenes > objects (p >= alpha_neg)
        """
        if len(cat_tuple) != 2:
            raise ValueError("Mixed selectivity currently only supports 2-category tuples")

        baseline_idx = self.baseline_idx
        scrambled_idx = self.scrambled_idx
        baseline_group = grouped[baseline_idx]
        scrambled_group = grouped[scrambled_idx]

        # Positive criteria: each category in tuple > baseline
        mask_list_pos = []
        for cat in cat_tuple:
            cat_idx = self.class_to_idx[cat]
            cat_group = grouped[cat_idx]

            # Test cat > baseline
            _, _, mask_vs_baseline = self.ttest_onesided(
                cat_group, baseline_group,
                self.fdr_method, self.alpha_pos
            )

            # Test cat > scrambled
            _, _, mask_vs_scrambled = self.ttest_onesided(
                cat_group, scrambled_group,
                self.fdr_method, self.alpha_pos
            )

            # Both must pass for this category
            mask_list_pos.append(mask_vs_baseline & mask_vs_scrambled)

        mask_pos = np.stack(mask_list_pos, axis=0).all(axis=0)

        # Negative criteria: ALL other categories NOT > baseline
        other_cats = [
            cat for cat in self.all_cats 
            if cat not in list(cat_tuple) + [self.baseline_cat, self.scrambled_cat]
        ]

        mask_list_neg = []
        for other_cat in other_cats:
            other_idx = self.class_to_idx[other_cat]
            other_group = grouped[other_idx]

            _, pvals_corr, _ = self.ttest_onesided(
                other_group, baseline_group,
                self.fdr_method, self.alpha_neg
            )

            mask_not_sel = (pvals_corr >= self.alpha_neg)
            mask_list_neg.append(mask_not_sel)

        mask_neg = np.stack(mask_list_neg, axis=0).all(axis=0) if mask_list_neg else np.ones_like(mask_pos, dtype=bool)

        # Both criteria must be met
        mask_mixed = mask_pos & mask_neg

        return {"mask": mask_mixed}

    @staticmethod
    def _check_mutual_exclusivity(unit_ids: dict, layer: str, verbose: bool=True):
        """Check that selectivity categories are mutually exclusive."""
        all_assigned = []
        for _, units in unit_ids.items():
            all_assigned.extend(units)

        n_unique = len(set(all_assigned))
        n_total = len(all_assigned)
        is_exclusive = (n_unique == n_total)

        if verbose:
            if is_exclusive:
                print(f"✓ {layer}: Mutual exclusivity confirmed")
            else:
                print(f"✗ {layer}: Mutual exclusivity VIOLATED ({n_total - n_unique} duplicates)")

                sel_types = list(unit_ids.keys())
                for i in range(len(sel_types)):
                    for j in range(i+1, len(sel_types)):
                        overlap = set(unit_ids[sel_types[i]]) & set(unit_ids[sel_types[j]])
                        if len(overlap) > 0:
                            print(f"  - {sel_types[i]}-{sel_types[j]} overlap: {len(overlap)} units")

        return {"is_exclusive": is_exclusive}

    def find_sel_units(self, activs: dict, labels: np.ndarray,
                       check_exclusivity: bool=True, out_name: str="floc_res"):
        """
        Identify selective units using baseline approach with mutual exclusivity.

        Runs with different single_cats/mixed_tuples must set different out_name,
        or they overwrite each other's results.

        For 2-category case (face, body) with baseline (object) and scrambled:
        - Face-selective: face>object AND face>scrambled AND face>body
          AND NOT(body>object) AND NOT(others>object)
        - Body-selective: body>object AND body>scrambled AND body>face
          AND NOT(face>object) AND NOT(others>object)
        - Mixed-selective: face>object AND face>scrambled AND body>object
          AND body>scrambled AND NOT(others>object)
        """
        res = {}
        excl_report = {}

        for layer in tqdm(self.layers, desc="Layers"):
            X, y = activs[layer], labels
            grouped = {self.class_to_idx[cat]: X[y == self.class_to_idx[cat]] 
                      for cat in self.all_cats}

            n_units = X.shape[1]
            unit_ids = {}
            stats_dict = {}

            # Test pure selectivity
            if len(self.single_cats) == 2:
                cat1, cat2 = self.single_cats
                pure_specs = [(cat1, cat2), (cat2, cat1)]
            else:
                pure_specs = [(cat, None) for cat in self.single_cats]
            pure_specs += [(cat, None) for cat in self.solo_cats]

            for target_cat, competing_cat in pure_specs:
                res_cat = self._compute_pure_sel(grouped, target_cat, competing_cat)
                unit_ids[target_cat] = np.where(res_cat["mask"])[0]

                dvals = self.dprime(X, y, self.class_to_idx[target_cat])
                stats_dict[target_cat] = {"dvals": dvals, "mask": res_cat["mask"]}

            # Test mixed selectivity
            for cat_tuple in self.mixed_tuples:
                tuple_name = self.mixed_tuple_names[cat_tuple]
                res_mixed = self._compute_mixed_sel(grouped, cat_tuple)
                unit_ids[tuple_name] = np.where(res_mixed["mask"])[0]

                tuple_ids = [self.class_to_idx[cat] for cat in cat_tuple]
                y_mixed = np.isin(y, tuple_ids).astype(int)
                dvals_mixed = self.dprime(X, y_mixed, target=1)
                stats_dict[tuple_name] = {"dvals": dvals_mixed, "mask": res_mixed["mask"]}

            res[layer] = {
                "unit_ids": unit_ids,
                "stats": stats_dict,
                "n_selective": {
                    sel_type: len(units)
                    for sel_type, units in unit_ids.items()
                },
                "total_units": n_units,
            }

            print(f"{layer}: " + ", ".join([f"{sel_type}={len(units)}" 
                                            for sel_type, units in unit_ids.items()]))

            if check_exclusivity:
                excl_report[layer] = self._check_mutual_exclusivity(
                    unit_ids, layer, verbose=False
                )

        if check_exclusivity:
            all_exclusive = all(rep["is_exclusive"] for rep in excl_report.values())
            print("\n" + "="*60)
            if all_exclusive:
                print("ALL LAYERS: Mutual exclusivity confirmed across all layers")
            else:
                print("WARNING: Mutual exclusivity violated in some layers")
                violated_layers = [lay for lay, rep in excl_report.items() 
                                 if not rep["is_exclusive"]]
                print(f"Violated in: {violated_layers}")
            print("="*60 + "\n")

        save_dir = PROJECT_ROOT / "selectivity" / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / f"{out_name}.pkl"
        save_pickle(res, fname)
        print(f"\nResults saved to: {fname}")


# ----------------------------------- Utils ---------------------------------- #
def filter_sel_activs(model_name: str, activs: dict, layer: str,
                      n_top: int=None, controlled: bool=False,
                      sort_nonsel_desc: str="mixed"):
    """
    Activations of each unit type in one layer, read from the saved fLoc results.

    Selective types are sorted by descending d'. The non-selective group is every
    remaining unit, sorted by ascending d' of `sort_nonsel_desc`.
    With controlled=True all four groups are truncated to the size of the smallest
    selective type.
    """
    X = activs[layer]
    res = load_pickle(PROJECT_ROOT / "selectivity" / model_name / "floc_res.pkl")
    sel_ids = res[layer]["unit_ids"]
    stats = res[layer]["stats"]
    sel_types = ["face", "body", "mixed"]

    # Sort each selective unit type by descending d'
    sorted_ids = {}
    for s in sel_types:
        ids = sel_ids[s]
        if ids.size:
            d = stats[s]["dvals"]
            sorted_ids[s] = ids[np.argsort(d[ids])[::-1]]
        else:
            sorted_ids[s] = np.array([], int)

    # Get non-selective units and sorted by ascending d'
    all_u = np.arange(X.shape[1])
    sel_u = np.concatenate(list(sorted_ids.values()))
    nonsel_u = np.setdiff1d(all_u, sel_u)
    ref_d = stats[sort_nonsel_desc]["dvals"]
    sorted_ids["nonselective"] = nonsel_u[np.argsort(ref_d[nonsel_u])]

    # Get equal‐sized unit groups or only top‐n units
    if controlled:
        top = n_top or min(len(sorted_ids[s]) for s in sel_types if len(sorted_ids[s])>0)
        for s in (*sel_types, "nonselective"):
            sorted_ids[s] = sorted_ids[s][:top]
    elif n_top is not None:
        for s in sel_types:
            sorted_ids[s] = sorted_ids[s][:n_top]

    return {s: X[:, sorted_ids[s]] for s in (*sel_types, "nonselective") if sorted_ids[s].size}

def compute_sel_resp(model_name: str, activs: dict, labels: np.ndarray,
                     class_to_idx: dict, img_db: str="validation"):
    """Compute mean z-scored response of each unit to each category."""
    res = {}
    for layer, X in activs.items():
        # z-score each unit across all images, then average within each category
        unit_mean = X.mean(axis=0, dtype=np.float64)
        unit_sd = X.std(axis=0, ddof=1, dtype=np.float64)
        unit_sd = np.where(unit_sd > EPS, unit_sd, np.nan)

        res[layer] = {
            f"{cat}_z": ((X[labels == idx].mean(axis=0, dtype=np.float64) - unit_mean)
                         / unit_sd).tolist()
            for cat, idx in class_to_idx.items()
        }

    save_pickle(res, PROJECT_ROOT / "selectivity" / model_name / f"{img_db}_resp.pkl")

def compute_sel_dprime(model_name: str, activs: dict, labels: np.ndarray,
                       class_to_idx: dict, img_db: str="validation",
                       baseline_cat: str=None):
    """
    Compute face, body, and mixed selectivity (d') for each unit in layer.
    If baseline_cat is None: compute face/body vs. all other categories (one vs. all)
    If baseline_cat is set: compute face/body vs. baseline category only
    """
    face_idx = class_to_idx["face"]
    body_idx = class_to_idx["body"]

    res = {}
    for layer, X in activs.items():
        if baseline_cat is None:
            # One vs. all - direct usage
            dvals_f = DNNfloc.dprime(X, labels, face_idx)
            dvals_b = DNNfloc.dprime(X, labels, body_idx)

            # Mixed: create binary labels (1=face/body, 0=others)
            y_mixed = ((labels == face_idx) | (labels == body_idx)).astype(int)
            dvals_m = DNNfloc.dprime(X, y_mixed, target=1)
        else:
            # One vs. baseline
            baseline_idx = class_to_idx[baseline_cat]

            # Face vs. baseline
            mask = (labels == face_idx) | (labels == baseline_idx)
            dvals_f = DNNfloc.dprime(X[mask], labels[mask], face_idx)

            # Body vs. baseline
            mask = (labels == body_idx) | (labels == baseline_idx)
            dvals_b = DNNfloc.dprime(X[mask], labels[mask], body_idx)

            # Mixed vs. baseline
            mask = (labels == face_idx) | (labels == body_idx) | (labels == baseline_idx)
            y_mixed = np.isin(labels[mask], [face_idx, body_idx]).astype(int)
            dvals_m = DNNfloc.dprime(X[mask], y_mixed, target=1)

        res[layer] = {
            "face_d": dvals_f.tolist(),
            "body_d": dvals_b.tolist(),
            "mixed_d": dvals_m.tolist()
        }

    save_pickle(res, PROJECT_ROOT / "selectivity" / model_name / f"{img_db}_dprime.pkl")

def stats_floc_dprime_summary(model_name: str, layers: list=None,
                              controlled: bool=True, n_top: int=None,
                              method: str="layer-mean"):
    """Compute grand mean ± SD d' per selective unit type across layers."""
    res = load_pickle(PROJECT_ROOT / "selectivity" / model_name / "floc_res.pkl")
    cats = ("face", "body", "mixed")

    if layers is None:
        layers = list(res.keys())

    # Collect per-layer selections and d' values
    pooled = {c: [] for c in cats}
    per_layer_means = {c: [] for c in cats}

    for lay in layers:
        if lay not in res:
            continue
        sel_ids = res[lay]["unit_ids"]
        stats = res[lay]["stats"]

        # Sort by within-type d' (descending)
        sorted_ids = {}
        for c in cats:
            ids = np.array(sel_ids.get(c, []), dtype=int)
            if ids.size:
                d_all = np.asarray(stats[c]["dvals"])
                sorted_ids[c] = ids[np.argsort(d_all[ids])[::-1]]
            else:
                sorted_ids[c] = np.array([], dtype=int)

        # Decide how many to keep per type in this layer
        if controlled:
            avail = [len(sorted_ids[c]) for c in cats if len(sorted_ids[c]) > 0]
            top = (n_top if n_top is not None else (min(avail) if avail else 0))
            use_ids = {c: sorted_ids[c][:top] for c in cats}
        else:
            use_ids = sorted_ids

        # Gather unit-level d' and layer means
        for c in cats:
            ids = use_ids[c]
            if ids.size == 0:
                continue
            d_vals = np.asarray(stats[c]["dvals"])[ids]
            pooled[c].append(d_vals)
            per_layer_means[c].append(float(np.mean(d_vals)))

    # Aggregate according to method
    out = {"method": method, "controlled": controlled, "per_type": {}}
    for c in cats:
        if method == "units":
            if len(pooled[c]) == 0:
                mu, sd, n_units, n_layers = np.nan, np.nan, 0, 0
            else:
                all_units = np.concatenate(pooled[c])
                mu = float(all_units.mean())
                sd = float(all_units.std(ddof=1)) if all_units.size > 1 else 0.0
                n_units = int(all_units.size)
                n_layers = int(len(pooled[c]))
        elif method == "layer-mean":
            vals = per_layer_means[c]
            if len(vals) == 0:
                mu, sd, n_units, n_layers = np.nan, np.nan, 0, 0
            else:
                mu = float(np.mean(vals))
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                n_units = int(sum(len(v) for v in pooled[c])) if pooled[c] else 0
                n_layers = int(len(vals))
        else:
            raise ValueError("method must be 'units' or 'layer-mean'.")

        out["per_type"][c] = {
            "mean": mu, "sd": sd, "n_units": n_units, "n_layers": n_layers
        }

    # Pretty print
    hdr = "Grand d' across layers"
    ctrl_txt = "controlled equal-N per layer" if controlled else "uncontrolled (all available)"
    print(f"\n=== {hdr} — {method}; {ctrl_txt} ===")
    for c, label in (("face", "Face"), ("body", "Body"), ("mixed", "Mixed")):
        s = out["per_type"][c]
        if s["n_layers"] == 0:
            print(f"{label:<5}: — (no layers)")
        else:
            print(f"{label:<5}: {s['mean']:.2f} ± {s['sd']:.2f} "
                  f"(layers={s['n_layers']}, units={s['n_units']})")

def compute_mean_resp_nsd(model_dict: dict, device: str, nsd_ids: np.ndarray=None,
                          batch_size: int=128, num_workers: int=16):
    """
    For each model, compute the mean response of each selectivity type to each
    NSD image in the model's last layer.
    """
    nsd_h5_path = DATA_ROOT / "datasets" / "nsd" / "nsd_stimuli.hdf5"

    if nsd_ids is None:
        img_ids_by_subj = load_pickle(PROJECT_ROOT / "nsd" / "img_ids.pkl")
        nsd_ids = np.concatenate([np.asarray(v) for v in img_ids_by_subj.values()])
    nsd_ids = np.unique(np.asarray(nsd_ids, dtype=np.int64))

    ds = NSDImageLoader(nsd_h5_path, nsd_ids, size=(224, 224))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=_nsd_img_worker_init,
    )

    for model_name, layers in model_dict.items():
        layer = layers[-1]
        print(f"Processing {model_name} {layer}")

        ml = ModelLoader(model_name, DATA_ROOT, device)
        fe = FeatureExtractor(ml, [layer])
        with torch.no_grad():
            activs = fe.extract(dl, to_memory=True)[0]
        if not isinstance(activs, dict):
            activs = {layer: activs[0]}

        activs_sel = filter_sel_activs(
            model_name, activs, layer,
            n_top=None,
            controlled=True,
        )

        # Mean response per image per unit type
        mean_resp = {
            sel_type: X_sel.mean(axis=1)
            for sel_type, X_sel in activs_sel.items()
        }

        out_path = (
            PROJECT_ROOT / "nsd" / "models" / "mean_resp_sel" /
            f"resp_{model_name}_{layer}.npy"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, {"nsd_ids": nsd_ids, "mean_resp": mean_resp})
        print(f"Saved → {out_path}")

    ds.close()
