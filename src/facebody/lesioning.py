from itertools import combinations
import numpy as np
import torch

from facebody.config import PROJECT_ROOT
from facebody.train_readouts import (
    TASK_CONFIGS, READOUT_CONFIGS,
    train_task_readout, load_task_readout
)
from myutils.utils import save_pickle, load_pickle, seed_everything

# ------------------------------- Bootstrap ---------------------------------- #
class BootstrapAnalyzer:
    """Handles bootstrap confidence intervals for lesioning analysis."""
    def __init__(self, n_iter: int=10000):
        self.n_iter = n_iter

    def mean_ci(self, values: np.ndarray, seed: int=0):
        """Bootstrap CI for mean."""
        values = np.asarray(values, dtype=float)
        rng = np.random.default_rng(seed)
        n = values.size

        boot_means = np.empty(self.n_iter, dtype=float)
        for i in range(self.n_iter):
            idx = rng.integers(0, n, size=n)
            boot_means[i] = values[idx].mean()

        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        return values.mean(), ci_low, ci_high, boot_means

    def drop_ci(self, base_flags: np.ndarray, les_flags: np.ndarray,
                rng=None):
        """Bootstrap CI for accuracy drop."""
        base_flags = np.asarray(base_flags, dtype=float)
        les_flags = np.asarray(les_flags, dtype=float)
        assert base_flags.shape == les_flags.shape

        N = base_flags.shape[0]
        rng = np.random.default_rng() if rng is None else rng

        drops = np.empty(self.n_iter, dtype=float)
        for i in range(self.n_iter):
            ids = rng.integers(0, N, size=N)
            drops[i] = base_flags[ids].mean() - les_flags[ids].mean()

        lo, hi = np.percentile(drops, [2.5, 97.5])
        return drops.mean(), lo, hi
    
    def paired_diff_ci(self, base_flags: np.ndarray, flags_a: np.ndarray, 
                       flags_b: np.ndarray, rng=None):
        """Bootstrap CI for paired difference in drops."""
        base_flags = np.asarray(base_flags, dtype=float)
        flags_a = np.asarray(flags_a, dtype=float)
        flags_b = np.asarray(flags_b, dtype=float)
        assert base_flags.shape == flags_a.shape == flags_b.shape

        N = base_flags.shape[0]
        rng = np.random.default_rng() if rng is None else rng
        
        diffs = np.empty(self.n_iter, dtype=float)
        for i in range(self.n_iter):
            idx = rng.integers(0, N, size=N)
            drop_a = base_flags[idx].mean() - flags_a[idx].mean()
            drop_b = base_flags[idx].mean() - flags_b[idx].mean()
            diffs[i] = drop_a - drop_b
        
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
        return diffs.mean(), lo, hi, float(p)
    
    def hierarch_drop(self, splits_data: list):
        """Hierarchical bootstrap across splits."""
        n_splits = len(splits_data)
        boot_drops = []

        for _ in range(self.n_iter):
            sampled_indices = np.random.choice(n_splits, size=n_splits, replace=True)
            resampled_drops = []

            for split_idx in sampled_indices:
                split_data = splits_data[split_idx]
                base = split_data["baseline_flags"]
                les = split_data["lesioned_flags"]

                n_imgs = len(base)
                img_indices = np.random.choice(n_imgs, size=n_imgs, replace=True)
                split_drop = base[img_indices].mean() - les[img_indices].mean()
                resampled_drops.append(split_drop)

            boot_drops.append(np.mean(resampled_drops))

        boot_drops = np.array(boot_drops)
        mean_drop = np.mean(boot_drops)
        ci_low, ci_high = np.percentile(boot_drops, [2.5, 97.5])
        p_val = 2 * min((boot_drops <= 0).mean(), (boot_drops >= 0).mean())

        return mean_drop, ci_low, ci_high, p_val

    def hierarch_paired(self, splits_data_a: list, splits_data_b: list):
        """Hierarchical bootstrap for paired comparison with Cohen's d."""
        n_splits = len(splits_data_a)
        assert len(splits_data_b) == n_splits

        # Compute split-level drops
        split_drops_a = []
        split_drops_b = []

        for split_idx in range(n_splits):
            # Drop for condition A
            split_a = splits_data_a[split_idx]
            base_a = split_a['baseline_flags']
            les_a = split_a['lesioned_flags']
            drop_a = base_a.mean() - les_a.mean()
            split_drops_a.append(drop_a)

            # Drop for condition B
            split_b = splits_data_b[split_idx]
            base_b = split_b['baseline_flags']
            les_b = split_b['lesioned_flags']
            drop_b = base_b.mean() - les_b.mean()
            split_drops_b.append(drop_b)

        # Compute Cohen's d from split-level differences
        split_drops_a = np.array(split_drops_a)
        split_drops_b = np.array(split_drops_b)
        split_diffs = split_drops_a - split_drops_b

        mean_diff_observed = split_diffs.mean()
        sd_diff = split_diffs.std(ddof=1)
        cohens_d = mean_diff_observed / sd_diff if sd_diff > 0 else np.nan

        # Hierarchical bootstrap for CI
        boot_diffs = []
        for _ in range(self.n_iter):
            sampled_indices = np.random.choice(n_splits, size=n_splits, replace=True)
            drops_a, drops_b = [], []

            for split_idx in sampled_indices:
                split_a = splits_data_a[split_idx]
                base_a = split_a['baseline_flags']
                les_a = split_a['lesioned_flags']
                n_imgs = len(base_a)
                img_idx = np.random.choice(n_imgs, size=n_imgs, replace=True)
                drops_a.append(base_a[img_idx].mean() - les_a[img_idx].mean())

                split_b = splits_data_b[split_idx]
                base_b = split_b['baseline_flags']
                les_b = split_b['lesioned_flags']
                drops_b.append(base_b[img_idx].mean() - les_b[img_idx].mean())

            boot_diffs.append(np.mean(drops_a) - np.mean(drops_b))

        boot_diffs = np.array(boot_diffs)
        mean_diff = np.mean(boot_diffs)
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        p_val = 2 * min((boot_diffs <= 0).mean(), (boot_diffs >= 0).mean())

        return mean_diff, ci_low, ci_high, p_val, cohens_d


# ----------------------------- Lesioning analyzer --------------------------- #
class LesioningAnalyzer:
    """Lesion selective units and measure accuracy drops."""
    def __init__(self, model_loader, classifier, extract_feats, test_loader,
                 lesion_scheme: str="controlled", top_k: int=5,
                 device: str="cuda"):
        self.model_loader = model_loader
        self.model_name = model_loader.model_name
        self.backbone = model_loader.model.to(device).eval()
        self.classifier = classifier.to(device).eval()
        self.extract_feats = extract_feats
        self.test_loader = test_loader
        self.lesion_scheme = lesion_scheme
        self.top_k = top_k
        self.device = torch.device(device)
        self.hooks = {}

        sel_path = PROJECT_ROOT / "selectivity" / self.model_name / "floc_res.pkl"
        self.sel_res = load_pickle(sel_path)
        self.bootstrap = BootstrapAnalyzer(n_iter=10000)

    def get_lesion_units(self, layer: str, sel: str):
        """Get unit ids to lesion for given layer and selectivity type."""
        sel_res_layer = self.sel_res.get(layer, None)
        if sel_res_layer is None:
            print(f"No selective units stored for layer {layer}.")
            return None

        stats = sel_res_layer["stats"]
        d_mixed = np.array(stats["mixed"]["dvals"])

        sel_types = ["face", "body", "mixed"]
        sel_ids_by_type = {
            s: np.array(sel_res_layer["unit_ids"].get(s, []), dtype=int)
            for s in sel_types
        }

        def _sorted_selective_ids(s):
            ids = sel_ids_by_type[s]
            if ids.size == 0:
                return ids
            dvals = np.array(stats[s]["dvals"])
            dvals_sel = dvals[ids]
            return ids[np.argsort(dvals_sel)[::-1]]

        if sel == "nonselective":
            all_units = np.arange(d_mixed.shape[0], dtype=int)
            if any(ids.size > 0 for ids in sel_ids_by_type.values()):
                sel_union = np.concatenate(
                    [ids for ids in sel_ids_by_type.values() if ids.size > 0]
                )
            else:
                sel_union = np.array([], dtype=int)
            nonsel_units = np.setdiff1d(all_units, sel_union)
            if nonsel_units.size == 0:
                print(f"No nonselective units found in {layer}.")
                return None

            dvals_nonsel = d_mixed[nonsel_units]
            sorted_nonsel = nonsel_units[np.argsort(dvals_nonsel)]
            return self._apply_lesion_scheme(sorted_nonsel, sel_ids_by_type)

        sel_units = sel_ids_by_type.get(sel, None)
        if sel_units is None or sel_units.size == 0:
            print(f"No {sel}-selective units found in {layer}.")
            return None

        unit_ids = _sorted_selective_ids(sel)
        return self._apply_lesion_scheme(unit_ids, sel_ids_by_type)

    def _apply_lesion_scheme(self, unit_ids: np.ndarray, sel_ids_by_type: dict):
        """Apply lesion scheme (all/controlled/topX%)."""
        unit_ids = np.asarray(unit_ids, dtype=int)

        if self.lesion_scheme == "all":
            return unit_ids

        if self.lesion_scheme == "controlled":
            counts = [sel_ids_by_type[s].size for s in ["face", "body", "mixed"]
                     if sel_ids_by_type[s].size > 0]
            if not counts:
                return unit_ids
            return unit_ids[:min(counts)]

        if self.lesion_scheme.startswith("top") and self.lesion_scheme.endswith("%"):
            perc = float(self.lesion_scheme[3:-1]) / 100.0
            n_lesion = max(1, int(unit_ids.size * perc))
            return unit_ids[:n_lesion]

        raise ValueError(f"Unknown lesion_scheme: {self.lesion_scheme}")

    def apply_hooks(self, lesion_ids_by_layer: dict):
        """Apply hooks to zero out units in specified layers."""
        self.remove_hooks()

        for layer_name, lesion_ids in lesion_ids_by_layer.items():
            lesion_ids = np.array(lesion_ids)

            def hook_fn(module, input, output, lesion_ids=lesion_ids):
                out = output.clone()
                B = out.size(0)
                flat = out.view(B, -1)
                flat[:, lesion_ids] = 0
                return out

            self.hooks[layer_name] = self.model_loader._register_hook(layer_name, hook_fn)

    def remove_hooks(self):
        """Remove all lesioning hooks."""
        for h in self.hooks.values():
            h.remove()
        self.hooks = {}

    @torch.no_grad()
    def eval_topk_flags(self, dataloader):
        """Evaluate top-k correctness per image."""
        self.backbone.eval()
        self.classifier.eval()
        all_correct = []

        for imgs, labels in dataloader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            feats = self.extract_feats(imgs).to(self.device)
            logits = self.classifier(feats)

            _, preds = torch.topk(logits, k=self.top_k, dim=1)
            match = (preds == labels.unsqueeze(1)).any(dim=1)
            all_correct.append(match.cpu().numpy().astype(np.int32))

        return np.concatenate(all_correct, axis=0)

    def run_lesioning_global(self,
                             sel_types: tuple=("face", "body", "mixed", "nonselective"),
                             layers=None):
        """Run lesioning analysis across all layers."""
        self.remove_hooks()

        baseline_flags = self.eval_topk_flags(self.test_loader)
        baseline_acc = baseline_flags.mean()

        res = {
            "baseline_topk": baseline_acc,
            "baseline_std": baseline_flags.std(ddof=1),
            "_baseline_flags": baseline_flags.copy(),
            "per_sel": {}
        }

        target_layers = sorted(self.sel_res.keys()) if layers is None else \
                       [l for l in layers if l in self.sel_res]

        lesioned_flags_by_sel = {}
        for sel in sel_types:
            print(f"Lesioning {sel}-selective units (scheme={self.lesion_scheme}) ...")

            lesion_ids_by_layer = self._collect_lesion_units(sel, target_layers)
            if not lesion_ids_by_layer:
                continue

            self.apply_hooks(lesion_ids_by_layer)
            lesioned_flags = self.eval_topk_flags(self.test_loader)
            lesioned_flags_by_sel[sel] = lesioned_flags

            mean_drop, lower, upper = self.bootstrap.drop_ci(baseline_flags, lesioned_flags)

            res["per_sel"][sel] = dict(
                baseline_topk=float(baseline_acc),
                baseline_std=float(baseline_flags.std(ddof=1)),
                lesioned_topk=float(lesioned_flags.mean()),
                _lesioned_flags=lesioned_flags.copy(),
                drop=float(mean_drop),
                drop_point=float(baseline_acc - lesioned_flags.mean()),
                ci_low=float(lower),
                ci_high=float(upper),
                ci_half=float((upper - lower) / 2.0),
                n_units=sum(len(ids) for ids in lesion_ids_by_layer.values()),
            )

            self.remove_hooks()

        res["pairwise"] = self._compute_pairwise(baseline_flags, lesioned_flags_by_sel)
        return res

    def _collect_lesion_units(self, sel, target_layers: list):
        """Collect lesion units across layers."""
        lesion_ids_by_layer = {}
        for layer_name in target_layers:
            lesion_units = self.get_lesion_units(layer_name, sel)
            if lesion_units is not None and len(lesion_units) > 0:
                lesion_ids_by_layer[layer_name] = lesion_units
        return lesion_ids_by_layer

    def _compute_pairwise(self, baseline_flags: np.ndarray,
                          lesioned_flags_by_sel: dict):
        """Compute pairwise comparisons between selectivity types."""
        pairwise = {}
        names = list(lesioned_flags_by_sel.keys())

        for i, a in enumerate(names):
            for b in names[i + 1:]:
                mean_diff, lo, hi, p = self.bootstrap.paired_diff_ci(
                    baseline_flags, lesioned_flags_by_sel[a], lesioned_flags_by_sel[b]
                )
                pairwise[(a, b)] = dict(
                    mean_diff_drop=float(mean_diff),
                    ci_low=float(lo),
                    ci_high=float(hi),
                    p_two_sided=p,
                )
                print(f"{a} vs {b}: Δdrop={mean_diff:.2f} [{lo:.2f}, {hi:.2f}], p={p:.3f}")

        return pairwise


# --------------------------- Results summarization -------------------------- #
def summarize_baseline_across_repeats(res: list):
    """Summarize baseline performance across splits."""
    bootstrap = BootstrapAnalyzer()
    baselines = np.array([r["baseline_topk"] for r in res], dtype=float)
    mean_base, ci_low, ci_high, _ = bootstrap.mean_ci(baselines)
    return dict(
        mean=float(mean_base),
        std=float(baselines.std(ddof=1)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_repeats=int(baselines.size),
    )

def summarize_single_type_hierarch(res: list, sel: str):
    """Summarize drops for one selectivity type with hierarchical bootstrap."""
    bootstrap = BootstrapAnalyzer()
    splits_data = [
        {
            'baseline_flags': r['_baseline_flags'],
            'lesioned_flags': r['per_sel'][sel]['_lesioned_flags'],
            'baseline_acc': r['baseline_topk']
        }
        for r in res if sel in r["per_sel"]
    ]

    if not splits_data:
        return None

    # Absolute drops
    mean_abs, lo_abs, hi_abs, p_abs = bootstrap.hierarch_drop(splits_data)

    # Relative drops
    splits_data_rel = [
        {
            'baseline_flags': sd['baseline_flags'] / sd['baseline_acc'],
            'lesioned_flags': sd['lesioned_flags'] / sd['baseline_acc']
        }
        for sd in splits_data
    ]
    mean_rel, lo_rel, hi_rel, p_rel = bootstrap.hierarch_drop(splits_data_rel)

    return dict(
        abs=dict(mean_drop=float(mean_abs), ci_low=float(lo_abs), ci_high=float(hi_abs),
                p_two_sided=float(p_abs), n_splits=len(splits_data)),
        rel=dict(mean_drop=float(mean_rel), ci_low=float(lo_rel), ci_high=float(hi_rel),
                p_two_sided=float(p_rel), n_splits=len(splits_data)),
    )

def summarize_pairwise_hierarch(res: list):
    """Summarize pairwise comparisons with hierarchical bootstrap."""
    bootstrap = BootstrapAnalyzer()
    sel_types = list(res[0]["per_sel"].keys())
    summary_pairs = {}

    for a, b in combinations(sel_types, 2):
        splits_a = [
            {'baseline_flags': r['_baseline_flags'],
             'lesioned_flags': r['per_sel'][a]['_lesioned_flags'],
             'baseline_acc': r['baseline_topk']}
            for r in res if a in r["per_sel"] and b in r["per_sel"]
        ]
        splits_b = [
            {'baseline_flags': r['_baseline_flags'],
             'lesioned_flags': r['per_sel'][b]['_lesioned_flags'],
             'baseline_acc': r['baseline_topk']}
            for r in res if a in r["per_sel"] and b in r["per_sel"]
        ]

        if not splits_a:
            continue

        # Absolute - now captures Cohen's d
        mean_abs, lo_abs, hi_abs, p_abs, d_abs = bootstrap.hierarch_paired(splits_a, splits_b)

        # Relative - now captures Cohen's d
        splits_a_rel = [{'baseline_flags': s['baseline_flags'] / s['baseline_acc'],
                        'lesioned_flags': s['lesioned_flags'] / s['baseline_acc']}
                       for s in splits_a]
        splits_b_rel = [{'baseline_flags': s['baseline_flags'] / s['baseline_acc'],
                        'lesioned_flags': s['lesioned_flags'] / s['baseline_acc']}
                       for s in splits_b]
        mean_rel, lo_rel, hi_rel, p_rel, d_rel = bootstrap.hierarch_paired(splits_a_rel, splits_b_rel)

        summary_pairs[(a, b)] = dict(
            abs=dict(mean_diff=float(mean_abs), ci_low=float(lo_abs), ci_high=float(hi_abs),
                    p_two_sided=float(p_abs), cohens_d=float(d_abs), n_splits=len(splits_a)),
            rel=dict(mean_diff=float(mean_rel), ci_low=float(lo_rel), ci_high=float(hi_rel),
                    p_two_sided=float(p_rel), cohens_d=float(d_rel), n_splits=len(splits_a)),
        )

    return summary_pairs

def summarize_repeated_lesioning(res: list):
    """Aggregate results across CV repeats."""
    if not res:
        return {}

    sel_types = list(res[0]["per_sel"].keys())
    summary = dict(
        baseline=summarize_baseline_across_repeats(res),
        per_sel={sel: summarize_single_type_hierarch(res, sel) for sel in sel_types},
        pairwise=summarize_pairwise_hierarch(res),
    )
    return summary

def print_lesioning_summary(model_name: str, task: str):
    """Pretty-print lesioning summary."""
    path = PROJECT_ROOT / "lesioning" / model_name / task / "lesion_global.pkl"
    summary = load_pickle(path)["summary"]

    base = summary["baseline"]
    print("Baseline performance across splits:")
    print(f"  M={base['mean']*100:.2f}, SD={base['std']*100:.2f}, "
          f"CI=[{base['ci_low']*100:.2f}, {base['ci_high']*100:.2f}], "
          f"n_repeats={base['n_repeats']}")

    print("\nLesioning drops by unit type:")
    for sel, comp in summary["per_sel"].items():
        if comp is None:
            continue
        abs_s, rel_s = comp["abs"], comp["rel"]
        print(f"  {sel:12s}: abs Δdrop={abs_s['mean_drop']*100:.2f}"
              f"[{abs_s['ci_low']*100:.2f}, {abs_s['ci_high']*100:.2f}], p={abs_s['p_two_sided']:.3f}; "
              f"rel Δdrop={rel_s['mean_drop']*100:.2f}% of baseline "
              f"[{rel_s['ci_low']*100:.2f}, {rel_s['ci_high']*100:.2f}], p={rel_s['p_two_sided']:.3f}")

    print("\nPairwise differences:")
    for (a, b), comp in summary["pairwise"].items():
        abs_s, rel_s = comp["abs"], comp["rel"]
        d_abs_str = f"{abs_s['cohens_d']:.2f}" if np.isfinite(abs_s['cohens_d']) else "--"
        d_rel_str = f"{rel_s['cohens_d']:.2f}" if np.isfinite(rel_s['cohens_d']) else "--"

        print(f"  {a:8s} vs {b:8s}: abs Δdrop={abs_s['mean_diff']*100:.2f}"
              f"[{abs_s['ci_low']*100:.2f}, {abs_s['ci_high']*100:.2f}], "
              f"d={d_abs_str}, p={abs_s['p_two_sided']:.3f}; "
              f"rel Δdrop={rel_s['mean_diff']*100:.4f}% "
              f"[{rel_s['ci_low']*100:.2f}, {rel_s['ci_high']*100:.2f}], "
              f"d={d_rel_str}, p={rel_s['p_two_sided']:.3f}")


# -------------------------- Run lesioning analysis -------------------------- #
def run_lesioning_analysis(model_name: str, readout_layer: str, device: str,
                           task: str, base_seed: int=0, n_repeats: int=10):
    """Run repeated CV lesioning analysis for a task."""
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task}")

    task_cfg = TASK_CONFIGS[task]
    dataset_spec = task_cfg["spec"]
    top_k = task_cfg["top_k"]

    readout_cfg = READOUT_CONFIGS[task]
    lr = readout_cfg["lr"]
    weight_decay = readout_cfg["weight_decay"]

    out_dir = PROJECT_ROOT / "lesioning" / model_name / task
    res = []

    for rep in range(n_repeats):
        seed = base_seed + rep
        seed_everything(seed)
        rep_out_dir = out_dir / f"rep{rep:02d}"

        # Train readout
        train_task_readout(
            model_name=model_name, feature_layer=readout_layer, device=device,
            dataset_spec=dataset_spec, batch_size=64, n_workers=8,
            train_frac=0.6, val_frac=0.2, test_frac=0.2, n_epochs=10,
            lr=lr, weight_decay=weight_decay,
            seed=seed, out_dir=rep_out_dir
        )

        # Run lesioning
        readout = load_task_readout(out_dir=rep_out_dir, device=device)
        lesion = LesioningAnalyzer(
            model_loader=readout["model_loader"],
            classifier=readout["classifier"],
            extract_feats=readout["extract_feats"],
            test_loader=readout["test_loader"],
            lesion_scheme="controlled",
            top_k=top_k,
            device=device,
        )
        lesion_res = lesion.run_lesioning_global(
            sel_types=("face", "body", "mixed", "nonselective")
        )
        lesion_res["seed"] = seed
        res.append(lesion_res)

    summary = summarize_repeated_lesioning(res)
    save_pickle(dict(repeats=res, summary=summary), out_dir / "lesion_global.pkl")

    print(f"\n=== Summary for task='{task}' ===")
    print_lesioning_summary(model_name, task)

    return res, summary
