import numpy as np

from .types import EncodingConfig
from .data import NSDRepository
from .activations import ActivationStore, ActivationExtractConfig, NSDActivationExtractor
from .predictors import NMFConfig, NMFStore, BlockConfig, BlockBuilder
from .ridgecv import RidgeCVConfig, RidgeCVEngine, set_single_thread_env
from .parallel import ParallelConfig, run_parallel
from .analyses import SeparateFitsAnalysis, FaceBodyVarPartAnalysis, DeltaMAnalysis

# ------------------------------- Alpha storage ------------------------------ #
class AlphaStore:
    """Cache tuned ridge alphas."""
    def __init__(self, cache: dict=None):
        self.cache = {} if cache is None else dict(cache)

    def get(self, key: tuple, default: float=float("nan")):
        a = float(self.cache.get(key, default))
        return float(default if not np.isfinite(a) else a)

    def set(self, key: tuple, alpha: float):
        self.cache[key] = float(alpha)


# -------------------------- Main encoding analysis -------------------------- #
class EncodingPipeline:
    """Main NSD encoding pipeline."""
    def __init__(self, cfg: EncodingConfig, repo: NSDRepository=None):
        self.cfg = cfg
        self.repo = NSDRepository() if repo is None else repo

        set_single_thread_env()

        # Load NSD data
        self.fmri = self.repo.load(self.repo.fmri_path())
        self.img_ids = self.repo.load(self.repo.img_ids_path())

        # Activations
        self.act_store = ActivationStore(self.repo.activs_cache_dir())
        act_cfg = ActivationExtractConfig(
            layers=cfg.layers,
            batch_size=cfg.batch_size_activs,
            num_workers=cfg.num_workers_images,
            prefetch_factor=cfg.prefetch_factor,
            persistent_workers=cfg.persistent_workers,
            device=cfg.device,
        )
        self.extractor = NSDActivationExtractor(self.repo, cfg.model_name, act_cfg)

        # NMF and block builder
        nmf_cache_path = self.repo.nmf_cache_path(cfg.model_name)
        nmf_cache = self.repo.load(nmf_cache_path) if nmf_cache_path.exists() else {}
        self.nmf_store = NMFStore(NMFConfig(n_components=cfg.n_nmf_components), cache=nmf_cache)
        self.block_builder = BlockBuilder(BlockConfig(cfg.model_name, cfg.controlled), self.nmf_store)

        # Ridge CV
        self.ridge = RidgeCVEngine(
            RidgeCVConfig(
                k_outer=cfg.k_outer,
                k_inner=cfg.k_inner,
                alpha_grid=cfg.alpha_grid,
                use_positive=cfg.use_positive,
            )
        )

        # Alpha cache
        alpha_path = self.repo.alpha_cache_path(cfg.model_name)
        alpha_cache = self.repo.load(alpha_path) if alpha_path.exists() else {}
        self.alpha_store = AlphaStore(alpha_cache)

        # Parallel processing
        self.par_cfg = ParallelConfig(max_proc=cfg.max_proc, batch_size=8)

        # Main analyses
        self.sep = SeparateFitsAnalysis()
        self.vp = FaceBodyVarPartAnalysis()
        self.dm = DeltaMAnalysis()

    # ---------------------------------- Helpers --------------------------------- #
    def _folds(self, y: np.ndarray):
        return self.ridge.folds(y.shape[0])

    def _iter_subj_roi(self, layer: str):
        for subj in self.cfg.subjects:
            activs = self.act_store.get(subj)
            if layer not in activs:
                continue
            for roi in self.cfg.rois:
                roi_data = self.fmri[subj][roi]
                if roi_data is None:
                    continue
                y = roi_data["responses"]
                yield subj, roi, y, self._folds(y)

    # ----------------------------------- Steps ---------------------------------- #
    def ensure_activs(self, save_to_disk: bool=False):
        missing = []
        for subj in self.cfg.subjects:
            try:
                self.act_store.get(subj)
            except Exception:
                missing.append(subj)

        if missing:
            self.extractor.extract_and_cache(
                subjects=self.cfg.subjects,
                img_ids_by_subj=self.img_ids,
                act_store=self.act_store,
                save_to_disk=save_to_disk,
            )

    def ensure_nmf(self, force: bool=False):
        for subj in self.cfg.subjects:
            activs = self.act_store.get(subj)
            for layer in self.cfg.layers:
                if layer in activs:
                    self.block_builder.fit_nmf_subject_layer(subj, layer, activs, force=force)

        self.repo.save(self.nmf_store.cache, self.repo.nmf_cache_path(self.cfg.model_name))

    def tune_alpha(self, force: bool=False):
        tasks = []

        for layer in self.cfg.layers:
            for subj in self.cfg.subjects:
                activs = self.act_store.get(subj)
                if layer not in activs:
                    continue
                blocks = self.block_builder.block_dict(subj, layer, activs)

                for roi in self.cfg.rois:
                    roi_data = self.fmri[subj][roi]
                    if roi_data is None:
                        continue

                    y = roi_data["responses"]
                    outer_folds = self._folds(y)

                    for sel, X in blocks.items():
                        key = (subj, roi, layer, sel)

                        if X.shape[1] == 0:
                            self.alpha_store.set(key, float("nan"))
                            continue
                        if (key in self.alpha_store.cache) and (not force):
                            continue

                        tasks.append((key, X, y, outer_folds))

        def _tune_task(key, X, y, outer_folds):
            alpha = self.ridge.tune_alpha_nested(X, y, outer_folds, use_positive=self.cfg.use_positive)
            return key, alpha

        out = run_parallel(_tune_task, tasks, "Tune alpha", self.par_cfg)
        for k, a in out:
            self.alpha_store.set(k, a)

        self.repo.save(self.alpha_store.cache, self.repo.alpha_cache_path(self.cfg.model_name))

    # --------------------------------- Analyses --------------------------------- #
    def run_sep(self, layer: str):
        tasks = []
        for subj, roi, y, folds in self._iter_subj_roi(layer):
            activs = self.act_store.get(subj)
            blocks = self.block_builder.block_dict(subj, layer, activs)
            for sel in ["f", "b", "m", "ns"]:
                X = blocks[sel]
                alpha = self.alpha_store.get((subj, roi, layer, sel))
                tasks.append((subj, roi, sel, X, y, folds, alpha))

        def _fit(subj, roi, sel, X, y, folds, alpha):
            r2 = self.ridge.cv_r2(X, y, folds, alpha, use_positive=self.cfg.use_positive)
            return subj, roi, sel, r2

        out = run_parallel(_fit, tasks, f"{layer} | SEP", self.par_cfg)

        res = {s: {r: None for r in self.cfg.rois} for s in self.cfg.subjects}
        for subj, roi, sel, r2 in out:
            if res[subj][roi] is None:
                res[subj][roi] = {"r2": {}}
            res[subj][roi]["r2"][sel] = r2
        return res

    def run_fb_varpart(self, layer: str):
        tasks = []
        for subj, roi, y, folds in self._iter_subj_roi(layer):
            activs = self.act_store.get(subj)
            Xf, Xb, _, _ = self.block_builder.blocks_tuple(subj, layer, activs)
            alpha_fb = self.alpha_store.get((subj, roi, layer, "fb"))
            tasks.append((subj, roi, Xf, Xb, y, folds, alpha_fb))

        def _fit(subj, roi, Xf, Xb, y, folds, alpha_fb):
            vp = self.vp.run_one(Xf, Xb, y, folds, alpha_fb, use_positive=self.cfg.use_positive)
            return subj, roi, vp

        out = run_parallel(_fit, tasks, f"{layer} | FB varpart", self.par_cfg)

        res = {s: {r: None for r in self.cfg.rois} for s in self.cfg.subjects}
        for subj, roi, vp in out:
            res[subj][roi] = {
                "r2_f": vp.r2_f,
                "r2_b": vp.r2_b,
                "r2_fb": vp.r2_fb,
                "u_f": vp.u_f,
                "u_b": vp.u_b,
                "s_fb": vp.s_fb,
            }
        return res

    def run_delta_m(self, layer: str):
        tasks = []
        for subj, roi, y, folds in self._iter_subj_roi(layer):
            activs = self.act_store.get(subj)
            Xf, Xb, Xm, _ = self.block_builder.blocks_tuple(subj, layer, activs)
            alpha_fbm = self.alpha_store.get((subj, roi, layer, "fbm"))
            tasks.append((subj, roi, Xf, Xb, Xm, y, folds, alpha_fbm))

        def _fit(subj, roi, Xf, Xb, Xm, y, folds, alpha_fbm):
            dm = self.dm.run_one(Xf, Xb, Xm, y, folds, alpha_fbm, use_positive=self.cfg.use_positive)
            return subj, roi, dm

        out = run_parallel(_fit, tasks, f"{layer} | ΔM", self.par_cfg)

        res = {s: {r: None for r in self.cfg.rois} for s in self.cfg.subjects}
        for subj, roi, dm in out:
            res[subj][roi] = {"r2_fb": dm.r2_fb, "r2_fbm": dm.r2_fbm, "delta_m": dm.delta_m}
        return res

    def run(self, analyses: tuple=("sep", "fb_varpart", "delta_m"),
            save_intermediates: bool=True):
        self.ensure_activs(save_to_disk=save_intermediates)
        self.ensure_nmf(force=False)
        self.tune_alpha(force=False)

        res = {"sep": {}, "fb_varpart": {}, "delta_m": {}}
        for layer in self.cfg.layers:
            if "sep" in analyses:
                res["sep"][layer] = self.run_sep(layer)
            if "fb_varpart" in analyses:
                res["fb_varpart"][layer] = self.run_fb_varpart(layer)
            if "delta_m" in analyses:
                res["delta_m"][layer] = self.run_delta_m(layer)

        for analysis, data in res.items():
            if data:
                self.repo.merge_save(
                    data, self.repo.results_path(self.cfg.model_name, analysis)
                )

        self.repo.save(
            self.alpha_store.cache, self.repo.alpha_cache_path(self.cfg.model_name)
        )

        return res
