"""Voxelwise encoding pipeline: Activations, NMF predictors, separate fits, variance partitioning, and permutation tests."""

import h5py
import numpy as np
from tqdm.auto import tqdm

from .types import VoxelwiseEncodingConfig
from .data import NSDRepository
from .activations import ActivationStore, ActivationExtractConfig, NSDActivationExtractor
from .predictors import NMFConfig, NMFStore, BlockConfig, BlockBuilder, fit_nmf
from .nnridge import VoxelRidgeConfig, VoxelwiseRidge
from .parallel import ParallelConfig, run_parallel
from .permute import permutation_null

from facebody.dnn.floc import filter_sel_activs
from facebody.fmri.streams import load_streams_resp

BLOCKS = ("f", "b", "m", "ns")

def _fit_nmf_task(cfg: NMFConfig, key: tuple, X: np.ndarray):
    """One NMF fit, returned with its cache key (runs in a worker process)."""
    return key, fit_nmf(cfg, X)

# ---------------------------- Voxel-wise encoding --------------------------- #
class VoxelwiseEncodingPipeline:
    """Fit every voxel in visual cortex for each subject and layer."""
    def __init__(self, cfg: VoxelwiseEncodingConfig, repo: NSDRepository=None):
        self.cfg = cfg
        self.repo = NSDRepository() if repo is None else repo
        self.img_ids = self.repo.load(self.repo.img_ids_path())

        # Activations
        self.act_store = ActivationStore(self.repo.activs_cache_dir(cfg.model_name))
        self.extractor = NSDActivationExtractor(
            self.repo, cfg.model_name,
            ActivationExtractConfig(
                layers=cfg.layers,
                batch_size=cfg.batch_size_activs,
                num_workers=cfg.num_workers_images,
                prefetch_factor=cfg.prefetch_factor,
                persistent_workers=cfg.persistent_workers,
                device=cfg.device,
            ),
        )

        # NMF and block builder
        nmf_cache_path = self.repo.nmf_cache_path(cfg.model_name)
        nmf_cache = self.repo.load(nmf_cache_path) if nmf_cache_path.exists() else {}
        self.nmf_store = NMFStore(NMFConfig(n_components=cfg.n_nmf_components), cache=nmf_cache)
        self.block_builder = BlockBuilder(BlockConfig(cfg.model_name, cfg.controlled), self.nmf_store)
        self.par_cfg = ParallelConfig(max_proc=cfg.max_proc, batch_size=1)

        # Ridge
        self.ridge = VoxelwiseRidge(
            VoxelRidgeConfig(
                k_outer=cfg.k_outer,
                k_inner=cfg.k_inner,
                alpha_grid=cfg.alpha_grid,
                vox_chunk=cfg.vox_chunk,
                device=cfg.device,
            )
        )

    # ---------------------------------- Helpers --------------------------------- #
    def _responses(self, subj: str):
        """Visual cortex responses for one subject."""
        return load_streams_resp(subj, self.repo.project_root)

    def _blocks(self, subj: str, layer: str):
        """NMF-projected predictor blocks, cached to disk."""
        tag = "ctrl" if self.cfg.controlled else "all"
        path = (self.repo.encoding_dir(self.cfg.model_name) / "blocks_cache"
                / f"{subj}_{layer}_{tag}.npz")
        if path.exists():
            z = np.load(path)
            Xf, Xb, Xm, Xns = (z[k] for k in BLOCKS)
        else:
            activs = self.act_store.get(subj)
            Xf, Xb, Xm, Xns = self.block_builder.blocks_tuple(subj, layer, activs)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **dict(zip(BLOCKS, (Xf, Xb, Xm, Xns))))

        widths = {k: X.shape[1] for k, X in zip(BLOCKS, (Xf, Xb, Xm, Xns))}
        if self.cfg.controlled and len(set(widths.values())) > 1:
            print(f"  {subj} {layer}: unequal block widths {widths} -- unit types differ "
                  f"in model capacity, not just selectivity")
        return dict(zip(BLOCKS, (Xf, Xb, Xm, Xns))), widths

    # ----------------------------------- Steps ---------------------------------- #
    def ensure_activs(self, save_to_disk: bool=False):
        """Extract activations for any subject whose cache is absent or missing a layer."""
        from dataclasses import replace as dc_replace

        need, cached_layers = [], set()
        for subj in self.cfg.subjects:
            try:
                have = self.act_store.get(subj)
            except Exception:
                need.append(subj)
                continue
            cached_layers |= set(have)
            if any(layer not in have for layer in self.cfg.layers):
                need.append(subj)

        if not need:
            return

        layers = sorted(cached_layers | set(self.cfg.layers))
        print(f"extracting activations for {need} | layers: {layers}")
        self.extractor.cfg = dc_replace(self.extractor.cfg, layers=layers)
        self.extractor.extract_and_cache(
            subjects=self.cfg.subjects,
            img_ids_by_subj=self.img_ids,
            act_store=self.act_store,
            save_to_disk=save_to_disk,
        )

    def ensure_nmf(self, force: bool=False):
        """Fit NMF predictors, in parallel across subject, layer, and unit type."""
        tasks = []
        for subj in self.cfg.subjects:
            activs = self.act_store.get(subj)
            for layer in self.cfg.layers:
                if layer not in activs:
                    continue
                sel = filter_sel_activs(self.cfg.model_name, activs, layer,
                                        controlled=self.cfg.controlled)
                for unit_type in BlockBuilder.UNIT_TYPES:
                    key = (subj, layer, unit_type)
                    if (key in self.nmf_store.cache) and (not force):
                        continue
                    tasks.append((self.nmf_store.cfg, key,
                                  sel.get(unit_type, np.zeros((activs[layer].shape[0], 0)))))

        if tasks:
            out = run_parallel(_fit_nmf_task, tasks, "Fit NMF", self.par_cfg)
            for key, model in out:
                self.nmf_store.cache[key] = model

            iters = [m.n_iter_ for _, m in out if m is not None]
            if iters:
                stuck = sum(i >= self.nmf_store.cfg.max_iter for i in iters)
                print(f"  NMF iterations: median {int(np.median(iters))}, max {max(iters)}")
                if stuck:
                    print(f"  {stuck}/{len(iters)} fits hit max_iter without converging -- "
                          f"the basis is not at an optimum and every fit pays the full "
                          f"iteration budget. Raise NMFConfig.max_iter or loosen tol.")

        self.repo.save(self.nmf_store.cache, self.repo.nmf_cache_path(self.cfg.model_name))

    # --------------------------------- Analyses --------------------------------- #
    def run_subject_layer(self, subj: str, layer: str, Y: np.ndarray):
        """
        Separate fits and variance partitioning for one subject and layer.

        Three fits per voxel: one per unit type on its own; face+body jointly, giving
        u_f (face unique, r2_fb - r2_b), u_b (body unique) and s_fb (shared,
        r2_f + r2_b - r2_fb); and face+body+mixed, giving delta_m, what mixed units add
        on top. Alphas are tuned per voxel within each fit.
        """
        blocks, widths = self._blocks(subj, layer)
        Xf, Xb, Xm = blocks["f"], blocks["b"], blocks["m"]
        nF, nB = Xf.shape[1], Xb.shape[1]

        out = {"r2": {}, "alpha": {}, "p_per_block": widths}

        # Separate fits: one alpha per voxel per unit type
        for sel in BLOCKS:
            res = self.ridge.fit(blocks[sel], Y, desc=f"{subj} {layer} | {sel}")
            out["r2"][sel] = res.r2["full"]
            out["alpha"][sel] = res.alpha

        # Variance partitioning: one alpha per voxel, tuned on fb
        Xfb = np.hstack([Xf, Xb])
        res = self.ridge.fit(Xfb, Y, sub_slices={"f": slice(0, nF), "b": slice(nF, nF + nB)},
                             full_key="fb", desc=f"{subj} {layer} | varpart")
        r2_f, r2_b, r2_fb = res.r2["f"], res.r2["b"], res.r2["fb"]
        out["r2"]["fb"] = r2_fb
        out["varpart"] = {"r2_f": r2_f, "r2_b": r2_b, "r2_fb": r2_fb,
                          "u_f": r2_fb - r2_b, "u_b": r2_fb - r2_f,
                          "s_fb": r2_f + r2_b - r2_fb}
        out["alpha"]["fb"] = res.alpha

        # Delta-M: what mixed units add on top of face and body units
        Xfbm = np.hstack([Xf, Xb, Xm])
        res = self.ridge.fit(Xfbm, Y, sub_slices={"fb": slice(0, nF + nB)}, full_key="fbm",
                             desc=f"{subj} {layer} | delta-M")
        out["delta_m"] = {"r2_fb": res.r2["fb"], "r2_fbm": res.r2["fbm"],
                          "delta_m": res.r2["fbm"] - res.r2["fb"]}
        out["r2"]["fbm"] = res.r2["fbm"]
        out["alpha"]["fbm"] = res.alpha

        if layer in self.cfg.perm_layers:
            out["perm"] = permutation_null(
                self.ridge, Xfbm, Y, res.alpha, n_perm=self.cfg.n_perm,
                desc=f"{subj} {layer} | perm",
            )
        return out

    def run(self, layers: list=None, subjects: list=None):
        """Fit every subject and layer, writing one HDF5 per layer."""
        layers = self.cfg.layers if layers is None else layers
        subjects = self.cfg.subjects if subjects is None else subjects

        missing = {s: [l for l in layers if l not in self.act_store.get(s)]
                   for s in subjects}
        missing = {s: l for s, l in missing.items() if l}
        if missing:
            raise KeyError(
                f"layers missing from the cached activations: {missing}\n"
                f"  cached for {subjects[0]}: {sorted(self.act_store.get(subjects[0]))}\n"
                f"  re-extract with: en.extractor.extract_and_cache(...) or delete "
                f"{self.repo.activs_cache_dir(self.cfg.model_name)}/*_activs.pkl "
                f"and re-run ensure_activs")

        bar = tqdm(total=len(layers) * len(subjects), desc="Encoding", unit="subj-layer")
        for layer in layers:
            path = self.repo.voxelwise_path(self.cfg.model_name, layer)
            path.parent.mkdir(parents=True, exist_ok=True)

            for subj in subjects:
                Y = self._responses(subj)
                bar.set_postfix(subj=subj, layer=layer, vox=Y.shape[1])
                res = self.run_subject_layer(subj, layer, Y)
                self._save(path, subj, res)
                bar.update(1)
        bar.close()

        return self.repo.voxelwise_path(self.cfg.model_name, layers[0]).parent

    # ---------------------------------- Storage --------------------------------- #
    def _save(self, path, subj: str, res: dict):
        """Write one subject's results, replacing any group already stored for them."""
        with h5py.File(path, "a") as f:
            if subj in f:
                del f[subj]
            g = f.create_group(subj)

            names = [*BLOCKS, "fb", "fbm"]
            g.create_dataset("r2", data=np.stack([res["r2"][k] for k in names]).astype(np.float32))
            g.create_dataset("alpha", data=np.stack([res["alpha"][k] for k in names]).astype(np.float32))
            g.attrs["blocks"] = names

            for key in ("varpart", "delta_m"):
                sub = res[key]
                g.create_dataset(key, data=np.stack(list(sub.values())).astype(np.float32))
                g[key].attrs["components"] = list(sub)

            if "perm" in res:
                for key, val in res["perm"].items():
                    g.create_dataset(f"perm/{key}", data=val)

            g.attrs["p_per_block"] = [res["p_per_block"][k] for k in BLOCKS]
            g.attrs["alpha_grid"] = np.asarray(self.cfg.alpha_grid, float)
            g.attrs["k_outer"] = self.cfg.k_outer
            g.attrs["k_inner"] = self.cfg.k_inner
            g.attrs["n_perm"] = self.cfg.n_perm
            g.attrs["model_name"] = self.cfg.model_name
            g.attrs["controlled"] = bool(self.cfg.controlled)
            g.attrs["n_nmf_components"] = self.cfg.n_nmf_components
