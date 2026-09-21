"""Guided Grad-CAM saliency maps driven by the units of one selectivity type."""

from pathlib import Path
import random
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

from facebody.config import DATA_ROOT, PROJECT_ROOT
from facebody.myutils.models import ModelLoader
from facebody.myutils.feature_visualization import GuidedGradCAM
from facebody.myutils.utils import load_pickle, seed_everything

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

# ------------------------------ Unit selection ------------------------------ #
class UnitMaskBuilder:
    """Build masks over target activation tensor for set of unit ids."""
    @staticmethod
    def make(fmap: torch.Tensor, sel_ids: list):
        sel = np.asarray(list(sel_ids), dtype=int)
        if sel.size == 0:
            raise ValueError("No selective unit ids provided.")
        if fmap.ndim == 4:
            return UnitMaskBuilder._make_conv_mask(fmap, sel)
        if fmap.ndim == 2:
            return UnitMaskBuilder._make_fc_mask(fmap, sel)
        raise ValueError(f"Unsupported fmap ndim={fmap.ndim}")

    @staticmethod
    def _make_conv_mask(fmap: torch.Tensor, sel: np.ndarray):
        """Create mask for conv layer fmap [N, C, H, W]."""
        _, C, H, W = fmap.shape
        mask = torch.zeros((C, H, W), dtype=torch.bool, device=fmap.device)

        spatial_size = H * W
        ch = sel // spatial_size
        rem = sel % spatial_size
        ys = rem // W
        xs = rem % W

        valid = (ch >= 0) & (ch < C) & (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        ch, ys, xs = ch[valid], ys[valid], xs[valid]

        mask[ch, ys, xs] = True

        if not mask.any():
            raise ValueError("Conv mask is empty after filtering.")
        return mask

    @staticmethod
    def _make_fc_mask(fmap: torch.Tensor, sel: np.ndarray):
        """Create mask for fc layer fmap [N, C]."""
        _, C = fmap.shape
        mask = torch.zeros((C,), dtype=torch.bool, device=fmap.device)

        valid = (sel >= 0) & (sel < C)
        idx = torch.as_tensor(sel[valid], dtype=torch.long, device=fmap.device)
        mask[idx] = True

        if not mask.any():
            raise ValueError("FC mask is empty after filtering.")
        return mask

def _apply_controlled_matching(by_type: dict, sel_types: tuple):
    """Truncate unit types to size of smallest type."""
    min_size = min((len(by_type[s]) for s in sel_types if len(by_type[s]) > 0), default=0)
    if min_size > 0:
        return {k: v[:min_size] for k, v in by_type.items()}
    return by_type

def sort_ids_layer(model_name: str, layer: str, controlled: bool, sel_types: tuple):
    """
    Unit ids per type for one layer, sorted by d'.

    Selective types are sorted by descending d' of their own type; the non-selective
    remainder is sorted by ascending d' of the reference type (mixed, when present).
    """
    floc_res = load_pickle(PROJECT_ROOT / "selectivity" / model_name / "floc_res.pkl")
    sel_ids = floc_res[layer]["unit_ids"]
    stats = floc_res[layer]["stats"]

    ref_key = "mixed" if "mixed" in stats else next(iter(stats.keys()))
    d_ref = np.asarray(stats[ref_key]["dvals"])
    n_units = d_ref.shape[0]

    by_type = {}
    selected_all = []

    for sel in sel_types:
        ids = np.asarray(sel_ids.get(sel, []), dtype=int)
        if ids.size > 0:
            d = np.asarray(stats[sel]["dvals"])
            ids = ids[np.argsort(d[ids])[::-1]]
            selected_all.append(ids)
        by_type[sel] = ids

    sel_union = np.concatenate(selected_all) if selected_all else np.array([], dtype=int)
    all_units = np.arange(n_units, dtype=int)
    nonsel = np.setdiff1d(all_units, sel_union)

    by_type["nonselective"] = nonsel[np.argsort(d_ref[nonsel])]

    if controlled:
        by_type = _apply_controlled_matching(by_type, sel_types)

    return by_type, n_units


# ------------------------------ Guided GradCAM ------------------------------ #
def objective_from_mask(acts_tgt: torch.Tensor, mask_tgt: torch.Tensor):
    """Build scalar objective from target activations and mask."""
    return (acts_tgt * mask_tgt.unsqueeze(0)).sum()

def norm_stack_inplace(maps: dict):
    """Global min-max normalization across all provided maps (in-place)."""
    stack = np.stack(list(maps.values()), axis=0)
    gmin, gmax = float(stack.min()), float(stack.max())
    rng = (gmax - gmin) if gmax > gmin else 1.0
    for k in maps:
        maps[k] = ((maps[k] - gmin) / rng).astype(np.float32)

def normalize_maps(maps: dict, mode: str="per_type", pct: float=99):
    """Return a rescaled copy of a {sel_type: heatmap} dict."""
    if mode == "joint":
        out = dict(maps)
        norm_stack_inplace(out)
        return out
    if mode != "per_type":
        raise ValueError(f"mode must be 'per_type' or 'joint', got {mode!r}")

    out = {}
    for k, m in maps.items():
        m = m.astype(np.float32) - float(m.min())
        hi = float(np.percentile(m, pct))
        out[k] = np.clip(m / (hi if hi > 0 else 1.0), 0.0, 1.0)
    return out

class LayerResolver:
    """Resolve pretty layer names."""
    @staticmethod
    def resolve(ml: ModelLoader, layer_pretty: str, label: str):
        layer = ml.get_module_by_raw_name(ml.layer_mapping.get(layer_pretty, layer_pretty))
        if layer is None:
            raise ValueError(f"{label} layer '{layer_pretty}' not found.")
        return layer

    @staticmethod
    def resolve_cam(ml: ModelLoader, cam_layer_pretty: str):
        if cam_layer_pretty is None:
            return None
        cam_layer = ml.get_module_by_raw_name(ml.layer_mapping.get(cam_layer_pretty, cam_layer_pretty))
        if cam_layer is None:
            raise ValueError(f"CAM layer '{cam_layer_pretty}' not found.")
        if not isinstance(cam_layer, nn.Conv2d):
            raise ValueError("CAM layer must be a Conv2d.")
        return cam_layer

class ImageSampler:
    """Image folder utilities."""
    @staticmethod
    def get_img_paths(folder: str, exts: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        folder = Path(folder).expanduser()
        paths = [p for p in folder.iterdir() if p.suffix.lower() in exts]
        if not paths:
            raise FileNotFoundError(f"No images with {exts} found in {folder}")
        return sorted(paths)

    @staticmethod
    def select(folder: str, n_imgs: int):
        paths = ImageSampler.get_img_paths(folder)
        if len(paths) < n_imgs:
            raise ValueError(f"Requested {n_imgs} images but only found {len(paths)} at {folder}")
        return random.sample(paths, n_imgs)

    @staticmethod
    def load_rgb(img_path: Path):
        with Image.open(img_path) as im:
            return im.convert("RGB")

@dataclass(frozen=True)
class GuidedGradCAMAnalyzer:
    """Run Guided Grad-CAM for each unit type on a sample of images."""
    model_name: str
    target_layer_pretty: str
    cam_layer_pretty: str
    device: str
    controlled: bool
    sel_types: tuple
    imgs_path: str
    n_imgs: int
    norm_types: bool
    alpha: float
    post_blur_sigma: float
    seed: int

    def run(self):
        """One entry per image: the image itself and its heatmap per unit type."""
        seed_everything(self.seed)

        ml = ModelLoader(self.model_name, DATA_ROOT, self.device)
        model = ml.model.eval()

        target_layer = LayerResolver.resolve(ml, self.target_layer_pretty, "Target")
        cam_layer = LayerResolver.resolve_cam(ml, self.cam_layer_pretty)

        ggc = GuidedGradCAM(
            model=model,
            target_layer=target_layer,
            cam_layer=cam_layer,
            device=self.device,
            upsample_mode="bicubic",
            align_corners=False,
            post_blur_sigma=self.post_blur_sigma,
            fusion_alpha=self.alpha,
        )

        sel_by_type, _ = sort_ids_layer(
            self.model_name,
            self.target_layer_pretty,
            self.controlled,
            self.sel_types,
        )

        img_paths = ImageSampler.select(self.imgs_path, self.n_imgs)

        res = []
        for img_path in img_paths:
            pil_img = ImageSampler.load_rgb(img_path)
            img_tensor = DEFAULT_TRANSFORM(pil_img).unsqueeze(0).to(self.device)

            heatmaps = self._compute_heatmaps_for_image(ggc, img_tensor, sel_by_type)

            if self.norm_types:
                norm_stack_inplace(heatmaps)

            res.append({"pil_img": pil_img, "heatmaps": heatmaps})

        return res

    def _compute_heatmaps_for_image(self, ggc: GuidedGradCAM, img_tensor: torch.Tensor, sel_by_type: dict):
        """Heatmap per unit type for one image, each driven by that type's units."""
        heatmaps = {}
        for sel in self.sel_types:
            sel_ids = sel_by_type[sel]

            def objective_fn(acts_tgt, sel_ids=sel_ids):
                mask = UnitMaskBuilder.make(acts_tgt, sel_ids)
                return objective_from_mask(acts_tgt, mask)

            maps = ggc(img_tensor, objective_fn)
            heatmaps[sel] = maps["guided_gradcam"]
        return heatmaps

def compute_guided_gradcam(
    model_name: str,
    target_layer_pretty: str,
    cam_layer_pretty: str=None,
    device: str="cuda",
    sel_types: tuple=None,
    controlled: bool=True,
    imgs_path: str=None,
    n_imgs: int=3,
    norm_types: bool=True,
    alpha: float=0.7,
    post_blur_sigma: float=3.0,
    seed: int=0,
    ):
    """Generate Guided-GradCAM saliency maps for each unit type."""
    runner = GuidedGradCAMAnalyzer(
        model_name=model_name,
        target_layer_pretty=target_layer_pretty,
        cam_layer_pretty=cam_layer_pretty,
        device=device,
        controlled=controlled,
        sel_types=sel_types,
        imgs_path=imgs_path,
        n_imgs=n_imgs,
        norm_types=norm_types,
        alpha=alpha,
        post_blur_sigma=post_blur_sigma,
        seed=seed,
    )
    return runner.run()
