from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import pickle as pkl

from .alexnet_barlow_twins import BarlowTwins, AlexNetGN, AlexNetBTReadout
# ------------------------------- Preprocessing ------------------------------ #
# Standard torchvision ImageNet transform
IMAGENET_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
])

class CaffeBGRTransform:
    """PIL image -> BGR float tensor on a 0-255 scale with per-channel mean subtraction."""
    def __init__(self, mean_bgr=(91.4953, 103.8827, 131.0912)):
        self.mean_bgr = np.asarray(mean_bgr, dtype=np.float32)

    def __call__(self, img):
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)[:, :, ::-1]  # RGB -> BGR
        arr = arr.astype(np.float32) - self.mean_bgr
        return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))

    def __repr__(self):
        return f"{type(self).__name__}(mean_bgr={tuple(self.mean_bgr)})"

# VGGFace2 (Caffe-converted) transform
VGGFACE2_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    CaffeBGRTransform(),
])

def _assert_state_dict_ok(model_name: str, missing, unexpected, allow_missing=()):
    """Fail loudly on a partial weight load."""
    bad_missing = [k for k in missing
                   if not any(k.endswith(suffix) for suffix in allow_missing)]
    if bad_missing or list(unexpected):
        raise RuntimeError(
            f"{model_name}: state_dict mismatch - weights NOT fully loaded.\n"
            f"  missing ({len(bad_missing)}):    {bad_missing[:10]}\n"
            f"  unexpected ({len(list(unexpected))}): {list(unexpected)[:10]}"
        )


# --------------------------- Utilities & Mixins --------------------------- #
class ModelUtilsMixin:
    @staticmethod
    def disable_inplace_relu(module: nn.Module):
        if isinstance(module, nn.ReLU) and module.inplace:
            module.inplace = False

    @staticmethod
    def load_state_dict_static(data_dir: Path, model_name: str, prefix: str=""):
        in_path = data_dir / "models" / model_name / "state_dict.pt"
        try:
            raw = torch.load(in_path, map_location="cpu", weights_only=True)
        except Exception:
            raw = torch_load_tolerant(in_path)

        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]

        return {f"{prefix}{k.replace('module.', '')}": v for k, v in raw.items()}


# ------------------------------- Factories ------------------------------- #
class BaseModelFactory(ABC):
    preprocess = None # Standard ImageNet transform

    @abstractmethod
    def build(self):
        pass

class TorchvisionFactory(BaseModelFactory):
    """Load standard Torchvision models."""
    def __init__(self, arch: str, weights: str, n_classes: int):
        self.arch = arch
        self.weights = weights
        self.n_classes = n_classes

    def build(self):
        arch_safe = self.arch.replace("-", "_")
        weight_enum = None
        if self.weights == "imagenet":
            weight_enum = ModelLoader.IMAGENET_WEIGHTS[self.arch].DEFAULT
        return getattr(models, arch_safe)(weights=weight_enum, num_classes=self.n_classes)

class ModelZooFactory(BaseModelFactory):
    """Load model trained on Ecoset, VGGFace2, or Places365."""
    def __init__(self, data_dir: Path, arch: str, model_name: str, n_classes: int):
        self.data_dir = data_dir
        self.arch = arch
        self.model_name = model_name
        self.n_classes = n_classes

    def build(self):
        model = getattr(models, self.arch)(weights=None, num_classes=self.n_classes)
        sd = ModelUtilsMixin.load_state_dict_static(self.data_dir, self.model_name)
        model.load_state_dict(sd)
        return model

class VGGFace2Factory(BaseModelFactory):
    """ResNet-50 trained from scratch on VGGFace2 (Cao et al. 2018, Caffe-converted)."""
    MEAN_BGR = (91.4953, 103.8827, 131.0912)
    N_IDENTITIES = 8631

    def __init__(self, data_dir: Path, arch: str, model_name: str):
        self.data_dir = Path(data_dir)
        self.model_dir = self.data_dir / "models" / model_name
        self.arch = arch
        self.model_name = model_name
        self.preprocess = VGGFACE2_TF

    def _check_converted(self):
        if (self.model_dir / "state_dict.pt").is_file():
            return
        raise FileNotFoundError(
            f"No state_dict.pt in {self.model_dir}")

    def build(self):
        if self.arch != "resnet50":
            raise ValueError(f"VGGFace2Factory supports resnet50 only, got '{self.arch}'")
        self._check_converted()

        sd = ModelUtilsMixin.load_state_dict_static(self.data_dir, self.model_name)
        n_ids = infer_num_classes_from_state_dict(sd)
        if n_ids != self.N_IDENTITIES:
            raise ValueError(
                f"{self.model_name}: expected {self.N_IDENTITIES} VGGFace2 identities, "
                f"checkpoint has {n_ids}. Wrong checkpoint?")

        model = models.resnet50(weights=None, num_classes=n_ids)
        model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        missing, unexpected = model.load_state_dict(sd, strict=False)

        _assert_state_dict_ok(self.model_name, missing, unexpected,
                              allow_missing=("num_batches_tracked",))
        print(f"[VGGFace2] loaded state_dict.pt: {len(sd)} tensors, {n_ids} identities, "
              f"Caffe BGR preprocessing, Caffe stem pool")
        return model

def infer_num_classes_from_state_dict(sd: dict):
    """Infer number of classes from a classifier/readout weight matrix."""
    # Common last-layer keys
    for k in ("classifier.6.weight", "fc.6.weight", "fc.weight", "readout.weight", "linear.weight"):
        if k in sd and sd[k].ndim == 2:
            return sd[k].shape[0]

    # Generic fallback: take the last (by name) 2D weight that looks like a Linear
    candidates = [(k, v.shape[0]) for k, v in sd.items() if k.endswith(".weight") and getattr(v, "ndim", 0) == 2]
    if not candidates:
        raise RuntimeError("Could not infer n_classes: no 2D Linear weights found in state_dict.")
    candidates.sort(key=lambda kv: kv[0])
    return candidates[-1][1]

class FolderCheckpointFactory(BaseModelFactory):
    """Load torchvision arch and weights from models/<model_name>/state_dict.pt.
       n_classes is inferred from the state_dict."""
    def __init__(self, data_dir: Path, arch: str, model_name: str):
        self.data_dir = data_dir
        self.arch = arch
        self.model_name = model_name

    def build(self):
        sd = ModelUtilsMixin.load_state_dict_static(self.data_dir, self.model_name)
        n_classes = infer_num_classes_from_state_dict(sd)
        model = getattr(models, self.arch)(weights=None, num_classes=n_classes)

        model.load_state_dict(sd, strict=True)
        return model

class BarlowFactory(BaseModelFactory):
    """Load Barlow Twins models."""
    def __init__(self, data_dir: Path, backbone: nn.Module, model_name: str, n_classes: int):
        self.data_dir = data_dir
        self.backbone = backbone
        self.model_name = model_name
        self.n_classes = n_classes

    def build(self):
        model = BarlowTwins(self.backbone)
        if "untrained" not in self.model_name:
            prefix = "backbone." if "ecoset" in self.model_name else ""
            sd = ModelUtilsMixin.load_state_dict_static(self.data_dir, self.model_name, prefix)
            model.load_state_dict(sd, strict=("ecoset" not in self.model_name))
        return model

def _infer_proj_dims_from_state(projector_sd: dict):
    lin_keys = sorted(
        [k for k, v in projector_sd.items()
         if k.endswith(".weight") and v.ndim == 2 and k.split(".")[0].isdigit()],
        key=lambda k: int(k.split(".")[0])
    )
    if not lin_keys:
        raise RuntimeError("Projector state dict has no Linear weights.")
    lin_idxs = [int(k.split(".")[0]) for k in lin_keys]
    in_dim = projector_sd[lin_keys[0]].shape[1]
    outs = [projector_sd[k].shape[0] for k in lin_keys]
    return [in_dim] + outs, lin_idxs

def _build_projector_from_state(projector_sd: dict):
    dims, lin_idxs = _infer_proj_dims_from_state(projector_sd)
    last_lin_idx = lin_idxs[-1]
    has_tail_bn = any(k.split(".")[0].isdigit()
                      and int(k.split(".")[0]) == last_lin_idx + 1
                      for k in projector_sd.keys())
    layers: list[nn.Module] = []
    L = len(dims) - 1
    for i in range(L):
        layers.append(nn.Linear(dims[i], dims[i+1], bias=False))
        if i < L - 1 or has_tail_bn:
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class BarlowReadoutFactory(BaseModelFactory):
    """Load Barlow Twins readout."""
    def __init__(self, data_dir: Path, model_name: str):
        self.data_dir = data_dir
        self.model_name = model_name

    def build(self):
        path = self.data_dir / "models" / self.model_name / "state_dict_readout.pt"
        if not path.is_file():
            raise FileNotFoundError(f"No readout bundle at {path}")
        sd = torch.load(path, map_location="cpu", weights_only=True)

        alex = AlexNetGN(in_channel=3, out_dim=128, l2norm=True)
        backbone = alex.backbone

        ro_w = sd["readout.weight"]
        n_classes, in_features = ro_w.shape
        bias = "readout.bias" in sd

        projector = None
        if any(k.startswith("projector.") for k in sd.keys()):
            proj_sd = {k[len("projector."):]: v for k, v in sd.items() if k.startswith("projector.")}
            projector = _build_projector_from_state(proj_sd)
            projector.load_state_dict(proj_sd, strict=True)

        model = AlexNetBTReadout(backbone, in_features=in_features, n_classes=n_classes,
                                 projector=projector, bias=bias)
        model.load_state_dict(sd, strict=True)
        return model


# ------------------------------- Model Loader ------------------------------- #
class ModelLoader(ModelUtilsMixin):
    """Load pretrained models."""
    N_CLASSES = {
        "ecoset": 565, "imagenet": 1000,
        "vggface2": 8631, "places365": 365,
        "untrained": 1000,
    }
    TORCHVISION_WEIGHTS = {"imagenet", "untrained"}
    IMAGENET_WEIGHTS = {
        "alexnet": models.AlexNet_Weights,
        "vgg16": models.VGG16_Weights,
        "resnet18": models.ResNet18_Weights,
        "resnet50": models.ResNet50_Weights,
        "vit-b-16": models.ViT_B_16_Weights,
        "vit-l-16": models.ViT_L_16_Weights,
    }

    def __init__(self, model_name: str, data_dir: str, device: str="cuda"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.data_dir = Path(data_dir)
        self.arch, self.weights = model_name.split("_", 1)
        self.barlow = "barlow-twins" in self.model_name
        self._alexnet_backbone = AlexNetGN().backbone if self.barlow else None

        folder_ckpt = (self.data_dir / "models" / self.model_name / "state_dict.pt").is_file()

        # VGGFace2 Caffe-converted
        self.vggface2_caffe = (self.weights == "vggface2" and self.arch == "resnet50")

        if self.barlow:
            readout_bundle = self.data_dir / "models" / self.model_name / "state_dict_readout.pt"
            if readout_bundle.is_file():
                factory = BarlowReadoutFactory(self.data_dir, self.model_name)
            else:
                n_classes = self.N_CLASSES.get(self.weights, 1000)
                factory = BarlowFactory(self.data_dir, self._alexnet_backbone, self.model_name, n_classes)

        elif self.vggface2_caffe:
            factory = VGGFace2Factory(self.data_dir, self.arch, self.model_name)

        elif folder_ckpt:
            factory = FolderCheckpointFactory(self.data_dir, self.arch, self.model_name)

        elif self.weights in ["ecoset", "vggface2", "places365", "casia-webface"]:
            n_classes = self.N_CLASSES[self.weights]
            factory = ModelZooFactory(self.data_dir, self.arch, self.model_name, n_classes)

        else:
            # Catch-all. TorchvisionFactory only has real weights for 'imagenet';
            # anything else here would be a randomly initialised net masquerading as
            # a trained one, so refuse rather than return it.
            if self.weights not in self.TORCHVISION_WEIGHTS:
                raise ValueError(
                    f"{self.model_name}: no factory matched and '{self.weights}' is not "
                    f"a torchvision weight set, so this would return a randomly "
                    f"initialised {self.arch}. Check the name for a typo, or that "
                    f"{self.data_dir / 'models' / self.model_name / 'state_dict.pt'} "
                    f"exists. Use '{self.arch}_untrained' if you really want random weights."
                )
            n_classes = self.N_CLASSES.get(self.weights, 1000)
            factory = TorchvisionFactory(self.arch, self.weights, n_classes)

        self._factory = factory
        print(f"[ModelLoader] {self.model_name} -> {type(factory).__name__}")
        self.model = factory.build()
        self.model.apply(self.disable_inplace_relu)
        self.model.to(self.device).eval()
        self._raw_layers, self._pretty_layers = self._get_layer_names()
        print(f"Loaded model onto {device}")

    _ATOMIC_TYPES = (
        "TopoConv2d", "BlurPoolConv2d", "TopoBlock",
        "EncoderBlock", "ResidualAttentionBlock", "Block",
        "ConvNeXtBlock",
        )

    @staticmethod
    def _iter_named_modules(model: nn.Module, parent_name: str=""):
        """Single traversal yielding aligned (raw_name, module_type) pairs."""
        out = []
        for name, module in model.named_children():
            raw = f"{parent_name}.{name}" if parent_name else name
            mtype = type(module).__name__
            if mtype in ModelLoader._ATOMIC_TYPES:
                out.append((raw, mtype))
            elif list(module.named_children()):
                out.extend(ModelLoader._iter_named_modules(module, raw))
            else:
                out.append((raw, str(module).split("(")[0]))
        return out

    def _get_layer_names(self):
        """Get raw and pretty layer names."""
        pairs_all = self._iter_named_modules(self.model)

        # Filter out downsample projection layers
        pairs = [(r, m) for r, m in pairs_all if ".downsample." not in r]
        raw, mods = zip(*pairs) if pairs else ([], [])
        raw, mods = list(raw), list(mods)

        pretty = []
        ctr = 0
        for mod in mods:
            mod_l = mod.lower()

            # Transformer "conv" layers
            if mod_l in ("encoderblock", "residualattentionblock", "block", "convnextblock"):
                ctr += 1
                pretty.append(f"block{ctr}")

            # Conv layers
            elif "conv" in mod_l:
                ctr += 1
                pretty.append(f"conv{ctr}")

            # Normalization layers
            elif any(k in mod_l for k in ("batchnorm", "groupnorm", "layernorm")):
                pretty.append(f"batchnorm{ctr}")

            # Local response norm (AlexNet-style)
            elif "localresponsenorm" in mod_l:
                pretty.append(f"lrn{ctr}")

            # Activation and pooling layers
            elif "relu" in mod_l:
                pretty.append(f"relu{ctr}")
            elif "quickgelu" in mod_l:
                pretty.append(f"quickgelu{ctr}")
            elif "gelu" in mod_l:
                pretty.append(f"gelu{ctr}")
            elif "maxpool" in mod_l:
                pretty.append(f"maxpool{ctr}")
            elif "avgpool" in mod_l:
                pretty.append(f"avgpool{ctr}")

            # Linear, dropout, and flatten layers
            elif "dropout" in mod_l:
                pretty.append(f"dropout{ctr}")
            elif "linear" in mod_l:
                ctr += 1
                pretty.append(f"fc{ctr}")
            elif "flatten" in mod_l:
                pretty.append("flatten")

            else:
                pretty.append(mod_l)

        return raw, pretty

    @property
    def layer_mapping(self):
        """Get layer mapping."""
        return dict(zip(self._pretty_layers, self._raw_layers))

    def get_module_by_raw_name(self, raw_name: str):
        """Get module based on pretty layer name."""
        parts, mod = raw_name.split("."), self.model
        for p in parts:
            try:
                mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
            except Exception:
                raise ValueError(f"Cannot descend into {p}")
        return mod

    def _register_hook(self, layer_name: str, hook_fn):
        """Register forward hook on module based on pretty layer name."""
        raw = self.layer_mapping.get(layer_name)
        if raw is None:
            raise ValueError(f"{layer_name} not in mapping")
        return self.get_module_by_raw_name(raw).register_forward_hook(hook_fn)

    @property
    def preprocess(self):
        """Model-specific preprocessing transform."""
        factory = getattr(self, "_factory", None)
        return getattr(factory, "preprocess", None) or IMAGENET_TF


# ------------------------ Tolerant checkpoint loading ----------------------- #
class _MissingPickleObj:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass

class _TolerantUnpickler(pkl.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, ImportError, AttributeError):
            return _MissingPickleObj

class _TolerantPickleModule:
    Unpickler = _TolerantUnpickler
    Pickler = pkl.Pickler
    load = pkl.load
    loads = pkl.loads
    dump = pkl.dump
    dumps = pkl.dumps
    UnpicklingError = pkl.UnpicklingError
    HIGHEST_PROTOCOL = pkl.HIGHEST_PROTOCOL
    DEFAULT_PROTOCOL = pkl.DEFAULT_PROTOCOL

def torch_load_tolerant(path):
    """
    torch.load a checkpoint, ignoring un-importable non-tensor objects in the
    pickle stream.
    """
    return torch.load(
        path, map_location="cpu", weights_only=False,
        pickle_module=_TolerantPickleModule, encoding="latin1",
    )
