"""Train linear readouts on frozen DNN features for face, person, and action tasks used by lesioning analysis."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets

from facebody.config import DATA_ROOT, PROJECT_ROOT
from facebody.myutils.models import ModelLoader
from facebody.myutils.dataset import BboxCroppedImageDataset
from facebody.myutils.utils import get_imagenet_transform

EPS = 1e-10

# -------------------------------- Task config ------------------------------- #
@dataclass
class DatasetSpec:
    kind: str
    root: str
    xml_dir: str=None

TASK_CONFIGS = {
    "face": dict(
        spec=DatasetSpec(
            kind="imagefolder",
            root=str(DATA_ROOT / "datasets" / "VGGFace2" / "test"),
        ),
        top_k=5,
    ),
    "person": dict(
        spec=DatasetSpec(
            kind="celeb-reid",
            root=str(DATA_ROOT / "datasets" / "celeb-reid_train"),
        ),
        top_k=5,
    ),
    "action": dict(
        spec=DatasetSpec(
            kind="stanford40",
            root=str(DATA_ROOT / "datasets" / "Stanford40Actions" / "images"),
            xml_dir=str(DATA_ROOT / "datasets" / "Stanford40Actions" / "annotations"),
        ),
        top_k=1,
    ),
}


# ------------------------------ Readout config ------------------------------ #
READOUT_CONFIGS = {
    "face": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
    "person": {
        "lr": 1e-4, 
        "weight_decay": 1e-5,
    },
    "action": {
        "lr": 1e-4,
        "weight_decay": 1e-5,
    }
}


# ------------------------- Stratified data splitter ------------------------- #
class StratifiedSplitter:
    """Stratified train/val/test splitting."""
    def __init__(self, train_frac: float=0.6, val_frac: float=0.2,
                 test_frac: float=0.2, seed: int=0):
        assert abs(train_frac + val_frac + test_frac - 1.0) < EPS
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed

    def split(self, targets: np.ndarray):
        """Return train/val/test indices."""
        rng = np.random.default_rng(self.seed)
        train_ids, val_ids, test_ids = [], [], []

        for c in np.unique(targets):
            cls_idx = np.where(targets == c)[0]
            rng.shuffle(cls_idx)

            n = len(cls_idx)
            n_train = int(round(n * self.train_frac))
            n_val = int(round(n * self.val_frac))

            train_ids.extend(cls_idx[:n_train].tolist())
            val_ids.extend(cls_idx[n_train:n_train + n_val].tolist())
            test_ids.extend(cls_idx[n_train + n_val:].tolist())

        return {"train": train_ids, "val": val_ids, "test": test_ids}


# ----------------------------- Dataset builder ------------------------------ #
class DatasetBuilder:
    """Builds datasets from specifications."""
    def __init__(self, spec: DatasetSpec):
        self.spec = spec

    def build(self, train_tfm, eval_tfm):
        """Return (classes, targets, make_split_fn)."""
        if self.spec.kind == "celeb-reid":
            return self._build_bbox_like()
        elif self.spec.kind == "stanford40":
            return self._build_stanford40()
        return self._build_imagefolder(train_tfm, eval_tfm)

    def _build_bbox_like(self):
        """Build Celeb-reID with padding."""
        base = datasets.ImageFolder(root=self.spec.root, transform=None)
        classes = base.classes
        targets = np.array(base.targets, dtype=int)

        def make_split(ids, tfm):
            return BboxCroppedImageDataset(
                base, xml_dir=None, apply_crop=True,
                transform=tfm, ids=ids
            )

        return classes, targets, make_split

    def _build_stanford40(self):
        """Build Stanford40 with bbox cropping and padding."""
        base = datasets.ImageFolder(root=self.spec.root, transform=None)
        classes = base.classes
        targets = np.array(base.targets, dtype=int)

        def make_split(ids, tfm):
            return BboxCroppedImageDataset(
                base, xml_dir=self.spec.xml_dir, apply_crop=True,
                transform=tfm, ids=ids
            )

        return classes, targets, make_split

    def _build_imagefolder(self, train_tfm, eval_tfm):
        """Build standard ImageFolder datasets."""
        full_train_ds = datasets.ImageFolder(root=self.spec.root, transform=train_tfm)
        full_eval_ds = datasets.ImageFolder(root=self.spec.root, transform=eval_tfm)

        classes = full_train_ds.classes
        targets = np.array(full_train_ds.targets, dtype=int)

        def make_split(ids, tfm):
            base_ds = full_train_ds if tfm is train_tfm else full_eval_ds
            return Subset(base_ds, ids)

        return classes, targets, make_split

def make_stratified_loaders(spec: DatasetSpec, batch_size: int=32,
                            n_workers: int=4, train_frac: float=0.6,
                            val_frac: float=0.2, test_frac: float=0.2,
                            seed: int=0):
    """Build train/val/test loaders."""
    train_tfm = eval_tfm = get_imagenet_transform()

    builder = DatasetBuilder(spec)
    classes, targets, make_split = builder.build(train_tfm, eval_tfm)

    splitter = StratifiedSplitter(train_frac, val_frac, test_frac, seed)
    split_ids = splitter.split(targets)

    train_ds = make_split(split_ids["train"], train_tfm)
    val_ds = make_split(split_ids["val"], eval_tfm)
    test_ds = make_split(split_ids["test"], eval_tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=n_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, classes, split_ids


# ------------------------- Feature extractor ---------------------------- #
class FeatureExtractor:
    """Extracts features from specific layer."""
    def __init__(self, model_loader: ModelLoader, layer: str, device: str="cuda"):
        self.model = model_loader.model.to(device).eval()
        self.device = device
        self.feat_dict = {"feats": None}

        def hook_fn(module, inp, out):
            self.feat_dict["feats"] = out.flatten(1)

        self.hook_handle = model_loader._register_hook(layer, hook_fn)
        self.feature_dim = self._infer_feature_dim()

    def _infer_feature_dim(self):
        """Infer feature dimensionality from dummy forward pass."""
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        return self.feat_dict["feats"].shape[1]

    @torch.no_grad()
    def extract(self, images):
        """Extract features from images."""
        self.feat_dict["feats"] = None
        _ = self.model(images)
        return self.feat_dict["feats"]

    def remove_hook(self):
        """Remove feature extraction hook."""
        self.hook_handle.remove()


# ------------------------- Linear readout trainer --------------------------- #
class LinearReadoutTrainer:
    """Trains linear classifier on frozen features."""
    def __init__(self, feature_extractor: FeatureExtractor, n_classes: int,
                 lr: float=1e-3, weight_decay: float=1e-4, device: str="cuda"):
        self.extractor = feature_extractor
        self.device = device
        self.classifier = nn.Linear(feature_extractor.feature_dim, n_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=weight_decay)
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        self.best_state = None
        self.best_val_acc = 0.0

    def train(self, train_loader, val_loader, test_loader, n_epochs: int=10):
        """Train classifier and return test metrics."""
        for epoch in range(n_epochs):
            train_loss, train_acc = self._run_epoch(train_loader, train_mode=True)
            val_loss, val_acc = self._run_epoch(val_loader, train_mode=False)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch + 1}/{n_epochs} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state = {k: v.cpu().clone()
                                  for k, v in self.classifier.state_dict().items()}

        if self.best_state is not None:
            self.classifier.load_state_dict(self.best_state)

        test_loss, test_acc = self._run_epoch(test_loader, train_mode=False)
        print(f"Test loss={test_loss:.4f}, test_acc={test_acc:.3f}")

        return {"loss": test_loss, "acc": test_acc}

    def _run_epoch(self, loader, train_mode: bool):
        """Run one epoch of training or evaluation."""
        self.classifier.train(mode=train_mode)
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                feats = self.extractor.extract(imgs)

            logits = self.classifier(feats)
            loss = self.criterion(logits, labels)

            if train_mode:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size

        return total_loss / total, correct / total


# ---------------------------- Training pipeline ----------------------------- #
def train_task_readout(model_name: str, feature_layer: str, device: str,
                       dataset_spec: DatasetSpec, batch_size: int=32,
                       n_workers: int=4, train_frac: float=0.6,
                       val_frac: float=0.2, test_frac: float=0.2,
                       n_epochs: int=10, lr: float=1e-3,
                       weight_decay: float=1e-4,
                       seed: int=0, out_dir: Path=PROJECT_ROOT / "lesioning"):
    """Train linear readout and save results."""
    loader = ModelLoader(model_name, data_dir=DATA_ROOT, device=device)

    train_loader, val_loader, test_loader, classes, split_ids = make_stratified_loaders(
        dataset_spec, batch_size, n_workers, train_frac, val_frac, test_frac, seed
    )

    extractor = FeatureExtractor(loader, feature_layer, device)
    trainer = LinearReadoutTrainer(extractor, len(classes), lr, weight_decay, device)
    test_metrics = trainer.train(train_loader, val_loader, test_loader, n_epochs)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trainer.classifier.state_dict(), out_dir / "linear_readout.pt")

    metadata = dict(
        model_name=model_name, feature_layer=feature_layer,
        test_metrics=test_metrics, history=trainer.history,
        classes=classes, split_ids=split_ids,
        train_frac=train_frac, val_frac=val_frac, test_frac=test_frac,
        n_epochs=n_epochs, lr=lr, weight_decay=weight_decay, seed=seed,
        dataset_spec=asdict(dataset_spec)
    )

    with (out_dir / "linear_readout_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

def load_task_readout(out_dir: Path, device: str, batch_size: int=32,
                      n_workers: int=4):
    """Load trained readout."""
    out_dir = Path(out_dir)
    with (out_dir / "linear_readout_metadata.json").open("r") as f:
        meta = json.load(f)

    spec = DatasetSpec(**meta["dataset_spec"])
    train_tfm = eval_tfm = get_imagenet_transform()

    builder = DatasetBuilder(spec)
    classes, targets, make_split = builder.build(train_tfm, eval_tfm)

    split_ids = meta["split_ids"]
    train_ds = make_split(split_ids["train"], train_tfm)
    val_ds = make_split(split_ids["val"], eval_tfm)
    test_ds = make_split(split_ids["test"], eval_tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=n_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=n_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=n_workers, pin_memory=True)

    ml = ModelLoader(meta["model_name"], data_dir=DATA_ROOT, device=device)
    extractor = FeatureExtractor(ml, meta["feature_layer"], device)

    classifier = nn.Linear(extractor.feature_dim, len(classes)).to(device)
    state = torch.load(out_dir / "linear_readout.pt", map_location=device, weights_only=True)
    classifier.load_state_dict(state)
    classifier.eval()

    return dict(
        model_loader=ml, classifier=classifier, extract_feats=extractor.extract,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        classes=classes, metadata=meta
    )
