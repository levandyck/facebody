import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from pathlib import Path

from facebody.myutils.models import AlexNetGN
from facebody.config import DATA_ROOT

# Projector index map (assuming projector: [Lin0, BN1, ReLU2, Lin3, BN4, ReLU5, Lin6])
TAP2IDX = {"fc6": 0, "relu6": 2, "fc7": 3, "relu7": 5}

# ------------------------------- Model pieces ------------------------------- #
class LinearProbe(nn.Module):
    def __init__(self, backbone: nn.Module, projector: nn.Sequential,
                 cut_idx: int, n_classes: int, bias: bool=False):
        super().__init__()
        self.backbone = backbone
        self.flatten = nn.Flatten()
        self.projector = None
        if projector is not None and cut_idx is not None:
            self.projector = projector[: cut_idx + 1]

        # Infer feature dim
        with torch.no_grad():
            x = torch.zeros(2, 3, 224, 224)
            h = self.flatten(self.backbone(x))
            if self.projector is not None:
                h = self.projector(h)
            in_dim = h.shape[1]
        self.readout = nn.Linear(in_dim, n_classes, bias=bias)

        # Freeze trunk and lock BN stats
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        if self.projector is not None:
            for p in self.projector.parameters():
                p.requires_grad = False
            self.projector.eval()
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):
        h = self.flatten(self.backbone(x))
        if self.projector is not None:
            h = self.projector(h)
        return self.readout(h)


# ---------------------------------- Loaders --------------------------------- #
def load_backbone(backbone_ckpt: Path):
    alex = AlexNetGN(in_channel=3, out_dim=128, l2norm=True)
    backbone = alex.backbone
    sd = torch.load(backbone_ckpt, map_location="cpu")
    backbone.load_state_dict(sd, strict=True)
    return backbone

def load_projector_if_any(projector_ckpt: Path):
    if projector_ckpt is None:
        return None
    pkg = torch.load(projector_ckpt, map_location="cpu")
    proj = nn.Sequential(
        nn.Linear(256 * 6 * 6, 4096, bias=False),
        nn.BatchNorm1d(4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 4096, bias=False),
        nn.BatchNorm1d(4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 4096, bias=False),
    )
    proj.load_state_dict(pkg["projector"], strict=True)
    return proj


# ----------------------------------- Data ----------------------------------- #
def build_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         normalize]
    )
    val_tf = transforms.Compose(
        [transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         normalize]
    )
    return train_tf, val_tf

def build_dataloaders(ecoset_root: Path, batch_size: int, num_workers: int, device: str):
    train_tf, val_tf = build_transforms()
    train_set = torchvision.datasets.ImageFolder(ecoset_root / "train", transform=train_tf)
    val_set = torchvision.datasets.ImageFolder(ecoset_root / "val", transform=val_tf)

    pin = "cuda" in device and torch.cuda.is_available()
    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    # prefetch_factor is only valid when num_workers > 0
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        prefetch_factor=(4 if num_workers > 0 else None),
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )
    return train_loader, val_loader


# ------------------------------ Loss / Metrics ------------------------------ #
def sparse_ce_loss(logits: torch.Tensor, targets: torch.Tensor, readout_weight: torch.Tensor,
                   l1_pos: float, l1_neg: float, sparse_pos: bool):
    ce = F.cross_entropy(logits, targets)
    if not sparse_pos:
        z = torch.tensor(0.0, device=logits.device)
        return ce, ce, z, z
    W = readout_weight
    l1p = l1_pos * W.clamp(min=0).abs().sum()
    l1n = l1_neg * W.clamp(max=0).abs().sum()
    return ce + l1p + l1n, ce, l1p, l1n

@torch.no_grad()
def topk_acc(logits: torch.Tensor, y: torch.Tensor, k: int=1):
    _, idx = logits.topk(k, 1, True, True)
    return idx.eq(y.view(-1, 1)).any(dim=1).float().mean().item() * 100.0


# ------------------------------- Train / Eval ------------------------------- #
def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer,
                    scaler: "torch.amp.GradScaler", device: torch.device, device_type: str,
                    sched: torch.optim.lr_scheduler._LRScheduler, l1_pos: float, l1_neg: float,
                    sparse_pos: bool, amp_enabled: bool):
    model.train() # Only readout has grads; BN already eval-locked
    tr_loss = tr_acc1 = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, enabled=amp_enabled):
            logits = model(images)
            loss, _, _, _ = sparse_ce_loss(
                logits, targets, model.readout.weight, l1_pos, l1_neg, sparse_pos
            )

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()

        bs = images.size(0)
        tr_loss += loss.item() * bs
        tr_acc1 += topk_acc(logits.detach(), targets, k=1) * bs

    n = len(loader.dataset)
    return {"loss": tr_loss / n, "acc1": tr_acc1 / n}

@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, device_type: str,
             l1_pos: float, l1_neg: float, sparse_pos: bool, amp_enabled: bool):
    model.eval()
    va_loss = va_acc1 = va_acc5 = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device_type, enabled=amp_enabled):
            logits = model(images)
            loss, *_ = sparse_ce_loss(
                logits, targets, model.readout.weight, l1_pos, l1_neg, sparse_pos
            )
        bs = images.size(0)
        va_loss += loss.item() * bs
        va_acc1 += topk_acc(logits, targets, k=1) * bs
        va_acc5 += topk_acc(logits, targets, k=5) * bs

    n = len(loader.dataset)
    return {"loss": va_loss / n, "acc1": va_acc1 / n, "acc5": va_acc5 / n}


# ------------------------------------ Run ----------------------------------- #
def train_readout(
    # Data / paths
    ecoset_root: Path=DATA_ROOT / "datasets" / "ecoset",
    backbone_ckpt: Path=DATA_ROOT / "models" / "alexnet-barlow-twins_ecoset" / "checkpoint" / "alexnet-barlow-twins_ecoset.pt",
    projector_ckpt: Path=DATA_ROOT / "models" / "alexnet-barlow-twins_ecoset" / "projector.pt",
    save_path: Path=DATA_ROOT / "models" / "alexnet-barlow-twins_ecoset" / "state_dict_readout.pt",
    # Model
    n_classes: int=565,
    tap: str="relu7", # One of: "backbone","fc6","relu6","fc7","relu7"
    bias: bool=False,
    # Train
    device: str="cuda:1",
    batch_size: int=512,
    epochs: int=10,
    max_lr: float=0.05,
    initial_lr: float=0.001,
    pct_start: float=0.3,
    num_workers: int=8,
    # Sparsity
    sparse_pos: bool=True,
    l1_pos_lambda: float=1e-4,
    l1_neg_lambda: float=1e-4,
    # Misc
    seed: int=None,
    cudnn_benchmark: bool=True,
    print_every_epoch: bool=True):
    """Train a linear readout on frozen AlexNet backbone (optionally cut through projector)."""
    assert tap in {"backbone", "fc6", "relu6", "fc7", "relu7"}, f"Invalid tap: {tap}"

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Device setup
    device_obj = torch.device(device if device != "cuda" else ("cuda" if torch.cuda.is_available() else "cpu"))
    device_type = "cuda" if ("cuda" in device and torch.cuda.is_available()) else "cpu"
    amp_enabled = device_type == "cuda"

    # cuDNN
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    # Data
    train_loader, val_loader = build_dataloaders(
        ecoset_root=ecoset_root, batch_size=batch_size, num_workers=num_workers, device=device
    )

    # Model
    backbone = load_backbone(backbone_ckpt)
    projector = None if tap == "backbone" else load_projector_if_any(projector_ckpt)
    cut_idx = None if tap == "backbone" else TAP2IDX[tap]
    model = LinearProbe(backbone, projector, cut_idx, n_classes, bias=bias).to(device_obj)

    # Optim / Sched / Scaler
    opt = torch.optim.SGD(model.readout.parameters(), lr=max_lr, momentum=0.9, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=pct_start,
        div_factor=max_lr / initial_lr,
    )
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    history = {"train": [], "val": []}
    for epoch in range(epochs):
        tr = train_one_epoch(
            model,
            train_loader,
            opt,
            scaler,
            device_obj,
            device_type,
            sched,
            l1_pos=l1_pos_lambda,
            l1_neg=l1_neg_lambda,
            sparse_pos=sparse_pos,
            amp_enabled=amp_enabled,
        )
        va = evaluate(
            model,
            val_loader,
            device_obj,
            device_type,
            l1_pos=l1_pos_lambda,
            l1_neg=l1_neg_lambda,
            sparse_pos=sparse_pos,
            amp_enabled=amp_enabled,
        )
        history["train"].append(tr)
        history["val"].append(va)
        if print_every_epoch:
            print(
                f"Epoch {epoch+1:02d}/{epochs}  "
                f"train loss {tr['loss']:.3f} top1 {tr['acc1']:.2f} | "
                f"val loss {va['loss']:.3f} top1 {va['acc1']:.2f} top5 {va['acc5']:.2f}"
            )

    # Save probe (readout only) or full probe module
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        if print_every_epoch:
            print(f"Saved to {save_path.name}")

    # Return summary
    out = {
        "final_train": history["train"][-1],
        "final_val": history["val"][-1],
        "history": history,
        "state_dict_path": str(save_path) if save_path is not None else None,
        "tap": tap,
    }
    return out
