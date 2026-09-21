from pathlib import Path
import torch
import numpy as np
import h5py
from tqdm import tqdm

# ----------------------------- Feature Extractor ---------------------------- #
class FeatureExtractor:
    """Extract activations from specified layers using forward-hooks."""
    def __init__(self, model_loader, layers: list, spatial_avg: str=None,
                 token_reduce: str=None):
        self.device = model_loader.device
        self.model = model_loader.model.to(self.device).eval()
        self.layer_mapping = model_loader.layer_mapping
        self.layers = layers
        self.spatial_avg = spatial_avg # None, "mean", "max", "l2"
        self.token_reduce = token_reduce # None, "cls", "mean", "patch_mean"
        self.clip = getattr(model_loader, "clip", False)

    def extract(self, dataloader: torch.utils.data.DataLoader,
                to_memory: bool=True, out_h5: str=None):
        """Extract unit activations."""
        feats = {layer: [] for layer in self.layers}
        labels = []

        # Register hooks
        handles = self._register_hooks(feats)

        try:
            # Forward pass and collect
            feats, labels = self._run_forward(dataloader, feats)
        finally:
            # Remove hooks
            self._remove_hooks(handles)

        # Preprocess features and labels
        feats_np: dict[str, np.ndarray] = {}
        for layer, chunks in feats.items():
            arrs = []
            for tensor in chunks:
                t = tensor.detach()
                if t.dtype == torch.bfloat16:
                    t = t.to(torch.float32)
                arr = t.cpu().numpy()
                if arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)
                arrs.append(arr)
            feats_np[layer] = np.concatenate(arrs, axis=0)
        labels_np = np.concatenate([label.detach().cpu().numpy() for label in labels], axis=0)

        # Load to memory or save to disk
        if to_memory:
            print("Loaded activations and labels into memory")
            return feats_np, labels_np
        if out_h5 is None:
            raise ValueError("out_h5 must be specified when to_memory is False")
        out_path = Path(out_h5)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_h5(feats_np, labels_np, out_path)
        print(f"Saved activations and labels to {out_path}")
        return out_path

    def _register_hooks(self, feats_dict: dict):
        """Register forward hooks."""
        handles = []
        modules = dict(self.model.named_modules())
        for layer in self.layers:
            raw = self.layer_mapping.get(layer)
            mod = modules.get(raw)
            if mod is None:
                raise KeyError(f"Layer '{layer}' not in model")
            handles.append(mod.register_forward_hook(self._make_hook(layer, feats_dict)))
        return handles

    def _make_hook(self, name: str, feats_dict: dict):
        """Make hook."""
        def hook(_, __, output):
            x = output[0] if isinstance(output, tuple) else output

            # CLIP transformer blocks are seq-first: [seq_len, batch, hidden_dim]
            if self.clip and x.ndim == 3:
                x = x.permute(1, 0, 2).contiguous()

            # Reduce transformer token sequences (N, seq, D) -> (N, D)
            if x.ndim == 3 and self.token_reduce:
                if self.token_reduce == "cls":
                    x = x[:, 0, :]
                elif self.token_reduce == "mean":
                    x = x.mean(1)
                elif self.token_reduce == "patch_mean":
                    x = x[:, 1:, :].mean(1)
                else:
                    raise ValueError(
                        "token_reduce must be None, 'cls', 'mean', or 'patch_mean'.")

            # Pool over spatial dims of conv feature maps (N, C, H, W) -> (N, C)
            if x.ndim == 4 and self.spatial_avg:
                if self.spatial_avg == "mean":
                    x = x.mean((2, 3))
                elif self.spatial_avg == "max":
                    x = x.amax((2, 3))
                elif self.spatial_avg == "l2":
                    x = x.norm(p=2, dim=(2, 3))
                else:
                    raise ValueError("spatial_avg must be None, 'mean', 'max', or 'l2'.")

            feats_dict[name].append(x.detach().cpu())
        return hook

    def _run_forward(self, dataloader: torch.utils.data.DataLoader,
                     feats_dict: dict):
        """Run forward pass."""
        labels = []
        with torch.no_grad():
            for imgs, labs in tqdm(dataloader, desc="Extracting activations"):
                labels.append(labs)
                self.model(imgs.to(self.device))
        return feats_dict, labels

    def _remove_hooks(self, handles):
        """Remove forward hooks."""
        for h in handles:
            h.remove()

    def _concatenate(self, feats_dict: dict):
        """Concatenate unit activations."""
        return {layer: torch.cat(batch, dim=0) for layer, batch in feats_dict.items()}

    def _save_h5(self, feats: np.ndarray, labels: np.ndarray, out_path: Path):
        """Save unit activations."""
        with h5py.File(out_path, "w") as f:
            for layer, arr in feats.items():
                f.create_dataset(layer, data=arr, compression="gzip")
            f.create_dataset("labels", data=labels)
