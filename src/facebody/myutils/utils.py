import os
from pathlib import Path
import numpy as np
import pickle as pkl
import h5py
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# --------------------------------- File I/O --------------------------------- #
def save_pickle(obj, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("wb") as f:
        pkl.dump(obj, f)
    print(f"Saved file: {filepath}")

def load_pickle(filepath: Path):
    with filepath.open("rb") as f:
        data = pkl.load(f)
    print(f"Loaded file: {filepath}")
    return data

def load_activs_labels_h5(filepath: Path):
    with h5py.File(filepath, "r") as f:
        labels = f["labels"][:]
        activs = {
            layer: f[layer][:].astype(np.float32)
            for layer in f
            if layer != "labels"
        }
    return activs, labels


# ------------------------------ Parallelization ----------------------------- #
def submit_parallel_jobs(func, args, joblib_kwargs: dict = {"n_jobs": -1, "verbose": 10}):
    parallel = Parallel(**joblib_kwargs)
    return parallel(delayed(func)(*arg) for arg in args)


# --------------------------------- Plotting --------------------------------- #
def create_square_subplots(n_subplots: int, subplot_size: int):
    n_rows = int(np.sqrt(n_subplots))
    n_cols = int(np.ceil(n_subplots / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=[n_cols*subplot_size, n_rows*subplot_size])
    axes = np.array(axes).flatten()
    for i in range(n_subplots, n_rows*n_cols):
        fig.delaxes(axes[i])
    return fig, axes

def clean_axes(ax):
    """Remove top and right spines from axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# ----------------------------- Image transforms ----------------------------- #
def get_imagenet_transform():
    """Standard ImageNet preprocessing for pretrained models."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# ----------------------------------- Seeds ---------------------------------- #
def seed_everything(seed: int=None):
    import torch
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
