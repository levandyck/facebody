"""Configuration for voxel-wise encoding pipeline."""

from dataclasses import dataclass
import numpy as np

# ------------------------------- Result types ------------------------------- #
@dataclass(frozen=True)
class VoxelwiseEncodingConfig:
    """Fit every voxel in visual cortex, with ridge alpha tuned per voxel."""
    model_name: str
    layers: list
    subjects: list
    controlled: bool = True

    k_outer: int = 10
    k_inner: int = 5
    alpha_grid: tuple = tuple(np.logspace(-2, 7, 10))

    n_nmf_components: int = 100

    n_perm: int = 1000
    perm_layers: tuple = ("fc7",)

    device: str = "cuda"
    vox_chunk: int = 32768
    max_proc: int = 32
    batch_size_activs: int = 128
    num_workers_images: int = 32
    prefetch_factor: int = 4
    persistent_workers: bool = True
