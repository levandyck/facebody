from dataclasses import dataclass
import numpy as np

# ------------------------------- Result types ------------------------------- #
@dataclass(frozen=True)
class EncodingConfig:
    model_name: str
    layers: list
    subjects: list
    rois: list
    controlled: bool

    k_outer: int = 10
    k_inner: int = 5
    use_positive: bool = True
    alpha_grid: tuple = tuple((10.0 ** np.arange(-1, 4)).astype(float))

    n_nmf_components: int = 100

    device: str = "cuda"
    max_proc: int = 64
    batch_size_activs: int = 128
    num_workers_images: int = 32
    prefetch_factor: int = 4
    persistent_workers: bool = True

@dataclass
class VarPartResult:
    """Results from face-body variance partitioning."""
    r2_f: np.ndarray
    r2_b: np.ndarray
    r2_fb: np.ndarray
    u_f: np.ndarray
    u_b: np.ndarray
    s_fb: np.ndarray

@dataclass
class DeltaMResult:
    """Results from delta mixed analysis."""
    r2_fb: np.ndarray
    r2_fbm: np.ndarray
    delta_m: np.ndarray
