"""Step 5: Extract visual cortex responses and fit voxelwise encoding models."""

from facebody.config import PROJECT_ROOT, SUBJECTS
from facebody.model_sets import ENCODING_MODEL, ENCODING_LAYERS
from facebody.fmri.streams import ROIS, ROIS_COMPLETE, extract_streams_resp
from facebody.encoding import VoxelwiseEncodingPipeline, VoxelwiseEncodingConfig

device = "cuda:0"

# Extract responses for entire visual cortex
extract_streams_resp(
    SUBJECTS, ROIS, ROIS_COMPLETE,
    proj_dir=PROJECT_ROOT,
)

# Fit voxel-wise ridge regression
cfg = VoxelwiseEncodingConfig(
    ENCODING_MODEL, ENCODING_LAYERS, SUBJECTS,
    device=device,
    perm_layers=("fc7",),
    n_perm=1000,
)
en = VoxelwiseEncodingPipeline(cfg)
en.ensure_activs(save_to_disk=True)
en.ensure_nmf(force=False)
en.run()
