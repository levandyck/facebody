from myutils.nsd import (
    download_sessions,
    prepro_img_meta,
    prepro_resp,
    define_rois,
    extract_roi_responses,
    save_subj_img_ids,
)
from facebody.encoding import EncodingPipeline, EncodingConfig

device = "cuda:0"

# Specify model, layers, subjects, and rois
model_name = "alexnet_ecoset"
layers = [f"conv{i}" for i in range(1, 6)] + [f"fc{i}" for i in range(6, 8)]
subjects = [f"subj{i:02d}" for i in range(1, 9)]
rois = ["OFA", "FFA", "aTL-faces", "EBA", "FBA", "mTL-bodies", "FFA&FBA", "V1"]

# Download and preprocess NSD fMRI data
download_sessions(subjects)
prepro_img_meta()
prepro_resp(subjects)
define_rois(subjects)
extract_roi_responses(subjects)
save_subj_img_ids(subjects)

# Set up encoding analysis
cfg = EncodingConfig(
    model_name=model_name,
    layers=layers,
    controlled=True,
    subjects=subjects,
    rois=rois,
    device=device,
    use_positive=True,
)
en = EncodingPipeline(cfg)

# Run encoding analysis
en.ensure_activations(save_to_disk=True)
en.ensure_nmf(force=True)
en.tune_alpha(force=True)
res = en.run(analyses=("sep", "fb_varpart", "delta_m"))
