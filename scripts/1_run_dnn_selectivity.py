"""Step 1: Find selective units using functional localizer and validate them on held-out image set."""

from facebody.config import DATA_ROOT
from facebody.model_sets import MODELS
from facebody.dnn.floc import DNNfloc, compute_sel_resp, compute_sel_dprime
from facebody.myutils.models import ModelLoader
from facebody.myutils.dataset import ImageDataLoader
from facebody.myutils.feature_extractor import FeatureExtractor

device = "cuda:0"

# Find selective units
for model, layers in MODELS.items():
    ml = ModelLoader(model, DATA_ROOT, device)

    fl = ImageDataLoader(
        root=DATA_ROOT / "datasets" / "floc",
        class_names=["face", "body", "scene", "object", "scrambled"],
        transform=ml.preprocess,
    )

    fe = FeatureExtractor(ml, layers)
    activs, labels = fe.extract(fl)

    dnn_floc = DNNfloc(
        model, layers,
        class_to_idx=fl.dataset.class_to_idx,
        single_cats=["face", "body"],
        mixed_tuples=[("face", "body")],
        solo_cats=["scene"],
    )
    dnn_floc.find_sel_units(activs, labels)

# Confirm selectivity (d')
for model, layers in MODELS.items():
    ml = ModelLoader(model, DATA_ROOT, device)

    vl = ImageDataLoader(
        root=DATA_ROOT / "datasets" / "validation",
        transform=ml.preprocess,
    )

    fe = FeatureExtractor(ml, layers)
    activs, labels = fe.extract(vl)

    compute_sel_resp(
        model, activs, labels, class_to_idx=vl.dataset.class_to_idx,
        img_db="validation",
    )

    compute_sel_dprime(
        model, activs, labels, class_to_idx=vl.dataset.class_to_idx,
        img_db="validation", baseline_cat="object",
    )
