from myutils.models import ModelLoader
from myutils.dataset import ImageDataLoader
from myutils.feature_extractor import FeatureExtractor
from facebody.config import DATA_ROOT
from facebody.selectivity import DNNfloc, compute_sel_dprime

device = "cuda:0"

ALEXNET_LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]
VGG16_LAYERS = [f"conv{i}" for i in range(1, 14)] + [f"fc{i}" for i in range(14, 16)]

# ------------------------------ AlexNet models ------------------------------ #
model_names = [
    "alexnet_ecoset", "alexnet_imagenet", # Supervised
    "alexnet-barlow-twins_ecoset", "alexnet-barlow-twins_imagenet", # Self-supervised
    "alexnet_untrained", "alexnet-barlow-twins_untrained", # Untrained
]

# Find selective units
floc_loader = ImageDataLoader(
    root=DATA_ROOT / "datasets" / "floc",
    class_names=["face", "body", "scene", "object", "scrambled"],
)

for model in model_names:
    ml = ModelLoader(model, DATA_ROOT, device)

    fe = FeatureExtractor(ml, ALEXNET_LAYERS)
    activs, labels = fe.extract(floc_loader)

    dnn_floc = DNNfloc(
        model, ALEXNET_LAYERS,
        class_to_idx=floc_loader.dataset.class_to_idx,
        single_cats=["face", "body"],
        mixed_tuples=[("face", "body")],
    )
    dnn_floc.find_sel_units(activs, labels)

# Confirm selectivity (d')
vl = ImageDataLoader(root=DATA_ROOT / "datasets" / "validation")

for model in model_names:
    ml = ModelLoader(model, DATA_ROOT, device)

    fe = FeatureExtractor(ml, ALEXNET_LAYERS)
    activs, labels = fe.extract(vl)

    compute_sel_dprime(model, activs, labels, class_to_idx=vl.dataset.class_to_idx,
                       img_db="validation", baseline_cat=None)


# ------------------------------- VGG16 models ------------------------------- #
model_names = [
    "vgg16_ecoset", "vgg16_imagenet", # Supervised
    "vgg16_untrained", # untrained
]

# Find selective units
floc_loader = ImageDataLoader(
    root=DATA_ROOT / "datasets" / "floc",
    class_names=["face", "body", "scene", "object", "scrambled"],
)

for model in model_names:
    ml = ModelLoader(model, DATA_ROOT, device)

    fe = FeatureExtractor(ml, VGG16_LAYERS)
    activs, labels = fe.extract(floc_loader)

    dnn_floc = DNNfloc(
        model, VGG16_LAYERS,
        class_to_idx=floc_loader.dataset.class_to_idx,
        single_cats=["face", "body"],
        mixed_tuples=[("face", "body")],
    )
    dnn_floc.find_sel_units(activs, labels)

# Confirm selectivity (d')
vl = ImageDataLoader(root=DATA_ROOT / "datasets" / "validation")

for model in model_names:
    ml = ModelLoader(model, DATA_ROOT, device)

    fe = FeatureExtractor(ml, VGG16_LAYERS)
    activs, labels = fe.extract(vl)

    compute_sel_dprime(model, activs, labels, class_to_idx=vl.dataset.class_to_idx,
                       img_db="validation", baseline_cat=None)
