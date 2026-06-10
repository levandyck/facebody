from myutils.models import ModelLoader
from myutils.dataset import ImageDataLoader
from myutils.feature_extractor import FeatureExtractor
from facebody.config import DATA_ROOT
from facebody.selectivity import DNNfloc, compute_sel_dprime

device = "cuda:0"

ALEXNET_LAYERS = [f"conv{i}" for i in range(1, 6)] + [f"fc{i}" for i in range(6, 8)]
VGG16_LAYERS = [f"conv{i}" for i in range(1, 14)] + [f"fc{i}" for i in range(14, 16)]
RESNET50_LAYERS = [f"conv{i}" for i in range(1, 50)]

# Specify models and layers
MODEL_DICT = {
    "alexnet_ecoset": ALEXNET_LAYERS,
    "alexnet_imagenet": ALEXNET_LAYERS,
    "alexnet_untrained": ALEXNET_LAYERS,

    "alexnet-barlow-twins_ecoset": ALEXNET_LAYERS,
    "alexnet-barlow-twins_imagenet": ALEXNET_LAYERS,
    "alexnet-barlow-twins_untrained": ALEXNET_LAYERS,

    "vgg16_ecoset": VGG16_LAYERS,
    "vgg16_imagenet": VGG16_LAYERS,
    "vgg16_untrained": VGG16_LAYERS,

    "resnet50_ecoset": RESNET50_LAYERS,
    "resnet50_imagenet": RESNET50_LAYERS,
    "resnet50_untrained": RESNET50_LAYERS,
}

# Find selective units
floc_loader = ImageDataLoader(
    root=DATA_ROOT / "datasets" / "floc",
    class_names=["face", "body", "scene", "object", "scrambled"],
)

for model, layers in MODEL_DICT.items():
    ml = ModelLoader(model, DATA_ROOT, device)

    fe = FeatureExtractor(ml, layers)
    activs, labels = fe.extract(floc_loader)

    dnn_floc = DNNfloc(
        model, layers,
        class_to_idx=floc_loader.dataset.class_to_idx,
        single_cats=["face", "body"],
        mixed_tuples=[("face", "body")],
    )
    dnn_floc.find_sel_units(activs, labels)

# Confirm selective units
vl = ImageDataLoader(root=DATA_ROOT / "datasets" / "validation")

for model, layers in MODEL_DICT.items():
    ml = ModelLoader(model, DATA_ROOT, device)

    fe = FeatureExtractor(ml, layers)
    activs, labels = fe.extract(vl)

    compute_sel_dprime(
        model, activs, labels, class_to_idx=vl.dataset.class_to_idx,
        img_db="validation", baseline_cat="object",
    )
