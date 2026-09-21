"""Model sets."""

ALEXNET_LAYERS = [f"conv{i}" for i in range(1, 6)] + [f"fc{i}" for i in range(6, 8)]
VGG16_LAYERS = [f"conv{i}" for i in range(1, 14)] + [f"fc{i}" for i in range(14, 16)]
RESNET50_LAYERS = [f"conv{i}" for i in range(1, 50)]

# Architectures
ARCHS = {
    "alexnet": ALEXNET_LAYERS,
    "alexnet-barlow-twins": ALEXNET_LAYERS,
    "vgg16": VGG16_LAYERS,
    "resnet50": RESNET50_LAYERS,
}
DIETS = ("ecoset", "imagenet")

def model_dict(diets: tuple, archs: dict=None):
    """{f"{arch}_{diet}": layers}, architecture-major so model order stays stable."""
    archs = ARCHS if archs is None else archs
    return {f"{arch}_{diet}": layers
            for arch, layers in archs.items()
            for diet in diets}

TRAINED = model_dict(DIETS)
UNTRAINED = model_dict(("untrained",))
MODELS = {**TRAINED, **UNTRAINED}

# Encoding and lesioning analyses
ENCODING_MODEL = "alexnet_ecoset"
ENCODING_LAYERS = ["conv3", "conv4", "conv5", "fc6", "fc7"]
LESIONING_LAYER = "relu7"
LESIONING_TASKS = ["face", "person", "action"]

# Training-diet contrast
DIET_ARCHS = ("alexnet", "vgg16", "resnet50")
DIET_SUFFIX = {
    "ImageNet": "imagenet",
    "Ecoset": "ecoset",
    "VGGFace2": {"alexnet": "faces", "vgg16": "faces", "resnet50": "vggface2"},
    "Face-obfuscated": "imagenet-face-obfus",
}

def diet_dict():
    """{diet label: {model: layers}}, architectures matched across diets."""
    def suffix(spec, arch):
        return spec[arch] if isinstance(spec, dict) else spec

    return {label: {f"{arch}_{suffix(spec, arch)}": ARCHS[arch] for arch in DIET_ARCHS}
            for label, spec in DIET_SUFFIX.items()}

DIET_MODELS = {model: layers
               for models in diet_dict().values()
               for model, layers in models.items()}
