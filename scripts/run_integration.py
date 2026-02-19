from myutils.models import ModelLoader
from myutils.dataset import ImageDataLoader
from myutils.feature_extractor import FeatureExtractor
from facebody.integration import integration_analysis
from facebody.config import DATA_ROOT

device = "cuda:0"

# Specify model and layers
model_name = "alexnet_ecoset"
layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7"]

# Run integration analysis
il = ImageDataLoader(root=DATA_ROOT / "datasets" / "integration")
ml = ModelLoader(model_name, DATA_ROOT, device)
fe = FeatureExtractor(ml, layers)
activs, labels = fe.extract(il)
integration_analysis(model_name, layers, activs, labels, class_to_idx=il.dataset.class_to_idx)
