from .model_01 import Model01
from .model_02 import Model02

MODEL_REGISTRY = {"model_01": Model01, "model_02": Model02}


def build_model(model_config):
    model_class = MODEL_REGISTRY[model_config["name"]]

    return model_class(out_channels=model_config["out_channels"])
