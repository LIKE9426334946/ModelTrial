from .model_01 import Model01
from .model_02 import Model02
from .model_03 import Model03

MODEL_REGISTRY = {"model_01": Model01, "model_02": Model02, "model_03": Model03}


def build_model(model_config, data_config):
    model_name = model_config["name"]
    model_class = MODEL_REGISTRY[model_name]
    kwargs = {
        "in_channels": data_config["in_channels"],
        "out_channels": (
            1 if data_config["mode"] == "binary" else data_config["num_classes"]
        ),
    }

    if model_name == "model_03":
        kwargs["image_size"] = tuple(data_config["image_size"])

    return model_class(**kwargs)
