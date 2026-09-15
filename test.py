import yaml

with open("./config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print(config["datasets"]["root"])
print(config["datasets"]["num_classes"])
print(type(config["datasets"]["image_size"]))