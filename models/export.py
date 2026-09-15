import torch

from model_01 import Model01

model = Model01(1)
model.eval()

x = torch.randn(4, 3, 256, 256)
torch.onnx.export(
    model, x, "onnx/model01.onnx", input_names=["input"], output_names=["output"]
)
