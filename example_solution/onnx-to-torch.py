import onnxruntime as ort
from onnx2torch import ConvertModel
import torch

# Define the path to your ONNX model file
onnx_model_path = "baseline.onnx"

# Load the ONNX model using the onnx library
onnx_model = ort.load(onnx_model_path)

# Convert the ONNX model to a PyTorch model
pytorch_model = ConvertModel(onnx_model)

# Set the model to evaluation mode (essential for inference, especially for BatchNorm layers)
pytorch_model.eval()

print("ONNX model successfully converted to PyTorch model.")