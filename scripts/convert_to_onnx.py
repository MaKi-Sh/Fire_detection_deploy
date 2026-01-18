from ultralytics import YOLO

# Load a pre-trained or custom YOLOv11n PyTorch model
model = "\media\nvidia\0051-0DA51"

# Export the model to ONNX format
# The 'dynamic=True' argument enables dynamic input shapes,
# allowing different batch sizes, image sizes, etc.
path = model.export(format="onnx", dynamic=True)

print(f"Model exported to: {path}")

