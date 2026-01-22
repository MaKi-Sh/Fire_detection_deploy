from ultralytics import YOLO
import onnx
import numpy as np

# Load a pre-trained or custom YOLOv11n PyTorch model
model = YOLO("/media/nvidia/0051-D5A7/yolo11n.pt")

# Export the model to ONNX format
# - opset=12: opset that OpenCV 4.8 supports
# - dynamic=False: fixed input shapes for better compatibility
# - simplify=True: simplify the model for better compatibility
# - half=False: use FP32 for better compatibility
# - batch=1: explicit batch size
path = model.export(
    format="onnx",
    imgsz=640,
    dynamic=False,
    simplify=True,
    opset=12,
    half=False,
    batch=1
)

print(f"Model exported to: {path}")

# Post-process the ONNX model to fix Conv nodes for OpenCV compatibility
print("Post-processing ONNX model for OpenCV compatibility...")
onnx_model = onnx.load(path)

modified = False
for node in onnx_model.graph.node:
    if node.op_type == "Conv":
        # Check if kernel_shape attribute exists
        has_kernel_shape = any(attr.name == "kernel_shape" for attr in node.attribute)
        if not has_kernel_shape:
            # Try to infer kernel shape from weights
            for init in onnx_model.graph.initializer:
                if init.name == node.input[1]:  # Weight tensor
                    # Weights are in OIHW format, kernel is HW
                    kernel_shape = list(init.dims[2:])
                    node.attribute.append(
                        onnx.helper.make_attribute("kernel_shape", kernel_shape)
                    )
                    print(f"Added kernel_shape {kernel_shape} to node {node.name}")
                    modified = True
                    break

if modified:
    onnx.save(onnx_model, path)
    print(f"Fixed model saved to: {path}")
else:
    print("No modifications needed")

