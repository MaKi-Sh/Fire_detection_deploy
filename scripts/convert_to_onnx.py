from ultralytics import YOLO
import onnx
import numpy as np

MODEL_DIR = "/media/nvidia/0051-D5A7"

MODEL_NAMES = [
    "Fire-detect_yolo11m_baseline_best",
    "Fire-detect_yolo11m_fire-recall_opt_best",
    "Fire-detect_yolo11m_recall_opt_best",
    "Fire-detect_yolo11n_baseline_best",
    "Fire-detect_yolo11n_fire-recall_opt_best",
    "Fire-detect_yolo11n_recall_opt_best",
    "Fire-detect_yolo11s_baseline_best",
    "Fire-detect_yolo11s_fire-recall_opt_best",
    "Fire-detect_yolo11s_recall_opt_best",
]

for model_name in MODEL_NAMES:
    pt_path = f"{MODEL_DIR}/{model_name}.pt"
    print(f"\n{'='*60}")
    print(f"Processing: {pt_path}")
    print(f"{'='*60}")

    # Load model
    model = YOLO(pt_path)

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
            has_kernel_shape = any(attr.name == "kernel_shape" for attr in node.attribute)
            if not has_kernel_shape:
                for init in onnx_model.graph.initializer:
                    if init.name == node.input[1]:  # Weight tensor
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

print(f"\n{'='*60}")
print("All models converted successfully!")
print(f"{'='*60}")

