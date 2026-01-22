import cv2
from ultralytics import YOLO

model_path = "/media/nvidia/0051-D5A7/yolo11n.pt"
image_path = "/home/nvidia/Downloads/Fire.png"

model = YOLO(model_path)

image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not load image '{image_path}'")
    exit(1)

results = model.predict(image, verbose=False)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        print(f"Detected: {class_name} | Confidence: {confidence:.2f} | Box: ({x1}, {y1}) to ({x2}, {y2})")

cv2.imshow("Fire Detection - Press any key to close", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
