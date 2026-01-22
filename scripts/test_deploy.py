import cv2
from ultralytics import YOLO
import numpy as np

model_path = "/media/nvidia/0051-D5A7/yolo11s.pt"
model = YOLO(model_path)
video = cv2.VideoCapture(10)

try:
    while True: 
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame")
            break
        result = model.predict(frame, verbose=False)
        annotated = result[0].plot()
        cv2.imshow("Fire Detection", annotated)
        cv2.waitKey(1)
except KeyboardInterrupt:
    pass
finally:
    video.release()
    cv2.destroyAllWindows()
