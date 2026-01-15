import cv2
from ultralytics import YOLO 
import numpy as np 
import torch

model_path = "*model path*"
model = torch.load(model_path)

<<<<<<< HEAD
video = cv2.VideoCapture(0)
=======
model = YOLO("yolo11n.pt")
video = cv2.VideoCapture(10)

ret, frame = video.read()
>>>>>>> 47fc51adbdcd40f6c7f5f5b371a65dd01d2c5728

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
