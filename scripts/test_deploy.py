
import cv2
from ultralytics import YOLO 
import numpy as np 
import torch

model_path = "*model path*"
model = torch.load(model_path)

video = cv2.VideoCapture(0)

try:
    while True: 
        _, frame = video.read()
        result = model.predict(frame, verbose=False)
        annotated = result[0].plot()
        cv2.imshow("Fire Detection", annotated)
        cv2.waitKey(1)
except KeyboardInterrupt:
    pass
finally:
    video.release()
    cv2.destroyAllWindows()
