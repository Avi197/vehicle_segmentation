from ultralytics import YOLO
from detection.yolo_results import CustomResult


class Segmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, img):
        results = self.model(img)
        annotated_frame = results[0].plot()
        return annotated_frame
