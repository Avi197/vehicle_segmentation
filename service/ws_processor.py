import cv2
from object_model.box_cfg import BoxCfg
from service.init_service import segmentation_model


class CarWSProcessor:
    # WARMUP_DURATION = 10
    # COUNT_FAILED = -100

    def __init__(self):
        self.box = BoxCfg
        pass

    def detect(self, data):
        results = segmentation_model.predict(data)
        return results
