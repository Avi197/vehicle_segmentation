import os

import vnd_log
from detection.yolo_seg import Segmentation
from dconfig import config_object

segmentation_model = None
try:
    import torch

    model_path = os.path.join(config_object.DATA_DIR, 'models/yolov5_eyeglass/best.pt')
    segmentation_model = Segmentation(model_path)
    vnd_log.dlog_i("Finish importing segmentation model")
except Exception as exc:
    vnd_log.dlog_e("Error importing segmentation model")
    vnd_log.dlog_e(exc)


def check_model_status():
    data = []

    if segmentation_model is not None:
        data.append({"name": "segmentation model", "status": "OK"})
    else:
        return {"status": "error", "message": "Missing segmentation model"}

    return data
