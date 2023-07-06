import time

import cv2
import numpy as np
from ultralytics import YOLO

from detection.yolo_results import CustomResult
from object_model.box_cfg import BoxCfg


def draw_3d_box(width, height):
    box_config = BoxCfg(scale=2.5)

    front_box, back_box = box_config.create_box_right(width, height)
    blank_image = np.zeros((height, width, 3), dtype=np.uint8)

    # [front_face_right[1], front_face_right[4], front_face_right[3],
    #  back_face_right[3], back_face_right[2], back_face_right[1]]
    blank_image = cv2.line(blank_image, front_box[0], front_box[3], (0, 255, 0), 2)
    blank_image = cv2.line(blank_image, front_box[3], front_box[2], (0, 255, 0), 2)
    blank_image = cv2.line(blank_image, front_box[2], back_box[2], (0, 255, 0), 2)
    blank_image = cv2.line(blank_image, back_box[2], back_box[1], (0, 255, 0), 2)
    blank_image = cv2.line(blank_image, back_box[1], back_box[0], (0, 255, 0), 2)
    blank_image = cv2.line(blank_image, back_box[0], front_box[0], (0, 255, 0), 2)

    # blank_image = cv2.line(blank_image, front_box[0], front_box[1], (0, 255, 0), 2)
    # blank_image = cv2.line(blank_image, front_box[1], front_box[2], (0, 255, 0), 2)
    # blank_image = cv2.line(blank_image, front_box[2], front_box[3], (0, 255, 0), 2)
    # blank_image = cv2.line(blank_image, front_box[3], front_box[0], (0, 255, 0), 2)
    #
    # blank_image = cv2.line(blank_image, back_box[0], back_box[1], (0, 255, 0), 2)
    # blank_image = cv2.line(blank_image, back_box[1], back_box[2], (0, 255, 0), 2)
    #
    # blank_image = cv2.line(blank_image, front_box[0], back_box[0], (0, 255, 0), 2)
    # blank_image = cv2.line(blank_image, front_box[1], back_box[1], (0, 255, 0), 2)
    # blank_image = cv2.line(blank_image, front_box[2], back_box[2], (0, 255, 0), 2)

    return blank_image


def single_img(img_path):
    model = YOLO('yolov8n-seg.pt')

    img = cv2.imread(img_path)
    blank_image = draw_3d_box(img.shape[1], img.shape[0])
    start = time.time()
    results = model(img)
    new_result = CustomResult.from_result(results[0])
    result_img = new_result.custom_plot(boxes=False)
    print(time.time() - start)
    img_box = cv2.add(result_img, blank_image)

    img_resize = cv2.resize(img_box, (460, 819))
    cv2.imshow('test', img_resize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stream(cap):
    model = YOLO('yolov8n-seg.pt')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    blank_image = draw_3d_box(width, height)
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)
            new_result = CustomResult.from_result(results[0])
            # Visualize the results on the frame
            annotated_frame = new_result.custom_plot(boxes=False)

            # frame_box = cv2.add(annotated_frame, blank_image)
            frame_box = cv2.flip(annotated_frame, 1)
            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", frame_box)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frame_box = draw_3d_box(640, 480)
    cv2.imshow("YOLOv8 Inference", frame_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cap = cv2.VideoCapture(0)
    # img = '/opt/gitlab/vehicle_segmentation/917A8351-4B39-4FAD-9263-F5A60E7BA69E.jpg'
    # real_car = '/opt/data/pti_car/VIOS1/20230103095018128021.jpg'
    # single_img(img)
    # stream(cap)
