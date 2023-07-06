import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils import LOGGER, SimpleClass, deprecation_warn, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.engine.results import Results
from PIL import Image, ImageDraw

from object_model.box_cfg import BoxCfg

box = BoxCfg(scale=2.5)


def check_point_in_polygon(point, polygon):
    edges = list(zip(polygon, polygon[1:] + polygon[:1]))
    xp = point[0]
    yp = point[1]
    cnt = 0
    for edge in edges:
        (x1, y1), (x2, y2) = edge
        if (yp < y1) != (yp < y2) and xp < x1 + ((yp - y1) / (y2 - y1)) * (x2 - x1):
            cnt += 1
    return cnt % 2 == 1


def check_inside(masks):
    # vertices = np.array(
    #     [front_face_right[0], front_face_right[3], front_face_right[2],
    #      back_face_right[2], back_face_right[1], back_face_right[0]])
    # grid = np.zeros((480, 640))
    # mask_polygon = np.array(vertices).transpose()
    if masks is not None:
        front_face_right, back_face_right = box.create_box_right(masks.shape[1], masks.shape[2])
        polygon = [front_face_right[0], front_face_right[3], front_face_right[2],
                   back_face_right[2], back_face_right[1], back_face_right[0]]
        img = Image.new('L', (masks.shape[2], masks.shape[1]), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
        mask_polygon = np.array(img)

        for mask in masks:
            mask_area = mask.cpu().numpy()
            area = mask_area / 255 + mask_polygon
            area = np.where(area > 0, 1, 0)

            if np.concatenate(area).sum() == np.concatenate(mask_polygon).sum():
                # and np.concatenate(mask_area) > np.concatenate(mask_polygon) * 50 %
                return mask


def check_collision(masks):
    vertices = np.array(
        [front_face_right[0], front_face_right[3], front_face_right[2],
         back_face_right[2], back_face_right[1], back_face_right[0]])
    for mask in masks:
        all_point = []
        mask_points = np.array(np.where(mask.data.cpu() == 1))
        mask_points = list(zip(mask_points[0], mask_points[1]))
        for point in mask_points:
            is_inside = check_point_in_polygon(point, vertices)
            if is_inside:
                all_point.append(is_inside)
        if len(all_point) < len(mask_points):
            return mask
    return None


def remove_none_select(cls_list, cls_select, boxes, masks):
    if masks:
        masks = masks.data
        boxes = boxes.data

        # Remove all values not equal to 2 from the list
        indices_to_remove = [i for i, x in enumerate(list(map(int, cls_list))) if x != cls_select]
        # lst = [x for x in idx if x == cls_select]
        if len(indices_to_remove) > 0 and len(indices_to_remove) != len(cls_list):
            # Remove the corresponding layers from the tensor
            new_boxes = torch.cat([boxes[i:i + 1] for i in range(boxes.shape[0]) if i not in indices_to_remove], dim=0)
            new_masks = torch.cat([masks[i:i + 1] for i in range(masks.shape[0]) if i not in indices_to_remove], dim=0)
            return new_boxes, new_masks
    return None, None


def remove_small_mask():
    pass


class CustomResult(Results):
    @classmethod
    def from_result(cls, data: Results):
        new_instance = object.__new__(cls)
        for key, value in data.__dict__.items():
            setattr(new_instance, key, value)
        return new_instance

    def custom_plot(
            self,
            conf=True,
            line_width=None,
            font_size=None,
            font='Arial.ttf',
            pil=False,
            img=None,
            img_gpu=None,
            kpt_line=True,
            labels=True,
            boxes=True,
            masks=True,
            probs=True,
            check_box=None,
            **kwargs  # deprecated args TODO: remove support in 8.2
    ):
        """
        Plots the detection results on an input RGB image. Accepts a numpy array (cv2) or a PIL Image.

        Args:
            conf (bool): Whether to plot the detection confidence score.
            line_width (float, optional): The line width of the bounding boxes. If None, it is scaled to the image size.
            font_size (float, optional): The font size of the text. If None, it is scaled to the image size.
            font (str): The font to use for the text.
            pil (bool): Whether to return the image as a PIL Image.
            img (numpy.ndarray): Plot to another image. if not, plot to original image.
            img_gpu (torch.Tensor): Normalized image in gpu with shape (1, 3, 640, 640), for faster mask plotting.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot the label of bounding boxes.
            boxes (bool): Whether to plot the bounding boxes.
            masks (bool): Whether to plot the masks.
            probs (bool): Whether to plot classification probability

        Returns:
            (numpy.ndarray): A numpy array of the annotated image.
        """
        if img is None and isinstance(self.orig_img, torch.Tensor):
            LOGGER.warning('WARNING ⚠️ Results plotting is not supported for torch.Tensor image types.')
            return

        # Deprecation warn TODO: remove in 8.2
        if 'show_conf' in kwargs:
            deprecation_warn('show_conf', 'conf')
            conf = kwargs['show_conf']
            assert type(conf) == bool, '`show_conf` should be of boolean type, i.e, show_conf=True/False'

        if 'line_thickness' in kwargs:
            deprecation_warn('line_thickness', 'line_width')
            line_width = kwargs['line_thickness']
            assert type(line_width) == int, '`line_width` should be of int type, i.e, line_width=3'

        names = self.names
        # origin_img = deepcopy(img)
        img = cv2.add(self.orig_img, draw_3d_box(self.orig_img.shape[1], self.orig_img.shape[0]))
        self.orig_img = img
        annotator = Annotator(deepcopy(self.orig_img if img is None else img),
                              line_width,
                              font_size,
                              font,
                              pil,
                              example=names)

        # pred_boxes, show_boxes = self.boxes, boxes
        # pred_masks, show_masks = self.masks, masks
        # pred_probs, show_probs = self.probs, probs
        # keypoints = self.keypoints

        # if pred_masks and show_masks:
        if len(self.boxes.data) > 0 and self.masks is not None:
            idx_og = self.boxes.cls if self.boxes else range(len(self.masks))
            new_boxes, new_masks = remove_none_select(idx_og, 2, self.boxes, self.masks)
            start = time.time()

            new_masks = check_inside(new_masks)
            print(f'post process {time.time() - start}')
            if new_boxes is not None and new_masks is not None:

                self.update(new_boxes, new_masks)
                pred_boxes, show_boxes = self.boxes, boxes
                pred_masks, show_masks = self.masks, masks
                # if pred_boxes is not None and pred_masks is not None:
                if img_gpu is None:
                    img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                    img_gpu = torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device).permute(
                        2, 0, 1).flip(0).contiguous() / 255
                idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
                annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=img_gpu)
            # if masks_temp is None:
            #     annotator.im = origin_img

        return annotator.result()


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
