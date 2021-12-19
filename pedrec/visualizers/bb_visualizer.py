from typing import List

import cv2
import numpy as np

from pedrec.models.constants.class_mappings import COCO_CLASSES
from pedrec.models.constants.color_palettes import DETECTION_COLOR_PALETTE
from pedrec.utils.bb_helper import get_bb_class_idx, get_img_coordinates_from_bb


def draw_bb(img: np.ndarray, bb: List[float], title: str = None, thickness: int = 1,
            text_size: int = 1, text_thickness: int = 1):
    bb_tl_x, bb_tl_y, bb_br_x, bb_br_y = get_img_coordinates_from_bb(bb)
    # cls_conf = get_bb_score(bb)
    cls_id = get_bb_class_idx(bb)
    color = DETECTION_COLOR_PALETTE[cls_id].tuple_rgb
    top_left_tuple = (bb_tl_x, bb_tl_y)
    bottom_right_tuple = (bb_br_x, bb_br_y)
    cv2.rectangle(img, top_left_tuple, bottom_right_tuple, color, thickness)
    box_title = COCO_CLASSES[cls_id]
    if title is not None:
        box_title = title
    text_box_size = cv2.getTextSize(box_title, cv2.FONT_HERSHEY_PLAIN, text_size, text_thickness)[0]
    text_box_top_left = bb_tl_x, bb_tl_y - text_box_size[1] - 4
    text_box_bottom_right = (bottom_right_tuple[0], top_left_tuple[1])
    cv2.rectangle(img, text_box_top_left, text_box_bottom_right, color, -1)
    cv2.putText(img, box_title, (bb_tl_x, bb_tl_y - 4), cv2.FONT_HERSHEY_PLAIN,
                text_size, [225, 255, 255], text_thickness)
    return img


def draw_bbs(img: np.ndarray, bbs: np.ndarray, title: str = None, thickness: int = 1,
             text_size: int = 1, text_thickness: int = 1):
    """
    Draws the given bounding boxes in the image
    :param img: The image in which the bounding box should be drawn
    :param bbs: The bounding box list with bbs: (bb_center_x, bb_center_y, bb_width, bb_height, conf, class_id)
    :param title: The title of the bounding box (if none == class name)
    :param thickness: The thickness of the bb
    :param text_size:
    :param text_thickness:
    :return: The image with the visualized bounding box
    """
    for bb in bbs:
        img = draw_bb(img, bb, title, thickness, text_size, text_thickness)
    return img
