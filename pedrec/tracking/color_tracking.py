import enum

import numpy as np
import matplotlib.pyplot as plt
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT


def get_joint_area(joint, model_input, head_area_pixels: int = 30):
    pxls = int(head_area_pixels / 2)
    area_min_x = int(max(0, joint[0] - pxls))  # head area = 10x10px
    area_max_x = int(min(model_input.shape[1], joint[0] + pxls))  # head area = 10x10px

    area_min_y = int(max(0, joint[1] - pxls))  # head area = 10x10px
    area_max_y = int(min(model_input.shape[0], joint[1] + pxls))  # head area = 10x10px
    return model_input[area_min_y:area_max_y, area_min_x:area_max_x]


rgb_to_xyz_matrix = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])


class StandardIlluminant(enum.Enum):
    D50 = 1,  # Standard US
    D65 = 2  # Standard EU


class StandardObserver(enum.Enum):
    CIE2 = 1,
    CIE10 = 2


def get_xn_yn_zn(standard_illuminant: StandardIlluminant, standard_observer: StandardObserver):
    if standard_illuminant == StandardIlluminant.D50:
        if standard_observer == StandardObserver.CIE2:
            return 96.522, 100, 82.521
        else:
            return 96.720, 100, 81.427
    if standard_observer == StandardObserver.CIE2:
        return 95.047, 100, 108.883
    return 94.811, 100, 107.304


def xyz_to_lab(x, y, z, X_n, Y_n, Z_n):
    def f(t: float, delta: float = 6 / 29):
        if t > delta:
            return t ** (1. / 3)
        return (t / 3 * delta ** 2) + (4 / 29)

    f_x = f(x / X_n)
    f_y = f(y / Y_n)
    f_z = f(z / Z_n)
    L = 116 * f_y - 16
    a = 500 * (f_x - f_y)
    b = 200 * (f_y - f_z)
    return L, a, b


def get_area_color_similarity(skeleton_a, skeleton_b, img_a, img_b, joint_num: int):
    a = False
    area_a = get_joint_area(skeleton_a[joint_num], img_a)
    if a:
        plt.imshow(area_a)
        plt.show()
    mean_rgb_a = np.mean(area_a, axis=(0, 1))

    area_b = get_joint_area(skeleton_b[joint_num], img_b)
    if a:
        plt.imshow(area_b)
        plt.show()
    mean_rgb_b = np.mean(area_b, axis=(0, 1))
    return np.linalg.norm(mean_rgb_a - mean_rgb_b)


def get_color_similarity(skeleton_a, skeleton_b, img_a, img_b, similarity_threshold: float = 25.5):
    """
    requires sRGB values to be normalized between 0 and 1
    TODO: remove pure black values, caused by affine transformation padding
    """
    # head_mean_xyz_b = np.dot(rgb_to_xyz_matrix, head_mean_rgb_b)
    # head_mean_lab_b = xyz_to_lab(*head_mean_xyz_b, *xyz_n)
    # xyz_n = get_xn_yn_zn(StandardIlluminant.D65, StandardObserver.CIE2)  # sRGB D65 White reference + eye like observer
    head_color_distance = get_area_color_similarity(skeleton_a, skeleton_b, img_a, img_b,
                                                    SKELETON_PEDREC_JOINT.nose.value)
    torso_color_distance = get_area_color_similarity(skeleton_a, skeleton_b, img_a, img_b,
                                                     SKELETON_PEDREC_JOINT.spine_center.value)
    hip_color_distance = get_area_color_similarity(skeleton_a, skeleton_b, img_a, img_b,
                                                   SKELETON_PEDREC_JOINT.hip_center.value)

    mean_distance = (head_color_distance + torso_color_distance + hip_color_distance) / 3
    return 1 - (mean_distance / 255)
