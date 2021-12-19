from typing import List

import numpy as np
from torchvision import transforms

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT, SKELETON_PEDREC_TO_PEDRECEHPI3D
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.utils.skeleton_helper_3d import get_unit_skeleton_sequence

ehpi_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.400, 0.443, 0.401], std=[0.203, 0.269, 0.203])
])

def get_skeleton_sequence_in_ehpi_order(skeleton_sequence_3d: np.ndarray):
    skeleton_sequence_3d = np.transpose(skeleton_sequence_3d, (1, 0, 2))
    skeleton_sequence_ehpi_order = np.zeros(skeleton_sequence_3d.shape, dtype=np.float32)
    for pedrec_idx, ehpi_idx in enumerate(SKELETON_PEDREC_TO_PEDRECEHPI3D):
        skeleton_sequence_ehpi_order[:, ehpi_idx, :] = skeleton_sequence_3d[:, pedrec_idx, :]
    skeleton_sequence_ehpi_order = np.transpose(skeleton_sequence_ehpi_order, (1, 0, 2))
    return skeleton_sequence_ehpi_order

def get_ehpi_from_human_history(human: Human, history: List[Human],
                                temporal_field: ImageSize = ImageSize(32, 32),
                                use_2d: bool = False, min_score: float = 0.0):
    history_len = len(history)
    human_skeleton_2d_history = [human.skeleton_2d.copy()]
    skeleton = human.skeleton_3d.copy()
    skeleton[:, :3] += 1500
    skeleton[:, :3] /= 3000
    skeleton[skeleton[:, 3] < min_score] = 0
    human_skeleton_3d_history = [skeleton]
    for i in range(1, temporal_field.width):
        if history_len >= i:
            skeleton = history[-i].skeleton_3d.copy()
            skeleton[:, :3] += 1500
            skeleton[:, :3] /= 3000
            skeleton[skeleton[:, 3] < min_score] = 0
            human_skeleton_3d_history.insert(0, skeleton)
            human_skeleton_2d_history.insert(0, history[-1].skeleton_2d.copy())
        else:
            break

    skeleton_sequence_3d = np.array(human_skeleton_3d_history, dtype=np.float32)[:, :, :3]
    skeleton_sequence_3d = np.transpose(skeleton_sequence_3d, (1, 0, 2))
    skeleton_sequence_2d = np.array(human_skeleton_2d_history, dtype=np.float32)
    # skeleton_sequence_2d = np.transpose(skeleton_sequence_2d, (1, 0, 2))
    # mins = np.min(skeleton_sequence_3d[:, :, :], axis=0)
    # maxs = np.max(skeleton_sequence_3d[:, :, :], axis=0)
    # skeleton_sequence_3d = (skeleton_sequence_3d[:, :, :] - mins) / (maxs - mins)
    if skeleton_sequence_3d.shape[1] < temporal_field.width or skeleton_sequence_3d.shape[
        0] < temporal_field.height:
        missing_frames = temporal_field.width - skeleton_sequence_3d.shape[1]
        missing_joints = temporal_field.height - skeleton_sequence_3d.shape[0]
        skeleton_sequence_3d = np.pad(skeleton_sequence_3d, ((0, missing_joints), (missing_frames, 0), (0, 0)))

    if use_2d and (np.min(skeleton_sequence_3d) != 0 or np.max(skeleton_sequence_3d) != 0):
        skeleton_sequence_2d[skeleton_sequence_2d[:, :, 2] < min_score] = 0
        # # Append hip2d for global movement
        length = abs(skeleton_sequence_2d.shape[0] - temporal_field.width)
        skel_2d_valid_x = skeleton_sequence_2d[skeleton_sequence_2d[:, :, 0] > 0]
        skel_2d_valid_y = skeleton_sequence_2d[skeleton_sequence_2d[:, :, 1] > 0]
        if len(skel_2d_valid_y) > 0 and len(skel_2d_valid_y) > 0:
            skeleton_sequence_2d_min_x = np.min(skel_2d_valid_x[:, 0])
            skeleton_sequence_2d_min_y = np.min(skel_2d_valid_y[:, 1])
            hip_centers_2d = np.transpose(skeleton_sequence_2d, (1, 0, 2))[SKELETON_PEDREC_JOINT.hip_center.value, :, :2].copy()
            skeleton_sequence_3d[-1, length:, :2] = hip_centers_2d
            skeleton_sequence_3d[-1, length:, 0] = skeleton_sequence_3d[-1, length:, 0] - skeleton_sequence_2d_min_x
            skeleton_sequence_3d[-1, length:, 1] = skeleton_sequence_3d[-1, length:, 1] - skeleton_sequence_2d_min_y
            max_factor_x = 1 / np.max(skeleton_sequence_2d[:, :, 0])
            max_factor_y = 1 / np.max(skeleton_sequence_2d[:, :, 1])
            skeleton_sequence_3d[-1, length:, 0] = skeleton_sequence_3d[-1, length:, 0] * max_factor_x
            skeleton_sequence_3d[-1, length:, 1] = skeleton_sequence_3d[-1, length:, 1] * max_factor_y

    skeleton_sequence_3d = get_unit_skeleton_sequence(skeleton_sequence_3d, nan_value=0)
    skeleton_sequence_3d[:len(SKELETON_PEDREC_JOINT), :, :3] += 20
    skeleton_sequence_3d[:len(SKELETON_PEDREC_JOINT), :, :3] /= 40
    skeleton_sequence_3d = get_skeleton_sequence_in_ehpi_order(skeleton_sequence_3d)
    ehpi = skeleton_sequence_3d * 255
    ehpi = ehpi.astype(np.uint8)
    return ehpi

def get_ehpi2d_from_human_history(human: Human, history: List[Human], temporal_field: ImageSize = ImageSize(32, 32),min_score: float = 0.0):
    history_len = len(history)
    human_skeleton_2d_history = [human.skeleton_2d.copy()]
    for i in range(1, 32):
        if history_len >= i:
            human_skeleton_2d_history.insert(0, history[-1].skeleton_2d.copy())
        else:
            break

    skeleton_sequence_2d = np.array(human_skeleton_2d_history, dtype=np.float32)
    skeleton_sequence_2d = np.transpose(skeleton_sequence_2d, (1, 0, 2))
    # mins = np.min(skeleton_sequence_3d[:, :, :], axis=0)
    # maxs = np.max(skeleton_sequence_3d[:, :, :], axis=0)
    # skeleton_sequence_3d = (skeleton_sequence_3d[:, :, :] - mins) / (maxs - mins)
    if skeleton_sequence_2d.shape[1] < temporal_field.width or skeleton_sequence_2d.shape[
        0] < temporal_field.height:
        missing_frames = temporal_field.width - skeleton_sequence_2d.shape[1]
        missing_joints = temporal_field.height - skeleton_sequence_2d.shape[0]
        skeleton_sequence_2d = np.pad(skeleton_sequence_2d, ((0, missing_joints), (missing_frames, 0), (0, 0)))


    skeleton_sequence_2d[skeleton_sequence_2d[:, :, 2] < min_score] = 0
    ehpi_img = skeleton_sequence_2d
    tmp = np.copy(ehpi_img)
    # normalize
    curr_min_x = np.min(ehpi_img[0, :, :][ehpi_img[0, :, :] > 0])
    curr_min_y = np.min(ehpi_img[1, :, :][ehpi_img[1, :, :] > 0])

    # Set x start to 0
    ehpi_img[0, :, :] = ehpi_img[0, :, :] - curr_min_x
    # Set y start to 0
    ehpi_img[1, :, :] = ehpi_img[1, :, :] - curr_min_y

    # Set x to max image_size.width
    max_factor_x = 1 / np.max(ehpi_img[0, :, :])
    max_factor_y = 1 / np.max(ehpi_img[1, :, :])
    ehpi_img[0, :, :] = ehpi_img[0, :, :] * max_factor_x
    ehpi_img[1, :, :] = ehpi_img[1, :, :] * max_factor_y
    ehpi_img[0, :, :][tmp[0, :, :] == 0] = 0
    ehpi_img[1, :, :][tmp[1, :, :] == 0] = 0

    ehpi_img = ehpi_img * 255
    ehpi_img = ehpi_img.astype(np.uint8)
    return ehpi_img