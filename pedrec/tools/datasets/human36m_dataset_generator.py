import sys

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC, SKELETON_PEDREC_JOINTS

sys.path.append(".")
import math
import os
from typing import List

import cdflib
import ffmpeg
import h5py as h5py
import numpy as np
import pandas as pd
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.constants.human_mappings import GENDER, AGE, SIZE, BMI, SKIN_COLOR
from pedrec.tools.datasets.dataset_generator_helper import get_column_names, set_df_dtypes
from pedrec.utils.bb_helper import get_center_bb_from_coord_bb
from pedrec.utils.file_helper import get_filename_without_extension

pi_half = math.pi / 2

subjects_data = {
    "S1": {
        "gender": GENDER.FEMALE,
        "age": AGE.ADULT,
        "size": SIZE.MEDIUM,
        "weight": BMI.NORMAL,
        "skin_color": SKIN_COLOR.WHITE
    },
    "S5": {
        "gender": GENDER.FEMALE,
        "age": AGE.ADULT,
        "size": SIZE.SMALL,
        "weight": BMI.NORMAL,
        "skin_color": SKIN_COLOR.DARK_WHITE
    },
    "S6": {
        "gender": GENDER.MALE,
        "age": AGE.ADULT,
        "size": SIZE.LARGE,
        "weight": BMI.NORMAL,
        "skin_color": SKIN_COLOR.DARK_WHITE
    },
    "S7": {
        "gender": GENDER.FEMALE,
        "age": AGE.ADULT,
        "size": SIZE.SMALL,
        "weight": BMI.NORMAL,
        "skin_color": SKIN_COLOR.LIGHT_BROWN
    },
    "S8": {
        "gender": GENDER.MALE,
        "age": AGE.ADULT,
        "size": SIZE.MEDIUM,
        "weight": BMI.NORMAL,
        "skin_color": SKIN_COLOR.DARK_WHITE
    },
    "S9": {
        "gender": GENDER.MALE,
        "age": AGE.ADULT,
        "size": SIZE.LARGE,
        "weight": BMI.OVERWEIGHT,
        "skin_color": SKIN_COLOR.LIGHT_BROWN
    },
    "S11": {
        "gender": GENDER.MALE,
        "age": AGE.ADULT,
        "size": SIZE.MEDIUM,
        "weight": BMI.NORMAL,
        "skin_color": SKIN_COLOR.DARK_WHITE
    },
}

HUMAN36M_TO_COCO_JOINTS = [
    -1, #  nose = 0  - note, there is joint 14 which sometimes is reffered to as nose, but it is the head middle, thus it can't be used.
    -1, #  left_eye = 1
    -1, #  right_eye = 2
    -1, #  left_ear = 3
    -1, #  right_ear = 4
    17, #  left_shoulder = 5
    25, #  right_shoulder = 6
    18, #  left_elbow = 7
    26, #  right_elbow = 8
    19, #  left_wrist = 9
    27, #  right_wrist = 10
    6,  #  left_hip = 11
    1,  #  right_hip = 12
    7,  #  left_knee = 13
    2,  #  right_knee = 14
    8,  #  left_ankle = 15
    3,  #  right_ankle = 16
    0,  # hip_center = 17,
    12,  # spine_center = 18,
    13,  # neck = 19
    14,  # head_lower = 20
    15,  # head_upper = 21,
    10,  # left_foot_end = 22,
    5,  # right_foot_end = 23,
    22,  # left_hand_end = 24,
    30  # left_hand_end = 25
]


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / l2


def get_human36m_joints_3d(human36m_joints_3d_path: str):
    cdf = cdflib.CDF(human36m_joints_3d_path)
    human36m_joints_3d = np.array(cdf.varget("Pose"))
    if human36m_joints_3d.shape[0] != 1:
        raise ValueError("not exactly 1 human")
    return human36m_joints_3d[0]


def get_skeleton_coco_from_human36m_3d(human36m_joints_3d: np.ndarray):
    num_frames = human36m_joints_3d.shape[0]
    skeleton = np.zeros((num_frames, len(SKELETON_PEDREC_JOINTS), 6), dtype=np.float32)
    for coco_joint_num, human36m_joint_num in enumerate(HUMAN36M_TO_COCO_JOINTS):
        if human36m_joint_num == -1:  # joint not available
            continue
        human36m_idx = human36m_joint_num * 3
        skeleton[:, coco_joint_num, 0:3] = human36m_joints_3d[:, human36m_idx:human36m_idx + 3]  # joint positions
        skeleton[:, coco_joint_num, 1] *= -1 # flip + / - to have positive values upwards.
        skeleton[:, coco_joint_num, 3:6] = 1  # score, visibility and supported to 1, because they are always visible in H36M
    return skeleton


def get_skeleton_hip_centers_3d(human36m_joints_3d: np.ndarray):
    num_frames = human36m_joints_3d.shape[0]
    skeleton_hip_centers = np.zeros((num_frames, 1, 6), dtype=np.float32)
    skeleton_hip_centers[:, 0, 0:3] = human36m_joints_3d[:, 0:3]  # joint positions
    skeleton_hip_centers[:, 0, 1] *= -1  # flip + / - to have positive values upwards.
    skeleton_hip_centers[:, 0, 3:6] = 1  # score, visibility and supported to 1, because they are always visible in H36M
    return skeleton_hip_centers


def get_skeleton_pedrec_from_human36m_2d(human36m_joints_2d_path: str):
    cdf = cdflib.CDF(human36m_joints_2d_path)
    human36m_joints_2d = np.array(cdf.varget("Pose"))
    if human36m_joints_2d.shape[0] != 1:
        raise ValueError("not exactly 1 human")
    human36m_joints_2d = human36m_joints_2d[0]
    num_frames = human36m_joints_2d.shape[0]
    skeleton = np.zeros((num_frames, len(SKELETON_PEDREC_JOINTS), 5), dtype=np.float32)
    for coco_joint_num, human36m_joint_num in enumerate(HUMAN36M_TO_COCO_JOINTS):
        if human36m_joint_num == -1:  # joint not available
            continue
        human36m_idx = human36m_joint_num * 2
        skeleton[:, coco_joint_num, 0:2] = human36m_joints_2d[:, human36m_idx:human36m_idx+2]  # joint positions
        skeleton[:, coco_joint_num, 2:5] = 1  # score, visibility and supported to 1, because they are always visible in H36M
    return skeleton


def get_skeleton_hip_normalized(skeletons_coco: np.ndarray, skeleton_hips: np.ndarray):
    # skeletons_hip_normalized = np.zeros(skeletons_coco.shape, dtype=np.float)
    skeletons_hip_normalized = skeletons_coco.copy()
    skeletons_hip_normalized[:, :, 0] -= skeleton_hips[:, :, 0]
    skeletons_hip_normalized[:, :, 1] -= skeleton_hips[:, :, 1]
    # flip coordinatees
    skeletons_hip_normalized[:, :, 2] -= skeleton_hips[:, :, 2]

    # set invisible to 0
    skeletons_hip_normalized[:, :, 0] *= skeletons_hip_normalized[:, :, 4]
    skeletons_hip_normalized[:, :, 1] *= skeletons_hip_normalized[:, :, 4]
    skeletons_hip_normalized[:, :, 2] *= skeletons_hip_normalized[:, :, 4]
    return skeletons_hip_normalized


def get_env_position(hip_centers: np.ndarray):
    """
    Returns the position in the environment, usually from a root joint / or hip
    """
    env_position = np.squeeze(hip_centers.copy())[:, 0:3]
    # env_position[:, 2] = 1
    return env_position


def get_bbs(obj_path: str) -> List[np.ndarray]:
    bbs = []
    with h5py.File(obj_path, 'r') as f:
        masks = f["Masks"]
        for maskref in masks:
            mask = f[maskref[0]]
            mask = np.array(mask)
            h_mask = mask.max(0)
            w_mask = mask.max(1)

            top = h_mask.argmax()
            bottom = len(h_mask) - h_mask[::-1].argmax()

            left = w_mask.argmax()
            right = len(w_mask) - w_mask[::-1].argmax()

            bb2d = get_center_bb_from_coord_bb(np.array([left, top, right, bottom], dtype=np.float32))
            bb2d = np.squeeze(bb2d)
            bb2d[4] = 1  # confidence
            bbs.append(bb2d)
    return bbs

# check 'Sitting 1.54138969.mp4'
# 1 - 1383
# 2 - 2766
def run(dataset_base_dir: str, dataset_dirs: List[str], output_path: str, dataset_type: DatasetType):
    datas = []
    curr_scene_uid = -1
    # count = 0
    for dataset_dir in dataset_dirs:
        video_dir = os.path.join(dataset_base_dir, dataset_dir, "Videos")
        for filename in sorted(os.listdir(video_dir)):
            if filename.startswith('_ALL'):
                continue
            # if curr_scene_uid > 2:
            #     break
            filename_wo_extension = get_filename_without_extension(filename)

            img_dir_path_relative = os.path.join(dataset_dir, "Images", filename_wo_extension)
            img_dir_path = os.path.join(dataset_base_dir, img_dir_path_relative)
            if not os.path.exists(img_dir_path):
                os.makedirs(img_dir_path)
                ffmpeg.input(os.path.join(video_dir, filename)).output(f"{img_dir_path}/img_%5d.jpg").run()
            curr_scene_uid += 1
            curr_img_ids = []
            curr_img_types = []
            curr_img_dirs = []
            for img_filename in sorted(os.listdir(img_dir_path)):
                file_name, file_ext = os.path.splitext(img_filename)
                img_type = file_ext[1:].lower()  # remove .
                img_id = int(file_name[-5:])  # extract file_id
                curr_img_ids.append(img_id)
                curr_img_types.append(img_type)
                curr_img_dirs.append(img_dir_path_relative)

            pose_gt_2d = os.path.join(dataset_base_dir, dataset_dir, "MyPoseFeatures", "D2_Positions",
                                      f"{filename_wo_extension}.cdf")
            pose_gt_3d = os.path.join(dataset_base_dir, dataset_dir, "MyPoseFeatures", "D3_Positions_mono",
                                      f"{filename_wo_extension}.cdf")
            bb_gt_2d = os.path.join(dataset_base_dir, dataset_dir, "MySegmentsMat", "ground_truth_bb",
                                      f"{filename_wo_extension}.mat")

            human36m_joints_3d = get_human36m_joints_3d(pose_gt_3d)
            joints3d = get_skeleton_coco_from_human36m_3d(human36m_joints_3d)
            hips3d = get_skeleton_hip_centers_3d(human36m_joints_3d)
            skeleton_2d = get_skeleton_pedrec_from_human36m_2d(pose_gt_2d)
            if len(curr_img_ids) != skeleton_2d.shape[0]:
                # remove frames at the end, because the videos contain more frames than available annotations ...
                curr_img_ids = curr_img_ids[:skeleton_2d.shape[0]]
                curr_img_types = curr_img_types[:skeleton_2d.shape[0]]
                curr_img_dirs = curr_img_dirs[:skeleton_2d.shape[0]]

            subject_data = subjects_data[dataset_dir]

            # gt = gts_by_scene[scene_uid]
            dataset = ["H36M"] * len(curr_img_ids)
            dataset_types = [dataset_type.value] * len(curr_img_ids)
            scene_uid = [curr_scene_uid] * len(curr_img_ids)
            frame_nr_global = list(range(1, len(curr_img_ids) + 1))
            frame_nr_locals = list(range(0, len(curr_img_ids)))
            img_dirs = curr_img_dirs
            img_ids = curr_img_ids
            img_types = curr_img_types
            subject_ids = [dataset_dir] * len(curr_img_ids)
            genders = [subject_data["gender"].value] * len(curr_img_ids)
            skin_colors = [subject_data["skin_color"].value] * len(curr_img_ids)
            sizes = [subject_data["size"].value] * len(curr_img_ids)
            bmis = [subject_data["weight"].value] * len(curr_img_ids)
            ages = [subject_data["age"].value] * len(curr_img_ids)
            movements = [-1] * len(curr_img_ids)
            movement_speeds = [-1] * len(curr_img_ids)
            is_real_imgs = [True] * len(curr_img_ids)
            actions = [-1] * len(curr_img_ids)
            bbs = np.array(get_bbs(bb_gt_2d))
            env_positions = get_env_position(hips3d)
            body_orientations = [0] * len(curr_img_ids)
            head_orientations = [0] * len(curr_img_ids)
            skeleton_2d = skeleton_2d.reshape(skeleton_2d.shape[0], skeleton_2d.shape[1] * skeleton_2d.shape[2])
            skeleton_3d = get_skeleton_hip_normalized(joints3d, hips3d)
            skeleton_3d = skeleton_3d.reshape(skeleton_3d.shape[0], skeleton_3d.shape[1] * skeleton_3d.shape[2])
            data = [
                dataset,
                dataset_types,
                scene_uid,
                frame_nr_global,
                frame_nr_locals,
                img_dirs,
                img_ids,
                img_types,
                subject_ids,
                genders,
                skin_colors,
                sizes,
                bmis,
                ages,
                movements,
                movement_speeds,
                is_real_imgs,
                actions,
            ]
            for i in range(0, 6):
                data.append(bbs[:, i])
            for i in range(0, 3):
                data.append(env_positions[:, i])
            for i in range(0, 4):
                data.append(body_orientations)
            for i in range(0, 4):
                data.append(head_orientations)
            for i in range(0, skeleton_2d.shape[1]):
                data.append(skeleton_2d[:, i])
            for i in range(0, skeleton_3d.shape[1]):
                data.append(skeleton_3d[:, i])
            data = list(map(list, zip(*data)))
            datas += data
            a = 1
            # df = pd.DataFrame(data=data, columns=get_column_names())
            # print(df.memory_usage(deep=True))
            # set_df_dtypes(df)
            # print(df.memory_usage(deep=True))

    df = pd.DataFrame(data=datas, columns=get_column_names())
    print(df.memory_usage(deep=True))
    set_df_dtypes(df)
    print(df.memory_usage(deep=True))
    df.to_pickle(output_path)
    # save_gt(gts_by_scene, output_path)


if __name__ == "__main__":
    dataset_dirs = [
        "S1",
        "S5",
        "S6",
        "S7",
        "S8"
    ]
    run("data/datasets/Human3.6m/train/", dataset_dirs, "data/datasets/Human3.6m/train/h36m_train_pedrec.pkl", DatasetType.TRAIN)
    dataset_dirs = [
        "S9",
        "S11"
    ]
    run("data/datasets/Human3.6m/val/", dataset_dirs, "data/datasets/Human3.6m/val/h36m_val_pedrec.pkl", DatasetType.VALIDATE)