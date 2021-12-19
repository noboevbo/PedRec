import sys

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS

sys.path.append(".")
import csv
import math
import os
import re
from typing import List, Dict

import ijson
import numpy as np

from pedrec.models.constants.action_mappings import sim_name_to_action_mapping
from pedrec.models.constants.human_mappings import MOVEMENT_SPEED
from pedrec.models.data_structures import ImageSize
from pedrec.utils.bb_helper import get_center_bb_from_coord_bb
from pedrec.utils.skeleton_helper import get_middle_joint

"""
Comparision of generated sim datasets to validate if some given data is equal in both datasets
"""

pi_half = math.pi / 2
snake_case_pattern = re.compile(r'(?<!^)(?=[A-Z])')

# SIM_TO_COCO_JOINTS = [
#     0,  # nose = 0
#     15,  # left_eye = 1
#     14,  # right_eye = 2
#     17,  # left_ear = 3
#     16,  # right_ear = 4
#     5,  # left_shoulder = 5
#     2,  # right_shoulder = 6
#     6,  # left_elbow = 7
#     3,  # right_elbow = 8
#     7,  # left_wrist = 9
#     4,  # right_wrist = 10
#     11,  # left_hip = 11
#     8,  # right_hip = 12
#     12,  # left_knee = 13
#     9,  # right_knee = 14
#     13,  # left_ankle = 15
#     10  # right_ankle = 16
# ]

SIM_TO_COCO_JOINTS = [
    "Nose",  # nose = 0
    "LeftEye",  # left_eye = 1
    "RightEye",  # right_eye = 2
    "LeftEar",  # left_ear = 3
    "RightEar",  # right_ear = 4
    "LeftShoulder",  # left_shoulder = 5
    "RightShoulder",  # right_shoulder = 6
    "LeftElbow",  # left_elbow = 7
    "RightElbow",  # right_elbow = 8
    "LeftWrist",  # left_wrist = 9
    "RightWrist",  # right_wrist = 10
    "LeftHip",  # left_hip = 11
    "RightHip",  # right_hip = 12
    "LeftKnee",  # left_knee = 13
    "RightKnee",  # right_knee = 14
    "LeftAnkle",  # left_ankle = 15
    "RightAnkle"  # right_ankle = 16
]

# def get_skeleton_3d_hip_body_norm():
#     # TODO Normalisierte 3D Koodinaten vom Hip joint aus zurückgeben!! 11 22 r

def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / l2


def get_skeleton_coco_from_sim_3d(sim_joints: List[Dict[str, any]]):
    skeleton = np.zeros((len(SKELETON_PEDREC_JOINTS), 5), dtype=np.float32)
    for coco_joint_num, sim_joint_name in enumerate(SIM_TO_COCO_JOINTS):
        sim_joint = next(item for item in sim_joints if item["name"] == sim_joint_name)
        sim_joint_3d = sim_joint['pos']
        skeleton[coco_joint_num][0:3] = sim_joint_3d  # joint positions
        skeleton[coco_joint_num][3:5] = sim_joint['inCamImg']  # score and visibility to
    skeleton[:, 0:3] *= skeleton[:, 4].reshape(len(SKELETON_PEDREC_JOINTS), 1) * 1000  # convert meter to mm and set invisible coords to 0
    return skeleton


def get_skeleton_coco_from_sim_2d(sim_joints: List[Dict[str, any]]):
    skeleton = np.zeros((len(SKELETON_PEDREC_JOINTS), 4), dtype=np.float32)
    for coco_joint_num, sim_joint_name in enumerate(SIM_TO_COCO_JOINTS):
        sim_joint = next(item for item in sim_joints if item["name"] == sim_joint_name)
        sim_joint_2d = sim_joint['imgPos']
        skeleton[coco_joint_num][0:2] = sim_joint_2d  # joint positions
        skeleton[coco_joint_num][2:4] = sim_joint['inCamImg']  # score and visibility to 1
    skeleton[:, 0:2] *= skeleton[:, 3].reshape(len(SKELETON_PEDREC_JOINTS), 1)  # set invisible coords to 0
    return skeleton


def get_env_position(root_joint: Dict[str, any]):
    env_position = np.zeros((3), dtype=np.float32)  # x, z, score
    if root_joint['inCamImg'] == 0:
        return env_position
    root_pos = root_joint['pos']
    env_position[0] = root_pos[0]
    env_position[1] = root_pos[2]
    env_position[2] = 1
    return env_position


def get_orientation(joint: Dict[str, any]):
    body_orientation = np.zeros((4), dtype=np.float32)  # theta, phi, score, visible/provided
    if joint['inCamImg'] == 0:
        return body_orientation
    root_rot = normalize(np.array(joint['directionVec']))
    # unity to std: x = -z, y = x, z = y
    x, y, z = -root_rot[2], root_rot[0], root_rot[1]
    theta = math.acos(z)
    # direction vec originates from * z vec, thus the orientation is rotated -90 degrees, revert it by +90degrees
    # phi = math.atan2(y, x)

    phi = math.atan2(y, x) - (math.pi / 2)
    # normalize 0 < phi < 2*pi
    phi = phi % (2 * math.pi)
    if phi < 0:
        phi = phi + (2 * math.pi)

    body_orientation[0] = theta / math.pi  # normalized value
    body_orientation[1] = phi / (2 * math.pi)  # normalized value
    body_orientation[2] = 1
    body_orientation[3] = 1
    if np.max(body_orientation) > 1 or np.min(body_orientation) < 0:
        print("WTF2")

    return body_orientation
    # if annot_joints['']

def bb_spawns_over_full_img(bb, image_size):
    return bb[0] < 0 and bb[1] < 0 and bb[2] > image_size.width and bb[3] > image_size.height

def bb_is_outside_of_img(bb, image_size):
    return bb[0] >= image_size.width or bb[1] >= image_size.height or bb[2] <= 0 or bb[3] <= 0


def get_center_bb(obj, image_size):
    bb2d = obj['visibleBoundingBox']
    bb2d = np.array([bb2d['minX'], bb2d['minY'], bb2d['maxX'], bb2d['maxY']], dtype=np.float32)
    if bb_spawns_over_full_img(bb2d, image_size) or bb_is_outside_of_img(bb2d, image_size):
        return np.zeros((6,), dtype=np.float32)

    # Set min / maxs to img dimension min maxes if necessary
    if bb2d[0] < 0:
        bb2d[0] = 0
    if bb2d[1] < 0:
        bb2d[1] = 0
    if bb2d[2] > image_size.width:
        bb2d[2] = image_size.width
    if bb2d[3] > image_size.height:
        bb2d[3] = image_size.height

    bb2dl = get_center_bb_from_coord_bb(bb2d)
    bb2dl = np.squeeze(bb2dl)

    # set confidence
    bb2dl[4] = 1
    if obj['objectClass'] == 'person' and (
            bb2dl[2] <= 10 or bb2dl[3] <= 10 or (bb2dl[2] == image_size.width and bb2dl[3] == image_size.height)):
        # ignore non visible or to small bbs or bb spans over full img
        bb2dl[4] = 0
    if obj['objectClass'] == 'head' and (
            bb2dl[2] <= 0 or bb2dl[3] <= 0 or (bb2dl[2] == image_size.width and bb2dl[3] == image_size.height)):
        # ignore non visible or to small bbs or bb spans over full img
        bb2dl[4] = 0
    return bb2dl

def get_obj_bb_dict(obj_path: str, image_size: ImageSize):
    results = {}
    zeros = 0
    with open(obj_path, 'r') as obj_file:
        annot = ijson.items(obj_file, "item")
        for annot_frame in annot:
            results[annot_frame['frameNum']] = {}
            results_frame = results[annot_frame['frameNum']]
            for obj in annot_frame['objects']:
                if obj['uniqueObjectID'] not in results_frame:
                    results_frame[obj['uniqueObjectID']] = []
                uid_obj = results_frame[obj['uniqueObjectID']]
                uid_obj_parts = {'class': obj['objectClass']}
                if obj['isVisible'] == 1:
                    center_bb = get_center_bb(obj, image_size)
                    uid_obj_parts['bb2d'] = center_bb
                else:
                    uid_obj_parts['bb2d'] = np.zeros((6,), dtype=np.float32)
                uid_obj.append(uid_obj_parts)
    # TODO 3D?
    if zeros > 0:
        a = 1
    return results


def get_description_gt(file_path: str, descriptions: Dict[str, any]):
    annotations = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row_num, row in enumerate(reader):
            if row_num == 0:  # skip header
                continue
            frame_nr = int(row[0])
            uid = row[1]
            description = descriptions[row[2]]
            if frame_nr not in annotations:
                annotations[frame_nr] = {}
            annotations[frame_nr][uid] = description
    return annotations


def get_description_data(file_path: str, id_name: str):
    descriptions = {}
    with open(file_path, 'r') as file:
        for description in ijson.items(file, "item"):
            descriptions[description[id_name]] = description
    return descriptions


def get_frame_uid_gt(annot_path: str, curr_scene_uid: int):
    char_descriptions = get_description_data(os.path.join(annot_path, "characterDescriptions.json"), "uid")
    motion_descriptions = get_description_data(os.path.join(annot_path, "mocapDescriptions.json"), "id")
    character_frames = get_description_gt(os.path.join(annot_path, "characterDescriptionFrameData.csv"),
                                          char_descriptions)
    motion_frames = get_description_gt(os.path.join(annot_path, "mocapDescriptionFrameData.csv"), motion_descriptions)
    return get_action_gt(os.path.join(annot_path, "actions.csv"), character_frames, motion_frames, curr_scene_uid)


def get_action_gt(file_path: str, character_frames: Dict[int, any], motion_frames: Dict[int, any], curr_scene_uid: int):
    annotations = {}
    scene_fr_store = {}
    scene_global_frame_store = {}
    scene_nr_store = {}
    scene_uid = curr_scene_uid + 1
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row_num, row in enumerate(reader):
            if row_num == 0:  # skip header
                continue
            frame_nr = int(row[0])
            uid = row[1]
            if uid not in scene_global_frame_store:
                scene_global_frame_store[uid] = frame_nr
            for i in range(scene_global_frame_store[uid], frame_nr - 1):
                # there were some missing frame numbers in between, thus, create a new scene
                scene_nr_store[uid] = scene_uid
                scene_uid += 1
                # set fr store manually, to prevent double creation of scene uids
                scene_fr_store[uid] = animation_frame
            scene_global_frame_store[uid] = frame_nr
            actions = row[2]
            animation_frame = int(row[3])
            if uid not in scene_fr_store:
                scene_fr_store[uid] = animation_frame
                scene_nr_store[uid] = scene_uid
                scene_uid += 1
            if animation_frame < scene_fr_store[uid]:
                # new scene
                scene_nr_store[uid] = scene_uid
                scene_uid += 1
            scene_fr_store[uid] = animation_frame
            if frame_nr not in annotations:
                annotations[frame_nr] = {}
            annotations[frame_nr][uid] = {
                "frame_nr_global": frame_nr,
                "character": character_frames[frame_nr][uid],
                "motion": motion_frames[frame_nr][uid],
                "actions": actions,
                "scene_uid": scene_nr_store[uid]
            }
    return annotations, scene_uid


def get_uid_frame_gt(frame_uid_gts: Dict[int, any]):
    uid_frame_gt = {}
    for frame_nr, uid_gt in frame_uid_gts.items():
        for uid, gt in uid_gt.items():
            if uid not in uid_frame_gt:
                uid_frame_gt[uid] = {}
            gt["frame_nr_local"] = len(uid_frame_gt[uid])
            uid_frame_gt[uid][frame_nr] = gt
    return uid_frame_gt


def get_movement_speed(name: str):
    snake_case_name = snake_case_pattern.sub('_', name).upper()
    return MOVEMENT_SPEED[snake_case_name]


def get_actions(actions_str: str):
    action_names = actions_str.split(',')
    return [sim_name_to_action_mapping[action_name].value for action_name in action_names]


def get_na_row_values(columns):
    """
    returns na values for all columns starting from gender
    """
    row_values = [
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    False,
    -1
    ]
    actions_idx = columns.index('actions')
    row_values += [0] * (len(columns) - actions_idx)
    return row_values

def get_gts(image_size: ImageSize, dataset_base_dir: str, dataset_dirs: List[str]):
    bbs = []
    head_orientations = []
    body_orientations = []
    env_positions = []
    gts_2d = []
    gts_3d = []
    # na_row_values = get_na_row_values(columns)
    for dataset_dir in sorted(dataset_dirs):
        rel_annot_path = os.path.join(dataset_dir, "annotations")
        annot_path = os.path.join(dataset_base_dir, rel_annot_path)
        for filename in sorted(os.listdir(annot_path)):
            file_path = os.path.join(annot_path, filename)
            if os.path.isfile(file_path) and filename.startswith("pose-view"):
                with open(file_path, 'r') as file:
                    pose_annotations = ijson.items(file, "item")
                    obj_annotations = get_obj_bb_dict(file_path.replace("pose", "object"), image_size)
                    for pose_annotation_frame in pose_annotations:
                        frame_nr_global = pose_annotation_frame['frameNum']
                        uid = str(pose_annotation_frame['uid'])

                        obj_frame = obj_annotations[frame_nr_global][uid]
                        bb_gt = [obj for obj in obj_frame if obj['class'] == 'person']
                        if len(bb_gt) != 1:
                            raise ValueError(f"Expected 1 bb obj for person, found {len(bb_gt)}")
                        bb_gt = bb_gt[0]
                        if bb_gt['bb2d'][4] == 0:
                            # no valid bb found
                            continue
                        joints = pose_annotation_frame['joints']
                        meta_joints = pose_annotation_frame['metaJoints']
                        root_joint = [joint for joint in meta_joints if joint['name'] == 'root'][0]
                        face_joint = [joint for joint in meta_joints if joint['name'] == 'face'][0]
                        joints2d = get_skeleton_coco_from_sim_2d(joints)
                        joints3d = get_skeleton_coco_from_sim_3d(joints)
                        bbs.append(bb_gt['bb2d'])
                        # env position
                        env_positions.append(get_env_position(root_joint))
                        # body_orientation
                        body_orientations.append(get_orientation(root_joint))
                        # head_orientation
                        head_orientations.append(get_orientation(face_joint))
                        
                        gts_2d.append(joints2d)
                        gts_3d.append(joints3d)
    return np.array(gts_2d, dtype=np.float32), np.array(gts_3d, dtype=np.float32), np.array(bbs, dtype=np.float32), np.array(env_positions, dtype=np.float32), np.array(body_orientations, dtype=np.float32), np.array(head_orientations, dtype=np.float32),




if __name__ == "__main__":
    dataset_dirs = [
        # "ROM1b",
        # "ROM2b",
        # "ROM3b",
        "ROM4b"
    ]
    gt_2d_1, gt_3d_1, bbs_1, env_positions_1, body_orientations_1, head_orientations_1 = get_gts(ImageSize(1920, 1080), "data/datasets/ROMb", dataset_dirs)
    gt_2d_2, gt_3d_2, bbs_2, env_positions_2, body_orientations_2, head_orientations_2 = get_gts(ImageSize(1920, 1080), "data/datasets/ROMb1", dataset_dirs)
    eq_2d = np.array_equal(gt_2d_1, gt_2d_2)
    eq_3d = np.array_equal(gt_3d_1, gt_3d_2)
    eq_bbs = np.array_equal(bbs_1, bbs_2)
    eq_env_positions = np.array_equal(env_positions_1, env_positions_2)
    eq_body_orientations = np.array_equal(body_orientations_1, body_orientations_2)
    eq_head_orientations = np.array_equal(head_orientations_1, head_orientations_2)
    a = 1
