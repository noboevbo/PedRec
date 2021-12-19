from typing import List

import numpy as np

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_MIDDLE_AXIS_IDS, SKELETON_PEDREC_EXTREMITIES_IDS
from pedrec.utils.skeleton_helper import get_euclidean_joint_distances


def get_skeleton_diameter(skeleton: np.ndarray, coord_split_idx: int = 2):
    min_ = []
    max_ = []
    for i in range(0, coord_split_idx):
        min_.append(skeleton[:, i].min())
        max_.append(skeleton[:, i].max())

    return np.linalg.norm(np.array([min_, max_]))

def get_number_of_similar_joints(skeleton_a: np.ndarray, skeleton_b: np.ndarray, joint_ids: List[int],
                                    coord_split_idx: int, min_joint_score: float, max_acceptable_distance: float):
    distances = get_euclidean_joint_distances(skeleton_a, skeleton_b, joint_ids, coord_split_idx, min_joint_score)
    # print(distances)
    # print(max_acceptable_distance)
    num_distances = len(distances)
    return np.count_nonzero(distances < max_acceptable_distance), num_distances


def get_pose_similarity(human_a, human_b, allowed_distance_2d: float , allowed_distance_3d: float):
    # set a minimum number of joints for mean calculation,
    # to prevent situations where only a small number of joints have a high enough joint score
    min_middle_axis_joints = int(len(SKELETON_PEDREC_MIDDLE_AXIS_IDS) * 0.7)
    min_extremities_joints = int(len(SKELETON_PEDREC_EXTREMITIES_IDS) * 0.7)
    # 2d
    middle_axis_similar_joints, middle_axis_num_used_joints = get_number_of_similar_joints(
        human_a.skeleton_2d, human_b.skeleton_2d, SKELETON_PEDREC_MIDDLE_AXIS_IDS, 2, 0.15, allowed_distance_2d)
    middle_axis_correct_pct_2d = middle_axis_similar_joints / max(min_middle_axis_joints, middle_axis_num_used_joints)

    extremities_similar_joints, extremities_num_used_joints = get_number_of_similar_joints(
        human_a.skeleton_2d, human_b.skeleton_2d, SKELETON_PEDREC_EXTREMITIES_IDS, 2, 0.15, allowed_distance_2d)
    extremities_correct_pct_2d = extremities_similar_joints / max(min_extremities_joints, extremities_num_used_joints)
    # 3d
    middle_axis_similar_joints_3d, middle_axis_num_used_joints_3d = get_number_of_similar_joints(
        human_a.skeleton_3d, human_b.skeleton_3d, SKELETON_PEDREC_MIDDLE_AXIS_IDS, 3, 0.15, allowed_distance_3d)
    middle_axis_correct_pct_3d = middle_axis_similar_joints_3d / max(min_middle_axis_joints,
                                                                     middle_axis_num_used_joints_3d)
    extremities_similar_joints_3d, extremities_num_used_joints_3d = get_number_of_similar_joints(
        human_a.skeleton_3d, human_b.skeleton_3d, SKELETON_PEDREC_EXTREMITIES_IDS, 3, 0.15, allowed_distance_3d)
    extremities_correct_pct_3d = extremities_similar_joints_3d / max(min_extremities_joints,
                                                                     extremities_num_used_joints_3d)
    # print(f"{middle_axis_correct_pct_2d} - {extremities_correct_pct_2d} - {middle_axis_correct_pct_3d} - {extremities_correct_pct_3d}")
    return (middle_axis_correct_pct_2d + extremities_correct_pct_2d + middle_axis_correct_pct_3d + extremities_correct_pct_3d) / 4