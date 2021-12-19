from typing import List

import cv2
import numpy as np

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.utils.bb_helper import get_human_bb_from_joints, bb_iou_numpy
from pedrec.utils.skeleton_helper import get_skeleton_mean_score


def get_features_to_track_from_human_joints(humans: List[Human]) -> np.ndarray:
    """
    TODO: Only use good features (joints with high scores)
    TODO: Alternative tracking without redetection via pose_net, just use pure tracked joints .. for win performance issues
    :param humans:
    :return:
    """
    features_to_track: List[List[float]] = []
    for human in humans:
        for joint in human.skeleton_2d:
            features_to_track.append([joint[0], joint[1]])
    features_array = np.ndarray((len(features_to_track), 1, 2), dtype=np.float32)
    for feature_id, feature in enumerate(features_to_track):
        features_array[feature_id][0][0] = feature[0]
        features_array[feature_id][0][1] = feature[1]
    return features_array

def remove_duplicates(human_bbs):
    ignore = []
    num_humans = len(human_bbs)
    output = []
    for i in range(0, num_humans):
        if i in ignore:
            continue
        human_bb = human_bbs[i]
        if i+1 < num_humans:
            for j in range(i+1, num_humans):
                other_bb = human_bbs[j]
                if bb_iou_numpy(human_bb, other_bb) > 0.95:
                    ignore.append(j)
        output.append(human_bb)
    return output

def bb_tracking(human_bbs: List[np.ndarray], tracked_humans: List[Human], min_imu: float = 0.5):
    used_humans = []
    for idx, bb in enumerate(human_bbs):
        best_human = None
        best_imu = 0
        for human in tracked_humans:
            if human.uid in used_humans:
                continue

            imu = bb_iou_numpy(bb, human.bb)
            if imu > min_imu and imu > best_imu:
                best_imu = imu
                best_human = human
        if best_human is not None:
            used_humans.append(best_human.uid)
            human_bbs[idx] = np.append(best_human.bb, best_human.uid)
        else:
            human_bbs[idx] = np.append(bb, -1)
    return human_bbs


def add_undetected_bbs_from_tracking(human_bbs, tracked_humans):
    num_detected_tracked_bbs = len([human_bb for human_bb in human_bbs if human_bb[-1] != -1])
    if num_detected_tracked_bbs != len(tracked_humans):
        for human in tracked_humans:
            found = False
            for bb in human_bbs:
                if bb[-1] == human.uid:
                    found = True
                    break
            if not found:
                human_bbs.append(np.append(human.bb, human.uid))
    return human_bbs


class HumanTracker(object):
    def __init__(self, img_size: ImageSize):
        self.img_size = img_size
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.previous_frame_gray: np.ndarray = None

    def get_humans_by_tracking(self, frame: np.ndarray, previous_humans: List[Human]) -> \
            List[Human]:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        humans: List[Human] = []
        if previous_humans is not None and len(previous_humans) > 0:
            features_to_track = get_features_to_track_from_human_joints(previous_humans)
            points, st, err = cv2.calcOpticalFlowPyrLK(self.previous_frame_gray, frame_gray, features_to_track, None,
                                                       **self.lk_params)
            optical_flow_point_idx = 0
            for previous_human in previous_humans:
                skeleton = np.zeros((len(SKELETON_PEDREC_JOINTS), 4), dtype=np.float32)
                for joint_num, point_idx in enumerate(
                        range(optical_flow_point_idx, optical_flow_point_idx + len(SKELETON_PEDREC_JOINTS))):
                    x = points[point_idx][0][0]
                    y = points[point_idx][0][1]
                    old_joint = previous_human.skeleton_2d[joint_num]
                    skeleton[joint_num, 0] = int(x)
                    skeleton[joint_num, 1] = int(y)
                    skeleton[joint_num, 2] = old_joint[2]-0.1
                    skeleton[joint_num, 3] = 1
                humans.append(Human(uid=previous_human.uid,
                                    skeleton_2d=skeleton,
                                    skeleton_3d=previous_human.skeleton_3d,
                                    bb=get_human_bb_from_joints(skeleton,
                                                                self.img_size.width,
                                                                self.img_size.height,
                                                                confidence=get_skeleton_mean_score(
                                                                    skeleton),
                                                                class_idx=0,
                                                                expand=0),
                                    orientation=None,
                                    env_position=None,
                                    ))

                optical_flow_point_idx = optical_flow_point_idx + len(SKELETON_PEDREC_JOINTS)
        self.previous_frame_gray = frame_gray
        return humans
