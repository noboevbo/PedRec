import math
from typing import List, Tuple

import cv2
import numpy as np

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_IDS
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.tracking.color_tracking import get_color_similarity
from pedrec.tracking.pose_tracking import get_skeleton_diameter, get_pose_similarity
from pedrec.utils.bb_helper import bb_iou_numpy, get_coord_bb_from_center_bb, get_bb_width, get_bb_height
from pedrec.utils.skeleton_helper import get_euclidean_joint_distances


def get_bb_color(human, img):
    human_bb = get_coord_bb_from_center_bb(human.bb)
    area_min_x = int(max(0, human_bb[0]))  # head area = 10x10px
    area_max_x = int(min(img.shape[1], human_bb[2]))  # head area = 10x10px

    area_min_y = int(max(0, human_bb[1]))  # head area = 10x10px
    area_max_y = int(min(img.shape[0], human_bb[3]))  # head area = 10x10px
    bb_area = img[area_min_y:area_max_y, area_min_x:area_max_x]
    return np.mean(bb_area, axis=(0, 1))

def get_human_similarity(human_a: Human, human_b: Human, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    size_factor_2d = get_skeleton_diameter(human_a.skeleton_2d, 2) * 0.05
    size_factor_3d = get_skeleton_diameter(human_a.skeleton_3d, 3) * 0.05

    bb_similarity = bb_iou_numpy(human_a.bb, human_b.bb)
    if bb_similarity < 0.1:  # remove all bbs without real overlap
        print(f"REMOVE: {bb_similarity}")
        print(human_a.bb)
        print(human_b.bb)
        return 0

    color_a = get_bb_color(human_a, frame_a)
    color_b = get_bb_color(human_b, frame_b)
    distance = np.linalg.norm(color_a - color_b)
    color_similarity = 1 - (distance / 255)
    # color_similarity = get_color_similarity(human_a.skeleton_2d, human_b.skeleton_2d, frame_a, frame_b)
    print(color_similarity)
    if color_similarity > 0.6:
        return color_similarity
    print("HERE")
    # TODO: If color similarity is low but pose will be high, there's a possibility that one person occludes
    # another one, handle this
    return get_pose_similarity(human_a, human_b, size_factor_2d, size_factor_3d)


class HumanMerger(object):
    def __init__(self, img_size: ImageSize,
                 min_joint_score_for_similarity: float = 0.5):
        self.img_size = img_size
        self.min_joint_score_for_similarity = min_joint_score_for_similarity
        self.joint_acceptable_distance_scale_factor_human_size = 0.075
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.next_human_uid = 0
        self.previous_frame_gray: np.ndarray = None

    def get_duplicated_human_idxs(self, humans_a, img, similarity_threshold=0.6):
        humans_a.sort(key=lambda human: human.score, reverse=True)
        # num_humans = len(humans)
        human_ids_to_delete = []
        for human_a_idx, human_a in enumerate(humans_a):
            # Step 1: Merge duplicates in detected humans
            for human_b_idx in range(human_a_idx + 1, len(humans_a)):
                human_b = humans_a[human_b_idx]
                human_similarity = get_human_similarity(human_a, human_b, img, img)
                if human_similarity > similarity_threshold:
                    if human_b_idx not in human_ids_to_delete:
                        # Delete the similar humans with a lower score.
                        human_ids_to_delete.append(human_b_idx)

        return human_ids_to_delete


    def get_pose_similarity(self, human: Human, human2: Human) -> float:
        human_size = math.sqrt(math.pow(get_bb_width(human.bb), 2) + math.pow(get_bb_height(human.bb), 2))
        human2_size = math.sqrt(math.pow(get_bb_width(human2.bb), 2) + math.pow(get_bb_height(human2.bb), 2))

        if human_size > human2_size:
            max_acceptable_distance = human_size * self.joint_acceptable_distance_scale_factor_human_size
        else:
            max_acceptable_distance = human2_size * self.joint_acceptable_distance_scale_factor_human_size
        if max_acceptable_distance == 0:
            return 0.0
        joint_similarity_percentages = []
        joint_distances = get_euclidean_joint_distances(human.skeleton_2d, human2.skeleton_2d, SKELETON_PEDREC_IDS, 2,
                                                             min_joint_score=self.min_joint_score_for_similarity)
        for joint_distance in joint_distances:
            # TODO: max_acceptable_distance something more meaningfull
            score = 1 - (joint_distance / max_acceptable_distance)
            if score < 0:
                score = 0
            joint_similarity_percentages.append(score)
        if len(joint_similarity_percentages) == 0:
            return 0
        return sum(joint_similarity_percentages) / len(joint_similarity_percentages)


    def merge_humans(self, humans_detected: List[Human], humans_tracked: List[Human], assign_new_ids: bool = True,
                     similarity_threshold: float = 0.15):

        # TODO: Preselect by bounding box location


        # TODO: Foreach bounding box get pose similarity
        # Get Color similarity
        # Get 3D Pose similarity

        humans_detected.sort(key=lambda human: human.score, reverse=True)
        # num_humans = len(humans)
        humans_to_delete = []
        for human_a_idx, human_a in enumerate(humans_detected):
            # Step 1: Merge duplicates in detected humans
            for human_b_idx in range(human_a_idx + 1, len(humans_detected)):
                human_b = humans_detected[human_b_idx]
                pose_similarity = self.get_pose_similarity(human_a, human_b)
                # print(pose_similarity)
                if pose_similarity > similarity_threshold:
                    if human_b_idx not in humans_to_delete:
                        # Delete the similar humans with a lower score.
                        humans_to_delete.append(human_b_idx)
        for human in sorted(humans_to_delete, reverse=True):
            # print(f"{len(humans_detected)} - {humans_to_delete}")
            humans_detected.pop(human)
            
        # remove tracked duplicates
        humans_to_delete = []
        for human_a_idx, human_a in enumerate(humans_tracked):
            # Step 1: Merge duplicates in tracked humans
            for human_b_idx in range(human_a_idx + 1, len(humans_tracked)):
                human_b = humans_tracked[human_b_idx]
                pose_similarity = self.get_pose_similarity(human_a, human_b)
                # print(pose_similarity)
                if pose_similarity > similarity_threshold:
                    if human_b_idx not in humans_to_delete:
                        # Delete the similar humans with a lower score.
                        humans_to_delete.append(human_b_idx)
        for human in sorted(humans_to_delete, reverse=True):
            # print(f"{len(humans_tracked)} - {humans_to_delete}")
            humans_tracked.pop(human)
        # Step 2: Merge with tracked humans

        for human_a_idx, human_a in enumerate(humans_detected):
            highest_similarity_score = 0.0
            most_similar: Human = None
            similar: List[Human] = []
            for human_b in humans_tracked:
                pose_similarity = self.get_pose_similarity(human_a, human_b)
                if pose_similarity > similarity_threshold:
                    similar.append(human_b)
                    if pose_similarity > highest_similarity_score:
                        most_similar = human_b
                        highest_similarity_score = pose_similarity
            # Set the uid based on the post similar
            if most_similar is not None:
                if human_a.uid == -1 or human_a.uid > most_similar.uid:
                    human_a.uid = most_similar.uid
                for human in similar:
                    humans_tracked.remove(human)
            if human_a.uid == -1 and assign_new_ids:
                human_a.uid = self.next_human_uid
                self.next_human_uid += 1

        return humans_detected, humans_tracked


    # def merge_humans(self, humans_detected: List[Human], humans_tracked: List[Human], img: np.ndarray, assign_new_ids: bool = True,
    #                  similarity_threshold: float = 0.7):
    #     # Step 1. check similarity of detected humans and remove duplicates
    #     human_idxs_to_delete = self.get_duplicated_human_idxs(humans_detected, img)
    #     for human_idx in sorted(human_idxs_to_delete, reverse=True):
    #         humans_detected.pop(human_idx)
    #     # Step 2. get similarity score of detected and tracked humans and merge similars
    #     merged_tracked_human_idxs = []
    #     merged_humans = []
    #     for human_tracked_idx, human_idx in enumerate(humans_tracked):
    #         most_similar: Tuple[float, Human, int] = None
    #         for human_detected_idx, human_detected in enumerate(humans_detected):
    #             human_similarity = get_human_similarity(human_idx, human_detected, img, img)
    #             # print(human_similarity)
    #             if human_similarity >= similarity_threshold:
    #                 if most_similar is not None and most_similar[0] < human_similarity:
    #                     most_similar = (human_similarity, human_detected, human_detected_idx)
    #                 else:
    #                     most_similar = (human_similarity, human_detected, human_detected_idx)
    #         if most_similar is not None:
    #             humans_detected.pop(most_similar[2])
    #             merged_tracked_human_idxs.append(human_tracked_idx)
    #             most_similar_human = most_similar[1]
    #             if most_similar_human.uid == -1 or most_similar_human.uid > human_idx.uid:
    #                 most_similar_human.uid = human_idx.uid
    #             merged_humans.append(most_similar_human)
    #     # Step 3. remove merged tracked humans
    #     for human_idx in sorted(merged_tracked_human_idxs, reverse=True):
    #         humans_tracked.pop(human_idx)
    #
    #     # Step 4. assign new ids to all humans detected but not merged
    #     for human in humans_detected:
    #         if human.uid == -1:
    #             human.uid = self.next_human_uid
    #             self.next_human_uid += 1
    #
    #     # DELLETE DUPLICATES
    #     for human_tracked_idx, human_idx in enumerate(merged_humans):
    #         human_idxs_to_delete = []
    #         for human_detected_idx, human_detected in enumerate(humans_detected):
    #             human_similarity = get_human_similarity(human_idx, human_detected, img, img)
    #             if human_similarity > 0.7:
    #                 human_idxs_to_delete.append(human_detected_idx)
    #         # human_idxs_to_delete = self.get_duplicated_human_idxs(humans_detected, img)
    #         for human_idx in sorted(human_idxs_to_delete, reverse=True):
    #             humans_detected.pop(human_idx)
    #     # Step 5. return humans which are tracked but not detected as undetected, try redetect, then add to list of missing humans which is kept for n frames
    #     print(len(humans_tracked))
    #     return merged_humans + humans_detected, humans_tracked

        #
        #
        # # Step 2: Merge with tracked humans
        #
        # for human_a_idx, human_a in enumerate(humans_detected):
        #     highest_similarity_score = 0.0
        #     most_similar_human: Human = None
        #     similar: List[Human] = []
        #     for human_b in humans_tracked:
        #         pose_similarity = self.get_pose_similarity(human_a, human_b)
        #         if pose_similarity > similarity_threshold:
        #             similar.append(human_b)
        #             if pose_similarity > highest_similarity_score:
        #                 most_similar_human = human_b
        #                 highest_similarity_score = pose_similarity
        #     # Set the uid based on the post similar
        #     if most_similar_human is not None:
        #         if human_a.uid == -1 or human_a.uid > most_similar_human.uid:
        #             human_a.uid = most_similar_human.uid
        #         for human_to_delete in similar:
        #             humans_tracked.remove(human_to_delete)
        #     if human_a.uid == -1 and assign_new_ids:
        #         human_a.uid = self.next_human_uid
        #         self.next_human_uid += 1
        #
        # return humans_detected, humans_tracked
