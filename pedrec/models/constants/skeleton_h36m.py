from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT

SKELETON_H36M_JOINTS = [
    SKELETON_PEDREC_JOINT.hip_center,
    SKELETON_PEDREC_JOINT.right_hip,
    SKELETON_PEDREC_JOINT.right_knee,
    SKELETON_PEDREC_JOINT.right_ankle,
    SKELETON_PEDREC_JOINT.left_hip,
    SKELETON_PEDREC_JOINT.left_knee,
    SKELETON_PEDREC_JOINT.left_ankle,
    SKELETON_PEDREC_JOINT.spine_center,
    SKELETON_PEDREC_JOINT.neck,
    SKELETON_PEDREC_JOINT.head_lower,
    SKELETON_PEDREC_JOINT.head_upper,
    SKELETON_PEDREC_JOINT.left_shoulder,
    SKELETON_PEDREC_JOINT.left_elbow,
    SKELETON_PEDREC_JOINT.left_wrist,
    SKELETON_PEDREC_JOINT.right_shoulder,
    SKELETON_PEDREC_JOINT.right_elbow,
    SKELETON_PEDREC_JOINT.right_wrist
]

SKELETON_H36M_HANDFOOTENDS_JOINTS = [
    SKELETON_PEDREC_JOINT.hip_center,
    SKELETON_PEDREC_JOINT.right_hip,
    SKELETON_PEDREC_JOINT.right_knee,
    SKELETON_PEDREC_JOINT.right_ankle,
    SKELETON_PEDREC_JOINT.left_hip,
    SKELETON_PEDREC_JOINT.left_knee,
    SKELETON_PEDREC_JOINT.left_ankle,
    SKELETON_PEDREC_JOINT.spine_center,
    SKELETON_PEDREC_JOINT.neck,
    SKELETON_PEDREC_JOINT.head_lower,
    SKELETON_PEDREC_JOINT.head_upper,
    SKELETON_PEDREC_JOINT.left_shoulder,
    SKELETON_PEDREC_JOINT.left_elbow,
    SKELETON_PEDREC_JOINT.left_wrist,
    SKELETON_PEDREC_JOINT.right_shoulder,
    SKELETON_PEDREC_JOINT.right_elbow,
    SKELETON_PEDREC_JOINT.right_wrist,
    SKELETON_PEDREC_JOINT.left_hand_end,
    SKELETON_PEDREC_JOINT.right_hand_end,
    SKELETON_PEDREC_JOINT.left_foot_end,
    SKELETON_PEDREC_JOINT.right_foot_end
]

#
# class SKELETON_H36M_JOINT(Enum):
#     """
#     Joints with _ prefix are static joints and don't contribute any information, thus they should be ignored.
#     """
#     hip_center = 0  # Hips
#     right_hip = 1  # RightUpLeg
#     right_knee = 2  # RightLeg
#     right_ankle = 3  # RightFoot
#     right_foot_end = 4  # RightToeBase
#     _site_1 = 5  # Site - ????
#     left_hip = 6  # LeftUpLeg
#     left_knee = 7  # LeftLeg
#     left_ankle = 8  # LeftFoot
#     left_foot_end = 9  # LeftToeBase
#     _site_2 = 10  # Site - ????
#     _spine_1 = 11  # Spine
#     spine_center = 12  # Spine1
#     neck = 13  # Neck
#     head_lower = 14  # Head
#     head_upper = 15  # Site
#     _left_neck = 16  # LShoulder
#     left_shoulder = 17  # LeftArm
#     left_elbow = 18  # LeftForeArm
#     left_wrist = 19  # LeftHand
#     _left_hand_thumb = 20  # LeftHandThumb
#     _site_3 = 21  # Site
#     left_hand_end = 22  # L_Wrist_End
#     _site_4 = 23  # Site
#     _right_neck = 24  # RightShoulder
#     right_shoulder = 25  # RightArm
#     right_elbow = 26  # RightForeArm
#     right_wrist = 27  # RightHand
#     _right_hand_thumb = 28  # RightHandThumb
#     _site_5 = 29  # Site
#     right_hand_end = 30  # R_Wrist_End
#     _site_6 = 31  # Site
#
# SKELETON_H36M_JOINTS = [
#     SKELETON_H36M_JOINT.hip_center,
#     SKELETON_H36M_JOINT.right_hip,
#     SKELETON_H36M_JOINT.right_knee,
#     SKELETON_H36M_JOINT.right_ankle,
#     SKELETON_H36M_JOINT.right_foot_end,
#     SKELETON_H36M_JOINT._site_1,
#     SKELETON_H36M_JOINT.left_hip,
#     SKELETON_H36M_JOINT.left_knee,
#     SKELETON_H36M_JOINT.left_ankle,
#     SKELETON_H36M_JOINT.left_foot_end,
#     SKELETON_H36M_JOINT._site_2,
#     SKELETON_H36M_JOINT._spine_1,
#     SKELETON_H36M_JOINT.spine_center,
#     SKELETON_H36M_JOINT.neck,
#     SKELETON_H36M_JOINT.head_lower,
#     SKELETON_H36M_JOINT.head_upper,
#     SKELETON_H36M_JOINT._left_neck,
#     SKELETON_H36M_JOINT.left_shoulder,
#     SKELETON_H36M_JOINT.left_elbow,
#     SKELETON_H36M_JOINT.left_wrist,
#     SKELETON_H36M_JOINT._left_hand_thumb,
#     SKELETON_H36M_JOINT._site_3,
#     SKELETON_H36M_JOINT.left_hand_end,
#     SKELETON_H36M_JOINT._site_4,
#     SKELETON_H36M_JOINT._right_neck,
#     SKELETON_H36M_JOINT.right_shoulder,
#     SKELETON_H36M_JOINT.right_elbow,
#     SKELETON_H36M_JOINT.right_wrist,
#     SKELETON_H36M_JOINT._right_hand_thumb,
#     SKELETON_H36M_JOINT._site_5,
#     SKELETON_H36M_JOINT.right_hand_end,
#     SKELETON_H36M_JOINT._site_6
# ]
#
# SKELETON_H36M_LEFT = [
#     SKELETON_H36M_JOINT.left_hip.value,
#     SKELETON_H36M_JOINT.left_knee.value,
#     SKELETON_H36M_JOINT.left_ankle.value,
#     SKELETON_H36M_JOINT.left_foot_end.value,
#     SKELETON_H36M_JOINT._site_2.value,
#     SKELETON_H36M_JOINT._left_neck.value,
#     SKELETON_H36M_JOINT.left_shoulder.value,
#     SKELETON_H36M_JOINT.left_elbow.value,
#     SKELETON_H36M_JOINT.left_wrist.value,
#     SKELETON_H36M_JOINT._left_hand_thumb.value,
#     SKELETON_H36M_JOINT._site_3.value,
#     SKELETON_H36M_JOINT.left_hand_end.value,
#     SKELETON_H36M_JOINT._site_4.value
# ]
#
# SKELETON_H36M_RIGHT = [
#     SKELETON_H36M_JOINT.right_hip.value,
#     SKELETON_H36M_JOINT.right_knee.value,
#     SKELETON_H36M_JOINT.right_ankle.value,
#     SKELETON_H36M_JOINT.right_foot_end.value,
#     SKELETON_H36M_JOINT._site_1.value,
#     SKELETON_H36M_JOINT._right_neck.value,
#     SKELETON_H36M_JOINT.right_shoulder.value,
#     SKELETON_H36M_JOINT.right_elbow.value,
#     SKELETON_H36M_JOINT.right_wrist.value,
#     SKELETON_H36M_JOINT._right_hand_thumb.value,
#     SKELETON_H36M_JOINT._site_5.value,
#     SKELETON_H36M_JOINT.right_hand_end.value,
#     SKELETON_H36M_JOINT._site_6.value
# ]
#
# # static joints which do not contribute any valuable information to the skeleton and thus should be removed
# SKELETON_H36M_STATIC = [
#     SKELETON_H36M_JOINT._site_1.value,
#     SKELETON_H36M_JOINT._site_2.value,
#     SKELETON_H36M_JOINT._spine_1.value,
#     SKELETON_H36M_JOINT._left_neck.value,
#     SKELETON_H36M_JOINT._left_hand_thumb.value,
#     SKELETON_H36M_JOINT._site_3.value,
#     SKELETON_H36M_JOINT._site_4.value,
#     SKELETON_H36M_JOINT._right_neck.value,
#     SKELETON_H36M_JOINT._right_hand_thumb.value,
#     SKELETON_H36M_JOINT._site_5.value,
#     SKELETON_H36M_JOINT._site_6.value,
# ]
#
# # additional joints which are not part of the 17 joint base skeleton used in most papers, but are not static and thus
# # provide valuable information
# SKELETON_H36M_ADDITIONAL = [
#     SKELETON_H36M_JOINT.right_foot_end.value,
#     SKELETON_H36M_JOINT.left_foot_end.value,
#     SKELETON_H36M_JOINT.left_hand_end.value,
#     SKELETON_H36M_JOINT.right_hand_end.value,
# ]