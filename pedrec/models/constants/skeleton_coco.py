# from enum import Enum
#
# from pedrec.models.data_structures import Color
#
#
# class SKELETON_COCO_JOINT(Enum):
#     nose = 0
#     left_eye = 1
#     right_eye = 2
#     left_ear = 3
#     right_ear = 4
#     left_shoulder = 5
#     right_shoulder = 6
#     left_elbow = 7
#     right_elbow = 8
#     left_wrist = 9
#     right_wrist = 10
#     left_hip = 11
#     right_hip = 12
#     left_knee = 13
#     right_knee = 14
#     left_ankle = 15
#     right_ankle = 16
#
#
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT


SKELETON_COCO_JOINTS = [
    SKELETON_PEDREC_JOINT.nose,
    SKELETON_PEDREC_JOINT.left_eye,
    SKELETON_PEDREC_JOINT.right_eye,
    SKELETON_PEDREC_JOINT.left_ear,
    SKELETON_PEDREC_JOINT.right_ear,
    SKELETON_PEDREC_JOINT.left_shoulder,
    SKELETON_PEDREC_JOINT.right_shoulder,
    SKELETON_PEDREC_JOINT.left_elbow,
    SKELETON_PEDREC_JOINT.right_elbow,
    SKELETON_PEDREC_JOINT.left_wrist,
    SKELETON_PEDREC_JOINT.right_wrist,
    SKELETON_PEDREC_JOINT.left_hip,
    SKELETON_PEDREC_JOINT.right_hip,
    SKELETON_PEDREC_JOINT.left_knee,
    SKELETON_PEDREC_JOINT.right_knee,
    SKELETON_PEDREC_JOINT.left_ankle,
    SKELETON_PEDREC_JOINT.right_ankle
]
#
# JOINT_COLORS = {
#     SKELETON_COCO_JOINT.nose: Color(r=192, g=192, b=192),
#     SKELETON_COCO_JOINT.left_eye: Color(r=255, g=255, b=0),
#     SKELETON_COCO_JOINT.right_eye: Color(r=255, g=0, b=255),
#     SKELETON_COCO_JOINT.left_ear: Color(r=255, g=255, b=102),
#     SKELETON_COCO_JOINT.right_ear: Color(r=255, g=102, b=255),
#     SKELETON_COCO_JOINT.left_shoulder: Color(r=0, g=255, b=0),
#     SKELETON_COCO_JOINT.right_shoulder: Color(r=255, g=0, b=0),
#     SKELETON_COCO_JOINT.left_elbow: Color(r=51, g=255, b=51),
#     SKELETON_COCO_JOINT.right_elbow: Color(r=255, g=51, b=51),
#     SKELETON_COCO_JOINT.left_wrist: Color(r=102, g=255, b=102),
#     SKELETON_COCO_JOINT.right_wrist: Color(r=255, g=102, b=102),
#     SKELETON_COCO_JOINT.left_hip: Color(r=0, g=102, b=0),
#     SKELETON_COCO_JOINT.right_hip: Color(r=102, g=0, b=0),
#     SKELETON_COCO_JOINT.left_knee: Color(r=0, g=153, b=0),
#     SKELETON_COCO_JOINT.right_knee: Color(r=153, g=0, b=0),
#     SKELETON_COCO_JOINT.left_ankle: Color(r=0, g=204, b=0),
#     SKELETON_COCO_JOINT.right_ankle: Color(r=204, g=0, b=0)
# }
#
# SKELETON_COCO_JOINT_COLORS = [
#     JOINT_COLORS[SKELETON_COCO_JOINT.nose],
#     JOINT_COLORS[SKELETON_COCO_JOINT.left_eye],
#     JOINT_COLORS[SKELETON_COCO_JOINT.right_eye],
#     JOINT_COLORS[SKELETON_COCO_JOINT.left_ear],
#     JOINT_COLORS[SKELETON_COCO_JOINT.right_ear],
#     JOINT_COLORS[SKELETON_COCO_JOINT.left_shoulder],
#     JOINT_COLORS[SKELETON_COCO_JOINT.right_shoulder],
#     JOINT_COLORS[SKELETON_COCO_JOINT.left_elbow],
#     JOINT_COLORS[SKELETON_COCO_JOINT.right_elbow],
#     JOINT_COLORS[SKELETON_COCO_JOINT.left_wrist],
#     JOINT_COLORS[SKELETON_COCO_JOINT.right_wrist],
#     JOINT_COLORS[SKELETON_COCO_JOINT.left_hip],
#     JOINT_COLORS[SKELETON_COCO_JOINT.right_hip],
#     JOINT_COLORS[SKELETON_COCO_JOINT.left_knee],
#     JOINT_COLORS[SKELETON_COCO_JOINT.right_knee],
#     JOINT_COLORS[SKELETON_COCO_JOINT.left_ankle],
#     JOINT_COLORS[SKELETON_COCO_JOINT.right_ankle]
# ]
#
# LIMB_COLORS = {
#     'Nose-RShoulder': Color(r=252, g=157, b=154),
#     'Nose-LShoulder': Color(r=200, g=255, b=0),
#     'RShoulder-RElbow': Color(r=252, g=157, b=154),
#     'RElbow-RWrist': Color(r=252, g=157, b=154),
#     'LShoulder-LElbow': Color(r=200, g=255, b=0),
#     'LElbow-LWrist': Color(r=200, g=255, b=0),
#     'LShoulder-LHip': Color(r=124, g=244, b=154),
#     'RShoulder-RHip': Color(r=250, g=2, b=60),
#     'RHip-RKnee': Color(r=250, g=2, b=60),
#     'RKnee-RAnkle': Color(r=250, g=2, b=60),
#     'LHip-LKnee': Color(r=124, g=244, b=154),
#     'LKnee-LAnkle': Color(r=124, g=244, b=154),
#     'Nose-REye': Color(r=252, g=157, b=154),
#     'REye-REar': Color(r=252, g=157, b=154),
#     'Nose-LEye': Color(r=200, g=255, b=0),
#     'LEye-LEar': Color(r=200, g=255, b=0),
# }
#
# SKELETON_COCO_LIMB_COLORS = [
#     LIMB_COLORS['Nose-RShoulder'],
#     LIMB_COLORS['Nose-LShoulder'],
#     LIMB_COLORS['RShoulder-RElbow'],
#     LIMB_COLORS['RElbow-RWrist'],
#     LIMB_COLORS['LShoulder-LElbow'],
#     LIMB_COLORS['LElbow-LWrist'],
#     LIMB_COLORS['LShoulder-LHip'],
#     LIMB_COLORS['RShoulder-RHip'],
#     LIMB_COLORS['RHip-RKnee'],
#     LIMB_COLORS['RKnee-RAnkle'],
#     LIMB_COLORS['LHip-LKnee'],
#     LIMB_COLORS['LKnee-LAnkle'],
#     LIMB_COLORS['Nose-REye'],
#     LIMB_COLORS['REye-REar'],
#     LIMB_COLORS['Nose-LEye'],
#     LIMB_COLORS['LEye-LEar']
# ]
#
# SKELETON_COCO = [
#     (SKELETON_COCO_JOINT.nose.value, SKELETON_COCO_JOINT.right_shoulder.value),  # Nose-RShoulder
#     (SKELETON_COCO_JOINT.nose.value, SKELETON_COCO_JOINT.left_shoulder.value),  # Nose-LShoulder
#     (SKELETON_COCO_JOINT.right_shoulder.value, SKELETON_COCO_JOINT.right_elbow.value),  # RShoulder-RElbow
#     (SKELETON_COCO_JOINT.right_elbow.value, SKELETON_COCO_JOINT.right_wrist.value),  # RElbow-RWrist
#     (SKELETON_COCO_JOINT.left_shoulder.value, SKELETON_COCO_JOINT.left_elbow.value),  # LShoulder-LElbow
#     (SKELETON_COCO_JOINT.left_elbow.value, SKELETON_COCO_JOINT.left_wrist.value),  # LElbow-LWrist
#     (SKELETON_COCO_JOINT.left_shoulder.value, SKELETON_COCO_JOINT.left_hip.value),  # LShoulder-LHip
#     (SKELETON_COCO_JOINT.right_shoulder.value, SKELETON_COCO_JOINT.right_hip.value),  # RShoulder-RHip
#     (SKELETON_COCO_JOINT.right_hip.value, SKELETON_COCO_JOINT.right_knee.value),  # RHip-RKnee
#     (SKELETON_COCO_JOINT.right_knee.value, SKELETON_COCO_JOINT.right_ankle.value),  # RKnee-RAnkle
#     (SKELETON_COCO_JOINT.left_hip.value, SKELETON_COCO_JOINT.left_knee.value),  # LHip-LKnee
#     (SKELETON_COCO_JOINT.left_knee.value, SKELETON_COCO_JOINT.left_ankle.value),  # LKnee-LAnkle
#     (SKELETON_COCO_JOINT.nose.value, SKELETON_COCO_JOINT.right_eye.value),  # Nose-REye
#     (SKELETON_COCO_JOINT.right_eye.value, SKELETON_COCO_JOINT.right_ear.value),  # REye-REar
#     (SKELETON_COCO_JOINT.nose.value, SKELETON_COCO_JOINT.left_eye.value),  # Nose-LEye
#     (SKELETON_COCO_JOINT.left_eye.value, SKELETON_COCO_JOINT.left_ear.value),  # LEye-LEar
# ]
#
# SKELETON_COCO_LR_PAIRS = [
#     (SKELETON_COCO_JOINT.left_eye.value, SKELETON_COCO_JOINT.right_eye.value),  # leye - reye
#     (SKELETON_COCO_JOINT.left_ear.value, SKELETON_COCO_JOINT.right_ear.value),  # lear - rear
#     (SKELETON_COCO_JOINT.left_shoulder.value, SKELETON_COCO_JOINT.right_shoulder.value),  # lshoulder - rshoulder
#     (SKELETON_COCO_JOINT.left_elbow.value, SKELETON_COCO_JOINT.right_elbow.value),  # lelbow - relbow
#     (SKELETON_COCO_JOINT.left_wrist.value, SKELETON_COCO_JOINT.right_wrist.value),  # lwrist - rwrist
#     (SKELETON_COCO_JOINT.left_hip.value, SKELETON_COCO_JOINT.right_hip.value),  # lhip - rhip
#     (SKELETON_COCO_JOINT.left_knee.value, SKELETON_COCO_JOINT.right_knee.value),  # lknee - rknee
#     (SKELETON_COCO_JOINT.left_ankle.value, SKELETON_COCO_JOINT.right_ankle.value)  # lankle - rankle
# ]
#
# SKELETON_COCO_PARENT_CHILD_PAIRS = [
#     (SKELETON_COCO_JOINT.right_shoulder.value, SKELETON_COCO_JOINT.right_elbow.value),  # RShoulder-RElbow
#     (SKELETON_COCO_JOINT.right_elbow.value, SKELETON_COCO_JOINT.right_wrist.value),  # RElbow-RWrist
#
#     (SKELETON_COCO_JOINT.left_shoulder.value, SKELETON_COCO_JOINT.left_elbow.value),  # LShoulder-LElbow
#     (SKELETON_COCO_JOINT.left_elbow.value, SKELETON_COCO_JOINT.left_wrist.value),  # LElbow-LWrist
#
#     (SKELETON_COCO_JOINT.right_shoulder.value, SKELETON_COCO_JOINT.right_hip.value),  # RShoulder-RHip
#     (SKELETON_COCO_JOINT.left_shoulder.value, SKELETON_COCO_JOINT.left_hip.value),  # LShoulder-LHip
#
#     (SKELETON_COCO_JOINT.right_hip.value, SKELETON_COCO_JOINT.right_knee.value),  # RHip-RKnee
#     (SKELETON_COCO_JOINT.right_knee.value, SKELETON_COCO_JOINT.right_ankle.value),  # RKnee-RAnkle
#
#     (SKELETON_COCO_JOINT.left_hip.value, SKELETON_COCO_JOINT.left_knee.value),  # LHip-LKnee
#     (SKELETON_COCO_JOINT.left_knee.value, SKELETON_COCO_JOINT.left_ankle.value),  # LKnee-LAnkle
# ] + SKELETON_COCO_LR_PAIRS
