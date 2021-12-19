from dataclasses import dataclass
from enum import Enum

from pedrec.models.data_structures import Color

class SKELETON_PEDREC_JOINT(Enum):
    nose = 0
    left_eye = 1
    right_eye = 2
    left_ear = 3
    right_ear = 4
    left_shoulder = 5
    right_shoulder = 6
    left_elbow = 7
    right_elbow = 8
    left_wrist = 9
    right_wrist = 10
    left_hip = 11
    right_hip = 12
    left_knee = 13
    right_knee = 14
    left_ankle = 15
    right_ankle = 16
    hip_center = 17
    spine_center = 18
    neck = 19
    head_lower = 20
    head_upper = 21
    left_foot_end = 22
    right_foot_end = 23
    left_hand_end = 24
    right_hand_end = 25


SKELETON_PEDREC_JOINTS = [
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
    SKELETON_PEDREC_JOINT.right_ankle,
    SKELETON_PEDREC_JOINT.hip_center,
    SKELETON_PEDREC_JOINT.spine_center,
    SKELETON_PEDREC_JOINT.neck,
    SKELETON_PEDREC_JOINT.head_lower,
    SKELETON_PEDREC_JOINT.head_upper,
    SKELETON_PEDREC_JOINT.left_foot_end,
    SKELETON_PEDREC_JOINT.right_foot_end,
    SKELETON_PEDREC_JOINT.left_hand_end,
    SKELETON_PEDREC_JOINT.right_hand_end
]

SKELETON_PEDREC_JOINTS_EHPI3D = [
    # Head area
    SKELETON_PEDREC_JOINT.head_upper,
    SKELETON_PEDREC_JOINT.left_ear,
    SKELETON_PEDREC_JOINT.right_ear,
    SKELETON_PEDREC_JOINT.nose,
    SKELETON_PEDREC_JOINT.left_eye,
    SKELETON_PEDREC_JOINT.right_eye,
    SKELETON_PEDREC_JOINT.head_lower,
    SKELETON_PEDREC_JOINT.neck,

    # Upper Left
    SKELETON_PEDREC_JOINT.left_shoulder,
    SKELETON_PEDREC_JOINT.left_elbow,
    SKELETON_PEDREC_JOINT.left_wrist,
    SKELETON_PEDREC_JOINT.left_hand_end,

    SKELETON_PEDREC_JOINT.spine_center,

    # Upper Right
    SKELETON_PEDREC_JOINT.right_shoulder,
    SKELETON_PEDREC_JOINT.right_elbow,
    SKELETON_PEDREC_JOINT.right_wrist,
    SKELETON_PEDREC_JOINT.right_hand_end,

    # Lower Left
    SKELETON_PEDREC_JOINT.left_hip,
    SKELETON_PEDREC_JOINT.left_knee,
    SKELETON_PEDREC_JOINT.left_ankle,
    SKELETON_PEDREC_JOINT.left_foot_end,

    SKELETON_PEDREC_JOINT.hip_center,

    # Lower Right
    SKELETON_PEDREC_JOINT.right_hip,
    SKELETON_PEDREC_JOINT.right_knee,
    SKELETON_PEDREC_JOINT.right_ankle,
    SKELETON_PEDREC_JOINT.right_foot_end,
]

SKELETON_PEDREC_TO_PEDRECEHPI3D = [SKELETON_PEDREC_JOINTS_EHPI3D.index(joint) for joint in SKELETON_PEDREC_JOINTS]

JOINT_COLORS = {
    SKELETON_PEDREC_JOINT.nose: Color(r=192, g=192, b=192),
    SKELETON_PEDREC_JOINT.left_eye: Color(r=255, g=255, b=0),
    SKELETON_PEDREC_JOINT.right_eye: Color(r=255, g=0, b=255),
    SKELETON_PEDREC_JOINT.left_ear: Color(r=255, g=255, b=102),
    SKELETON_PEDREC_JOINT.right_ear: Color(r=255, g=102, b=255),
    SKELETON_PEDREC_JOINT.left_shoulder: Color(r=0, g=255, b=0),
    SKELETON_PEDREC_JOINT.right_shoulder: Color(r=255, g=0, b=0),
    SKELETON_PEDREC_JOINT.left_elbow: Color(r=51, g=255, b=51),
    SKELETON_PEDREC_JOINT.right_elbow: Color(r=255, g=51, b=51),
    SKELETON_PEDREC_JOINT.left_wrist: Color(r=102, g=255, b=102),
    SKELETON_PEDREC_JOINT.right_wrist: Color(r=255, g=102, b=102),
    SKELETON_PEDREC_JOINT.left_hip: Color(r=0, g=102, b=0),
    SKELETON_PEDREC_JOINT.right_hip: Color(r=102, g=0, b=0),
    SKELETON_PEDREC_JOINT.left_knee: Color(r=0, g=153, b=0),
    SKELETON_PEDREC_JOINT.right_knee: Color(r=153, g=0, b=0),
    SKELETON_PEDREC_JOINT.left_ankle: Color(r=0, g=204, b=0),
    SKELETON_PEDREC_JOINT.right_ankle: Color(r=204, g=0, b=0),
    SKELETON_PEDREC_JOINT.hip_center: Color(r=0, g=0, b=0),
    SKELETON_PEDREC_JOINT.spine_center: Color(r=51, g=51, b=51),
    SKELETON_PEDREC_JOINT.neck: Color(r=101, g=101, b=101),
    SKELETON_PEDREC_JOINT.head_lower: Color(r=148, g=148, b=148),
    SKELETON_PEDREC_JOINT.head_upper: Color(r=224, g=224, b=224),
    SKELETON_PEDREC_JOINT.left_foot_end: Color(r=0, g=255, b=0),
    SKELETON_PEDREC_JOINT.right_foot_end: Color(r=255, g=0, b=0),
    SKELETON_PEDREC_JOINT.left_hand_end: Color(r=152, g=255, b=152),
    SKELETON_PEDREC_JOINT.right_hand_end: Color(r=255, g=152, b=152)
}

SKELETON_PEDREC_JOINT_COLORS = [
    JOINT_COLORS[SKELETON_PEDREC_JOINT.nose],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.left_eye],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.right_eye],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.left_ear],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.right_ear],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.left_shoulder],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.right_shoulder],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.left_elbow],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.right_elbow],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.left_wrist],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.right_wrist],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.left_hip],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.right_hip],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.left_knee],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.right_knee],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.left_ankle],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.right_ankle],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.hip_center],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.spine_center],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.neck],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.head_lower],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.head_upper],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.left_foot_end],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.right_foot_end],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.left_hand_end],
    JOINT_COLORS[SKELETON_PEDREC_JOINT.right_hand_end]
]

LIMB_COLORS = {
    'right_shoulder-right_elbow': Color(r=252, g=157, b=154),
    'right_elbow-right_wrist': Color(r=252, g=157, b=154),
    'left_shoulder-left_elbow': Color(r=200, g=255, b=0),
    'left_elbow-left_wrist': Color(r=200, g=255, b=0),
    'right_hip-right_knee': Color(r=250, g=2, b=60),
    'right_knee-right_ankle': Color(r=250, g=2, b=60),
    'left_hip-left_knee': Color(r=124, g=244, b=154),
    'left_knee-left_ankle': Color(r=124, g=244, b=154),
    'nose-right_eye': Color(r=252, g=157, b=154),
    'right_eye-right_ear': Color(r=252, g=157, b=154),
    'nose-left_eye': Color(r=200, g=255, b=0),
    'left_eye-left_ear': Color(r=200, g=255, b=0),
    'hip_center-left_hip': Color(r=124, g=244, b=154),
    'hip_center-right_hip': Color(r=250, g=2, b=60),
    'hip_center-spine_center': Color(r=124, g=124, b=124),
    'spine_center-neck': Color(r=124, g=124, b=124),
    'neck-head_lower': Color(r=124, g=124, b=124),
    'head_lower-head_upper': Color(r=124, g= 124, b= 124),
    'neck-left_shoulder': Color(r=200, g=255, b=0),
    'neck-right_shoulder': Color(r=252, g=157, b=154),
    'head_lower-nose': Color(r=200, g=200, b=200),
    'left_ankle-left_foot_end': Color(r=124, g=244, b=154),
    'right_ankle-right_foot_end': Color(r=250, g=2, b=60),
    'left_wrist-left_hand_end': Color(r=200, g=255, b=0),
    'right_wrist-right_hand_end': Color(r=252, g=157, b=154)
}

SKELETON_PEDREC_LIMB_COLORS = [
    LIMB_COLORS['right_shoulder-right_elbow'],
    LIMB_COLORS['right_elbow-right_wrist'],
    LIMB_COLORS['left_shoulder-left_elbow'],
    LIMB_COLORS['left_elbow-left_wrist'],
    LIMB_COLORS['right_hip-right_knee'],
    LIMB_COLORS['right_knee-right_ankle'],
    LIMB_COLORS['left_hip-left_knee'],
    LIMB_COLORS['left_knee-left_ankle'],
    LIMB_COLORS['nose-right_eye'],
    LIMB_COLORS['right_eye-right_ear'],
    LIMB_COLORS['nose-left_eye'],
    LIMB_COLORS['left_eye-left_ear'],
    LIMB_COLORS['hip_center-left_hip'],
    LIMB_COLORS['hip_center-right_hip'],
    LIMB_COLORS['hip_center-spine_center'],
    LIMB_COLORS['spine_center-neck'],
    LIMB_COLORS['neck-head_lower'],
    LIMB_COLORS['head_lower-head_upper'],
    LIMB_COLORS['neck-left_shoulder'],
    LIMB_COLORS['neck-right_shoulder'],
    LIMB_COLORS['head_lower-nose'],
    LIMB_COLORS['left_ankle-left_foot_end'],
    LIMB_COLORS['right_ankle-right_foot_end'],
    LIMB_COLORS['left_wrist-left_hand_end'],
    LIMB_COLORS['right_wrist-right_hand_end'],
]

SKELETON_PEDREC = [
    (SKELETON_PEDREC_JOINT.right_shoulder.value, SKELETON_PEDREC_JOINT.right_elbow.value),
    (SKELETON_PEDREC_JOINT.right_elbow.value, SKELETON_PEDREC_JOINT.right_wrist.value),
    (SKELETON_PEDREC_JOINT.left_shoulder.value, SKELETON_PEDREC_JOINT.left_elbow.value),
    (SKELETON_PEDREC_JOINT.left_elbow.value, SKELETON_PEDREC_JOINT.left_wrist.value),
    (SKELETON_PEDREC_JOINT.right_hip.value, SKELETON_PEDREC_JOINT.right_knee.value),
    (SKELETON_PEDREC_JOINT.right_knee.value, SKELETON_PEDREC_JOINT.right_ankle.value),
    (SKELETON_PEDREC_JOINT.left_hip.value, SKELETON_PEDREC_JOINT.left_knee.value),
    (SKELETON_PEDREC_JOINT.left_knee.value, SKELETON_PEDREC_JOINT.left_ankle.value),
    (SKELETON_PEDREC_JOINT.nose.value, SKELETON_PEDREC_JOINT.right_eye.value),
    (SKELETON_PEDREC_JOINT.right_eye.value, SKELETON_PEDREC_JOINT.right_ear.value),
    (SKELETON_PEDREC_JOINT.nose.value, SKELETON_PEDREC_JOINT.left_eye.value),
    (SKELETON_PEDREC_JOINT.left_eye.value, SKELETON_PEDREC_JOINT.left_ear.value),
    (SKELETON_PEDREC_JOINT.hip_center.value, SKELETON_PEDREC_JOINT.left_hip.value),
    (SKELETON_PEDREC_JOINT.hip_center.value, SKELETON_PEDREC_JOINT.right_hip.value),
    (SKELETON_PEDREC_JOINT.hip_center.value, SKELETON_PEDREC_JOINT.spine_center.value),
    (SKELETON_PEDREC_JOINT.spine_center.value, SKELETON_PEDREC_JOINT.neck.value),
    (SKELETON_PEDREC_JOINT.neck.value, SKELETON_PEDREC_JOINT.head_lower.value),
    (SKELETON_PEDREC_JOINT.head_lower.value, SKELETON_PEDREC_JOINT.head_upper.value),
    (SKELETON_PEDREC_JOINT.neck.value, SKELETON_PEDREC_JOINT.left_shoulder.value),
    (SKELETON_PEDREC_JOINT.neck.value, SKELETON_PEDREC_JOINT.right_shoulder.value),
    (SKELETON_PEDREC_JOINT.head_lower.value, SKELETON_PEDREC_JOINT.nose.value),
    (SKELETON_PEDREC_JOINT.left_ankle.value, SKELETON_PEDREC_JOINT.left_foot_end.value),
    (SKELETON_PEDREC_JOINT.right_ankle.value, SKELETON_PEDREC_JOINT.right_foot_end.value),
    (SKELETON_PEDREC_JOINT.left_wrist.value, SKELETON_PEDREC_JOINT.left_hand_end.value),
    (SKELETON_PEDREC_JOINT.right_wrist.value, SKELETON_PEDREC_JOINT.right_hand_end.value)
]

SKELETON_PEDREC_LR_PAIRS = [
    (SKELETON_PEDREC_JOINT.left_eye.value, SKELETON_PEDREC_JOINT.right_eye.value),  # left_eye - right_eye
    (SKELETON_PEDREC_JOINT.left_ear.value, SKELETON_PEDREC_JOINT.right_ear.value),  # lear - right_ear
    (SKELETON_PEDREC_JOINT.left_shoulder.value, SKELETON_PEDREC_JOINT.right_shoulder.value),  # left_shoulder - right_shoulder
    (SKELETON_PEDREC_JOINT.left_elbow.value, SKELETON_PEDREC_JOINT.right_elbow.value),  # left_elbow - right_elbow
    (SKELETON_PEDREC_JOINT.left_wrist.value, SKELETON_PEDREC_JOINT.right_wrist.value),  # left_wrist - right_wrist
    (SKELETON_PEDREC_JOINT.left_hip.value, SKELETON_PEDREC_JOINT.right_hip.value),  # left_hip - right_hip
    (SKELETON_PEDREC_JOINT.left_knee.value, SKELETON_PEDREC_JOINT.right_knee.value),  # left_knee - right_knee
    (SKELETON_PEDREC_JOINT.left_ankle.value, SKELETON_PEDREC_JOINT.right_ankle.value),  # left_ankle - right_ankle
    (SKELETON_PEDREC_JOINT.left_foot_end.value, SKELETON_PEDREC_JOINT.right_foot_end.value),
    (SKELETON_PEDREC_JOINT.left_hand_end.value, SKELETON_PEDREC_JOINT.right_hand_end.value)
]

SKELETON_PEDREC_LEFT = [
    SKELETON_PEDREC_JOINT.left_eye.value,
    SKELETON_PEDREC_JOINT.left_ear.value,
    SKELETON_PEDREC_JOINT.left_shoulder.value,
    SKELETON_PEDREC_JOINT.left_elbow.value,
    SKELETON_PEDREC_JOINT.left_wrist.value,
    SKELETON_PEDREC_JOINT.left_hip.value,
    SKELETON_PEDREC_JOINT.left_knee.value,
    SKELETON_PEDREC_JOINT.left_ankle.value,
    SKELETON_PEDREC_JOINT.left_foot_end.value,
    SKELETON_PEDREC_JOINT.left_hand_end.value,
]

SKELETON_PEDREC_RIGHT = [
    SKELETON_PEDREC_JOINT.right_eye.value,
    SKELETON_PEDREC_JOINT.right_ear.value,
    SKELETON_PEDREC_JOINT.right_shoulder.value,
    SKELETON_PEDREC_JOINT.right_elbow.value,
    SKELETON_PEDREC_JOINT.right_wrist.value,
    SKELETON_PEDREC_JOINT.right_hip.value,
    SKELETON_PEDREC_JOINT.right_knee.value,
    SKELETON_PEDREC_JOINT.right_ankle.value,
    SKELETON_PEDREC_JOINT.right_foot_end.value,
    SKELETON_PEDREC_JOINT.right_hand_end.value,
]

SKELETON_PEDREC_PARENT_CHILD_PAIRS = [
    (SKELETON_PEDREC_JOINT.hip_center.value, SKELETON_PEDREC_JOINT.spine_center.value),
    (SKELETON_PEDREC_JOINT.spine_center.value, SKELETON_PEDREC_JOINT.neck.value),
    (SKELETON_PEDREC_JOINT.neck.value, SKELETON_PEDREC_JOINT.head_lower.value),
    (SKELETON_PEDREC_JOINT.head_lower.value, SKELETON_PEDREC_JOINT.head_upper.value),

    (SKELETON_PEDREC_JOINT.right_shoulder.value, SKELETON_PEDREC_JOINT.right_elbow.value),  # right_shoulder-right_elbow
    (SKELETON_PEDREC_JOINT.right_elbow.value, SKELETON_PEDREC_JOINT.right_wrist.value),  # right_elbow-right_wrist

    (SKELETON_PEDREC_JOINT.left_shoulder.value, SKELETON_PEDREC_JOINT.left_elbow.value),  # left_shoulder-left_elbow
    (SKELETON_PEDREC_JOINT.left_elbow.value, SKELETON_PEDREC_JOINT.left_wrist.value),  # left_elbow-left_wrist

    (SKELETON_PEDREC_JOINT.right_shoulder.value, SKELETON_PEDREC_JOINT.right_hip.value),  # right_shoulder-right_hip
    (SKELETON_PEDREC_JOINT.left_shoulder.value, SKELETON_PEDREC_JOINT.left_hip.value),  # left_shoulder-left_hip

    (SKELETON_PEDREC_JOINT.right_hip.value, SKELETON_PEDREC_JOINT.right_knee.value),  # right_hip-right_knee
    (SKELETON_PEDREC_JOINT.right_knee.value, SKELETON_PEDREC_JOINT.right_ankle.value),  # right_knee-right_ankle
    (SKELETON_PEDREC_JOINT.right_ankle.value, SKELETON_PEDREC_JOINT.right_foot_end.value),

    (SKELETON_PEDREC_JOINT.left_hip.value, SKELETON_PEDREC_JOINT.left_knee.value),  # left_hip-left_knee
    (SKELETON_PEDREC_JOINT.left_knee.value, SKELETON_PEDREC_JOINT.left_ankle.value),  # left_knee-left_ankle
    (SKELETON_PEDREC_JOINT.left_ankle.value, SKELETON_PEDREC_JOINT.left_foot_end.value)
] + SKELETON_PEDREC_LR_PAIRS

SKELETON_PEDREC_MIDDLE_AXIS = [
    SKELETON_PEDREC_JOINT.nose,
    SKELETON_PEDREC_JOINT.neck,
    SKELETON_PEDREC_JOINT.spine_center,
    SKELETON_PEDREC_JOINT.hip_center,
]

SKELETON_PEDREC_EXTREMITIES = [
    SKELETON_PEDREC_JOINT.right_elbow,
    SKELETON_PEDREC_JOINT.right_wrist,
    SKELETON_PEDREC_JOINT.left_elbow,
    SKELETON_PEDREC_JOINT.left_wrist,
    SKELETON_PEDREC_JOINT.right_knee,
    SKELETON_PEDREC_JOINT.right_ankle,
    SKELETON_PEDREC_JOINT.left_knee,
    SKELETON_PEDREC_JOINT.left_ankle
]

SKELETON_PEDREC_IDS = [joint.value for joint in SKELETON_PEDREC_JOINTS]
SKELETON_PEDREC_MIDDLE_AXIS_IDS = [joint.value for joint in SKELETON_PEDREC_MIDDLE_AXIS]
SKELETON_PEDREC_EXTREMITIES_IDS = [joint.value for joint in SKELETON_PEDREC_EXTREMITIES]

SKELETON_PEDREC_BUILDER = [
    # used to build a normalized skeleton based on parent / child
    # joint_a value, joint_b value, limb length
    # based on the Vitruvian Man, 4 units = 1 Head Length, max_height (arms above head = 38 units + buffer = 40 units)

    # middle axis
    (SKELETON_PEDREC_JOINT.hip_center.value, SKELETON_PEDREC_JOINT.spine_center.value, 5),
    (SKELETON_PEDREC_JOINT.spine_center.value, SKELETON_PEDREC_JOINT.neck.value, 6),
    (SKELETON_PEDREC_JOINT.neck.value, SKELETON_PEDREC_JOINT.head_lower.value, 1),
    (SKELETON_PEDREC_JOINT.head_lower.value, SKELETON_PEDREC_JOINT.head_upper.value, 4),

    # upper left
    (SKELETON_PEDREC_JOINT.neck.value, SKELETON_PEDREC_JOINT.left_shoulder.value, 3),
    (SKELETON_PEDREC_JOINT.left_shoulder.value, SKELETON_PEDREC_JOINT.left_elbow.value, 6),
    (SKELETON_PEDREC_JOINT.left_elbow.value, SKELETON_PEDREC_JOINT.left_wrist.value, 5),
    (SKELETON_PEDREC_JOINT.left_wrist.value, SKELETON_PEDREC_JOINT.left_hand_end.value, 3),

    # upper right
    (SKELETON_PEDREC_JOINT.neck.value, SKELETON_PEDREC_JOINT.right_shoulder.value, 3),
    (SKELETON_PEDREC_JOINT.right_shoulder.value, SKELETON_PEDREC_JOINT.right_elbow.value, 6),
    (SKELETON_PEDREC_JOINT.right_elbow.value, SKELETON_PEDREC_JOINT.right_wrist.value, 5),
    (SKELETON_PEDREC_JOINT.right_wrist.value, SKELETON_PEDREC_JOINT.right_hand_end.value, 3),

    # lower left
    (SKELETON_PEDREC_JOINT.hip_center.value, SKELETON_PEDREC_JOINT.left_hip.value, 2),
    (SKELETON_PEDREC_JOINT.left_hip.value, SKELETON_PEDREC_JOINT.left_knee.value, 8),
    (SKELETON_PEDREC_JOINT.left_knee.value, SKELETON_PEDREC_JOINT.left_ankle.value, 8),
    (SKELETON_PEDREC_JOINT.left_ankle.value, SKELETON_PEDREC_JOINT.left_foot_end.value, 4),

    # lower right
    (SKELETON_PEDREC_JOINT.hip_center.value, SKELETON_PEDREC_JOINT.right_hip.value, 2),
    (SKELETON_PEDREC_JOINT.right_hip.value, SKELETON_PEDREC_JOINT.right_knee.value, 8),
    (SKELETON_PEDREC_JOINT.right_knee.value, SKELETON_PEDREC_JOINT.right_ankle.value, 8),
    (SKELETON_PEDREC_JOINT.right_ankle.value, SKELETON_PEDREC_JOINT.right_foot_end.value, 4),

    # Head
    (SKELETON_PEDREC_JOINT.head_lower.value, SKELETON_PEDREC_JOINT.nose.value, 2.25),
    (SKELETON_PEDREC_JOINT.nose.value, SKELETON_PEDREC_JOINT.right_eye.value, 1),
    (SKELETON_PEDREC_JOINT.right_eye.value, SKELETON_PEDREC_JOINT.right_ear.value, 1.25),
    (SKELETON_PEDREC_JOINT.nose.value, SKELETON_PEDREC_JOINT.left_eye.value, 1),
    (SKELETON_PEDREC_JOINT.left_eye.value, SKELETON_PEDREC_JOINT.left_ear.value, 1.25),
]

# x = [0, 0, 0] + SKELETON_PEDREC_JOINTS_EHPI3D + [0, 0, 0, 0, 0, 0, 0] + [0, 0, 0]
#
# for i in range(0, 38, 2):
#     text = ""
#     for j in range(i, i+7):
#         val = x[j]
#         if val == 0:
#             text = f"{text}{val},"
#             continue
#         text = f"{text}{val.name},"
#
#     text = text[:-1]
#     print(text)