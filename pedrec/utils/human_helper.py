from typing import List, Dict

import numpy as np

from pedrec.models.human import Human
from pedrec.utils.bb_helper import get_human_bb_from_joints


def get_humans_from_pedrec_detections(bbs: List[np.ndarray], pedrec_detections: Dict[str, np.ndarray]) -> List[Human]:
    skeletons = pedrec_detections['skeletons']
    skeletons_3d = pedrec_detections['skeletons_3d']
    orientations = pedrec_detections['orientations']
    assert len(bbs) == len(skeletons)
    humans: List[Human] = []
    for bb, skeleton, skeleton_3d, orientation in zip(bbs, skeletons, skeletons_3d, orientations):
        # uid = int(bb[-1])
        # if uid == -1:
        #     uid = next_human_uid
        #     next_human_uid += 1
        humans.append(Human(bb=get_human_bb_from_joints(skeleton, class_idx=0, confidence=bb[4]),
                            skeleton_2d=skeleton,
                            skeleton_3d=skeleton_3d,
                            orientation=orientation,
                            uid=-1))
        # humans.append(Human(bb=bb, skeleton_2d=skeleton,
        #                     skeleton_3d=skeleton_3d, orientation=orientation, uid=-1))
    return humans
