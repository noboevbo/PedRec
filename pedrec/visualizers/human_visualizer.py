from typing import List

import numpy as np

from pedrec.models.human import Human
from pedrec.visualizers.bb_visualizer import draw_bb
from pedrec.visualizers.skeleton_visualizer import draw_skeleton


def draw_humans(img: np.ndarray, humans: List[Human], only_tracked: bool = False, contains_z: bool = True):
    for human in humans:
        if only_tracked and human.uid == -1:
            continue
        draw_skeleton(img, human.skeleton_2d, contains_z=contains_z)
        if human.uid != -1:
            draw_bb(img, human.bb, title=str(human.uid))
        else:
            draw_bb(img, human.bb)
