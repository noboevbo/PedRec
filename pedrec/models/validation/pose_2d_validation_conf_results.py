from dataclasses import dataclass
import numpy as np


@dataclass()
class Pose2DValidationConfResults(object):
    conf_acc: float
    conf_per_joint_acc: np.ndarray