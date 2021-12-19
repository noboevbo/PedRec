from dataclasses import dataclass
import numpy as np


@dataclass()
class Pose2DValidationPCKResults(object):
    pck_05: np.ndarray
    pck_2: np.ndarray
    pck_05_mean: float
    pck_2_mean: float