from dataclasses import dataclass
import numpy as np


@dataclass()
class Pose3DValidationResults(object):
    mpjpe: np.ndarray
    mpjpe_mean: float
    mean_joint_depth_distances: np.ndarray
    mean_joint_x_distances: np.ndarray
    mean_joint_y_distances: np.ndarray
    pct_correct_depth_per_pair: np.ndarray
    pct_correct_depth_mean: float
    pct_correct_joint_position_per_pair: np.ndarray
    pct_correct_joint_position_mean: np.ndarray
    num_examples: int
