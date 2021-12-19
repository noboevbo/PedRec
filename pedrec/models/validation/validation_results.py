from dataclasses import dataclass

from pedrec.models.validation.env_position_validation_results import EnvPositionValidationResults
from pedrec.models.validation.orientation_validation_results import OrientationValidationResults
from pedrec.models.validation.pose_2d_validation_conf_results import Pose2DValidationConfResults
from pedrec.models.validation.pose_2d_validation_pck_results import Pose2DValidationPCKResults
from pedrec.models.validation.pose_3d_validation_results import Pose3DValidationResults


@dataclass()
class ValidationResults(object):
    loss: float
    val_duration: float
    pose2d_pck: Pose2DValidationPCKResults = None
    pose2d_conf: Pose2DValidationConfResults = None
    pose3d: Pose3DValidationResults = None
    orientation: OrientationValidationResults = None
    env_position: EnvPositionValidationResults = None
