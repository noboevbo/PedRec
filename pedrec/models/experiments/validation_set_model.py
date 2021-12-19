from dataclasses import dataclass

from torch.utils.data import DataLoader

from pedrec.configs.dataset_configs import PedRecDatasetConfig


@dataclass()
class ValidationSet(object):
    name: str
    loader: DataLoader
    val_set_cfg: PedRecDatasetConfig
    validate_2D: bool = False
    validate_3D: bool = False
    validate_orientation: bool = False
    validate_pose_conf: bool = False
    validate_env_position: bool = False
