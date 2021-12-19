from dataclasses import dataclass
from typing import List

from pedrec.configs.dataset_configs import CocoDatasetConfig
from pedrec.models.constants.action_mappings import ACTION
from pedrec.models.data_structures import ImageSize


@dataclass
class CUDNNConfig:
    benchmark: bool
    deterministic: bool
    enabled: bool


@dataclass
class CUDAConfig:
    cudnn: CUDNNConfig
    use_gpu: bool


@dataclass
class InferenceConfig:
    img_size: ImageSize
    action_list: List[ACTION]
    input_fps: int
    buffer_size: int


action_list_c01 = [
    ACTION.STAND,
    ACTION.IDLE,
    ACTION.WALK,
    ACTION.JOG,
    ACTION.WAVE,
    ACTION.KICK_BALL,
    ACTION.THROW,
    ACTION.LOOK_FOR_TRAFFIC,
    ACTION.HITCHHIKE,
    ACTION.TURN_AROUND,
    ACTION.WORK,
    ACTION.ARGUE,
    ACTION.STUMBLE,
    ACTION.OPEN_DOOR,
    ACTION.FALL,
    ACTION.STAND_UP,
    ACTION.FIGHT,
]

action_list_c01_w_real = action_list_c01 + [
    ACTION.SIT,
    ACTION.JUMP,
    ACTION.WAVE_CAR_OUT
]


@dataclass
class AppConfig:
    ehpi_joint_min_score: float = 0.25
    inference = InferenceConfig(
        img_size=ImageSize(width=1920, height=1080),
        input_fps=30,
        buffer_size=63,
        action_list=action_list_c01_w_real
    )

    cuda = CUDAConfig(
        cudnn=CUDNNConfig(
            benchmark=True,
            deterministic=False,
            enabled=True
        ),
        use_gpu=True
    )

    coco_dataset_config = CocoDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=30,
        use_mebow_orientation=False
    )
