from logging import Logger
from typing import Union

import numpy as np
import torch

from pedrec.configs.yolo_v4_config import YoloV4Config
from pedrec.networks.net_yolo_v4.yolov4 import YoloV4


def get_detector(cfg: YoloV4Config, weights_path: str, logger: Logger, device: torch.device):
    detector = YoloV4(cfg, inference=True)
    detector.load_state_dict(torch.load(weights_path))
    model_parameters = filter(lambda p: p.requires_grad, detector.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info("Loaded YoloV4 (weights: {}). Num of trainable params: {}".format(weights_path, num_params))
    detector = detector.to(device)
    detector.eval()
    return detector


def init_pose_model(pose_model: Union[torch.nn.Module],
                    weights_path: str,
                    logger: Logger,
                    device: torch.device):
    pose_model.load_state_dict(torch.load(weights_path))
    model_parameters = filter(lambda p: p.requires_grad, pose_model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(
        "Loaded PoseHRNet (weights: {}). Num of trainable params: {}".format(weights_path, num_params))
    pose_model = pose_model.to(device)
    pose_model.eval()
    return pose_model
