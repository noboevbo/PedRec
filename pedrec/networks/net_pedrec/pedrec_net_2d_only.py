import torch.nn as nn

from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.networks.net_pedrec.pedrec_conv_transpose_base import PedRecConvTransposeBase
from pedrec.networks.net_pedrec.pedrec_pose_head_2d import PedRecPose2DHead
from pedrec.networks.net_resnet.resnet_feature_extractor import ResNetHeadless
from pedrec.utils.torch_utils.loss_functions import Pose2DL1Loss


class PedRecNet(nn.Module):
    def __init__(self, cfg: PedRecNetConfig):
        super(PedRecNet, self).__init__()
        self.cfg = cfg
        self.feature_extractor = ResNetHeadless(cfg.layer.block, cfg.layer.layers)
        self.conv_transpose_shared = PedRecConvTransposeBase(cfg, self.feature_extractor.inplanes, num_heads=1)
        self.head_pose_2d = PedRecPose2DHead(cfg, self.conv_transpose_shared.deconv_heads[0])

    def forward(self, x):
        x = self.feature_extractor(x)
        x_deconv = self.conv_transpose_shared(x)
        pose_coords_2d, pose_map_sigmoid_2d = self.head_pose_2d(x_deconv)

        return pose_coords_2d

    def init_weights(self):
        self.conv_transpose_shared.init_weights()
        self.head_pose_2d.init_weights()


class PedRecNetLossHead(nn.Module):
    def __init__(self, device):
        super(PedRecNetLossHead, self).__init__()
        self.pose_loss_2d = Pose2DL1Loss()
        self.device = device

    def forward(self, outputs, targets):
        pose_2d_preds = outputs
        pose_2d_targets = targets["skeleton"]

        return self.pose_loss_2d(pose_2d_preds, pose_2d_targets)
