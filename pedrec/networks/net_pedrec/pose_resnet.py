import torch
import torch.nn as nn

from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.networks.net_pedrec.pedrec_conv_transpose_base import PedRecConvTransposeBase
from pedrec.networks.net_pedrec.pedrec_net_helper import default_init
from pedrec.networks.net_resnet.resnet_feature_extractor import ResNetHeadless
from pedrec.utils.torch_utils.loss_functions import AngularErrorLoss, Pose2DL1Loss, Pose3DL1Loss, JointConfLoss, \
    Pose2DL2Loss, Pose3DL2Loss

class PoseResNetPose2DHead(nn.Module):
    def __init__(self, cfg: PedRecNetConfig, deconv_head: nn.Module):
        super(PoseResNetPose2DHead, self).__init__()
        self.deconv_with_bias = cfg.model.deconv_with_bias
        self.deconv_head = deconv_head
        self.pose_heatmap_layer = nn.Conv2d(
            in_channels=cfg.model.num_deconv_filters[-1],
            out_channels=cfg.model.num_joints,
            kernel_size=cfg.model.final_conv_kernel,
            stride=1,
            padding=1 if cfg.model.final_conv_kernel == 3 else 0
        )

    def forward(self, x_deconv):
        x_deconv_head = self.deconv_head(x_deconv)
        pose_map = self.pose_heatmap_layer(x_deconv_head)
        return pose_map

    def init_weights(self):
        for name, m in self.deconv_head.named_modules():
            default_init(m, name, self.deconv_with_bias)
        for m in self.pose_heatmap_layer.modules():
            default_init(m, "pose_2d_heatmap")

class PoseResNet(nn.Module):
    def __init__(self, cfg: PedRecNetConfig):
        super(PoseResNet, self).__init__()
        self.cfg = cfg
        self.feature_extractor = ResNetHeadless(cfg.layer.block, cfg.layer.layers)
        self.conv_transpose_shared = PedRecConvTransposeBase(cfg, self.feature_extractor.inplanes, num_heads=1)
        self.head_pose_2d = PoseResNetPose2DHead(cfg, self.conv_transpose_shared.deconv_heads[0])

    def forward(self, x):
        x = self.feature_extractor(x)
        x_deconv = self.conv_transpose_shared(x)
        pose_map = self.head_pose_2d(x_deconv)

        return pose_map

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
