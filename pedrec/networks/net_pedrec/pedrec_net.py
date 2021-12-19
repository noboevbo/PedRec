import torch
import torch.nn as nn

from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.networks.net_pedrec.pedrec_conv_transpose_base import PedRecConvTransposeBase
from pedrec.networks.net_pedrec.pedrec_orientation_head_shared import PedRecOrientationsHead
from pedrec.networks.net_pedrec.pedrec_pose_conf_head import PedRecPoseConfHead
from pedrec.networks.net_pedrec.pedrec_pose_head_2d import PedRecPose2DHead
from pedrec.networks.net_pedrec.pedrec_pose_head_3d import PedRecPose3DHead
from pedrec.networks.net_resnet.resnet_feature_extractor import ResNetHeadless
from pedrec.utils.torch_utils.loss_functions import AngularErrorLoss, Pose2DL1Loss, Pose3DL1Loss, JointConfLoss


class PedRecNet(nn.Module):
    def __init__(self, cfg: PedRecNetConfig):
        super(PedRecNet, self).__init__()
        self.cfg = cfg
        self.feature_extractor = ResNetHeadless(cfg.layer.block, cfg.layer.layers)
        self.conv_transpose_shared = PedRecConvTransposeBase(cfg, self.feature_extractor.inplanes, num_heads=2)
        self.head_pose_2d = PedRecPose2DHead(cfg, self.conv_transpose_shared.deconv_heads[0])
        self.head_pose_3d = PedRecPose3DHead(cfg, self.conv_transpose_shared.deconv_heads[1])
        self.head_orientation = PedRecOrientationsHead(cfg)
        self.head_conf = PedRecPoseConfHead(cfg)

    def forward(self, x):
        x = self.feature_extractor(x)
        x_deconv = self.conv_transpose_shared(x)
        pose_coords_2d, pose_map_2d = self.head_pose_2d(x_deconv)
        pose_coords_3d, pose_map_3d = self.head_pose_3d(x_deconv)
        pose_conf = torch.unsqueeze(self.head_conf(pose_map_2d, pose_map_3d), dim=2)
        pose_coords_2d = torch.cat([pose_coords_2d, pose_conf], dim=2)
        pose_coords_3d = torch.cat([pose_coords_3d, pose_conf], dim=2)
        orientations, theta_map, phi_map = self.head_orientation(x, pose_coords_3d)

        return pose_coords_2d, pose_coords_3d, orientations, pose_map_2d, pose_map_3d, theta_map, phi_map

    def init_weights(self):
        self.conv_transpose_shared.init_weights()
        self.head_pose_2d.init_weights()
        self.head_pose_3d.init_weights()
        self.head_orientation.init_weights()
        self.head_conf.init_weights()


class PedRecNetLossHead(nn.Module):
    def __init__(self, device, use_p3d_loss: bool = True, use_orientation_loss: bool = True,
                 use_conf_loss: bool = True):
        super(PedRecNetLossHead, self).__init__()
        self.pose_loss_2d = Pose2DL1Loss()
        self.pose_loss_3d = Pose3DL1Loss()
        self.orientation_loss = AngularErrorLoss()
        self.coord_conf_loss = JointConfLoss()
        self.sigmas = nn.Parameter(torch.ones(4))
        self.device = device
        self.use_p3d_loss = use_p3d_loss
        self.use_orientation_loss = use_orientation_loss
        self.use_conf_loss = use_conf_loss

    def reset_sigmas(self):
        nn.init.constant_(self.sigmas, 1)

    def forward(self, outputs, targets):
        pose_2d_preds, pose_3d_preds, orientation_preds = outputs[0:3]
        pose_2d_targets, pose_3d_targets, orientation_targets = targets["skeleton"], targets["skeleton_3d"], targets["orientation"]

        sigma_prod = 1
        loss = 0
        p2d_loss = self.pose_loss_2d(pose_2d_preds, pose_2d_targets)
        if not torch.isnan(p2d_loss):
            loss += (1 / (2 * self.sigmas[0] ** 2)) * p2d_loss
            sigma_prod = sigma_prod * (self.sigmas[0] ** 2)
        if self.use_p3d_loss:
            p3d_loss = self.pose_loss_3d(pose_3d_preds, pose_3d_targets)
            if not torch.isnan(p3d_loss):
                loss += (1 / (2 * self.sigmas[1] ** 2)) * p3d_loss
                sigma_prod = sigma_prod * (self.sigmas[1] ** 2)
        if self.use_orientation_loss:
            orientation_loss = self.orientation_loss(orientation_preds, orientation_targets)
            if not torch.isnan(orientation_loss):
                loss += (1 / (2 * self.sigmas[2] ** 2)) * orientation_loss
                sigma_prod = sigma_prod * (self.sigmas[2] ** 2)
        if self.use_conf_loss:
            coord_conf_loss = self.coord_conf_loss(pose_2d_preds, pose_2d_targets)
            if not torch.isnan(coord_conf_loss):
                loss += (1 / (self.sigmas[3] ** 2)) * coord_conf_loss
                sigma_prod = sigma_prod * (self.sigmas[3] ** 2)
        loss = loss + torch.log(1+sigma_prod)  # +1 to enforce positive loss
        return loss
