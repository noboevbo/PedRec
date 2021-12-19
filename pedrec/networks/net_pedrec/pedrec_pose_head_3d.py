import torch
import torch.nn as nn

from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.networks.net_pedrec.pedrec_net_helper import default_init
from pedrec.utils.torch_utils.torch_modules import SoftArgmax2d, DepthRegression


class PedRecPose3DHead(nn.Module):
    def __init__(self, cfg: PedRecNetConfig, deconv_head: nn.Module):
        super(PedRecPose3DHead, self).__init__()
        self.deconv_with_bias = cfg.model.deconv_with_bias
        self.deconv_head = deconv_head
        self.pose_3d_heatmap_layer = nn.Conv2d(
            in_channels=cfg.model.num_pose_3d_deconv_filters[-1],
            out_channels=cfg.model.num_joints,
            kernel_size=cfg.model.final_conv_kernel,
            stride=1,
            padding=1 if cfg.model.final_conv_kernel == 3 else 0
        )

        self.depth_heatmap_layer = nn.Conv2d(
            in_channels=cfg.model.num_pose_3d_deconv_filters[-1],
            out_channels=cfg.model.num_joints,
            kernel_size=cfg.model.final_conv_kernel,
            stride=1,
            padding=1 if cfg.model.final_conv_kernel == 3 else 0
        )

        self.pose_coords = SoftArgmax2d()
        self.depth = DepthRegression()

    def forward(self, x_deconv):
        x_deconv_head_3d = self.deconv_head(x_deconv)
        pose_3d_map = self.pose_3d_heatmap_layer(x_deconv_head_3d)
        # pose_3d_map_sigmoid = torch.sigmoid(pose_3d_map)
        depth_map = self.depth_heatmap_layer(x_deconv_head_3d)
        pose_3d_map_softmax, pose_3d_coords = self.pose_coords(pose_3d_map)
        depth = self.depth(depth_map, pose_3d_map_softmax)
        pose_3d_coords = torch.cat([pose_3d_coords, depth], dim=2)
        pose_3d_coords[:, :, 1] = 1 - pose_3d_coords[:, :, 1]
        return pose_3d_coords, pose_3d_map

    def init_weights(self):
        for name, m in self.deconv_head.named_modules():
            default_init(m, name, self.deconv_with_bias)
        for m in self.pose_3d_heatmap_layer.modules():
            default_init(m, "pose_3d_heatmap_layer")
        for m in self.depth_heatmap_layer.modules():
            default_init(m, "pose_3d_depth_layer")