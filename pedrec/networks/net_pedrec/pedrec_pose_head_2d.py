import torch.nn as nn

from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.networks.net_pedrec.pedrec_net_helper import default_init
from pedrec.utils.torch_utils.torch_modules import SoftArgmax2d


class PedRecPose2DHead(nn.Module):
    def __init__(self, cfg: PedRecNetConfig, deconv_head: nn.Module):
        super(PedRecPose2DHead, self).__init__()
        self.deconv_with_bias = cfg.model.deconv_with_bias
        self.deconv_head = deconv_head
        self.pose_heatmap_layer = nn.Conv2d(
            in_channels=cfg.model.num_deconv_filters[-1],
            out_channels=cfg.model.num_joints,
            kernel_size=cfg.model.final_conv_kernel,
            stride=1,
            padding=1 if cfg.model.final_conv_kernel == 3 else 0
        )

        self.pose_coords = SoftArgmax2d()

    def forward(self, x_deconv):
        x_deconv_head = self.deconv_head(x_deconv)
        pose_map = self.pose_heatmap_layer(x_deconv_head)
        pose_map_softmax, pose_coords = self.pose_coords(pose_map)
        return pose_coords, pose_map

    def init_weights(self):
        for name, m in self.deconv_head.named_modules():
            default_init(m, name, self.deconv_with_bias)
        for m in self.pose_heatmap_layer.modules():
            default_init(m, "pose_2d_heatmap")