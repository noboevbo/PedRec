import torch
import torch.nn as nn

from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.networks.net_pedrec.pedrec_net_helper import default_init
from pedrec.utils.torch_utils.torch_modules import SoftArgmax1d


class PedRecOrientationsHead(nn.Module):
    def __init__(self, cfg: PedRecNetConfig):
        super(PedRecOrientationsHead, self).__init__()
        self.deconv_with_bias = False
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.pose_deconv = self._make_deconv_layer()
        self.body_orientation = PedRecOrientationHead(cfg)
        self.head_orientation = PedRecOrientationHead(cfg)

    def forward(self, x, x_pose):
        # batch_size, 2048, 8, 6
        x = self.avgpool(x)
        # batch_size, 2048, 1, 1
        x = torch.flatten(x, 1)
        x_pose = x_pose.detach()
        x_pose = x_pose[:, :, :3]  # select x, y and z
        x_pose = torch.transpose(x_pose, 1, 2)
        # concatenate feature map with 3D pose coordinates as additional features
        pose_features = self.pose_deconv(x_pose)
        # x_pose_flat =   # detach pose coords to stop gradient flow because this is a label
        # pose_features = self.avgpool(pose_features)
        pose_features = torch.flatten(pose_features, 1)
        x = torch.cat([x, pose_features], dim=1)

        body_orientation, body_theta_map, body_phi_map = self.body_orientation(x)
        body_orientation = torch.unsqueeze(body_orientation, dim=1)
        body_theta_map = torch.unsqueeze(body_theta_map, dim=1)
        body_phi_map = torch.unsqueeze(body_phi_map, dim=1)
        head_orientation, head_theta_map, head_phi_map = self.head_orientation(x)
        head_orientation = torch.unsqueeze(head_orientation, dim=1)
        head_theta_map = torch.unsqueeze(head_theta_map, dim=1)
        head_phi_map = torch.unsqueeze(head_phi_map, dim=1)
        orientation = torch.cat((body_orientation, head_orientation), dim=1)

        theta_maps = torch.cat((body_theta_map, head_theta_map), dim=1)
        phi_maps = torch.cat((body_phi_map, head_phi_map), dim=1)
        return orientation, theta_maps, phi_maps

    def _make_deconv_layer(self):
        layers = []

        layers.append(nn.ConvTranspose1d(
            in_channels=3,
            out_channels=6,
            kernel_size=3
        ))
        layers.append(nn.BatchNorm1d(6))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.ConvTranspose1d(
            in_channels=6,
            out_channels=12,
            kernel_size=3
        ))
        layers.append(nn.BatchNorm1d(12))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.ConvTranspose1d(
            in_channels=12,
            out_channels=24,
            kernel_size=3
        ))
        layers.append(nn.BatchNorm1d(24))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def init_weights(self):
        for name, m in self.pose_deconv.named_modules():
            default_init(m, name, self.deconv_with_bias)
        self.body_orientation.init_weights()
        self.head_orientation.init_weights()


class PedRecOrientationHead(nn.Module):
    def __init__(self, cfg: PedRecNetConfig):
        super(PedRecOrientationHead, self).__init__()
        self.pose_size = 768
        self.phi = nn.Linear(512 * cfg.layer.block.expansion + self.pose_size, 360)
        self.theta = nn.Linear(512 * cfg.layer.block.expansion + self.pose_size, 180)

        self.body_orientation = SoftArgmax1d()

    def forward(self, x):
        theta_map = self.theta(x)
        theta_softmax, theta = self.body_orientation(theta_map)

        phi_map = self.phi(x)
        phi_softmax, phi = self.body_orientation(phi_map)

        orientation = torch.cat([theta, phi], dim=1)
        return orientation, theta_map, phi_map

    def init_weights(self):
        nn.init.xavier_normal_(self.phi.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.phi.bias.data, 0)

        nn.init.xavier_normal_(self.theta.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.theta.bias.data, 0)

