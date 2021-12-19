import torch
import torch.nn as nn
import torch.nn.functional as F

from pedrec.configs.pedrec_net_config import PedRecNetConfig


class PedRecPoseConfHead(nn.Module):
    def __init__(self, cfg: PedRecNetConfig):
        super(PedRecPoseConfHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pose_conf = nn.Linear(cfg.layer.block.expansion * 512, cfg.model.num_joints)
        self.conv1 = nn.Conv2d(52, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(42240, 128)
        self.fc2 = nn.Linear(128, cfg.model.num_joints)

    def forward(self, pose_softmax_2d, pose_softmax_3d):
        x = torch.cat([pose_softmax_2d.detach(), pose_softmax_3d.detach()], dim=1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv2.bias, 0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)