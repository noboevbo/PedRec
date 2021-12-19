import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

from pedrec.networks.net_resnet.resnet_feature_extractor import ResNetHeadless


class Ehpi3DNet(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()
        block = Bottleneck
        layers = [3, 4, 6, 3]
        self.feature_extractor = ResNetHeadless(block, layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(block.expansion * 512, num_actions)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
