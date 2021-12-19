# import torch
# import torch.nn as nn
#
# from pedrec.configs.pedrec_net_config import PedRecNetConfig
#
#
# class PedRecPoseConfHead(nn.Module):
#     def __init__(self, cfg: PedRecNetConfig):
#         super(PedRecPoseConfHead, self).__init__()
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(cfg.layer.block.expansion * 512, cfg.model.num_joints)
#
#     def forward(self, x):
#         x = self.avgpool(x.detach())
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         x = torch.sigmoid(x)
#         return x
#
#     def init_weights(self):
#         nn.init.xavier_uniform_(self.fc.weight)
#         nn.init.constant_(self.fc.bias, 0)
