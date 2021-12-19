from typing import List

import torch.nn as nn

from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.networks.net_pedrec.pedrec_net_helper import default_init


def get_padding_from_kernel(deconv_kernel):
    if deconv_kernel == 4:
        padding = 1
        output_padding = 0
    elif deconv_kernel == 3:
        padding = 1
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 0
        output_padding = 0

    return deconv_kernel, padding, output_padding


class PedRecConvTransposeBase(nn.Module):
    def __init__(self, cfg: PedRecNetConfig, inplanes: int, num_heads: int):
        """
        Deconvolutional layers
        """
        super(PedRecConvTransposeBase, self).__init__()
        self.deconv_with_bias = cfg.model.deconv_with_bias
        self.inplanes = inplanes

        self.deconv_layers, self.deconv_heads = self._make_deconv_layer(
            cfg.model.num_deconv_layers,
            cfg.model.num_deconv_filters,
            cfg.model.num_deconv_kernels,
            num_heads
        )

    def forward(self, x):
        x_deconv = self.deconv_layers(x)
        return x_deconv

    def _add_deconv_layer(self, layers: List[nn.Module], planes: int, kernel: int, padding: int, output_padding: int):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=self.inplanes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, num_heads):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers - 1):
            kernel, padding, output_padding = \
                get_padding_from_kernel(num_kernels[i])
            planes = num_filters[i]
            self._add_deconv_layer(layers, planes, kernel, padding, output_padding)
            self.inplanes = planes

        kernel, padding, output_padding = \
            get_padding_from_kernel(num_kernels[num_layers - 1])
        planes = num_filters[num_layers - 1]
        layers_heads = []
        for i in range(num_heads):
            head_layers = []
            self._add_deconv_layer(head_layers, planes, kernel, padding, output_padding)
            layers_heads.append(nn.Sequential(*head_layers))
        return nn.Sequential(*layers), layers_heads

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            default_init(m, name, self.deconv_with_bias)