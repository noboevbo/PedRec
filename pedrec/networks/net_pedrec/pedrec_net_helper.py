import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


def default_init(m: nn.Module, name: str, deconv_with_bias: bool = False):
    if isinstance(m, nn.Conv2d):
        logger.info(f'=> init {name}.weight as normal(0, 0.001)')
        logger.info(f'=> init {name}.bias as 0')
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.ConvTranspose2d):
        logger.info(f'=> init {name}.weight as normal(0, 0.001)')
        logger.info(f'=> init {name}.bias as 0')
        nn.init.normal_(m.weight, std=0.001)
        if deconv_with_bias:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        logger.info(f'=> init {name}.weight as 1')
        logger.info(f'=> init {name}.bias as 0')
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose1d):
        logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
        logger.info('=> init {}.bias as 0'.format(name))
        nn.init.normal_(m.weight, std=0.001)
        if deconv_with_bias:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        logger.info('=> init {}.weight as 1'.format(name))
        logger.info('=> init {}.bias as 0'.format(name))
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)