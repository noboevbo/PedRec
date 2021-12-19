from torch import nn

# nn layer types
batch_norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
bias_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)
no_wd_types = batch_norm_types + (nn.LayerNorm,)

# dataset mean / std
cifar_stats = ([0.491, 0.482, 0.447], [0.247, 0.243, 0.261])
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
utk_stats = ([0.4930, 0.3726, 0.3174], [0.3094, 0.2492, 0.2232])
mnist_stats = ([0.131], [0.308])
jhmdb_ehpi_stats = ([5.9460e-01, 4.2385e-01, 1.0000e-05], [2.1045e-01, 2.5349e-01, 1.0000e-05])
