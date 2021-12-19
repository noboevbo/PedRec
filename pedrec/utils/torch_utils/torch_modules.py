from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pedrec.utils.torch_utils.torch_helper import create_meshgrid, create_linspace


class DepthRegression(nn.Module):
    def __init__(self) -> None:
        super(DepthRegression, self).__init__()

    def forward(self, input: torch.Tensor, pose_heatmap: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = input.shape
        x: torch.Tensor = input.view(batch_size, channels, -1)
        x_sigmoid: torch.Tensor = torch.sigmoid(x)

        result = torch.sum(x_sigmoid * pose_heatmap, -1, keepdim=True)
        return result

class SoftArgmax1d(nn.Module):
    """
    Soft Argmax 1D
    """

    def __init__(self, norm_val = 1.0, normalized_coordinates: Optional[bool] = True) -> None:
        super(SoftArgmax1d, self).__init__()
        self.normalized_coordinates: Optional[bool] = normalized_coordinates
        self.norm_val = norm_val

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        # if not len(input.shape) == 2:
        #     raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
        #                      .format(input.shape))
        # unpack shapes and create view from input tensor
        # x: torch.Tensor = input.view(batch_size, channels, -1)
        # input_a = input[0][0].detach().cpu().numpy()
        #
        # plt.imshow(input_a, cmap='hot', interpolation='nearest')
        # plt.show()

        # Softmax
        x_soft: torch.Tensor = F.softmax(input * self.norm_val, dim=-1)

        pos_x = create_linspace(input, self.normalized_coordinates)

        expected_x: torch.Tensor = torch.sum(pos_x * x_soft, -1, keepdim=True)

        return x_soft, expected_x  # BxNx2


class SoftArgmax2d(nn.Module):
    r"""Creates a module that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Arguments:
        normalized_coordinates (Optional[bool]): wether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples::
        >>> input = torch.rand(1, 4, 2, 3)
        >>> m = tgm.losses.SpatialSoftArgmax2d()
        >>> coords = m(input)  # 1x4x2
        >>> x_coord, y_coord = torch.chunk(coords, dim=-1, chunks=2)
    """
    def __init__(self, norm_val: float = 1.0, normalized_coordinates: Optional[bool] = True) -> None:
        super(SoftArgmax2d, self).__init__()
        self.normalized_coordinates: Optional[bool] = normalized_coordinates
        self.norm_val = norm_val

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = input.shape
        x: torch.Tensor = input.view(batch_size, channels, -1)
        # input_a = input[0][0].detach().cpu().numpy()
        #
        # plt.imshow(input_a, cmap='hot', interpolation='nearest')
        # plt.show()

        # Softmax
        x_soft: torch.Tensor = F.softmax(x * self.norm_val, dim=-1)

        pos_y, pos_x = create_meshgrid(input, self.normalized_coordinates)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        expected_x: torch.Tensor = torch.sum(pos_x * x_soft, -1, keepdim=True)
        expected_y: torch.Tensor = torch.sum(pos_y * x_soft, -1, keepdim=True)

        output: torch.Tensor = torch.cat([expected_x, expected_y], dim=-1)
        return x_soft, output.view(batch_size, channels, 2)  # BxNx2


class SpatialSoftArgmax2d(nn.Module):
    r"""Creates a module that computes the Spatial Soft-Argmax 2D
    of a given input heatmap.

    Returns the index of the maximum 2d coordinates of the give map.
    The output order is x-coord and y-coord.

    Arguments:
        normalized_coordinates (Optional[bool]): wether to return the
          coordinates normalized in the range of [-1, 1]. Otherwise,
          it will return the coordinates in the range of the input shape.
          Default is True.

    Shape:
        - Input: :math:`(B, N, H, W)`
        - Output: :math:`(B, N, 2)`

    Examples::
        >>> input = torch.rand(1, 4, 2, 3)
        >>> m = tgm.losses.SpatialSoftArgmax2d()
        >>> coords = m(input)  # 1x4x2
        >>> x_coord, y_coord = torch.chunk(coords, dim=-1, chunks=2)
    """

    def __init__(self, normalized_coordinates: Optional[bool] = True) -> None:
        super(SpatialSoftArgmax2d, self).__init__()
        self.normalized_coordinates: Optional[bool] = normalized_coordinates
        self.eps: float = 1e-6

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # unpack shapes and create view from input tensor
        batch_size, channels, height, width = input.shape
        x: torch.Tensor = input.view(batch_size, channels, -1)

        # compute softmax with max substraction trick
        exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
        exp_x_sum = 1.0 / (exp_x.sum(dim=-1, keepdim=True) + self.eps)
        # test = exp_x_sum.detach().cpu().numpy() # == probabilities

        # create coordinates grid
        pos_y, pos_x = create_meshgrid(input, self.normalized_coordinates)
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        # compute the expected coordinates
        expected_y: torch.Tensor = torch.sum(
            (pos_y * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        expected_x: torch.Tensor = torch.sum(
            (pos_x * exp_x) * exp_x_sum, dim=-1, keepdim=True)
        output: torch.Tensor = torch.cat([expected_x, expected_y], dim=-1)
        return output.view(batch_size, channels, 2)  # BxNx2
