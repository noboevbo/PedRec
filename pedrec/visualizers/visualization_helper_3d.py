from typing import Tuple

import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def draw_grid_3d(ax: Axes3D, lim: float = 1.5, tick_step: float = 0.5):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ticks = np.arange(-lim, lim, step=tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)


def draw_origin_3d(ax: Axes3D, dimensions: Tuple[int, int, int] = (3, 3, 3)):
    x, y, z = np.zeros((3, 3))
    u, v, w = np.array([[dimensions[0] / 2, 0, 0], [0, dimensions[1] / 2, 0], [0, 0, dimensions[2] / 2]])
    color = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.1, color=color)