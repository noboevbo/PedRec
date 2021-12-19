import math
from typing import Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pedrec.visualizers.visualization_helper_3d import draw_origin_3d, draw_grid_3d


def add_orientation_to_axes(ax: Axes3D, theta: float, phi: float, size: float = 2,
                            color: Tuple[float, float, float] = (1, 0, 0)):
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)

    ax.quiver([0], [0], [0], [x], [y], [-z], colors=color, linewidths=size)


def get_orientation_figure(theta: float, phi: float, size: float = 2,
                            color: Tuple[float, float, float] = (1, 0, 0)):
    # Preparation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_grid_3d(ax, lim=1)
    draw_origin_3d(ax, dimensions=(1, 1, 1))
    add_orientation_to_axes(ax, theta, phi, size, color)
    return fig, ax


def plot_skeleton_3d(theta: float, phi: float, size: float = 2,
                            color: Tuple[float, float, float] = (1, 0, 0)):
    get_orientation_figure(theta, phi, size, color)
    plt.show()
