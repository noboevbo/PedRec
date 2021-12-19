import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC, SKELETON_PEDREC_JOINT_COLORS, SKELETON_PEDREC_LIMB_COLORS
from pedrec.visualizers.visualization_helper_3d import draw_origin_3d, draw_grid_3d


def add_skeleton_3d_to_axes(ax: Axes3D, skeleton_3d: np.ndarray, size: float = 2, min_score: float = 0.3):
    # Joints
    xs = skeleton_3d[:, 0]
    ys = skeleton_3d[:, 2]
    zs = skeleton_3d[:, 1]
    colors = []
    for idx, joint in enumerate(skeleton_3d):
        if joint[3] < min_score:  # score
            colors.append([0, 0, 0, 0])
        else:
            colors.append(SKELETON_PEDREC_JOINT_COLORS[idx].rgba_float_list)
    ax.scatter(xs, ys, zs, c=colors, s=size)

    # Limbs
    for idx, pair in enumerate(SKELETON_PEDREC):
        if (skeleton_3d[pair[0:2], 3] >= min_score).all():
            ax.plot(skeleton_3d[pair[0:2], 0], skeleton_3d[pair[0:2], 2], skeleton_3d[pair[0:2], 1], linewidth=size, c=SKELETON_PEDREC_LIMB_COLORS[idx].rgba_float_list)


def get_skeleton_3d_figure(skeleton_3d: np.ndarray):
    # Preparation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_grid_3d(ax)
    draw_origin_3d(ax)
    add_skeleton_3d_to_axes(ax, skeleton_3d)
    return fig, ax


def plot_skeleton_3d(skeleton_3d: np.ndarray):
    fig, ax = get_skeleton_3d_figure(skeleton_3d)
    plt.show()
