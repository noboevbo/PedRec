import math
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from pedrec.configs.dataset_configs import get_pedrec_dataset_cfg_default
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.datasets.pedrec_dataset import PedRecDataset
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.data_structures import ImageSize
from pedrec.visualizers.orientation_visualizer import add_orientation_to_axes
from pedrec.visualizers.skeleton_3d_visualizer import add_skeleton_3d_to_axes
from pedrec.visualizers.visualization_helper_3d import draw_grid_3d, draw_origin_3d

def plot_img(img):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)

def plot_pose3d(labels):
    # visualize 3d skeleton
    skeleton_3d = labels["skeleton_3d"]
    scale_factor = 3
    skeleton_3d[:, :3] *= scale_factor
    skeleton_3d[:, :3] -= (scale_factor / 2)

    fig_skeleton_3d = plt.figure()
    ax_3d = fig_skeleton_3d.add_subplot(1, 1, 1, projection='3d')
    ax_3d.set_title("3D Pose")
    draw_grid_3d(ax_3d)
    draw_origin_3d(ax_3d)
    add_skeleton_3d_to_axes(ax_3d, skeleton_3d, size=4)

def plot_orientation(labels):
    # Visualize orientation orientation
    orientation = labels["orientation"]
    body_theta = orientation[0, 0] * math.pi
    body_phi = orientation[0, 1] * 2 * math.pi
    head_theta = orientation[1, 0] * math.pi
    head_phi = orientation[1, 1] * 2 * math.pi
    fig_orientation = plt.figure()
    ax = fig_orientation.add_subplot(111, projection='3d')
    draw_grid_3d(ax, lim=1)
    draw_origin_3d(ax, dimensions=(1, 1, 1))
    add_orientation_to_axes(ax, body_theta, body_phi, color=(1, 0, 0))
    add_orientation_to_axes(ax, head_theta, head_phi, color=(0, 1, 0))
    # fig, ax = get_skeleton_3d_figure(skeleton_3d)

def run(img: np.ndarray, labels: Dict[str, any]):
    plot_img(img)
    plot_pose3d(labels)
    plot_orientation(labels)
    plt.show()

if __name__ == "__main__":
    cfg = PedRecNet50Config()
    # ROM Train
    dataset_name = "ROM"
    dataset_cfg = get_pedrec_dataset_cfg_default()
    dataset = PedRecDataset("data/datasets/ROM/", "rt_rom_01",
                            DatasetType.TRAIN, dataset_cfg, cfg.model.input_size, ImageSize(1920, 1080), None)

    img, labels = dataset[1950]
    run(img, labels)
