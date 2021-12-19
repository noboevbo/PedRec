import csv
import logging
import math
import os.path
import os.path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torchvision import transforms

from pedrec.configs.app_config import AppConfig
from pedrec.configs.dataset_configs import get_coco_dataset_cfg_default
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.datasets.coco_dataset import CocoDataset
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.utils.augmentation_helper import get_affine_transforms
from pedrec.utils.bb_helper import get_center_bb_from_coord_bb, \
    bb_to_center_scale
from pedrec.utils.demo_helper import init_pose_model
from pedrec.utils.torch_utils.torch_helper import get_device

pose_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_angular_distance(phi_gt_deg, phi_pred_deg):
    dist_phi = np.abs(phi_pred_deg - phi_gt_deg)
    return np.minimum(360 - dist_phi, dist_phi)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    # pedrecnet_weights = "data/models/pedrec/experiment_pedrec_direct_4_net.pth"

    # pedrecnet_weights = "data/models/pedrec/experiment_pedrec_p2d3d_c_o_h36m_sim_0_net.pth"
    # pedrecnet_weights = "data/models/pedrec/experiment_pedrec_p2d3d_c_o_h36m_mebow_0_net.pth"
    pedrecnet_weights = "data/models/pedrec/experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_net.pth"
    # pedrecnet_weights = "data/models/pedrec/experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_tud_0_net.pth"
    # pedrecnet_weights = "data/models/pedrec/experiment_pedrec_p2d3d_c_o_sim_0_net.pth"
    dataset_cfg = get_coco_dataset_cfg_default()
    dataset_cfg.use_mebow_orientation = True
    cfg = PedRecNet50Config()
    app_cfg = AppConfig()
    device = get_device(app_cfg.cuda.use_gpu)
    val_set = CocoDataset("data/datasets/COCO", DatasetType.VALIDATE, dataset_cfg,
                          cfg.model.input_size, pose_transform)
    # Pose
    pose_cfg = PedRecNet50Config()
    net = init_pose_model(PedRecNet(cfg), pedrecnet_weights, logger, device)
    distances = []
    for annotation in val_set:
        img, labels = annotation

        model_input = img.unsqueeze(0).to(device)
        trans_invs = torch.tensor(labels["trans_inv"], dtype=torch.float32).unsqueeze(0).to(device)
        img_size = torch.tensor([labels["img_size"]], dtype=torch.float32).to(device)

        output = net(model_input)
        # visibles ??
        orientation_pred = output[2].cpu().detach().numpy()

        orientation_pred[:, :, 0] *= math.pi
        orientation_pred[:, :, 1] *= 2 * math.pi

        orientation_gt = labels["orientation"]
        if orientation_gt[0, 4] != 1:
            continue
        orientation_gt[:, 0] *= math.pi
        orientation_gt[:, 1] *= 2 * math.pi

        body_phi_degree_gt = math.degrees(orientation_gt[0, 1])
        body_phi_degree_pred = math.degrees(orientation_pred[0, 0, 1])
        # orientation_bin = orientation_bin_from_orientation(body_phi_degree)
        distances.append(get_angular_distance(body_phi_degree_gt, body_phi_degree_pred))
        # corrects += int(orientation_bin == annotation["orientation_bin"])
        a = 1

    distances = np.array(distances, dtype=np.float32)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    acc_5 = len(np.where(distances <= 5)[0]) / distances.shape[0]
    acc_15 = len(np.where(distances <= 15)[0]) / distances.shape[0]
    acc22_5 = len(np.where(distances <= 22.5)[0]) / distances.shape[0]
    acc_30 = len(np.where(distances <= 30)[0]) / distances.shape[0]
    acc45 = len(np.where(distances <= 45)[0]) / distances.shape[0]
    print(f"Acc5: {acc_5}, Acc15: {acc_15},  Acc22.5: {acc22_5}, Acc30: {acc_30}, Acc45: {acc45}, Mean: {mean_distance}, std_distance: {std_distance}")
    # print(f"Correct: {corrects}, total: {len(train_data)}, percentage: {corrects / len(train_data)}")

        # x = 1

        # create model input
        # run model
        # save output gt / pred
    a = 1
