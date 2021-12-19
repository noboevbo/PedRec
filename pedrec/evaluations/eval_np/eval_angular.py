import math

import numpy as np
import torch


def get_angular_distances(target: np.ndarray, pred: np.ndarray):
    filter_theta = target[:, 3] == 1
    filter_phi = target[:, 4] == 1
    target_masked_theta = target[filter_theta]
    output_masked_theta = pred[filter_theta]
    theta_gt = target_masked_theta[:, 0]
    theta_pred = output_masked_theta[:, 0]
    
    target_masked_phi = target[filter_phi]
    output_masked_phi = pred[filter_phi]
    phi_gt = target_masked_phi[:, 1]
    phi_pred = output_masked_phi[:, 1]

    dist_phi = np.abs(phi_pred - phi_gt)
    dist_phi = np.minimum(2 * math.pi - dist_phi, dist_phi)
    dist_theta = np.abs(theta_pred - theta_gt)
    dist_theta = np.minimum(2 * math.pi - dist_theta, dist_theta)

    if dist_theta.shape[0] == dist_phi.shape[0]:
        spherical_distance = np.sqrt(
            2 - 2 * (np.sin(theta_gt) * np.sin(theta_pred) * np.cos(phi_gt - phi_pred) + np.cos(theta_gt) * np.cos(theta_pred)))
    else:
        spherical_distance = 0
    return dist_phi, dist_theta, spherical_distance

def get_angular_error_statistics(target: np.ndarray, pred: np.ndarray):
    """
    n = number of records
    :param gt_array: (n, num_joints, 2)
    :param pred_array: (n, num_joints, 2)
    :param visible_array: (n, num_joints) # 0 if invisible, 1 if visible
    :return:
    """
    num_visible_orientation_joints_theta = np.sum(target[:, 3])
    num_visible_orientation_joints_phi = np.sum(target[:, 4])
    # Set predicted values of all invisible joints to 0 (ignore)
    dist_phi, dist_theta, spherical_distance = get_angular_distances(target, pred)
    if num_visible_orientation_joints_phi > 0:
        dist_phi = np.sum(dist_phi, axis=0) / num_visible_orientation_joints_phi
    else:
        dist_phi = 0
    if num_visible_orientation_joints_theta > 0:
        dist_theta = np.sum(dist_theta, axis=0) / num_visible_orientation_joints_theta
    else:
        dist_theta = 0
    if num_visible_orientation_joints_phi > 0 and num_visible_orientation_joints_theta > 0:
        spherical_distance = np.sum(spherical_distance, axis=0) / num_visible_orientation_joints_theta
    else:
        spherical_distance = 0

    return dist_phi, dist_theta, spherical_distance
