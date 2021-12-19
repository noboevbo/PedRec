import numpy as np


def get_binary_multi_label_accuracy(target: np.ndarray, pred: np.ndarray, threshold: float = 0.5):
    pred_masked = pred.copy()
    pred_masked[pred > threshold] = 1
    pred_masked[pred <= threshold] = 0
    corrects = pred_masked == target
    corrects_per_joint = corrects.sum(axis=0)
    accuracy_per_joint = corrects_per_joint / target.shape[0]
    accuracy = np.sum(accuracy_per_joint) / target.shape[1]
    return accuracy, accuracy_per_joint


def get_joint_label_accuracy(target: np.ndarray, pred: np.ndarray, threshold: float = 0.5):
    """
    requires input shape n, num_joints, 5 (5 = x, y, score, visibile, supported)
    """
    pred_masked = pred.copy()
    pred_masked[pred > threshold] = 1
    pred_masked[pred <= threshold] = 0
    corrects_per_joint = []
    num_supported_joints = 0
    for joint_num in range(target.shape[1]):
        mask = target[:, joint_num, 4] == 1

        a = target[:, joint_num, 2][mask]
        b = pred_masked[:, joint_num, 2][mask]
        if a.shape[0] > 0:
            corrects = a == b
            corrects_per_joint.append(corrects.sum())
            num_supported_joints += 1
        else:
            corrects_per_joint.append(0)

    corrects_per_joint = np.array(corrects_per_joint)
    accuracy_per_joint = corrects_per_joint / target.shape[0]
    accuracy = np.sum(accuracy_per_joint) / num_supported_joints
    return accuracy, accuracy_per_joint
