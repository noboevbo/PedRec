import sys
import time
from pathlib import Path

import numpy as np
import sklearn.metrics
from sklearn.metrics import accuracy_score, hamming_loss
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from pedrec.configs.app_config import AppConfig, action_list_c01
from pedrec.configs.dataset_configs import get_sim_dataset_cfg_default, PedRecTemporalDatasetConfig
from pedrec.datasets.dataset_helper import worker_init_fn
from pedrec.datasets.pedrec_temporal_dataset import PedRecTemporalDataset
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.constants.sample_method import SAMPLE_METHOD
from pedrec.models.data_structures import ImageSize
from pedrec.models.experiments.experiment_paths import ExperimentPaths
from pedrec.networks.net_pedrec.ehpi_3d_net import Ehpi3DNet

sys.path.append(".")

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths_home
from pedrec.training.experiments.experiment_train_helper import init_experiment
from pedrec.utils.torch_utils.torch_helper import get_device, move_to_device


def initialize_from_imgnet(net, pose_resnet_weights_path: str):
    pose_resnet_state_dict = torch.load(pose_resnet_weights_path)
    net_weights = net.state_dict()
    for name, param in pose_resnet_state_dict.items():
        net_name = f"feature_extractor.{name}"
        if net_name in net_weights:
            net_weights[net_name] = param
        else:
            print("Skipped:" + name)
    net.load_state_dict(net_weights)


def get_val_loader(experiment_paths: ExperimentPaths, batch_size, action_list, dataset_cfg: PedRecTemporalDatasetConfig):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.400, 0.443, 0.401], std=[0.203, 0.269, 0.203])
    ])

    sim_val = PedRecTemporalDataset(experiment_paths.sim_c01_val_dir,
                                      experiment_paths.sim_c01_val_filename,
                                      DatasetType.VALIDATE,
                                      dataset_cfg,
                                      action_list,
                                      trans,
                                      pose_results_file=experiment_paths.sim_c01_val_results_filename)

    return DataLoader(sim_val, batch_size=batch_size, shuffle=False, num_workers=12, worker_init_fn=worker_init_fn)


def main(net_weights_path, dataset_cfg: PedRecTemporalDatasetConfig):

    app_cfg = AppConfig()
    experiment_paths = get_experiment_paths_home()
    init_experiment(42)
    device = get_device(use_gpu=True)
    net = Ehpi3DNet(len(app_cfg.inference.action_list))
    net.load_state_dict(torch.load(net_weights_path))
    net.to(device)

    val_loader = get_val_loader(experiment_paths, 48, app_cfg.inference.action_list, dataset_cfg)

    ## LR
    criterion = BCEWithLogitsLoss()

    start = time.time()
    loss_total = 0.0
    net.eval()

    gts = np.empty((0, len(app_cfg.inference.action_list)))
    preds = np.empty((0, len(app_cfg.inference.action_list)))
    probabilities = np.empty((0, len(app_cfg.inference.action_list)))
    with tqdm(total=len(val_loader)) as pbar:
        for i, val_data in enumerate(val_loader):
            inputs, labels = val_data

            inputs = inputs.to(device)
            labels = move_to_device(labels, device)

            outputs = net(inputs)
            # loss = criterion(outputs, labels)

            action_probabilities = torch.sigmoid(outputs).detach().cpu().numpy()
            action_preds = action_probabilities.copy()
            action_preds[action_probabilities[:, :] >= 0.8] = 1
            action_preds[action_probabilities[:, :] < 0.8] = 0
            labels = labels.detach().cpu().numpy()

            gts = np.concatenate([gts, labels], axis=0)
            preds = np.concatenate([preds, action_preds])
            probabilities = np.concatenate([probabilities, action_probabilities])

            # loss_total += loss.item()
            pbar.update()

    # np.save("data/latex/Dissertation/data/ehpi_c01_results/gts.npz", gts)
    # np.save("data/latex/Dissertation/data/ehpi_c01_results/preds.npz", preds)
    balanced_accs = []
    precisions = []
    recalls = []
    fscores = []
    aps = []
    print(f"### GT_RESULT_RATIO: {dataset_cfg.gt_result_ratio} ({Path(net_weights_path).stem})")
    print()
    print("Action & BAcc & AP & F1 & Precision & Recall & NPS \\\\\\midrule")
    # remove the labels of non c01 classes
    gts = gts[:, :len(action_list_c01)]
    preds = preds[:, :len(action_list_c01)]
    probabilities = probabilities[:, :len(action_list_c01)]
    for idx, action in enumerate(action_list_c01):
        mask_true = gts[:, idx] == 1
        # mask_false = gts[:, idx] == 0
        gts_masked_true = gts[mask_true]
        # if gts_masked_true.shape[0] == 0:
        #     continue
        # preds_masked_true = preds[mask_true]
        # gts_masked_false = gts[mask_false]
        # preds_masked_false = preds[mask_false]
        # TP = np.sum(preds_masked_true[:, idx])
        # FN = gts_masked_true.shape[0] - TP
        # FP = np.sum(preds_masked_false[:, idx])
        # TN = gts_masked_false.shape[0] - FP
        # acc = (TP + TN) / (gts.shape[0])
        # TPR = TP / (TP + FN)
        # TNR = TN / (TN + FP)
        # precision = TP / (TP+FP)
        # recall = TPR
        # balanced_acc = (TPR + TNR) / 2
        # balanced_accs.append(balanced_acc)
        # precisions.append(precision)
        # recalls.append(recall)
        # fscore = 2 * ((TPR * TNR) / (TPR + TNR))
        # fscores.append(fscore)

        # Use standard methods
        precision = sklearn.metrics.precision_score(gts[:, idx], preds[:, idx])
        recall = sklearn.metrics.recall_score(gts[:, idx], preds[:, idx])
        balanced_acc = sklearn.metrics.balanced_accuracy_score(gts[:, idx], preds[:, idx])
        ap = sklearn.metrics.average_precision_score(gts[:, idx], probabilities[:, idx])
        fscore = sklearn.metrics.f1_score(gts[:, idx], preds[:, idx])
        precisions.append(precision)
        recalls.append(recall)
        balanced_accs.append(balanced_acc)
        aps.append(ap)
        fscores.append(fscore)
        # precision = TP / (gts_masked_true.shape[0] + )
        print(f"{action} & ${balanced_acc*100:.1f}$ & ${ap*100:.1f}$ & ${fscore*100:.1f}$ & ${precision*100:.1f}$ & ${recall*100:.1f}$ & {gts_masked_true.shape[0]} \\\\")

    # mask_true = gts[:, :] == 1
    # mask_false = gts[:, :] == 0
    # gts_masked_true = gts[mask_true]
    # preds_masked_true = preds[mask_true]
    # gts_masked_false = gts[mask_false]
    # preds_masked_false = preds[mask_false]
    # TP = np.sum(preds_masked_true)
    # FN = gts_masked_true.shape[0] - TP
    # FP = np.sum(preds_masked_false)
    # TN = gts_masked_false.shape[0] - FP
    # TPR = TP / (TP + FN)
    # TNR = TN / (TN + FP)
    # precision = TP / (TP+FP)
    # recall = TPR
    # of1 = 2 * (precision * recall) / (precision + recall)

    # use standard methods
    of1 = sklearn.metrics.f1_score(gts, preds, average='micro')
    op = sklearn.metrics.precision_score(gts, preds, average='micro')
    or_ = sklearn.metrics.recall_score(gts, preds, average='micro')
    # of1_weighted = sklearn.metrics.f1_score(gts, preds, average='weighted')
    # of1_macro = sklearn.metrics.f1_score(gts, preds, average='macro')
    balanced_acc = np.array(balanced_accs, dtype=np.float32)
    balanced_acc = np.mean(balanced_acc)
    cp = np.array(precisions, dtype=np.float32)
    cp = np.mean(cp)
    cr = np.array(recalls, dtype=np.float32)
    cr = np.mean(cr)
    cf1 = np.array(fscores, dtype=np.float32)
    cf1 = np.mean(cf1)
    aps = np.array(aps, dtype=np.float32)
    map = np.mean(aps)
    # acc = accuracy_score(gts, preds)
    print(f"Total: mBAcc {balanced_acc*100:.2f}%, mAP: {map*100:.2f}%, CF1: {cf1*100:.2f}%, CP {cp*100:.2f}%, CR: {cr*100:.2f}%, OF1: {of1*100:.2f}%, OP {op*100:.2f}%, OR: {or_*100:.2f}%")
    return balanced_acc, map, cf1, cp, cr, of1, op, or_

def get_experiment_results(weights_path, pedrec_cfg, gt_result_ratio):
    balanced_acc, map, cf1, cp, cr, of1, op, or_ = main(weights_path, pedrec_cfg)
    experiment_name = Path(weights_path).stem\
        .replace('ehpi_3d_sim_c01_actionrec_', '')\
        .replace('gt', 'G')\
        .replace('pred', 'P')\
        .replace('ehpi2dvids', 'E')\
        .replace('no_unit_skel', 'N')\
        .replace('_', '+')
    md = f"| {experiment_name} | {gt_result_ratio} | {balanced_acc * 100:.2f} | {map * 100:.2f} | {of1 * 100:.2f}  | {op * 100:.2f}  | {or_ * 100:.2f} | {cf1 * 100:.2f}  | {cp * 100:.2f}  | {cr * 100:.2f}"
    latex = f"{experiment_name} & ${gt_result_ratio}$ &  ${balanced_acc * 100:.1f}$ & ${map * 100:.1f}$ & ${of1 * 100:.1f}$ & ${op * 100:.1f}$ & ${or_ * 100:.1f}$ & ${cf1 * 100:.1f}$ & ${cp * 100:.1f}$ & ${cr * 100:.1f}$ \\\\"
    return md, latex


def run_experiments(gt_result_ratio: float):
    results = []
    results_latex = []
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=False,
        scale_factor=0,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=gt_result_ratio,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False
    )
    weights_path = "data/models/ehpi3d/ehpi_3d_sim_c01_actionrec_pred.pth"
    md, latex = get_experiment_results(weights_path, pedrec_cfg, gt_result_ratio)
    results.append(md)
    results_latex.append(latex)

    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=False,
        scale_factor=0,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=gt_result_ratio,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False
    )
    weights_path = "data/models/ehpi3d/ehpi_3d_sim_c01_actionrec_gt.pth"
    md, latex = get_experiment_results(weights_path, pedrec_cfg, gt_result_ratio)
    results.append(md)
    results_latex.append(latex)

    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=False,
        scale_factor=0,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=gt_result_ratio,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False
    )
    weights_path = "data/models/ehpi3d/ehpi_3d_sim_c01_actionrec_gt_pred.pth"
    md, latex = get_experiment_results(weights_path, pedrec_cfg, gt_result_ratio)
    results.append(md)
    results_latex.append(latex)

    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=False,
        scale_factor=0,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=gt_result_ratio,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False
    )
    weights_path = "data/models/ehpi3d/ehpi_3d_sim_c01_actionrec_gt_pred_ehpi2dvids.pth"
    md, latex = get_experiment_results(weights_path, pedrec_cfg, gt_result_ratio)
    results.append(md)
    results_latex.append(latex)

    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=False,
        scale_factor=0,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=gt_result_ratio,
        use_unit_skeleton=False,
        min_joint_score=0,
        add_2d=False
    )
    weights_path = "data/models/ehpi3d/ehpi_3d_sim_c01_actionrec_gt_pred_no_unit_skeleton.pth"
    md, latex = get_experiment_results(weights_path, pedrec_cfg, gt_result_ratio)
    results.append(md)
    results_latex.append(latex)

    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=False,
        scale_factor=0,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=gt_result_ratio,
        use_unit_skeleton=True,
        min_joint_score=0.4,
        add_2d=False
    )
    weights_path = "data/models/ehpi3d/ehpi_3d_sim_c01_actionrec_gt_pred_zero_by_score.pth"
    md, latex = get_experiment_results(weights_path, pedrec_cfg, gt_result_ratio)
    results.append(md)
    results_latex.append(latex)

    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=gt_result_ratio,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False,
        frame_sampling=2
    )
    weights_path = "data/models/ehpi3d/ehpi_3d_sim_c01_actionrec_gt_pred_15fps.pth"
    md, latex = get_experiment_results(weights_path, pedrec_cfg, gt_result_ratio)
    results.append(md)
    results_latex.append(latex)

    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=False,
        scale_factor=0,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=gt_result_ratio,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False,
        frame_sampling=2
    )
    weights_path = "data/models/ehpi3d/ehpi_3d_sim_c01_actionrec_gt_pred_ehpi2dvids_15fps.pth"
    md, latex = get_experiment_results(weights_path, pedrec_cfg, gt_result_ratio)
    results.append(md)
    results_latex.append(latex)

    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=False,
        scale_factor=0,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=gt_result_ratio,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False,
        temporal_field=ImageSize(64, 32)
    )
    weights_path = "data/models/ehpi3d/ehpi_3d_sim_c01_actionrec_gt_pred_64frames.pth"
    md, latex = get_experiment_results(weights_path, pedrec_cfg, gt_result_ratio)
    results.append(md)
    results_latex.append(latex)

    return results, results_latex

if __name__ == '__main__':
    results = ["| Model | GT/Result Ratio | Balanced Acc | mAP | OF1 | OP | OR | CF1 | CP | CR"]
    results_latex = ["Model & G & mBAcc & mAP & OF1 & OP & OR & CF1 & CP & CR \\\\\\midrule"]
    result, result_latex = run_experiments(1)
    results += result
    results_latex += result_latex
    result, result_latex = run_experiments(0)
    results += result
    results_latex += result_latex

    for result in results:
        print(result)

    for result in results_latex:
        print(result)


# TODO: Eval mit Recognition Pipeline