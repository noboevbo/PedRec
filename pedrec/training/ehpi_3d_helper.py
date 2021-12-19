import sys
import time
from typing import Tuple

from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from pedrec.configs.app_config import AppConfig
from pedrec.configs.dataset_configs import PedRecTemporalDatasetConfig, VideoActionDatasetConfig
from pedrec.datasets.dataset_helper import worker_init_fn
from pedrec.datasets.pedrec_temporal_dataset import PedRecTemporalDataset
from pedrec.datasets.video_action_dataset import VideoActionDataset
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.experiments.experiment_paths import ExperimentPaths
from pedrec.networks.net_pedrec.ehpi_3d_net import Ehpi3DNet
from pedrec.utils.ehpi_helper import ehpi_transform
from pedrec.utils.torch_utils.lr_finder import LRFinder

sys.path.append(".")
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import OneCycleLR
from pedrec.training.experiments.experiment_train_helper import init_experiment
from pedrec.utils.torch_utils.torch_helper import get_device, split_no_wd_params, move_to_device


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


def get_train_loader(experiment_paths: ExperimentPaths, batch_size, action_list,
                     pedrec_cfg: PedRecTemporalDatasetConfig, vid_cfg: VideoActionDatasetConfig = None):
    sim_train = PedRecTemporalDataset(experiment_paths.sim_c01_dir,
                                      experiment_paths.sim_c01_filename,
                                      DatasetType.TRAIN,
                                      pedrec_cfg,
                                      action_list,
                                      ehpi_transform,
                                      pose_results_file=experiment_paths.sim_c01_results_filename)

    train_set = sim_train

    if vid_cfg is not None:
        ehpi_vid_dataset = VideoActionDataset("data/videos/ehpi_videos/", "pedrec_p2d3d_c_o_h36m_sim_mebow_0_results.pkl",
                                              DatasetType.TRAIN, vid_cfg, action_list, ehpi_transform)
        train_set = ConcatDataset([sim_train, ehpi_vid_dataset])

    return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12, worker_init_fn=worker_init_fn)


def ehpi3d_experiment(experiment_name: str, experiment_paths: ExperimentPaths, pedrec_cfg: PedRecTemporalDatasetConfig,
                      vid_cfg: VideoActionDatasetConfig = None):
    app_cfg = AppConfig()
    init_experiment(42)
    device = get_device(use_gpu=True)
    net = Ehpi3DNet(len(app_cfg.inference.action_list))
    # initialize_weights_with_same_name_and_shape(net,
    #                                             "data/models/ehpi3d/ehpi3d_c01_test_01.pth")
    # initialize_from_imgnet(net, experiment_paths.pose_resnet_weights_path)
    net.to(device)
    net_layers = [
        net,
    ]

    cycle(0, net, device, experiment_paths, net_layers, "data/models/ehpi3d/", experiment_name, app_cfg.inference.action_list,
          pedrec_cfg, vid_cfg)


def cycle(cycle_num: int, net, device, experiment_paths, net_layers, output_dir, experiment_name, action_list,
          pedrec_cfg: PedRecTemporalDatasetConfig, vid_cfg: VideoActionDatasetConfig = None):
    if cycle_num > 0:
        net.load_state_dict(torch.load(
            os.path.join(output_dir, f"{experiment_name}.pth")))

    split_params = split_no_wd_params(net_layers)
    params = ([{'params': p, 'weight_decay': 0 if wd else 1e-2} for (wd, p) in split_params])

    ####################################################################################################################
    ################################################# Datasets #########################################################
    ####################################################################################################################
    train_loader = get_train_loader(experiment_paths, 48, action_list, pedrec_cfg, vid_cfg)

    ## LR
    criterion = BCEWithLogitsLoss()
    # criterion = lambda predicted, target: get_msjpe(predicted, target)
    optimizer_x = torch.optim.AdamW(params, lr=1e-7, weight_decay=1e-2)
    lr_finder = LRFinder(net, optimizer_x, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=9e-01, num_iter=100)
    _, suggested_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
    lr_finder.reset()  # to reset the model and optimizer to their initial state
    # suggested_lr = 2e-04

    ####################################################################################################################
    ########################################### P2D + P3D Pretrain 01 ###################################################
    ####################################################################################################################
    epochs = 20
    scheduler_lrs = [
        suggested_lr, suggested_lr
    ]
    optimizer_params = {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "amsgrad": False
    }
    optimizer = torch.optim.AdamW(params, **optimizer_params)
    scheduler = OneCycleLR(optimizer, max_lr=scheduler_lrs, steps_per_epoch=len(train_loader),
                           epochs=epochs)

    for epoch in range(epochs):  # loop over the dastaset multiple times
        print(train(net, optimizer, criterion, scheduler, train_loader, device))

    torch.save(net.state_dict(), os.path.join(output_dir, f"{experiment_name}.pth"))


def train(net: torch.nn.Module, optimizer, criterion, scheduler, train_loader: DataLoader, device: torch.device) -> \
        Tuple[float, float]:
    start = time.time()
    loss_total = 0.0
    net.train()
    # for param in net.parameters():
    #     print(param.requires_grad)
    # param.requires_grad = False
    # count = 0
    with tqdm(total=len(train_loader)) as pbar:
        for i, train_data in enumerate(train_loader):
            # if count > 2:
            #     break
            # count += 1
            inputs, labels = train_data

            inputs = inputs.to(device)
            labels = move_to_device(labels, device)

            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            if scheduler is not None:
                scheduler.step()
            pbar.update()

    return loss_total / len(train_loader), time.time() - start
