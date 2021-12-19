import sys

from pedrec.utils.torch_utils.mtl_lr_finder import MTLLRFinder

sys.path.append(".")
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import OneCycleLR
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet, PedRecNetLossHead
from pedrec.models.experiments.experiment_description import ExperimentDescription
from pedrec.models.experiments.experiment_round_description import ExperimentRoundDescription
from pedrec.training.experiments.experiment_dataset_helper import get_validation_sets, get_train_loader
from pedrec.training.experiments.experiment_log_helper import get_experiment_protocol, save_log
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths_home
from pedrec.training.experiments.experiment_train_helper import init_experiment, \
    train_round, get_outputs_loss_mtl
from pedrec.utils.torch_utils.torch_helper import get_device, split_no_wd_params


class PedRecNetMTLWrapper(nn.Module):
    def __init__(self, model: nn.Module, loss_head: nn.Module):
        super(PedRecNetMTLWrapper, self).__init__()
        self.model = model
        self.loss_head = loss_head

    def forward(self, inputs, targets):
        outputs = self.model(inputs)
        loss = self.loss_head(outputs, targets)
        return outputs, loss


def get_preds_mtl(outputs: torch.Tensor):
    return {
        "skeleton": outputs[0].cpu().detach().numpy(),
        "skeleton_3d": None,
        "orientation": None
    }

def initialize_pose_resnet(net, pose_resnet_weights_path: str):
    pose_resnet_state_dict = torch.load(pose_resnet_weights_path)
    net_weights = net.state_dict()
    for name, param in pose_resnet_state_dict.items():
        if name.startswith("final"):
            net_name = name.replace("final_layer.", "model.head_pose_2d.pose_heatmap_layer.")
            # net_name = f"head.pose_3d_heatmap_layer.{name}"
        elif name.startswith("deconv_layers.6") or name.startswith("deconv_layers.7"):
            net_name = f"model.head_pose_2d.{name.replace('deconv_layers', 'deconv_head').replace('6', '0').replace('7', '1')}"
        elif name.startswith("deconv"):
            net_name = f"model.conv_transpose_shared.{name}"
        else:
            net_name = f"model.feature_extractor.{name}"
        if net_name in net_weights:
            if net_weights[net_name].shape != param.shape and len(param.shape) == 1:
                net_weights[net_name][:17] = param
            elif net_weights[net_name].shape != param.shape and len(param.shape) == 4:
                net_weights[net_name][:17, :, :, :] = param
            elif net_weights[net_name].shape != param.shape:
                print(f"Shape mismatch in {net_name}: {net_weights[net_name].shape} <-> {param.shape}")
            else:
                net_weights[net_name] = param
        else:
            print(f"Skipped: {name}, tried: {net_name}")
    net.load_state_dict(net_weights)


def main():
    os.environ['NUMEXPR_MAX_THREADS'] = '16'
    experiment_paths = get_experiment_paths_home()
    experiment_description = ExperimentDescription(
        net_name="PedRecNet2D",
        experiment_name="experiment_pedrec_p2d_coco_only",
        initialization_notes="Initialized from ResNet",
        experiment_paths=experiment_paths,
        net_cfg=PedRecNet50Config(),
        use_train_tud=False,
        use_train_sim=False,
        use_train_h36m=False,
        validate_3d_sim=False,
        validate_3d_h36m=False,
        # validate_orientation_sim=True,
        # validate_orientation_coco=True,
        # validate_joint_conf_sim=True,
        # validate_joint_conf_coco=True,
        # validate_joint_conf_h36m=True,

        sim_val_subsampling=10,
        batch_size=48
    )
    init_experiment(experiment_description.seed)
    device = get_device(use_gpu=True)
    net = PedRecNet(experiment_description.net_cfg)
    net.init_weights()
    loss_head = PedRecNetLossHead(device)
    loss_head.use_p3d_loss = False
    loss_head.use_conf_loss = False
    loss_head.use_orientation_loss = False
    net = PedRecNetMTLWrapper(net, loss_head)
    # initialize_weights_with_same_name_and_shape(net,
    #                                             experiment_description.experiment_paths.pose_2d_coco_only_weights_path)
    initialize_pose_resnet(net, experiment_description.experiment_paths.pose_resnet_weights_path)
    net.to(device)
    experiment_description.net_layer_names = [
        "net.model.feature_extractor",
        "net.model.conv_transpose_shared",
        "net.model.head_pose_2d",
        "net.model.head_pose_3d",
        "net.model.head_orientation",
        "net.model.head_conf"
    ]
    experiment_description.net_layers = [
        net.model.feature_extractor,
        net.model.conv_transpose_shared,
        net.model.head_pose_2d,
        net.model.head_pose_3d,
        net.model.head_orientation,
        net.model.head_conf
    ]
    cycle(0, experiment_description, net, device)

    protocol = get_experiment_protocol(experiment_description)
    save_log(protocol,
             os.path.join(experiment_paths.output_dir, f"{experiment_description.experiment_name}_protocol.md"))
    print(protocol)


def cycle(cycle_num: int, experiment_description: ExperimentDescription, net, device, frozen_1=None, frozen_2=None,
          freeze_hard: bool = False):
    if cycle_num > 0:
        net.load_state_dict(torch.load(
            os.path.join(experiment_description.experiment_paths.output_dir, f"{experiment_description.experiment_name}_{cycle_num-1}.pth")))

    split_params = split_no_wd_params(experiment_description.net_layers)
    params = ([{'params': p, 'weight_decay': 0 if wd else 1e-2} for (wd, p) in split_params])

    get_outputs_loss_func = lambda net, model_input, labels: get_outputs_loss_mtl(net, model_input, labels)
    get_gt_pred_func = get_preds_mtl

    ####################################################################################################################
    ################################################# Datasets #########################################################
    ####################################################################################################################
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader = get_train_loader(experiment_description, trans)
    validation_sets = get_validation_sets(experiment_description, trans)

    optimizer_x = torch.optim.AdamW(params, lr=1e-7, weight_decay=1e-2)
    lr_finder = MTLLRFinder(optimizer_x, net, device="cuda")
    lr_finder.range_test(train_loader, end_lr=0.05, num_iter=100)
    _, experiment_description.suggested_lr = lr_finder.plot()  # to inspect the loss-learning rate graph
    # test = lr_finder.plot()
    lr_finder.reset()  # to reset the model and optimizer to their initial state

    # experiment_description.suggested_lr = 4.33e-03

    # Append sigma AFTER LR range test.
    params.append({'params': net.loss_head.sigmas, 'weight_decay': 1e-2})

    ####################################################################################################################
    ########################################### P2D + P3D Pretrain 01 ###################################################
    ####################################################################################################################
    epochs = 10
    scheduler_lrs = [
        experiment_description.suggested_lr, experiment_description.suggested_lr,  # feature extractor
        experiment_description.suggested_lr, experiment_description.suggested_lr,  # net.model.conv_transpose_shared
        experiment_description.suggested_lr, experiment_description.suggested_lr,  # p2d
        0, 0,  # p3d
        0, 0,  # orientation
        0, 0,  # conf
        0  # sigmas
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
    experiment_round_description = ExperimentRoundDescription(
        num_epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        frozen_layers=[
            net.model.feature_extractor,
        ],
        max_lrs=scheduler_lrs,
        optimizer_parameters=optimizer_params
    )
    if frozen_1 is not None:
        experiment_description.frozen_layers = frozen_1
    experiment_description.experiment_rounds.append(experiment_round_description)
    experiment_round_description.validation_results = train_round(net, experiment_description,
                                                                  experiment_round_description, train_loader,
                                                                  validation_sets,
                                                                  get_outputs_loss_func, get_gt_pred_func, device,
                                                                  log=True, freeze_hard=freeze_hard)
    torch.save(net.state_dict(),
               os.path.join(experiment_description.experiment_paths.output_dir,
                            f"{experiment_description.experiment_name}_{cycle_num}_01.pth"))

    ####################################################################################################################
    ########################################### P2D + P3D Pretrain 02 ###################################################
    ####################################################################################################################
    epochs = 5
    scheduler_lrs = [
        experiment_description.suggested_lr / 20, experiment_description.suggested_lr / 20,  # feature extractor
        experiment_description.suggested_lr / 10, experiment_description.suggested_lr / 10,
        # net.model.conv_transpose_shared
        experiment_description.suggested_lr / 10, experiment_description.suggested_lr / 10,  # p2d
        0, 0,  # p3d
        0, 0,  # orientation
        0, 0,  # conf
        0  # sigmas
    ]
    optimizer = torch.optim.AdamW(params, **optimizer_params)
    scheduler = OneCycleLR(optimizer, max_lr=scheduler_lrs, steps_per_epoch=len(train_loader),
                           epochs=epochs)
    experiment_round_description = ExperimentRoundDescription(
        num_epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        frozen_layers=[],
        max_lrs=scheduler_lrs,
        optimizer_parameters=optimizer_params
    )
    if frozen_2 is not None:
        experiment_description.frozen_layers = frozen_2
    experiment_description.experiment_rounds.append(experiment_round_description)
    experiment_round_description.validation_results = train_round(net, experiment_description,
                                                                  experiment_round_description, train_loader,
                                                                  validation_sets,
                                                                  get_outputs_loss_func, get_gt_pred_func, device,
                                                                  log=True, freeze_hard=freeze_hard)
    torch.save(net.state_dict(),
               os.path.join(experiment_description.experiment_paths.output_dir,
                            f"{experiment_description.experiment_name}_{cycle_num}.pth"))


if __name__ == '__main__':
    main()
