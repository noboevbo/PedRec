import sys

sys.path.append(".")

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from pedrec.networks.net_pedrec.v5.p2d_p3d_o_c import PedRecNetV5, PedRecNetLossWrapperV5
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.models.experiments.experiment_description import ExperimentDescription
from pedrec.tools.networks.resnet_initializer import initialize_from_p2d_p3d
from pedrec.training.experiments.experiment_dataset_helper import get_validation_sets
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths_home
from pedrec.training.experiments.experiment_train_helper import init_experiment, \
    get_outputs_loss_mtl, validate_val_sets
from pedrec.utils.torch_utils.torch_helper import get_device


def get_preds_mtl(outputs: torch.Tensor):
    orientation = outputs[2].cpu().detach().numpy()
    return {
        "skeleton": outputs[0].cpu().detach().numpy(),
        "skeleton_3d": outputs[1].cpu().detach().numpy(),
        "orientation": orientation,
    }


def main():
    experiment_paths = get_experiment_paths_home()
    experiment_description = ExperimentDescription(
        net_name="PedRecNetMtlFullLossWrapperV5",
        experiment_name="v5_p2d_p3d_o_c",
        initialization_notes="Initialized from p2d. Deconv heads without combined deconvs as heatmap inputs.",
        experiment_paths=experiment_paths,
        net_cfg=PedRecNet50Config(),
        use_val_sim=False,
        use_val_coco=False,
        validate_orientation_sim=True,
        validate_joint_conf_sim=True,
        validate_joint_conf_coco=True,
        validate_joint_conf_h36m=True,
        sim_val_subsampling=10,
        batch_size=48
    )
    # TODO: Add train rtvalidate03
    # experiment_paths.sim_val_dir = experiment_paths.sim_train_dir
    # experiment_paths.sim_val_filename = experiment_paths.sim_train_filename
    init_experiment(experiment_description.seed)
    device = get_device(use_gpu=True)

    ####################################################################################################################
    ############################################ Initialize Network ####################################################
    ####################################################################################################################
    net = PedRecNetV5(experiment_description.net_cfg)
    net.init_weights()
    # initialize_pose_resnet(net, experiment_paths.pose_resnet_weights_path)  # load COCO pretrained weights
    initialize_from_p2d_p3d(net, experiment_paths.p2d_p3d_path)
    net = PedRecNetLossWrapperV5(net, device)
    net.to(device)
    net.load_state_dict(torch.load("data/models/pedrec/single_results/v5_p2d_p3d_o_c.pth"))

    get_outputs_loss_func = lambda net, model_input, labels: get_outputs_loss_mtl(net, model_input, labels)
    get_gt_pred_func = get_preds_mtl

    ####################################################################################################################
    ################################################# Datasets #########################################################
    ####################################################################################################################
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    validation_sets = get_validation_sets(experiment_description, trans)
    validation_results = validate_val_sets(net, validation_sets, get_outputs_loss_func, get_gt_pred_func,
                                           device, experiment_description.net_cfg.model.input_size)
    a = 1


if __name__ == '__main__':
    main()
