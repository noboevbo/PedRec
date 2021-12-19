import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from pedrec.configs.dataset_configs import get_h36m_val_dataset_cfg_default
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.datasets.pedrec_dataset import PedRecDataset
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.networks.net_pedrec.pedrec_net_mtl_wrapper import PedRecNetMTLWrapper
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths_home
from pedrec.utils.torch_utils.torch_helper import get_device

h36m_actions = [
    "Walking(?: .)*\.",
    "Greeting(?: .)*\.",
    "Discussion(?: .)*\.",
    "Phoning(?: .)*\.",
    "Waiting(?: .)*\.",
    "Directions(?: .)*\.",
    "Posing(?: .)*\.",
    "Purchases(?: .)*\.",
    "SittingDown(?: .)*\.",
    "Smoking(?: .)*\.",
    "Eating(?: .)*\.",
    "Sitting(?: .)*\.",
    "Photo(?: .)*\.",
    "WalkTogether(?: .)*\.",
    "WalkDog(?: .)*\.|WalkingDog(?: .)*\."
]


def get_msjpe(output, target):
    mask = target[:, :, 4] == 1
    a = output[mask]
    b = target[mask][:, :3]
    num_visible_joints = b.shape[0]
    if num_visible_joints < 1:
        return 0
    test = torch.norm(a[:, 0:3] - b[:, 0:3], dim=1)
    return torch.mean(test)


def validate_h36m(net, val_loaders, device, skeleton_3d_range: int):
    net.eval()

    all = []
    # all_single = []
    for action, val_loader in zip(h36m_actions, val_loaders):
        msjes = []
        msjpes_single = []
        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                # if count > 2:
                #     break
                # count += 1
                inputs, labels = val_data
                inputs = inputs.to(device)

                outputs = net(inputs)

                # results_single = get_msjpe(torch.unsqueeze(inputs[:, -1, :, :], dim=1), labels)
                results = get_msjpe(outputs[1], labels["skeleton_3d"].to(device))
                # msjpes_single.append(results_single.cpu().detach().numpy())
                msjes.append(results.cpu().detach().numpy())

        msej_mean = np.mean(np.array(msjes) * skeleton_3d_range)
        # msej_mean_single = np.mean(np.array(msjpes_single) * 1000)
        all.append(msej_mean)
        # all_single.append(msej_mean_single)
        print(f"{action}: {msej_mean}")
    full = round(np.mean(np.array(all)), 1)
    # full_single = round(np.mean(np.array(all_single)), 1)
    print(f"Val MSJPE: {full}")


def val_example():
    # Initialize net
    device = get_device(use_gpu=True)
    net_cfg = PedRecNet50Config()
    net = PedRecNet(net_cfg)
    net.init_weights()
    net.load_state_dict(torch.load(
        "data/models/pedrec/experiment_p2d_p3d_conv_shared_orientation_conf_frozenFE_net.pth"))
    net.to(device)

    # Load H36M validation set
    experiment_paths = get_experiment_paths_home()
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_loaders = []
    val_sets_length = 0
    val_cfg = get_h36m_val_dataset_cfg_default()
    val_cfg.subsample = 1
    for action in h36m_actions:
        val_set = PedRecDataset(experiment_paths.h36m_val_dir,
                                experiment_paths.h36m_val_filename,
                                DatasetType.VALIDATE, val_cfg,
                                net_cfg.model.input_size,
                                trans,
                                is_h36m=True,
                                action_filters=action)
        val_loader = DataLoader(val_set, batch_size=48, shuffle=False, num_workers=12)
        val_loaders.append(val_loader)
        set_length = len(val_set)
        print(f"{action}: {set_length}")
        val_sets_length += set_length
    print(f"Val sets length: {val_sets_length}")
    validate_h36m(net, val_loaders, device, val_cfg.skeleton_3d_range)


if __name__ == "__main__":
    val_example()
