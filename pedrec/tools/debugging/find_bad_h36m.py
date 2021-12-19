#%%

import logging

import numpy as np
import torch
from pedrec.datasets.h36m_dataset import Human36MDataset
from pedrec.evaluations.pckh import get_pck_results
from torch.backends import cudnn
from torchvision import transforms
from tqdm import tqdm

from pedrec.configs.app_config import AppConfig
from pedrec.configs.dataset_configs import CocoDatasetConfigDefault
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.evaluations.eval_helper import get_total_coords
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.data_structures import ImageSize
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.utils.demo_helper import init_pose_model
from pedrec.utils.log_helper import configure_logger
from pedrec.utils.torch_utils.torch_helper import get_device

#%%

## Standard configurations

configure_logger()
logger = logging.getLogger(__name__)
app_cfg = AppConfig()
dataset_cfg = CocoDatasetConfigDefault()
pose_cfg = PedRecNet50Config()
device = get_device(app_cfg.cuda.use_gpu)

cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Data loading code
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

inv_trans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

#%%

## Load dataset
# coco_dir = "data/datasets/COCO"
# val_set = CocoDataset(coco_dir, DatasetType.VALIDATE, dataset_cfg.cfg, pose_cfg.model, trans)
# val_loader = DataLoader(val_set, batch_size=64, num_workers=6)

val_set = Human36MDataset("data/datasets/Human3.6m/", "h36m_set.hdf5", DatasetType.VALIDATE, dataset_cfg.cfg, pose_cfg.model,
                            ImageSize(1000, 1000),
                            trans)
print(len(val_set))

#%%

## Load network
pose_weights = "data/models/pedrec/pedrec_net_v6_net.pth"
net: PedRecNet = init_pose_model(PedRecNet(pose_cfg), pose_weights, logger, device)
net.to(device)
print("network loaded!")

#%%

## Test on one img
examples = []
with tqdm(total=len(val_set)) as pbar:
    for i in range(0, len(val_set)):
        model_inputs, labels = val_set[i]
        model_inputs = model_inputs.unsqueeze(0)
        model_inputs = model_inputs.to(device)
        img_path = labels["img_path"]
        skeleton_labels = np.expand_dims(labels["skeleton"], axis=0)
        centers = np.expand_dims(labels["center"], axis=0)
        scales = np.expand_dims(labels["scale"], axis=0)
        rotations = np.expand_dims(labels["rotation"], axis=0)
        outputs = net(model_inputs)
        output_coords = outputs[0].detach().cpu().numpy()
        skeletons_gt = skeleton_labels.copy()
        skeletons_gt_global = get_total_coords(skeletons_gt, pose_cfg.model.input_size, centers, scales, rotations)
        skeletons_pred = output_coords
        skeletons_pred_global = get_total_coords(skeletons_pred, pose_cfg.model.input_size, centers, scales, rotations)

        ## Show PCKh / other metrics?

        gts = skeletons_gt_global[:, :, 0:2]
        preds = skeletons_pred_global[:, :, 0:2]
        visibles = skeletons_gt_global[:, :, 3]
        print("@0.05")
        pck = get_pck_results(gts, preds, visibles, 0.05)
        pck_wo_nans = pck[~np.isnan(pck)]
        pck_05 = np.sum(pck_wo_nans) / len(pck_wo_nans)
        print("@0.2")
        pck = get_pck_results(gts, preds, visibles, 0.2)
        pck_wo_nans = pck[~np.isnan(pck)]
        pck_2 = np.sum(pck_wo_nans) / len(pck_wo_nans)

        if pck_05 < 5 and pck_2 > 70:
            examples.append(i)
        pbar.update()


a = 1
