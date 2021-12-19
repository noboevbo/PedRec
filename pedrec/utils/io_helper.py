import numpy as np

from pedrec.configs.dataset_configs import PedRecDatasetConfigDefault
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.datasets.pedrec_dataset import PedRecDataset
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.models.data_structures import ImageSize


def skeleton_2d_to_tikz_array(skeleton_2d: np.ndarray):
    outputs = []
    for joint in SKELETON_PEDREC_JOINTS:
        joint_data = skeleton_2d[joint.value]
        outputs.append("/".join(map(str, map(int, joint_data[0:4]))))
    return ",".join(outputs)

def skeleton_3d_to_tikz_array(skeleton_2d: np.ndarray):
    outputs = []
    for joint in SKELETON_PEDREC_JOINTS:
        joint_data = skeleton_2d[joint.value]
        outputs.append("/".join(map(str, joint_data[0:5])))
    return ",".join(outputs)


if __name__ == "__main__":
    net_cfg = PedRecNet50Config()
    dataset_name = "RT3DValidateTest"
    dataset_path = "data/datasets/ROM"
    dataset_cfg = PedRecDatasetConfigDefault().cfg
    dataset = PedRecDataset(dataset_path, "rt_rom_01",
                            DatasetType.TRAIN, dataset_cfg, net_cfg.model.input_size,
                            ImageSize(1920, 1080), None)
    model_input, labels = dataset[0]
    skeleton = labels["skeleton"]
    skeleton[:, 0] *= model_input.shape[1]
    skeleton[:, 1] *= model_input.shape[0]
    a = skeleton_2d_to_tikz_array(skeleton)
    print(a)
    skeleton_3d = labels["skeleton_3d"]
    skeleton_3d[:, 0:3] *= 3000
    skeleton_3d[:, 0:3] -= 1500
    b = skeleton_3d_to_tikz_array(skeleton_3d)
    print(b)