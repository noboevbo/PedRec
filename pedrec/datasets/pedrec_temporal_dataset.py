import copy
import math
import os
from random import random
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from pedrec.configs.app_config import AppConfig
from pedrec.configs.dataset_configs import get_sim_dataset_cfg_default, \
    PedRecTemporalDatasetConfig
from pedrec.datasets.dataset_helper import get_ehpi_skeleton_sequence_3d
from pedrec.datasets.pedrec_df_loader import get_annotations_from_pedrec_df
from pedrec.models.constants.action_mappings import ACTION
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.constants.sample_method import SAMPLE_METHOD
from pedrec.models.data_structures import ImageSize


class PedRecTemporalDataset(Dataset):
    def __init__(self, dataset_root_path: str,
                 dataset_filename: str,
                 mode: DatasetType,
                 cfg: PedRecTemporalDatasetConfig,
                 action_list: List[ACTION],
                 transform: Callable,
                 action_filters: List[str] = None,
                 pose_results_file: str = None,
                 use_movement_instead_action: bool = False
                 ):
        """
        pose_results_file: Filename of results from an algorithm df, should contain pose2d / 3d results with same index
        as gt.
        - use_movement_instead_action: Uses the movement label used in RTSIMs instead of actions, movements are binary, thus it returns only the label for CE loss.
        - frame-sampling: currently only supports training data with same input fps
        """
        self.action_filter = action_filters
        self.use_movement_instead_action = use_movement_instead_action
        self.mode = mode
        self.cfg = copy.deepcopy(cfg)
        self.cfg.subsample = 1
        self.cfg.subsampling_strategy = SAMPLE_METHOD.SYSTEMATIC
        self.half_skeleton_range = self.cfg.skeleton_3d_range / 2
        self.dataset_root_path = dataset_root_path
        self.dataset_hdf5_path = os.path.join(self.dataset_root_path, dataset_filename)
        self.dataset_results_path = None
        self.results_provided = pose_results_file is not None
        if self.results_provided:
            self.dataset_results_path = os.path.join(self.dataset_root_path, pose_results_file)
        self.transform = transform
        self.index_mappings, self.annotations, self.info = get_annotations_from_pedrec_df(
            os.path.join(dataset_root_path, dataset_filename),
            self.cfg,
            ImageSize(192, 256),
            self.dataset_results_path,
            self.action_filter,
            is_h36m=False)
        self.index_mappings = list(self.index_mappings)

        self.action_label_mapping = {}
        for action_idx, action in enumerate(action_list):
            self.action_label_mapping[action.value] = action_idx

        self.num_full_dataset = len(self.index_mappings)
        self.indexes = list(range(0, self.num_full_dataset))
        if cfg.subsample != 1:
            if cfg.subsampling_strategy == SAMPLE_METHOD.SYSTEMATIC:
                self.indexes = range(0, self.num_full_dataset, cfg.subsample)
            elif cfg.subsampling_strategy == SAMPLE_METHOD.RANDOM:
                self.indexes = np.random.choice(self.indexes, math.floor(self.num_full_dataset / cfg.subsample),
                                                replace=False)
            else:
                raise NotImplementedError(f"Sampling strategy {cfg.subsampling_strategy.name} is not implemented.")
        self.__length = len(self.indexes)
        self.test = [0] * len(ACTION)
        self.test_count = 0

    def __len__(self):
        return self.__length

    def get_label(self, curr_frame_annotations, valid):
        label = np.zeros(len(self.action_label_mapping), dtype=np.float32)
        if not valid:
            return label
        for action in curr_frame_annotations.actions:
            if action == ACTION.WALK.value:
                # SELECT WALK / JOG / RUN based on movement description
                if curr_frame_annotations.movement == ACTION.WALK.value \
                        or curr_frame_annotations.movement == ACTION.JOG.value \
                        or curr_frame_annotations.movement == ACTION.RUN.value:
                    # print(curr_frame_annotations.movement)
                    label[self.action_label_mapping[curr_frame_annotations.movement]] = 1
                else:
                    label[self.action_label_mapping[action]] = 1
            else:
                label[self.action_label_mapping[action]] = 1
        if ACTION.WALK.value not in curr_frame_annotations.actions \
                and ACTION.JOG.value not in curr_frame_annotations.actions \
                and ACTION.RUN.value not in curr_frame_annotations.actions \
                and ACTION.STUMBLE.value not in curr_frame_annotations.actions \
                and ACTION.FALL.value not in curr_frame_annotations.actions \
                and ACTION.STAND_UP.value not in curr_frame_annotations.actions:
            # if there is no walk / jog / run annotated we assume standing
            label[self.action_label_mapping[ACTION.STAND.value]] = 1
        if curr_frame_annotations.movement == ACTION.STAND.value:
            if ACTION.WALK.value not in curr_frame_annotations.actions \
                    and ACTION.JOG.value not in curr_frame_annotations.actions \
                    and ACTION.RUN.value not in curr_frame_annotations.actions:
                label[self.action_label_mapping[ACTION.STAND.value]] = 1
        return label

    def __getitem__(self, dataset_idx):
        index = self.index_mappings[self.indexes[dataset_idx]]
        curr_frame_annotations_orig = self.annotations.get_entry(index, self.cfg)
        curr_frame_annotations = copy.deepcopy(curr_frame_annotations_orig)

        # get skeleton gt from GT or Pose Recognition Results
        if self.results_provided and random() >= self.cfg.gt_result_ratio:
            skeletons_3d = self.annotations.skeleton3ds_results
            skeletons_2d = self.annotations.skeleton2ds_results
        else:
            skeletons_3d = self.annotations.skeleton3ds
            skeletons_2d = self.annotations.skeleton2ds

        skeleton_sequence_3d, valid = get_ehpi_skeleton_sequence_3d(index=index,
                                                                    temporal_field=self.cfg.temporal_field,
                                                                    curr_frame_annotations=curr_frame_annotations,
                                                                    skeletons_3d=skeletons_3d,
                                                                    skeletons_2d=skeletons_2d,
                                                                    mode=self.mode,
                                                                    flip=self.cfg.flip,
                                                                    half_skeleton_range=self.half_skeleton_range,
                                                                    skeleton_3d_range=self.cfg.skeleton_3d_range,
                                                                    use_2d=self.cfg.add_2d,
                                                                    min_score=self.cfg.min_joint_score,
                                                                    use_unit_skeleton=self.cfg.use_unit_skeleton,
                                                                    frame_sampling=self.cfg.frame_sampling)

        label = self.get_label(curr_frame_annotations=curr_frame_annotations, valid=valid)
        # skeleton_sequence_3d = np.transpose(skeleton_sequence_3d, (2, 0, 1))
        ehpi = skeleton_sequence_3d * 255
        ehpi = ehpi.astype(np.uint8)

        if self.transform:
            ehpi = self.transform(ehpi)

        # return ehpi, label, curr_frame_annotations

        return ehpi, label


if __name__ == "__main__":
    app_cfg = AppConfig()
    dataset_name = "RT3ROM"
    dataset_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=0,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False,
        temporal_field = ImageSize(64, 32)
    )
    # dataset_file = "rt_conti_01_val_FIN.pkl"
    # dataset_results_file = "C01F_pred_df_experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_allframes.pkl"
    dataset = PedRecTemporalDataset("data/datasets/Conti01", "rt_conti_01_val_FIN.pkl",
                                    DatasetType.VALIDATE, dataset_cfg, app_cfg.inference.action_list, None,
                                    pose_results_file="C01F_pred_df_experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_allframes.pkl")



    # ehpis = np.zeros((len(dataset), 32, 32, 3), dtype=np.float32)
    # for idx, entry in enumerate(dataset):
    #     ehpi, _ = entry
    #     ehpis[idx, :, :, :] = ehpi / 255
    #
    # r = np.reshape(ehpis[:, :, :, 0], -1)
    # g = np.reshape(ehpis[:, :, :, 1], -1)
    # b = np.reshape(ehpis[:, :, :, 2], -1)
    #
    # mean_r = np.mean(r, axis=0)
    # mean_g = np.mean(g, axis=0)
    # mean_b = np.mean(b, axis=0)
    # std_r = np.std(r, axis=0)
    # std_g = np.std(g, axis=0)
    # std_b = np.std(b, axis=0)
    #
    # print(f"mean: [{mean_r}, {mean_g}, {mean_b}], std: [{std_r}, {std_g}, {std_b}]")
    # ehpi, action = dataset[103658]
    for ehpi, action, x in dataset:
        a = 1
        # if action[0] == 1:
        #     imgplot = plt.imshow(ehpi)
        # plt.show()
    # ehpi, action, x = dataset[40]


    b = 1
