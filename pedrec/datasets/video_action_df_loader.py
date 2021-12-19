from typing import Tuple, List

import numpy as np
import pandas as pd

from pedrec.configs.dataset_configs import VideoActionDatasetConfig
from pedrec.models.constants.sample_method import SAMPLE_METHOD
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.models.datasets.video_action_dataset_entry_annotations import VideoActionDatasetEntries
from pedrec.utils.pandas_helper import get_subsampled_df
from pedrec.utils.pedrec_dataset_helper import get_filter_skeleton2d, get_filter_body_orientation, \
    get_filter_skeleton3d, get_filter_head_orientation


def set_scene_start_end(df: pd.DataFrame):
    scene_groups = df.groupby("path")
    df['scene_start'] = 0
    df['scene_end'] = 0
    for path, vid_df in scene_groups:
        mask = df['path'] == path
        df.loc[mask, 'scene_start'] = vid_df.index.min()
        df.loc[mask, 'scene_end'] = vid_df.index.max()

def filter_out(df: pd.DataFrame, filters: List[str]):
    # hard coded exclusion of SIM data, only real required for now.
    # df = df[~df['path'].str.contains("SIM")]
    mask = (~df['path'].str.contains(filters[0]))
    for i in range(1, len(filters)):
        mask = (mask & (~df['path'].str.contains(filters[i])))
    df = df[mask]
    df = df.reset_index()
    return df

def filter_in(df: pd.DataFrame, filters: List[str]):
    mask = df['path'].str.contains(filters[0])
    for i in range(1, len(filters)):
        mask = (mask | (df['path'].str.contains(filters[i])))
    df = df[mask]
    df = df.reset_index()
    return df

def get_annotations_from_video_action_df(df_path: str,
                                   cfg: VideoActionDatasetConfig,
                                   excluded_folders: str,
                                   include_folders: List[str]) -> VideoActionDatasetEntries:
    assert excluded_folders is None or include_folders is None
    df = get_subsampled_df(df_path, cfg)
    if include_folders is not None:
        df = filter_in(df, include_folders)
    elif excluded_folders is not None:
        df = filter_out(df, excluded_folders)

    set_scene_start_end(df)
    # df2 = pd.read_pickle("data/datasets/ROMb/v5_pose_results.pkl")
    df_full_length = len(df)

    df_used_length = len(df)
    filter_skeleton2d = get_filter_skeleton2d(df)
    filter_skeleton3d = get_filter_skeleton3d(df)
    filter_body_orientation = get_filter_body_orientation(df)
    filter_head_orientation = get_filter_head_orientation(df)

    # test = df[filter_skeleton2d]
    annotations = VideoActionDatasetEntries(
        img_dirs=df['path'].to_numpy(dtype=str),
        scene_starts=df['scene_start'].to_numpy(dtype=np.uint32),
        scene_ends=df['scene_end'].to_numpy(dtype=np.uint32),
        frame_nr_locals=df['frame'].to_numpy(dtype=np.uint32),
        actions=df['action'],
        body_orientations=df[filter_body_orientation].to_numpy(dtype=np.float32).reshape(df_used_length, 1, 4),
        head_orientations=df[filter_head_orientation].to_numpy(dtype=np.float32).reshape(df_used_length, 1, 4),
        skeleton2ds=df[filter_skeleton2d].to_numpy(dtype=np.float32).reshape(df_used_length, len(SKELETON_PEDREC_JOINTS), 5),
        skeleton3ds=df[filter_skeleton3d].to_numpy(dtype=np.float32).reshape(df_used_length, len(SKELETON_PEDREC_JOINTS), 6),
    )
    return annotations

if __name__ == "__main__":
    x = get_annotations_from_video_action_df("data/videos/ehpi_videos/pedrec_results.pkl",
                                       VideoActionDatasetConfig(subsample=1, subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
                                                                use_unit_skeleton=True,
                                                                min_joint_score=0,
                                                                add_2d=False,
                                                                flip=False,
                                                                skeleton_3d_range=3000),
                                             excluded_folders = ["SIM", "2019_ITS_Journal_Eval2"],
                                             include_folders=None)
    # df = pd.read_pickle()
    a = 1

