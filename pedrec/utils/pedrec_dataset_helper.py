import numpy as np
import pandas as pd

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS


def get_filter_skeleton2d(df: pd.DataFrame):
    return [col for col in df if col.startswith('skeleton2d')]


def get_filter_skeleton3d(df: pd.DataFrame):
    return [col for col in df if col.startswith('skeleton3d')]


def get_filter_bb(df: pd.DataFrame):
    return [col for col in df if col.startswith('bb')]


def get_filter_body_orientation(df: pd.DataFrame):
    return [col for col in df if col.startswith('body_orientation')]


def get_filter_head_orientation(df: pd.DataFrame):
    return [col for col in df if col.startswith('head_orientation')]


def get_filter_env(df: pd.DataFrame):
    return [col for col in df if col.startswith('env')]


def get_gt_arrays(df: pd.DataFrame):
    df_len = len(df)
    filter_skeleton2d = get_filter_skeleton2d(df)
    filter_skeleton3d = get_filter_skeleton3d(df)
    filter_bb = get_filter_bb(df)
    filter_body_orientation = get_filter_body_orientation(df)
    filter_head_orientation = get_filter_head_orientation(df)
    filter_env = get_filter_env(df)

    bbs = df[filter_bb].to_numpy(dtype=np.float32)
    env_positions = df[filter_env].to_numpy(dtype=np.float32)
    body_orientations = df[filter_body_orientation].to_numpy(dtype=np.float32).reshape(df_len, 1, 4)
    head_orientations = df[filter_head_orientation].to_numpy(dtype=np.float32).reshape(df_len, 1, 4)
    skeleton2ds = df[filter_skeleton2d].to_numpy(dtype=np.float32).reshape(df_len, len(SKELETON_PEDREC_JOINTS), 5)
    skeleton3ds = df[filter_skeleton3d].to_numpy(dtype=np.float32).reshape(df_len, len(SKELETON_PEDREC_JOINTS), 6)

    return skeleton2ds, skeleton3ds, body_orientations, head_orientations, bbs, env_positions


def get_pred_arrays(df: pd.DataFrame):
    df_len = len(df)
    filter_skeleton2d = get_filter_skeleton2d(df)
    filter_skeleton3d = get_filter_skeleton3d(df)
    filter_body_orientation = get_filter_body_orientation(df)
    filter_head_orientation = get_filter_head_orientation(df)

    body_orientations = df[filter_body_orientation].to_numpy(dtype=np.float32).reshape(df_len, 1, 4)
    head_orientations = df[filter_head_orientation].to_numpy(dtype=np.float32).reshape(df_len, 1, 4)
    skeleton2ds = df[filter_skeleton2d].to_numpy(dtype=np.float32).reshape(df_len, len(SKELETON_PEDREC_JOINTS), 5)
    skeleton3ds = df[filter_skeleton3d].to_numpy(dtype=np.float32).reshape(df_len, len(SKELETON_PEDREC_JOINTS), 6)

    return skeleton2ds, skeleton3ds, body_orientations, head_orientations
