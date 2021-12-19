import pandas as pd

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS


def set_df_dtypes(df: pd.DataFrame):
    df["dataset"] = df["dataset"].astype("category")
    df["dataset_type"] = df["dataset_type"].astype("category")
    df["scene_id"] = df["scene_id"].astype("category")
    df["scene_start"] = df["scene_start"].astype("category")
    df["scene_end"] = df["scene_end"].astype("category")
    df["frame_nr_global"] = df["frame_nr_global"].astype("uint32")
    df["frame_nr_local"] = df["frame_nr_local"].astype("uint32")
    df["img_dir"] = df["img_dir"].astype("category")
    df["img_id"] = df["img_id"].astype("uint32")
    df["img_type"] = df["img_type"].astype("category")
    df["subject_id"] = df["subject_id"].astype("category")
    df["gender"] = df["gender"].astype("category")
    df["skin_color"] = df["skin_color"].astype("category")
    df["size"] = df["size"].astype("category")
    df["bmi"] = df["bmi"].astype("category")
    df["age"] = df["age"].astype("category")
    df["movement"] = df["movement"].astype("category")
    df["movement_speed"] = df["movement_speed"].astype("category")
    df["is_real_img"] = df["is_real_img"].astype("bool")

    df["bb_center_x"] = df["bb_center_x"].astype("float32")
    df["bb_center_y"] = df["bb_center_y"].astype("float32")
    df["bb_width"] = df["bb_width"].astype("float32")
    df["bb_height"] = df["bb_height"].astype("float32")
    df["bb_score"] = df["bb_score"].astype("float32")
    df["bb_class"] = df["bb_class"].astype("category")

    df["env_position_x"] = df["env_position_x"].astype("float32")
    df["env_position_y"] = df["env_position_y"].astype("float32")
    df["env_position_z"] = df["env_position_z"].astype("float32")

    df["body_orientation_phi"] = df["body_orientation_phi"].astype("float32")
    df["body_orientation_theta"] = df["body_orientation_theta"].astype("float32")
    df["body_orientation_score"] = df["body_orientation_score"].astype("float32")
    df["body_orientation_visible"] = df["body_orientation_visible"].astype("category")

    df["head_orientation_phi"] = df["head_orientation_phi"].astype("float32")
    df["head_orientation_theta"] = df["head_orientation_theta"].astype("float32")
    df["head_orientation_score"] = df["head_orientation_score"].astype("float32")
    df["head_orientation_visible"] = df["head_orientation_visible"].astype("category")

    for joint in SKELETON_PEDREC_JOINTS:
        df[f"skeleton2d_{joint.name}_x"] = df[f"skeleton2d_{joint.name}_x"].astype("float32")
        df[f"skeleton2d_{joint.name}_y"] = df[f"skeleton2d_{joint.name}_y"].astype("float32")
        df[f"skeleton2d_{joint.name}_score"] = df[f"skeleton2d_{joint.name}_score"].astype("float32")
        df[f"skeleton2d_{joint.name}_visible"] = df[f"skeleton2d_{joint.name}_visible"].astype("category")
        df[f"skeleton2d_{joint.name}_supported"] = df[f"skeleton2d_{joint.name}_visible"].astype("category")

        df[f"skeleton3d_{joint.name}_x"] = df[f"skeleton3d_{joint.name}_x"].astype("float32")
        df[f"skeleton3d_{joint.name}_y"] = df[f"skeleton3d_{joint.name}_y"].astype("float32")
        df[f"skeleton3d_{joint.name}_z"] = df[f"skeleton3d_{joint.name}_z"].astype("float32")
        df[f"skeleton3d_{joint.name}_score"] = df[f"skeleton3d_{joint.name}_score"].astype("float32")
        df[f"skeleton3d_{joint.name}_visible"] = df[f"skeleton3d_{joint.name}_visible"].astype("category")
        df[f"skeleton3d_{joint.name}_supported"] = df[f"skeleton3d_{joint.name}_visible"].astype("category")


def get_column_names():
    column_names = [
        "dataset",
        "dataset_type",

        "scene_id",
        "scene_start",
        "scene_end",
        "frame_nr_global",
        "frame_nr_local",

        "img_dir",
        "img_id",
        "img_type",

        "subject_id",
        "gender",
        "skin_color",
        "size",
        "bmi",
        "age",
        "movement",
        "movement_speed",
        "is_real_img",
        "actions",

        "bb_center_x",
        "bb_center_y",
        "bb_width",
        "bb_height",
        "bb_score",
        "bb_class",

        "env_position_x",
        "env_position_y",
        "env_position_z",

        "body_orientation_theta",
        "body_orientation_phi",
        "body_orientation_score",
        "body_orientation_visible",

        "head_orientation_theta",
        "head_orientation_phi",
        "head_orientation_score",
        "head_orientation_visible",
    ]

    for joint in SKELETON_PEDREC_JOINTS:
        column_names.append(f"skeleton2d_{joint.name}_x")
        column_names.append(f"skeleton2d_{joint.name}_y")
        column_names.append(f"skeleton2d_{joint.name}_score")
        column_names.append(f"skeleton2d_{joint.name}_visible")
        column_names.append(f"skeleton2d_{joint.name}_supported")
    for joint in SKELETON_PEDREC_JOINTS:
        column_names.append(f"skeleton3d_{joint.name}_x")
        column_names.append(f"skeleton3d_{joint.name}_y")
        column_names.append(f"skeleton3d_{joint.name}_z")
        column_names.append(f"skeleton3d_{joint.name}_score")
        column_names.append(f"skeleton3d_{joint.name}_visible")
        column_names.append(f"skeleton3d_{joint.name}_supported")

    return column_names
