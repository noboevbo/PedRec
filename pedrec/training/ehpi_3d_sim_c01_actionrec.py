import os
import sys

from pedrec.models.data_structures import ImageSize

sys.path.append(".")

from pedrec.configs.dataset_configs import PedRecTemporalDatasetConfig, VideoActionDatasetConfig
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths_home
from pedrec.models.constants.sample_method import SAMPLE_METHOD
from pedrec.training.ehpi_3d_helper import ehpi3d_experiment


def gt():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=1,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False
    )

    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=None)


def pred():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
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
        add_2d=False
    )

    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_pred", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=None)

def gt_pred():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=0.65,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False
    )

    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_pred", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=None)

def gt_pred_ehpi2dvids():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=0.65,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False
    )

    vid_cfg = VideoActionDatasetConfig(
        flip=True,
        skeleton_3d_range=3000,
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False
    )
    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_pred_ehpi2dvids", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=vid_cfg)


def gt_pred_no_unit_skeleton():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=0.65,
        use_unit_skeleton=False,
        min_joint_score=0,
        add_2d=False
    )

    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_pred_no_unit_skeleton", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=None)


def gt_pred_zero_by_score():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=0.65,
        use_unit_skeleton=True,
        min_joint_score=0.4,
        add_2d=False
    )

    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_pred_zero_by_score", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=None)

# 15 fps
def gt_15fps():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=1,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False,
        frame_sampling=2
    )

    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_15fps", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=None)


def gt_pred_15fps():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=0.65,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False,
        frame_sampling=2
    )

    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_pred_15fps", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=None)

def gt_pred_ehpi2dvids_15fps():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=0.65,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False,
        frame_sampling=2
    )

    vid_cfg = VideoActionDatasetConfig(
        flip=True,
        skeleton_3d_range=3000,
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False,
        frame_sampling=2
    )
    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_pred_ehpi2dvids_15fps", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=vid_cfg)


def gt_pred_no_unit_skeleton_15fps():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=0.65,
        use_unit_skeleton=False,
        min_joint_score=0,
        add_2d=False,
        frame_sampling=2
    )

    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_pred_no_unit_skeleton_15fps", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=None)


def gt_pred_zero_by_score_15fps():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=0.65,
        use_unit_skeleton=True,
        min_joint_score=0.4,
        add_2d=False,
        frame_sampling=2
    )

    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_pred_zero_by_score_15fps", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=None)


# 64 frames
def gt_64frames():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=1,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False,
        temporal_field=ImageSize(64, 32)
    )

    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_64frames", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=None)


def gt_pred_64frames():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=0.65,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False,
        temporal_field=ImageSize(64, 32)
    )

    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_pred_64frames", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=None)

def gt_pred_ehpi2dvids_64frames():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=0.65,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False,
        temporal_field=ImageSize(64, 32)
    )

    vid_cfg = VideoActionDatasetConfig(
        flip=True,
        skeleton_3d_range=3000,
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        use_unit_skeleton=True,
        min_joint_score=0,
        add_2d=False,
        temporal_field=ImageSize(64, 32)
    )
    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_pred_ehpi2dvids_64frames", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=vid_cfg)


def gt_pred_no_unit_skeleton_64frames():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=0.65,
        use_unit_skeleton=False,
        min_joint_score=0,
        add_2d=False,
        temporal_field = ImageSize(64, 32)
    )

    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_pred_no_unit_skeleton_64frames", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=None)


def gt_pred_zero_by_score_64frames():
    experiment_paths = get_experiment_paths_home()
    pedrec_cfg = PedRecTemporalDatasetConfig(
        flip=True,
        scale_factor=0.25,
        rotation_factor=0,
        skeleton_3d_range=3000,
        img_pattern="view_{cam_name}-frame_{id}.{type}",
        subsample=1,
        subsampling_strategy=SAMPLE_METHOD.SYSTEMATIC,
        gt_result_ratio=0.65,
        use_unit_skeleton=True,
        min_joint_score=0.4,
        add_2d=False,
        temporal_field=ImageSize(64, 32)
    )

    ehpi3d_experiment("ehpi_3d_sim_c01_actionrec_gt_pred_zero_by_score_64frames", experiment_paths, pedrec_cfg=pedrec_cfg, vid_cfg=None)


if __name__ == '__main__':
    os.environ['NUMEXPR_MAX_THREADS'] = '16'
    # gt()
    # gt_pred()
    # gt_pred_ehpi2dvids()
    # gt_pred_no_unit_skeleton()
    # gt_pred_zero_by_score()
    # gt_pred_15fps()
    # gt_pred_ehpi2dvids_15fps()
    # gt_pred_ehpi2dvids_64frames()
    pred()
