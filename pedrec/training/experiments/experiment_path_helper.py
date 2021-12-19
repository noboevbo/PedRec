from pedrec.models.experiments.experiment_paths import ExperimentPaths


def get_experiment_paths_home() -> ExperimentPaths:
    return ExperimentPaths(
        pose_resnet_weights_path="data/models/human_pose_baseline/pose_resnet_50_256x192.pth.tar",
        pose_2d_coco_only_weights_path="data/models/pedrec/single_results/experiment_pedrec_p2d_coco_only_0.pth",
        output_dir="data/models/pedrec/single_results",
        coco_dir="data/datasets/COCO",
        tud_dir="data/datasets/cvpr10_multiview_pedestrians/",
        sim_train_dir="data/datasets/ROMb",
        sim_val_dir="data/datasets/RT3DValidate",
        h36m_train_dir="data/datasets/Human3.6m/train",
        h36m_val_dir="data/datasets/Human3.6m/val",
        sim_c01_dir="data/datasets/Conti01/",
        sim_c01_filename="rt_conti_01_train_FIN.pkl",
        sim_c01_results_filename="C01F_train_pred_df_experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_allframes.pkl",
        sim_c01_val_dir="data/datasets/Conti01/",
        sim_c01_val_filename="rt_conti_01_val_FIN.pkl",
        sim_c01_val_results_filename="C01F_pred_df_experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_allframes.pkl",

        pedrec_2d_path="data/models/pedrec/single_results/experiment_pedrec_p2d_coco_only_0.pth",
        pedrec_2d_c_path="data/models/pedrec/single_results/experiment_pedrec_p2d_c_0.pth",
        pedrec_2d_h36m_path="data/models/pedrec/single_results/experiment_pedrec_p2d_h36m_0.pth",
        pedrec_2d_sim_path="data/models/pedrec/single_results/experiment_pedrec_p2d_sim_0.pth",
        pedrec_2d_h36m_sim_path="data/models/pedrec/single_results/experiment_pedrec_p2d_h36m_sim_0.pth",
        pedrec_2d3d_h36m_path="data/models/pedrec/single_results/experiment_pedrec_p2d3d_h36m_0.pth",
        pedrec_2d3d_sim_path="data/models/pedrec/single_results/experiment_pedrec_p2d3d_sim_0.pth",
        pedrec_2d3d_h36m_sim_path="data/models/pedrec/single_results/experiment_pedrec_p2d3d_h36m_sim_0.pth",
        pedrec_2d3d_c_h36m_path="data/models/pedrec/single_results/experiment_pedrec_p2d3d_c_h36m_0.pth",
        pedrec_2d3d_c_sim_path="data/models/pedrec/single_results/experiment_pedrec_p2d3d_c_sim_0.pth",
        pedrec_2d3d_c_h36m_sim_path="data/models/pedrec/single_results/experiment_pedrec_p2d3d_c_h36m_sim_0.pth",
        pedrec_2d3d_c_o_h36m_mebow_path="data/models/pedrec/single_results/experiment_pedrec_p2d3d_c_o_h36m_mebow_0.pth",
        pedrec_2d3d_c_o_sim_path="data/models/pedrec/single_results/experiment_pedrec_p2d3d_c_o_sim_0.pth",
        pedrec_2d3d_c_o_h36m_sim_path="data/models/pedrec/single_results/experiment_pedrec_p2d3d_c_o_h36m_sim_0.pth",
        pedrec_2d3d_c_o_h36m_sim_mebow_path="data/models/pedrec/single_results/experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0.pth",
        pedrec_full_path="data/models/pedrec/single_results/experiment_pedrec_p2d3d_c_o_h36m_sim_0.pth",
    )