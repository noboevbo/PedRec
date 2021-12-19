from dataclasses import dataclass


@dataclass()
class ExperimentPaths(object):
    pose_resnet_weights_path: str
    pose_2d_coco_only_weights_path: str
    pedrec_2d_path: str
    pedrec_2d_h36m_path: str
    pedrec_2d_sim_path: str
    pedrec_2d_c_path: str
    pedrec_2d_h36m_sim_path: str
    pedrec_2d3d_h36m_path: str
    pedrec_2d3d_sim_path: str
    pedrec_2d3d_h36m_sim_path: str
    pedrec_2d3d_c_h36m_path: str
    pedrec_2d3d_c_sim_path: str
    pedrec_2d3d_c_h36m_sim_path: str
    pedrec_2d3d_c_o_h36m_mebow_path: str
    pedrec_2d3d_c_o_sim_path: str
    pedrec_2d3d_c_o_h36m_sim_path: str
    pedrec_2d3d_c_o_h36m_sim_mebow_path: str
    pedrec_full_path: str
    output_dir: str
    coco_dir: str
    tud_dir: str
    sim_train_dir: str
    sim_val_dir: str
    h36m_train_dir: str
    h36m_val_dir: str
    sim_c01_dir: str
    sim_c01_filename: str
    sim_c01_results_filename: str
    sim_c01_val_dir: str
    sim_c01_val_filename: str
    sim_c01_val_results_filename: str
    pretrained_model_path: str = None
    sim_train_filename: str = "rt_rom_01b.pkl"
    sim_val_filename: str = "rt_validate_3d.pkl"
    h36m_val_filename: str = "h36m_val_pedrec.pkl"
    h36m_train_filename: str = "h36m_train_pedrec.pkl"
