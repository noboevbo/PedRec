from dataclasses import dataclass


@dataclass
class PedRecDatasetInfo(object):
    full_length: int
    used_length: int
    subsampling: int
    provides_bbs: bool = False
    provides_skeleton_2ds: bool = False
    provides_skeleton_3ds: bool = False
    provides_env_positions: bool = False
    provides_body_orientations: bool = False
    provides_head_orientations: bool = False
    provides_scene_ids: bool = False
    provides_actions: bool = False
    provides_movements: bool = False
    provides_movement_speeds: bool = False
    provides_genders: bool = False
    provides_skin_colors: bool = False
    provides_sizes: bool = False
    provides_weights: bool = False
    provides_ages: bool = False
    provides_frame_nr_locals: bool = False
    provides_frame_nr_globals: bool = False
    provides_is_real_img: bool = False
