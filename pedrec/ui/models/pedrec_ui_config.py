import os
import pickle
from dataclasses import dataclass


@dataclass
class PedRecUIConfig():
    show_pose_2d: bool = True
    show_object_bb: bool = True
    show_human_bb: bool = True
    show_head_orientation_2d: bool = True
    show_body_orientation_2d: bool = True
    show_sees_car_flag: bool = True
    show_actions: bool = True

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.__dict__ = pickle.load(f)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
