# from dataclasses import dataclass
# from typing import Dict
#
#
# @dataclass()
# class SceneInfo:
#     id: int = None
#     name: str = None
#     start_idx: int = None
#     end_idx: int = None
#     subject_id: str = None
#     gender: int = None
#     skin_color: int = None
#     size: int = None
#     weight: int = None
#     age: int = None
#
#     def from_json(self, json_dict: Dict[str, any]):
#         for key in json_dict:
#             setattr(self, key, json_dict[key])
