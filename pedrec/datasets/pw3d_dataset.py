# """3DPW dataset."""
# import copy
# import json
# import os
#
# import numpy as np
# import scipy.misc
# import torch.utils.data as data
# from pycocotools.coco import COCO
#
#
# class PW3D(data.Dataset):
#     """ 3DPW dataset.
#
#     Parameters
#     ----------
#     ann_file: str,
#         Path to the annotation json file.
#     root: str, default './data/pw3d'
#         Path to the PW3D dataset.
#     train: bool, default is True
#         If true, will set as training mode.
#     skip_empty: bool, default is False
#         Whether skip entire image if no valid label is found.
#     """
#     CLASSES = ['person']
#
#     EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]
#
#     num_joints = 17
#     bbox_3d_shape = (2, 2, 2)
#     joints_name_17 = (
#         'Pelvis',                               # 0
#         'L_Hip', 'L_Knee', 'L_Ankle',           # 3
#         'R_Hip', 'R_Knee', 'R_Ankle',           # 6
#         'Torso', 'Neck',                        # 8
#         'Nose', 'Head',                         # 10
#         'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
#         'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
#     )
#     joints_name_24 = (
#         'pelvis', 'left_hip', 'right_hip',      # 2
#         'spine1', 'left_knee', 'right_knee',    # 5
#         'spine2', 'left_ankle', 'right_ankle',  # 8
#         'spine3', 'left_foot', 'right_foot',    # 11
#         'neck', 'left_collar', 'right_collar',  # 14
#         'jaw',                                  # 15
#         'left_shoulder', 'right_shoulder',      # 17
#         'left_elbow', 'right_elbow',            # 19
#         'left_wrist', 'right_wrist',            # 21
#         'left_thumb', 'right_thumb'             # 23
#     )
#     joints_name_14 = (
#         'R_Ankle', 'R_Knee', 'R_Hip',           # 2
#         'L_Hip', 'L_Knee', 'L_Ankle',           # 5
#         'R_Wrist', 'R_Elbow', 'R_Shoulder',     # 8
#         'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 11
#         'Neck', 'Head'
#     )
#     skeleton = (
#         (1, 0), (2, 1), (3, 2),  # 2
#         (4, 0), (5, 4), (6, 5),  # 5
#         (7, 0), (8, 7),  # 7
#         (9, 8), (10, 9),  # 9
#         (11, 7), (12, 11), (13, 12),  # 12
#         (14, 7), (15, 14), (16, 15),  # 15
#     )
#
#     def __init__(self,
#                  cfg,
#                  ann_file,
#                  root='./data/pw3d',
#                  train=True,
#                  skip_empty=True,
#                  dpg=False,
#                  lazy_import=False):
#         self._cfg = cfg
#
#         self._ann_file = os.path.join(root, 'json', ann_file)
#         self._lazy_import = lazy_import
#         self._root = root
#         self._skip_empty = skip_empty
#         self._train = train
#         self._dpg = dpg
#
#         self._scale_factor = cfg.DATASET.SCALE_FACTOR
#         self._color_factor = cfg.DATASET.COLOR_FACTOR
#         self._rot = cfg.DATASET.ROT_FACTOR
#         self._input_size = cfg.MODEL.IMAGE_SIZE
#         self._output_size = cfg.MODEL.HEATMAP_SIZE
#
#         self._occlusion = cfg.DATASET.OCCLUSION
#
#         self._crop = cfg.MODEL.EXTRA.CROP
#         self._sigma = cfg.MODEL.EXTRA.SIGMA
#         self._depth_dim = cfg.MODEL.EXTRA.DEPTH_DIM
#
#         self._check_centers = False
#
#         self.num_class = len(self.CLASSES)
#
#         self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
#         self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
#
#         self.augment = cfg.MODEL.EXTRA.AUGMENT
#
#         self._loss_type = cfg.LOSS['TYPE']
#
#         self.upper_body_ids = (7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
#         self.lower_body_ids = (0, 1, 2, 3, 4, 5, 6)
#
#         self.kinematic = cfg.MODEL.EXTRA.get('KINEMATIC', False)
#         self.classfier = cfg.MODEL.EXTRA.get('WITHCLASSFIER', False)
#
#         self.root_idx_17 = self.joints_name_17.index('Pelvis')
#         self.lshoulder_idx_17 = self.joints_name_17.index('L_Shoulder')
#         self.rshoulder_idx_17 = self.joints_name_17.index('R_Shoulder')
#         self.root_idx_smpl = self.joints_name_24.index('pelvis')
#         self.lshoulder_idx_24 = self.joints_name_24.index('left_shoulder')
#         self.rshoulder_idx_24 = self.joints_name_24.index('right_shoulder')
#
#         self._items, self._labels = self._lazy_load_json()
#
#         # self.transformation = SimpleTransform3DSMPL(
#         #     self, scale_factor=self._scale_factor,
#         #     color_factor=self._color_factor,
#         #     occlusion=self._occlusion,
#         #     input_size=self._input_size,
#         #     output_size=self._output_size,
#         #     depth_dim=self._depth_dim,
#         #     bbox_3d_shape=self.bbox_3d_shape,
#         #     rot=self._rot, sigma=self._sigma,
#         #     train=self._train, add_dpg=self._dpg,
#         #     loss_type=self._loss_type)
#
#     def __getitem__(self, idx):
#         # get image id
#         img_path = self._items[idx]
#         img_id = int(self._labels[idx]['img_id'])
#
#         # load ground truth, including bbox, keypoints, image size
#         label = copy.deepcopy(self._labels[idx])
#         img = scipy.misc.imread(img_path, mode='RGB')
#
#         # transform ground truth into training label and apply data augmentation
#         target = self.transformation(img, label)
#
#         img = target.pop('image')
#         bbox = target.pop('bbox')
#         return img, target, img_id, bbox
#
#     def __len__(self):
#         return len(self._items)
#
#     def _lazy_load_json(self):
#         """Load all image paths and labels from json annotation files into buffer."""
#
#         items = []
#         labels = []
#
#         db = COCO(self._ann_file)
#         cnt = 0
#
#         for aid in db.anns.keys():
#             ann = db.anns[aid]
#
#             img_id = ann['image_id']
#
#             img = db.loadImgs(img_id)[0]
#             width, height = img['width'], img['height']
#
#             sequence_name = img['sequence']
#             img_name = img['file_name']
#             abs_path = os.path.join(
#                 self._root, 'imageFiles', sequence_name, img_name)
#
#             beta = np.array(ann['smpl_param']['shape']).reshape(10)
#             theta = np.array(ann['smpl_param']['pose']).reshape(24, 3)
#
#             x, y, w, h = ann['bbox']
#             # xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(ann['bbox']), width, height)
#             # if xmin > xmax - 5 or ymin > ymax - 5:
#             #     continue
#
#             f = np.array(img['cam_param']['focal'], dtype=np.float32)
#             c = np.array(img['cam_param']['princpt'], dtype=np.float32)
#
#             joint_cam_17 = np.array(ann['h36m_joints'], dtype=np.float32).reshape(17, 3)
#             joint_vis_17 = np.ones((17, 3))
#             joint_img_17 = np.zeros((17, 3))
#
#             joint_relative_17 = joint_cam_17 - joint_cam_17[self.root_idx_17, :]
#
#             joint_cam = np.array(ann['smpl_joint_cam'])
#             if joint_cam.size == 24 * 3:
#                 joint_cam_29 = np.zeros((29, 3))
#                 joint_cam_29[:24, :] = joint_cam.reshape(24, 3)
#             else:
#                 joint_cam_29 = joint_cam.reshape(29, 3)
#
#             joint_img = np.array(ann['smpl_joint_img'], dtype=np.float32).reshape(24, 3)
#             if joint_img.size == 24 * 3:
#                 joint_img_29 = np.zeros((29, 3))
#                 joint_img_29[:24, :] = joint_img.reshape(24, 3)
#             else:
#                 joint_img_29 = joint_img.reshape(29, 3)
#
#             joint_img_29[:, 2] = joint_img_29[:, 2] - joint_img_29[self.root_idx_smpl, 2]
#
#             joint_vis_24 = np.ones((24, 3))
#             joint_vis_29 = np.zeros((29, 3))
#             joint_vis_29[:24, :] = joint_vis_24
#
#             root_cam = joint_cam_29[self.root_idx_smpl]
#
#             items.append(abs_path)
#             labels.append({
#                 # 'bbox': (xmin, ymin, xmax, ymax),
#                 'img_id': cnt,
#                 'img_path': abs_path,
#                 'img_name': img_name,
#                 'width': width,
#                 'height': height,
#                 'joint_img_17': joint_img_17,
#                 'joint_vis_17': joint_vis_17,
#                 'joint_cam_17': joint_cam_17,
#                 'joint_relative_17': joint_relative_17,
#                 'joint_img_29': joint_img_29,
#                 'joint_vis_29': joint_vis_29,
#                 'joint_cam_29': joint_cam_29,
#                 'beta': beta,
#                 'theta': theta,
#                 'root_cam': root_cam,
#                 'f': f,
#                 'c': c
#             })
#             cnt += 1
#
#         return items, labels
