import logging
import sys
sys.path.append('.')
import time
from typing import List

import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication

from pedrec.configs.app_config import AppConfig, action_list_c01_w_real
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.configs.yolo_v4_config import YoloV4Config
from pedrec.models.constants.action_mappings import ACTION
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.networks.net_pedrec.ehpi_3d_net import Ehpi3DNet
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.networks.net_yolo_v4.yolo_v4_helper import do_detect
from pedrec.tracking.human_merger import HumanMerger
from pedrec.tracking.human_tracker import HumanTracker, bb_tracking, add_undetected_bbs_from_tracking, remove_duplicates
from pedrec.ui.pedrec_app import PedRecApp
from pedrec.ui.pedrec_worker import PedRecWorker
from pedrec.utils.bb_helper import split_human_bbs, get_bbs_above_score
from pedrec.utils.demo_helper import get_detector, init_pose_model
from pedrec.utils.ehpi_helper import get_ehpi_from_human_history, ehpi_transform
from pedrec.utils.human_helper import get_humans_from_pedrec_detections
from pedrec.utils.image_content_buffer import ImageContent, ImageContentBuffer
from pedrec.utils.input_providers.input_provider_base import InputProviderBase
from pedrec.utils.input_providers.video_provider import VideoProvider
from pedrec.utils.log_helper import configure_logger
from pedrec.utils.pose_deconv_helper import (pedrec_recognizer, do_redetect_pose_recognition)
from pedrec.utils.time_helper import timed
from pedrec.utils.torch_utils.torch_helper import get_device


def get_action_classes(action_probabilities: np.ndarray, action_list: List[ACTION], thresh: float = 0.7):
    actions = []
    for probability, action in zip(action_probabilities, action_list):
        if probability > thresh:
            actions.append(action)
    return actions

class PedRecNetDemoWorker(PedRecWorker):
    def __init__(self, parent: QApplication, input_provider: InputProviderBase, app_cfg: AppConfig,
                 model_file_path: str):
        super().__init__(parent)
        configure_logger()
        self.logger = logging.getLogger(__name__)
        self.app_cfg = app_cfg
        self.device = get_device(self.app_cfg.cuda.use_gpu)

        yolo_weights = "data/models/yolo_v4/yolov4.pth"
        pedrecnet_weights = model_file_path

        # Detector
        self.yolo_cfg = YoloV4Config()
        self.detector = get_detector(self.yolo_cfg, yolo_weights, self.logger, self.device)

        # Pose
        self.pose_cfg = PedRecNet50Config()
        self.pose_recognizer = init_pose_model(PedRecNet(self.pose_cfg), pedrecnet_weights, self.logger, self.device)
        self.movement_recognizer = Ehpi3DNet(len(app_cfg.inference.action_list)).to(self.device)
        self.movement_recognizer.load_state_dict(
            torch.load("data/models/ehpi3d/ehpi_3d_sim_c01_actionrec_gt_pred_64frames.pth"))
        self.movement_recognizer.eval()
        # Tracking
        self.human_tracker = HumanTracker(img_size=self.app_cfg.inference.img_size)
        self.human_merger = HumanMerger(self.app_cfg.inference.img_size)
        self.image_content_buffer: ImageContentBuffer = ImageContentBuffer(
            buffer_size=self.app_cfg.inference.buffer_size)

        self.input_provider = input_provider
        self.temporal_field = ImageSize(width=64, height=32)

    def run_impl(self, frame_nr: int, img: np.ndarray):
        # if frame_nr <= 6000:
        #     return
        start = time.time()
        last_humans = self.image_content_buffer.get_last_humans()
        tracked_humans = self.human_tracker.get_humans_by_tracking(img, previous_humans=last_humans)

        human_bbs, other_bbs = self.get_bbs(img, tracked_humans)

        pose_time, pose_preds = timed(
            lambda: pedrec_recognizer(self.pose_recognizer, self.pose_cfg, img, human_bbs, self.device))
        humans = get_humans_from_pedrec_detections(human_bbs, pose_preds)
        humans = self.get_humans_with_tracking(img, humans, tracked_humans)
        humans = [human for human in humans if human.score > 0.65]  # remove low score humans and unnatural high

        ehpis = []
        for human in humans:
            history = self.image_content_buffer.get_human_data_buffer_by_id(human.uid)
            self.smooth_data(human, history)
            human.ehpi = get_ehpi_from_human_history(human, history, temporal_field=self.temporal_field)
            ehpis.append(ehpi_transform(human.ehpi))

        if len(ehpis) > 0:
            ehpis = torch.stack(ehpis).to(self.device)
            action_probabilities = torch.sigmoid(self.movement_recognizer(ehpis)).detach().cpu().numpy()
            # actions = np.argwhere(action_probabilities > 0.1)
            # action_output = torch.argmax(F.log_softmax(self.movement_recognizer(ehpis)), dim=1).detach().cpu().numpy()
            for human, action_probs in zip(humans, action_probabilities):
                human.action_probabilities = action_probs
                human.actions = get_action_classes(action_probs, self.app_cfg.inference.action_list, thresh=0.7)

        image_content = ImageContent(humans=humans, objects=other_bbs)
        self.image_content_buffer.add(image_content)

        required_time = time.time() - start

        fps = int(1.0 / required_time)
        self.data_updated.emit(frame_nr, img, humans, np.array(other_bbs), fps)

    def get_bbs(self, img: np.ndarray, tracked_humans: List[Human]):
        sized = cv2.resize(img, (self.yolo_cfg.model.input_size.width, self.yolo_cfg.model.input_size.height))
        detection_time, bbs = timed(
            lambda: do_detect(self.detector, sized, self.app_cfg.inference.img_size, 0.4, 0.6, self.device, self.logger,
                              tracked_humans))
        human_bbs, other_bbs = split_human_bbs(bbs[0])
        human_bbs = bb_tracking(human_bbs, tracked_humans)
        human_bbs = add_undetected_bbs_from_tracking(human_bbs, tracked_humans)
        human_bbs = get_bbs_above_score(human_bbs, 0.6)
        human_bbs = remove_duplicates(human_bbs)
        return human_bbs, other_bbs

    def get_humans_with_tracking(self, img: np.ndarray, humans: List[Human], tracked_humans: List[Human]):
        humans, undetected_humans = self.human_merger.merge_humans(humans, tracked_humans, assign_new_ids=True)
        redetected_humans = do_redetect_pose_recognition(self.pose_recognizer, self.pose_cfg, img, undetected_humans,
                                                         self.device)
        humans.extend(redetected_humans)
        return humans

    def smooth_data(self, human: Human, history: List[Human], num_smoothing: int = 2):
        if len(history) >= num_smoothing:
            for i in range(1, num_smoothing):
                human.skeleton_3d += history[-i].skeleton_3d
                human.orientation += history[-i].orientation
            human.orientation /= num_smoothing
            human.skeleton_3d /= num_smoothing

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app_config = AppConfig()
    app_config.inference.action_list = action_list_c01_w_real
    model_file_path = "data/models/pedrec/experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_net.pth"

    # Demo Videos (free to use license)
    vid_file = "data/demo/multi_person_crossing_street.mp4"

    # Human3.6m
    # app_config.inference.img_size = ImageSize(1000, 1002)
    # vid_file = "data/datasets/Human3.6m/train/S1/Videos/Directions 1.58860488.mp4"

    # own
    # vid_file = "demo/05070850_9672.m4v"

    input_provider = VideoProvider(vid_file, app_config.inference.img_size, mirror=False)
    # input_provider = ImgDirProvider("path_to_dir_with_image_sequence",
    #                                 image_size=ImageSize(1920, 1080), fps=60)

    # Images
    # app_config.inference.img_size = ImageSize(249, 269)
    # img_file = "some single image"
    # input_provider = ImgProvider(img_file, app_config.inference.img_size)
    worker = PedRecNetDemoWorker(app, input_provider, app_config, model_file_path)
    ex = PedRecApp(app, worker, app_config)
    sys.exit(app.exec_())
