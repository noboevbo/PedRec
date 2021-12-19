import os
import time

import cv2
import numpy as np

from pedrec.models.data_structures import ImageSize
from pedrec.utils.file_helper import get_img_paths_from_folder
from pedrec.utils.input_providers.input_provider_base import InputProviderBase


class ImgProvider(InputProviderBase):
    def __init__(self,
                 img_path: str,
                 image_size: ImageSize = None):
        """
        Provides images from a given image directory.
        :param img_dir: The directory which contains the images
        :param fps: Forced fps for the image delivery. May be limited by disk io...
        """
        self.img_path = img_path
        assert os.path.isfile(self.img_path), f'Image {self.img_path} not found!'
        self.image_size = image_size
        self.stopped = False

    def get_data(self) -> np.ndarray:
        for i in range(0, 1):
            img = cv2.cvtColor(cv2.imread(self.img_path), cv2.COLOR_BGR2RGB)
            if self.image_size is not None:
                img = cv2.resize(img, (self.image_size.width, self.image_size.height))
            yield img
