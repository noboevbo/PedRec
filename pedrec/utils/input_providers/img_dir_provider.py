import os
import time

import cv2
import numpy as np

from pedrec.models.data_structures import ImageSize
from pedrec.utils.file_helper import get_img_paths_from_folder
from pedrec.utils.input_providers.input_provider_base import InputProviderBase


class ImgDirProvider(InputProviderBase):
    def __init__(self,
                 img_dir: str,
                 fps: int = None,
                 loop: bool = False,
                 image_size: ImageSize = None,
                 print_current_path: bool = False):
        """
        Provides images from a given image directory.
        :param img_dir: The directory which contains the images
        :param fps: Forced fps for the image delivery. May be limited by disk io...
        """
        self.img_dir = img_dir
        assert os.path.exists(img_dir), 'Image directory not found!'
        self.img_paths = sorted(get_img_paths_from_folder(img_dir))
        self.fps = fps
        self.loop = loop
        self.image_size = image_size
        self.print_current_path = print_current_path
        self.stopped = False

    def get_data(self) -> np.ndarray:
        assert len(self.img_paths) > 0, 'No images found'
        while True:
            if self.stopped:
                return None
            for img_path in self.img_paths:
                if self.fps is not None:
                    start = time.time()
                if self.print_current_path:
                    print(img_path)
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                if self.image_size is not None:
                    img = cv2.resize(img, (self.image_size.width, self.image_size.height))
                yield img

                if self.fps is not None:
                    time.sleep(max(1. / self.fps - (time.time() - start), 0))
            if not self.loop:
                break

    def stop(self):
        self.stopped = True
