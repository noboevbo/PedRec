from collections import deque
from dataclasses import dataclass, field
from typing import Dict
from typing import List

import numpy as np

from pedrec.models.human import Human


@dataclass
class ImageContent(object):
    """
    Contains the content in a image observed by algorithm(s)
    """
    humans: List[Human] = field(default_factory=list)
    objects: List[np.ndarray] = field(default_factory=list)


class HumansBufferEntry(object):
    __slots__ = ['human_id', 'last_added', 'human_content_buffer']

    def __init__(self, human_id: int, buffer_count: int, human: Human, buffer_size: int):
        self.human_id: int = human_id
        self.last_added: int = buffer_count
        self.human_content_buffer: List[Human] = deque(maxlen=buffer_size)
        self.human_content_buffer.append(human)

    def update(self, human: Human, buffer_count: int):
        if human is not None:
            self.last_added = buffer_count
        self.human_content_buffer.append(human)


class ImageContentBuffer(object):
    buffer_count: int = 0
    __image_contents: List[ImageContent] = None
    __humans_buffers: Dict[str, HumansBufferEntry]

    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.buffer_count = 0
        self.__image_contents = deque(maxlen=buffer_size)
        self.__humans_buffers = {}

    def add(self, image_content: ImageContent):
        self.buffer_count += 1
        self.__image_contents.append(image_content)
        updated_keys = []
        if image_content.humans is not None:
            for human in image_content.humans:
                if human.uid in self.__humans_buffers.keys():
                    self.__humans_buffers[human.uid].update(human, self.buffer_count)
                else:
                    self.__humans_buffers[human.uid] = HumansBufferEntry(human_id=human.uid, human=human,
                                                                         buffer_count=self.buffer_count,
                                                                         buffer_size=self.buffer_size)
                updated_keys.append(human.uid)

        for key, buffer in self.__humans_buffers.items():
            if key in updated_keys:
                continue
            buffer.update(None, self.buffer_count)
        self.__clean_humans_buffers()

    def __clean_humans_buffers(self):
        keys_to_delete = []
        for key, buffer in self.__humans_buffers.items():
            if buffer.last_added < self.buffer_count - self.buffer_size:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del self.__humans_buffers[key]

    def get_human_data_buffer_by_id(self, human_id: str) -> List[Human]:
        if human_id in self.__humans_buffers:
            return self.__humans_buffers[human_id].human_content_buffer
        return []

    def get_humans_buffers(self) -> Dict[str, List[Human]]:
        result: Dict[str, List[Human]] = {}
        for key, humans_buffer in self.__humans_buffers.items():
            result[key] = humans_buffer.human_content_buffer
        return result

    def get_last(self):
        if len(self.__image_contents) == 0:
            return None
        return self.__image_contents[-1]

    def get_last_humans(self):
        if len(self.__image_contents) == 0:
            return None
        return self.__image_contents[-1].humans
