from dataclasses import dataclass

from pedrec.models.data_structures import ImageSize


@dataclass
class YoloV4Model:
    input_size: ImageSize
    num_classes: int


@dataclass
class YoloV4Config:
    model = YoloV4Model(
        input_size=ImageSize(width=608, height=320),
        num_classes=80
    )
