from dataclasses import dataclass


@dataclass()
class DatasetDescription(object):
    name: str
    subsampling: int
    full_length: int
    used_length: int
