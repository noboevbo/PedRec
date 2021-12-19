from dataclasses import dataclass
import numpy as np


@dataclass()
class EnvPositionValidationResults(object):
    distance_mm: float