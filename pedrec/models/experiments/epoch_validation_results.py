from dataclasses import dataclass
from typing import Dict

from pedrec.evaluations.validate import ValidationResults


@dataclass()
class EpochValidationResults(object):
    epoch: int
    train_loss: float
    train_time: float
    validation_results: Dict[str, ValidationResults]
