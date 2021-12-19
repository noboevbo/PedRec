import os
import time
from typing import Callable, List, Dict, Tuple

import torch.nn as nn
import torch.utils.data.distributed
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from pedrec.evaluations.validate import validate, ValidationResults
from pedrec.models.data_structures import ImageSize
from pedrec.models.experiments.epoch_validation_results import EpochValidationResults
from pedrec.models.experiments.experiment_description import ExperimentDescription
from pedrec.models.experiments.experiment_round_description import ExperimentRoundDescription
from pedrec.models.experiments.validation_set_model import ValidationSet
from pedrec.training.experiments.experiment_log_helper import log_epoch_validation_results
from pedrec.utils.log_helper import configure_logger
from pedrec.utils.torch_utils.torch_helper import move_to_device, set_fixed_seeds, unfreeze_layers, freeze_layers


def get_subsampled_dataset(dataset: Dataset, subsample: int) -> Dataset:
    if subsample == 1:
        return dataset
    dataset_len = len(dataset)
    subsampled_dataset = Subset(dataset, list(range(0, dataset_len, subsample)))
    print(f"train full size: {dataset_len}, subsampled: {len(subsampled_dataset)}")
    return subsampled_dataset


def get_preds_single(outputs: torch.Tensor):
    """
    Helper for single models, outputs outputs[0] as every possible gt
    """
    output = outputs.cpu().detach().numpy()
    return {
        "skeleton": output,
        "skeleton_3d": output,
        "orientation": output,
    }


def get_preds_mtl(outputs: torch.Tensor):
    orientation = None
    if len(outputs) > 2:
        orientation = outputs[3].cpu().detach().numpy()
    return {
        "skeleton": outputs[0].cpu().detach().numpy(),
        "skeleton_3d": outputs[1].cpu().detach().numpy(),
        "orientation": orientation
    }



def get_outputs_loss_single(net: nn.Module, model_input: torch.Tensor, labels: torch.Tensor, loss_func: Callable):
    outputs = net(model_input)
    loss = loss_func(outputs, labels)
    return outputs, loss


def get_outputs_loss_mtl(net: nn.Module, model_input: torch.Tensor, labels: torch.Tensor):
    return net(model_input, labels)


def train(net: nn.Module, optimizer, scheduler, train_loader: DataLoader, device: torch.device,
          get_outputs_loss_func: Callable) -> Tuple[float, float]:
    start = time.time()
    loss_total = 0.0
    net.train()
    # for param in net.parameters():
    #     print(param.requires_grad)
        # param.requires_grad = False
    # count = 0
    with tqdm(total=len(train_loader)) as pbar:
        for i, train_data in enumerate(train_loader):
            # if count > 2:
            #     break
            # count += 1
            inputs, labels = train_data

            inputs = inputs.to(device)
            labels = move_to_device(labels, device)

            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            outputs, loss = get_outputs_loss_func(net, inputs, labels)
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
            if scheduler is not None:
                scheduler.step()
            pbar.update()

    return loss_total / len(train_loader), time.time() - start


def train_round(net: nn.Module, experiment_description: ExperimentDescription,
                experiment_round_description: ExperimentRoundDescription, train_loader: DataLoader,
                validation_sets: List[ValidationSet], get_outputs_loss_func: Callable, get_gt_pred_func: Callable,
                device: torch.device, log: bool = True, scheduler_after_every_batch: bool = True, freeze_hard: bool = False) -> List[EpochValidationResults]:
    unfreeze_layers(net.children())
    for layer in experiment_round_description.frozen_layers:
        freeze_layers(layer.children(), hard=freeze_hard)
    epoch_results: List[EpochValidationResults] = []
    for epoch in range(experiment_round_description.num_epochs):  # loop over the dastaset multiple times
        scheduler = None
        if scheduler_after_every_batch:
            scheduler = experiment_round_description.scheduler
        # validation_results = validate_val_sets(net, validation_sets, get_outputs_loss_func, get_gt_pred_func,
        #                                        device, experiment_description.net_cfg.model.input_size)
        train_loss, train_time = train(net, experiment_round_description.optimizer,
                                       scheduler, train_loader,
                                       device, get_outputs_loss_func)
        validation_results = validate_val_sets(net, validation_sets, get_outputs_loss_func, get_gt_pred_func,
                                              device, experiment_description.net_cfg.model.input_size)
        results = EpochValidationResults(epoch=epoch, train_loss=train_loss, train_time=train_time,
                                         validation_results=validation_results)
        if not scheduler_after_every_batch:
            experiment_round_description.scheduler.step(epoch+1)
        epoch_results.append(results)
        if log:
            log_epoch_validation_results(results)
    return epoch_results



################## Validation Methods ####################
def validate_val_sets(net, validation_sets: List[ValidationSet], get_outputs_loss_func: Callable,
                     get_preds_func: Callable, device: torch.device, model_input_size: ImageSize) -> Dict[str, ValidationResults]:
    results: Dict[str, ValidationResults] = {}
    for val_set in validation_sets:
        skeleton_3d_range = 0
        if val_set.val_set_cfg is not None:
            skeleton_3d_range = val_set.val_set_cfg.skeleton_3d_range
        val_result = validate(net, val_set.loader, get_outputs_loss_func, get_preds_func, device, model_input_size,
                              validate_2D=val_set.validate_2D,
                              validate_3D=val_set.validate_3D,
                              validate_orientation=val_set.validate_orientation,
                              validate_pose_conf=val_set.validate_pose_conf,
                              validate_env_position=val_set.validate_env_position,
                              skeleton_3d_range=skeleton_3d_range)
        results[val_set.name] = val_result
    return results

def init_experiment(seed: int):
    set_fixed_seeds(seed)
    configure_logger()
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True