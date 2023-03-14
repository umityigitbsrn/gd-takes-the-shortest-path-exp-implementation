import torch
from typing import Dict, Callable
import numpy as np


def normalized_misfit(labels: torch.Tensor, data: torch.Tensor, init_model: torch.nn.Module, model: torch.nn.Module) \
        -> float:
    init_model.eval()
    model.eval()

    # label transformation
    # transformed_labels = labels.nonzero()[:, 1]
    transformed_labels = labels

    predictions = model(data)
    init_predictions = init_model(data)

    # transformed_predictions = torch.argmax(predictions, dim=1)
    # transformed_init_predictions = torch.argmax(init_predictions, dim=1)

    transformed_predictions = predictions
    transformed_init_predictions = init_predictions

    misfit = torch.linalg.norm((transformed_labels - transformed_predictions).type(torch.FloatTensor))
    init_misfit = torch.linalg.norm((transformed_labels - transformed_init_predictions).type(torch.FloatTensor))
    return (misfit / init_misfit).item()


def normalized_distance(init_model: torch.nn.Module, model: torch.nn.Module) -> Dict[str, float]:
    init_model.eval()
    model.eval()

    normalized_distance_dict = {}

    for ((init_name, init_param), (name, param)) in zip(init_model.named_parameters(), model.named_parameters()):
        if init_name.find('weight') != -1:
            distance = torch.linalg.norm((param - init_param).reshape((param.shape[0], -1)))
            init_distance = torch.linalg.norm(init_param.reshape((init_param.shape[0], -1)))
            normalized_distance_dict[init_name] = (distance / init_distance).item()

    return normalized_distance_dict


def normalized_misfit_function_based(labels: np.ndarray,
                                     kernel_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                                     data: np.ndarray,
                                     init_weight: np.ndarray,
                                     weight: np.ndarray) -> float:
    misfit = np.linalg.norm(labels - kernel_function(data, weight))
    init_misfit = np.linalg.norm(labels - kernel_function(data, init_weight))
    return misfit / init_misfit


def normalized_distance_weight_based(init_weight: np.ndarray,
                                     weight: np.ndarray) -> float:
    distance = np.linalg.norm(weight - init_weight)
    init_distance = np.linalg.norm(init_weight)
    return distance / init_distance
