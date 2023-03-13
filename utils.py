import torch
from typing import Dict


def normalized_misfit(labels: torch.Tensor, data: torch.Tensor, init_model: torch.nn.Module, model: torch.nn.Module) \
        -> float:
    init_model.eval()
    model.eval()

    # label transformation
    transformed_labels = labels.nonzero()[:, 1]

    predictions = model(data)
    init_predictions = init_model(data)

    transformed_predictions = torch.argmax(predictions, dim=1)
    transformed_init_predictions = torch.argmax(init_predictions, dim=1)

    misfit = torch.linalg.vector_norm((transformed_labels - transformed_predictions).type(torch.FloatTensor), ord=2)
    init_misfit = torch.linalg.vector_norm((transformed_labels - transformed_init_predictions).type(torch.FloatTensor)
                                           , ord=2)
    return (misfit / init_misfit).item()


def normalized_distance(init_model: torch.nn.Module, model: torch.nn.Module) -> Dict[str, float]:
    init_model.eval()
    model.eval()

    normalized_distance_dict = {}

    for ((init_name, init_param), (name, param)) in zip(init_model.named_parameters(), model.named_parameters()):
        if init_name.find('weight') != -1:
            distance = torch.linalg.norm((param - init_param).reshape((param.shape[0], -1)), ord='fro')
            init_distance = torch.linalg.norm(init_param.reshape((init_param.shape[0], -1)), ord='fro')
            normalized_distance_dict[init_name] = (distance / init_distance).item()

    return normalized_distance_dict
