import torch


def normalized_misfit(labels: torch.Tensor, data: torch.Tensor, init_model: torch.nn.Module, model: torch.nn.Module) \
        -> torch.Tensor:
    init_model.eval()
    model.eval()
    misfit = torch.linalg.vector_norm((labels - model(data)), ord=2, dim=1)
    init_misfit = torch.linalg.vector_norm((labels - init_model(data)), ord=2, dim=1)
    return torch.div(misfit, init_misfit)


def normalized_distance(init_weight, weight):
    pass
