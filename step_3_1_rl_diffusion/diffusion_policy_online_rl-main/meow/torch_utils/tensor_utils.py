import torch

def _add_batch_dim(t:torch.Tensor):
    if len(t.shape) == 1:
        return t.unsqueeze(0)
    return t