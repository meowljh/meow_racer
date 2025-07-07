import torch


def extract(input, t:torch.Tensor, x:torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape 
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1) #(B, 1 for _ in range(len(shape)-1))

    return out.reshape(*reshape)