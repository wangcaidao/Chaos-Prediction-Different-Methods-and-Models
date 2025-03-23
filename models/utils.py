import torch.nn.functional as F
import torch

def _get_act(act):
    if act == 'tanh':
        func = torch.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = torch.relu
    elif act == 'elu':
        func = F.elu
    elif act == 'leaky_relu':
        func = F.leaky_relu
    else:
        raise ValueError(f'{act} is not supported')
    return func

