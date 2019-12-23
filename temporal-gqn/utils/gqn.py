'''
miscellaneous functions: gqn 
'''
import torch

''' for gqn '''
def recursive_to_device(x, device):
    if type(x) is tuple:
        return tuple([recursive_to_device(item, device) for item in x])
    elif type(x) is list:
        return [recursive_to_device(item, device) for item in x]
    elif type(x) is torch.Tensor:
        return x.to(device)
    else:
        return x

def recursive_clone_structure(x):
    if type(x) is tuple:
        return tuple([recursive_clone_structure(item) for item in x])
    elif type(x) is list:
        return [recursive_clone_structure(item) for item in x]
    else:
        return x

def recursive_detach(x):
    if type(x) is tuple:
        return tuple([recursive_detach(item) for item in x])
    elif type(x) is list:
        return [recursive_detach(item) for item in x]
    else:
        return x.detach()