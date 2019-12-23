import torch
import numpy as np

def stratified_resampling(w_torch):
    """
    Stratified resampling for batch
    Copied and modified from https://github.com/blei-lab/variational-smc/blob/master/variational_smc.py
    """
    w = w_torch.detach().cpu().numpy()
    B = w.shape[0]
    K = w.shape[1]

    ancestors = []
    ind = np.arange(K)
    for b in range(B):
        bins = np.cumsum(w[b])
        u = (ind + np.random.rand(K)) / K
        ancestors += [list(np.digitize(u, bins))]

    return ancestors