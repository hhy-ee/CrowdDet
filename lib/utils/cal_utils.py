import torch
import numpy as np
from pynverse import inversefunc

@torch.no_grad()
def pf_inv_mapping(nf_para, bin, rep):
    bin = bin.clone().cpu().detach().numpy()
    flow = nf_para.clone().cpu().detach().numpy()
    nf_u, nf_w, nf_b = flow[..., 0], flow[..., 1], flow[..., 2]
    for n in range(nf_u.shape[0]):
        for l in range(rep):
            unl, wnl, bnl = nf_u[n, l], nf_w[n, l], nf_b[n, l]
            nflow_func = lambda x: x + np.tanh(wnl * x + bnl) * unl
            bin[n, :] = inversefunc(nflow_func, y_values=bin[n, :])
    return bin
        


