import torch
import numpy as np

@torch.no_grad()
def pf_inv_mapping(nf_para, bin, rep, acc):
    nf_u, nf_w, nf_b = nf_para[..., 0], nf_para[..., 1], nf_para[..., 2]
    for n in range(nf_u.shape[0]):
        for l in range(rep):
            unl, wnl, bnl = nf_u[n, l], nf_w[n, l], nf_b[n, l]
            nflow_func = lambda x: x + torch.tanh(wnl * x + bnl) * unl
            bin[n,:] = inversefunc(nflow_func, bin[n,:], acc)
    return bin

@torch.no_grad()
def inversefunc(func, y_values, accuracy):
    results = torch.zeros_like(y_values)
    fix_point = y_values[0]
    for j in range(y_values.shape[0]):
        while 1:
            fix_point = y_values[j] - (func(fix_point) - fix_point)
            if  torch.abs(func(fix_point) - y_values[j]) < accuracy:
                break
        results[j] = fix_point
    return results
        


