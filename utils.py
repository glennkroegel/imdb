import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import six
import json
import os
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def one_hot(labels, num_classes):
    y = torch.eye(num_classes)
    y = y[labels]
    return y

# def accuracy(input, targs):
#     "Compute accuracy with `targs` when `input` is bs * n_classes."
#     n = targs.shape[0]
#     inp = input.argmax(dim=-1).view(-1)
#     targs = targs.view(n,-1)
#     acc = (inp==targs).float().mean()
#     return acc

def accuracy(inputs, targets):
    with torch.no_grad():
        inps = inputs.detach().view(-1)
        targs = targets.detach().view(-1)
        pred = torch.round(inps)
        acc = (pred == targs).float().mean()
    return acc

def bce_acc(input, targs):
    bs = targs.shape[0]
    inp = torch.round(input).view(bs, 1)
    targs = targs.view(bs, 1)
    ix0 = (targs == 0).nonzero().squeeze()
    ix1 = (targs == 1).nonzero().squeeze()
    acc0 = (inp[ix0] == targs[ix0]).float().mean().item()
    acc1 = (inp[ix1] == targs[ix1]).float().mean().item()
    acc = (inp==targs).float().mean().item()
    return acc, acc0, acc1