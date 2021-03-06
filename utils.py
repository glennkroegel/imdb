import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import six
import json
import os
from torchtext import datasets, data
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.data.utils import get_tokenizer
from transformers import DistilBertTokenizer
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def one_hot(labels, num_classes):
    y = torch.eye(num_classes)
    y = y[labels]
    return y

def length_collate(batch):
    pass

def make_imdb(batch_size=32, device=device, vectors=None):
  TEXT = data.Field(include_lengths=False, lower=True, batch_first=False)
  LABEL = data.LabelField()
  train, test = datasets.IMDB.splits(TEXT, LABEL)

  TEXT.build_vocab(train, test, vectors=vectors, max_size=20000) 
  LABEL.build_vocab(train, test)
  train_iter, test_iter = data.BucketIterator.splits(
              (train, test), batch_size=batch_size, device=device, repeat=False)

  return train_iter, test_iter, TEXT, LABEL

# def make_small_imdb(batch_size=32, device=-1, vectors=None):
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   # TEXT = data.Field(include_lengths=False, lower=True, batch_first=True)
#   TEXT = data.Field(tokenize=get_tokenizer("basic_english"),
#                     init_token='<sos>',
#                     eos_token='<eos>',
#                     lower=True, 
#                     batch_first=True)
#   LABEL = data.LabelField()

#   datafields = [('text', TEXT), ('label', LABEL)]
#   train, test = TabularDataset.splits(path='.', train='train.csv', validation='cv.csv', 
#   format='csv', skip_header=True, fields=datafields)

#   TEXT.build_vocab(train, test, vectors=vectors, max_size=10000) 
#   LABEL.build_vocab(train, test)
#   train_iter, test_iter = BucketIterator.splits((train, test), batch_sizes=(128, 128), device=device, 
#   sort_key=lambda x: len(x.text), sort_within_batch=False, repeat=False)

#   return train_iter, test_iter, TEXT, LABEL

def make_small_imdb(batch_size=8, device=-1, vectors=None):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # TEXT = data.Field(include_lengths=False, lower=True, batch_first=True)
  TEXT = data.Field(tokenize=get_tokenizer("basic_english"),
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True, 
                    batch_first=False)
  LABEL = data.LabelField()

  datafields = [('text', TEXT), ('label', LABEL)]
  train, test = TabularDataset.splits(path='.', train='train.csv', validation='cv.csv', 
  format='csv', skip_header=True, fields=datafields)

  TEXT.build_vocab(train, test, vectors=vectors, max_size=30000) 
  LABEL.build_vocab(train, test)
  train_iter, test_iter = BucketIterator.splits((train, test), batch_sizes=(128, 128), device=device, 
  sort_key=lambda x: len(x.text), sort_within_batch=False, repeat=False)

  return train_iter, test_iter, TEXT, LABEL

# def accuracy(input, targs):
#     "Compute accuracy with `targs` when `input` is bs * n_classes."
#     n = targs.shape[0]
#     inp = input.argmax(dim=-1).view(-1)
#     targs = targs.view(n,-1)
#     acc = (inp==targs).float().mean()
#     return acc

def accuracy(inputs, targets):
    with torch.no_grad():
        pred = inputs.detach().argmax(dim=1)
        acc = (pred == targets).float().mean()
    return acc

def bce_accuracy(inputs, targets):
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