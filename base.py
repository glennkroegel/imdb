'''
created_by: Glenn Kroegel
date: 31 January 2020

https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms
https://github.com/mttk/rnn-classifier/blob/master/model.py
'''

import pandas as pd
import numpy as np
from collections import defaultdict, Counter

import torch
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import math
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import *
import shutil

from utils import count_parameters, accuracy, make_imdb, make_small_imdb

NUM_EPOCHS = 100

# TODO: Add mask for padding

#################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
status_properties = ['loss', 'accuracy']

class Attention(nn.Module):
    def __init__(self, query_dim):
        super(Attention, self).__init__()
        self.scale = 1/math.sqrt(query_dim)

    def forward(self, query, keys, values, mask=None):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]
        bs = keys.size(1)
        T = keys.size(0)
        query = query.view(bs, 1, -1)
        keys = keys.view(bs, -1, T)
        e = torch.bmm(query, keys)
        if mask is not None:
            e.masked_fill_(mask.unsqueeze(1), value=0)
        alpha = F.softmax(e.mul_(self.scale), dim=2)
        values = values.transpose(0, 1)
        attn = torch.bmm(alpha, values).squeeze()
        return alpha, attn

class BaseLineRNN(nn.Module):
    def __init__(self, vocab_sz, hidden_sz, pad_idx, emb_dim=10, rnn_layers=1):
        super(BaseLineRNN, self).__init__()
        self.emb_dim = emb_dim
        self.rnn_layers = rnn_layers
        self.bidir = True

        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_idx, max_norm=1)
        self.rnn = nn.GRU(input_size=10, hidden_size=hidden_sz, num_layers=self.rnn_layers, bidirectional=self.bidir)
        # self.l_proj = nn.Linear(hidden_sz, hidden_sz)
        self.l_out = nn.Linear(hidden_sz*2, 2)
        self.act = nn.ReLU()

    def init_hidden(self, batch_sz):
        ''' (num_layers, batch_size, hidden_size)'''
        hidden = torch.zeros(self.rnn_layers, batch_sz, hidden_sz)
        return hidden

    def forward(self, x):
        # bs = x.size(0)
        # sl = x.size(1)
        # x = x.view(sl,bs)
        x = self.emb(x)
        # x = x.view(sl, bs, -1)
        # h0 = self.init_hidden(bs)
        outp, hidden = self.rnn(x)
        if self.bidir:
            x = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            x = hidden.view(bs, -1)
        # x = self.act(self.l_proj(hidden))
        x = self.l_out(x)
        # x = torch.sigmoid(x)
        return x

class RNNAttn(nn.Module):
    def __init__(self, vocab_sz, hidden_sz, pad_idx, emb_dim=10, rnn_layers=1):
        super(RNNAttn, self).__init__()
        self.emb_dim = emb_dim
        self.rnn_layers = rnn_layers
        self.bidir = False

        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=pad_idx, max_norm=1)
        self.rnn = nn.GRU(input_size=10, hidden_size=hidden_sz, num_layers=self.rnn_layers, bidirectional=self.bidir)
        self.attn = Attention(query_dim=hidden_sz)
        self.l_out = nn.Linear(hidden_sz, 2)
        self.act = nn.ReLU()

    def init_hidden(self, batch_sz):
        ''' (num_layers, batch_size, hidden_size)'''
        hidden = torch.zeros(self.rnn_layers, batch_sz, hidden_sz)
        return hidden

    def forward(self, x):
        bs = x.size(1)
        x = self.emb(x)
        outp, hidden = self.rnn(x)
        if self.bidir:
            x = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            x = hidden.view(bs, -1)
        alpha, x = self.attn(query=x, keys=outp, values=outp)
        x = self.l_out(x)
        return x, alpha

###########################################################################

# CONV models

class BaseConv(nn.Module):
    def __init__(self, emb_dim=30):
        super(BaseConv, self).__init__()
        self.emb_dim = emb_dim

        self.emb = nn.Embedding(vocab_sz, emb_dim)
        self.conv1 = nn.Conv1d(emb_dim, emb_dim, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(30)
        self.l_out = nn.Linear(300, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        bs = x.size(0)
        sl = x.size(1)
        x = self.emb(x)
        x = x.view(bs, self.emb_dim, -1)
        x = self.act(self.conv1(x))
        x = self.pool(x)
        x = x.view(bs, -1)
        x = self.l_out(x)
        x = torch.sigmoid(x)
        return x

class ConvAttn(nn.Module):
    def __init__(self, emb_dim=30):
        super(ConvAttn, self).__init__()
        self.emb_dim = emb_dim
        self.emb = nn.Embedding(vocab_sz, emb_dim)
        self.conv1 = nn.Conv1d(emb_dim, emb_dim, 3, padding=1, bias=False)
        self.attn = Attention(emb_dim)
        self.l_out = nn.Linear(emb_dim, 1, bias=False)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        bs = x.size(0)
        sl = x.size(1)
        mask = (x == tokenizer.pad_token_id)
        x = self.emb(x)
        x = x.view(bs, self.emb_dim, -1)
        x = self.act(self.conv1(x))
        x = x.view(sl, bs, -1)
        q = x[0].view(bs, -1)
        alpha, attn = self.attn(query=q, keys=x, values=x, mask=mask)
        x = self.bn(attn)
        x = self.l_out(x)
        x = torch.sigmoid(x)
        return x

###########################################################################

def save_checkpoint(state, is_best, filename='checkpoint_simple.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def train(loader, model, criterion, optimizer):
    model.train()
    props = {k:0 for k in status_properties}
    for i, data in enumerate(loader):
        model.zero_grad()
        x = data.text
        targets = data.label
        preds,_ = model(x)
        loss = criterion(preds, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        props['loss'] += loss.item()
        props['accuracy'] += accuracy(preds, targets)
    L = len(loader)
    props = {k:v/L for k,v in props.items()}
    return props

def evaluate(loader, model, criterion, optimizer):
    model.eval()
    props = {k:0 for k in status_properties}
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.text
            targets = data.label
            preds,_ = model(x)
            loss = criterion(preds, targets)
            props['loss'] += loss.item()
            props['accuracy'] += accuracy(preds, targets)
    L = len(loader)
    props = {k:v/L for k,v in props.items()}
    return props

def status(epoch, train_props, cv_props, epochs):
    s0 = 'epoch {0}/{1}\n'.format(epoch, epochs)
    s1, s2 = '',''
    for k,v in train_props.items():
        s1 = s1 + 'train_'+ k + ': ' + str(v) + ' '
    for k,v in cv_props.items():
        s2 = s2 + 'cv_'+ k + ': ' + str(v) + ' '
    print(s0 + s1 + s2)

def main():
    train_iter, val_iter, TEXT, LABEL = make_small_imdb()
    vocab_sz = len(TEXT.vocab)
    print(vocab_sz)
    hidden_sz = 50
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]

    model = RNNAttn(vocab_sz, hidden_sz, pad_idx).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    epochs = 20
    best_loss = np.inf

    for epoch in tqdm(range(epochs)):
        train_props = train(train_iter, model, criterion, optimizer)
        cv_props = evaluate(val_iter, model, criterion, optimizer)
        status(epoch, train_props, cv_props, epochs)

        cv_loss = cv_props['loss']
        if cv_loss < best_loss:
            print('Saving model..')
            torch.save(model, 'model.pt')
            best_loss = cv_loss

if __name__ == "__main__":
    main()