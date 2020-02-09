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

from utils import count_parameters, accuracy

NUM_EPOCHS = 100

# TODO: Add mask for padding

#################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
status_properties = ['loss', 'accuracy']

#################################################################################################

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokenize = lambda x: tokenizer.tokenize(x)

#################################################################################################

TEXT = Field(sequential=True, 
             tokenize=tokenize, 
             use_vocab=True,
             init_token=tokenizer.cls_token,
             pad_token=tokenizer.pad_token, 
             unk_token=tokenizer.unk_token, 
             pad_first=False, 
             batch_first=True)

LABEL = Field(use_vocab=False, sequential=False)

datafields = [('text', TEXT), ('label', LABEL)]

trn, cv = TabularDataset.splits(path='.', train='train.csv', validation='cv.csv', 
format='csv', skip_header=True, fields=datafields)

TEXT.build_vocab(trn, cv)
stoi = dict(tokenizer.vocab)
itos = list(stoi.keys())
TEXT.vocab.stoi = stoi
TEXT.vocab.itos = itos

train_iter, val_iter = BucketIterator.splits((trn, cv), batch_sizes=(64, 64), device=device, 
sort_key=lambda x: len(x.text), sort_within_batch=True, repeat=False)

vocab_sz = len(tokenizer.vocab)
print(vocab_sz)
hidden_sz = 50

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
        query = query.unsqueeze(1)
        keys = keys.view(bs, -1, T)
        e = torch.bmm(query, keys)
        if mask is not None:
            e.masked_fill_(mask.unsqueeze(1), value=0)
        alpha = F.softmax(e.mul_(self.scale), dim=-1)
        values = values.transpose(0, 1)
        attn = torch.bmm(alpha, values).squeeze()
        return alpha, attn

class RNNAttn(nn.Module):
    def __init__(self, emb_dim=10, rnn_layers=1):
        super(RNNAttn, self).__init__()
        self.emb_dim = emb_dim
        self.rnn_layers = rnn_layers

        self.emb = nn.Embedding(vocab_sz, emb_dim)
        self.rnn = nn.GRU(input_size=10, hidden_size=hidden_sz, num_layers=self.rnn_layers)
        self.attn = Attention(query_dim=hidden_sz)
        self.l_proj = nn.Linear(hidden_sz, hidden_sz)
        self.l_out = nn.Linear(hidden_sz, 1)
        self.act = nn.ReLU()

    def init_hidden(self, batch_sz):
        ''' (num_layers, batch_size, hidden_size)'''
        hidden = torch.zeros(self.rnn_layers, batch_sz, hidden_sz)
        return hidden

    def forward(self, x):
        bs = x.size(0)
        sl = x.size(1)
        mask = (x == tokenizer.pad_token_id)
        x = self.emb(x)
        x = x.view(sl, bs, -1)
        h0 = self.init_hidden(bs)
        outp, hidden = self.rnn(x, h0.to(device))
        hidden = hidden.view(bs, -1)
        alpha, attn = self.attn(query=hidden, keys=outp, values=outp, mask=mask)
        x = self.act(self.l_proj(attn))
        x = self.l_out(x)
        x = torch.sigmoid(x)
        return x, alpha

class BaseLineRNN(nn.Module):
    def __init__(self, emb_dim=10, rnn_layers=1):
        super(BaseLineRNN, self).__init__()
        self.emb_dim = emb_dim
        self.rnn_layers = rnn_layers

        self.emb = nn.Embedding(vocab_sz, emb_dim)
        self.rnn = nn.GRU(input_size=10, hidden_size=hidden_sz, num_layers=self.rnn_layers)
        self.l_proj = nn.Linear(hidden_sz, hidden_sz)
        self.l_out = nn.Linear(hidden_sz, 1)
        self.act = nn.ReLU()

    def init_hidden(self, batch_sz):
        ''' (num_layers, batch_size, hidden_size)'''
        hidden = torch.zeros(self.rnn_layers, batch_sz, hidden_sz)
        return hidden

    def forward(self, x):
        bs = x.size(0)
        sl = x.size(1)
        x = self.emb(x)
        x = x.view(sl, bs, -1)
        # h0 = self.init_hidden(bs)
        outp, hidden = self.rnn(x)
        hidden = hidden.view(bs, -1)
        x = self.act(self.l_proj(hidden))
        x = self.l_out(x)
        x = torch.sigmoid(x)
        return x

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

class Learner():
    '''Training loop'''
    def __init__(self, epochs=NUM_EPOCHS):
        self.model = ConvAttn().to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-4)
        self.epochs = epochs
        self.best_loss = np.inf

        resume = False
        if resume:
            print('resuming..')
            checkpoint = torch.load('checkpoint_simple.pth.tar')
            self.model.load_state_dict(checkpoint['state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_loss = checkpoint['best_loss']

        self.train_loss = []
        self.cv_loss = []

        print('Model Parameters: ', count_parameters(self.model))

    def iterate(self, loader, model, criterion, optimizer, training=True):
        if training:
            model.train()
        else:
            model.eval()
        props = {k:0 for k in status_properties}
        for i, data in enumerate(loader):
            if i % 1 == 0:
                pass
            x = data.text
            targets = data.label.float()
            # targets = targets.to(device)
            preds = model(x)
            loss = criterion(preds.view(-1), targets.view(-1))
            acc = accuracy(preds.view(-1), targets.view(-1))
            props['loss'] += loss.item()
            props['accuracy'] += acc.item()
            if training:
                optimizer.zero_grad()
                if i % 20 == 0:
                    print(i)
                loss.backward()
                optimizer.step()
                # clip_grad_norm_(model.parameters(), 0.5)
            L = len(loader)
        props = {k:v/L for k,v in props.items()}
        return props

    def step(self):
        '''Actual training loop.'''
        for epoch in tqdm(range(self.epochs)):
            train_props = self.iterate(train_iter, self.model, self.criterion, self.optimizer, training=True)
            self.scheduler.step(epoch)
            lr = self.scheduler.get_lr()[0]
            # if train_props['loss'] > 3.0:
            #     exit()
            self.train_loss.append(train_props['loss'])
            # cross validation
            with torch.no_grad():
                cv_props = self.iterate(val_iter, self.model, self.criterion, self.optimizer, training=False)
                L = len(val_iter)
                self.cv_loss.append(cv_props['loss'])
                is_best = False
                if epoch % 1 == 0:
                    self.status(epoch, train_props, cv_props)
                if cv_props['loss'] < self.best_loss:
                    print('dumping model...')
                    path = 'model' + '.pt'
                    torch.save(self.model, path)
                    self.best_loss = cv_props['loss']
                    is_best = True
                save_checkpoint(
                    {'epoch': epoch + 1,
                    'lr': lr, 
                    'state_dict': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict(), 
                    'best_loss': self.best_loss}, is_best=is_best)
                is_best=False

    def status(self, epoch, train_props, cv_props):
        s0 = 'epoch {0}/{1}\n'.format(epoch, self.epochs)
        s1, s2 = '',''
        for k,v in train_props.items():
            s1 = s1 + 'train_'+ k + ': ' + str(v) + ' '
        for k,v in cv_props.items():
            s2 = s2 + 'cv_'+ k + ': ' + str(v) + ' '
        print(s0 + s1 + s2)

if __name__ == "__main__":
    try:
        mdl = Learner()
        mdl.step()
    except KeyboardInterrupt:
        pd.to_pickle(mdl.train_loss, 'train_loss.pkl')
        pd.to_pickle(mdl.cv_loss, 'cv_loss.pkl')
        print('Stopping')