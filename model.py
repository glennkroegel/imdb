'''
created_by: Glenn Kroegel
date: 31 January 2020
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
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import *
import shutil

from utils import count_parameters, accuracy
# from losses import BCEDiceLoss, DiceLoss
NUM_EPOCHS = 10

#################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
status_properties = ['loss']

#################################################################################################

model_class = DistilBertModel
tokenizer_class = DistilBertTokenizer
weights = 'distilbert-base-uncased'

tokenizer = tokenizer_class.from_pretrained(weights)
tokenize = lambda x: tokenizer.tokenize(x)

#################################################################################################

TEXT = Field(sequential=True, 
             tokenize=tokenize, 
             use_vocab=True, 
             pad_token=tokenizer.pad_token, 
             unk_token=tokenizer.unk_token, 
             pad_first=True, 
             batch_first=True)

LABEL = Field(use_vocab=False, sequential=False)

datafields = [('text', TEXT), ('label', LABEL)]

trn, cv = TabularDataset.splits(path='.', train='train.csv', validation='cv.csv', 
format='csv', skip_header=True, fields=datafields)

TEXT.build_vocab()
stoi = dict(tokenizer.vocab)
itos = list(stoi.keys())
TEXT.vocab.stoi = stoi
TEXT.vocab.itos = itos

train_iter, val_iter = BucketIterator.splits((trn, cv), batch_sizes=(6, 6), device=device, 
sort_key=lambda x: len(x.text), sort_within_batch=False, repeat=False)

#################################################################################################

class DistilBert(nn.Module):
    def __init__(self):
        super(DistilBert, self).__init__()
        self.clf = DistilBertModel.from_pretrained(weights)

    def forward(self, x):
        x = self.clf(x)[0]
        return x


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        pass

    def forward(self, x):
        x = x.sum(axis=1)
        x = x/x.norm()
        return x

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(DistilBert(),
        Head(), nn.Linear(768, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.enc(x)
        return x

###########################################################################

def save_checkpoint(state, is_best, filename='checkpoint_simple.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Learner():
    '''Training loop'''
    def __init__(self, epochs=NUM_EPOCHS):
        self.model = Encoder().to(device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=1e-5)
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
                # print(i)
            x = data.text
            targets = data.label.float()
            # targets = targets.to(device)
            preds = model(x)
            loss = criterion(preds.view(-1), targets.view(-1))
            props['loss'] += loss.item()
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
