import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import math
from utils import *
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class DistilBert(nn.Module):
    def __init__(self):
        super(DistilBert, self).__init__()
        self.clf = DistilBertModel.from_pretrained(weights)

    def forward(self, x):
        x = self.clf(x)[0]
        x = x.permute(1, 0, 2)
        x = x[0]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Embed(nn.Module):
    def __init__(self, vocab_sz, emb_dim):
        super(Embed, self).__init__()
        self.embed = nn.Embedding(vocab_sz, emb_dim)

    def forward(self, x):
        x = self.embed(x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, n_head=2):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=2*d_model)
        self.transformer = nn.TransformerEncoder(self.enc_layer, num_layers=3, norm=nn.LayerNorm(d_model,))

    def forward(self, x):
        return self.transformer(x)

class Model(nn.Module):
    def __init__(self, d_model, vocab_sz, emb_dim):
        super(Model, self).__init__()
        self.emb = Embed(vocab_sz, emb_dim)
        self.pos = PositionalEncoding(d_model=d_model)
        self.encoder = Encoder(d_model, n_head=4)
        self.l_out = nn.Linear(d_model, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.emb.embed.weight.data.uniform_(-initrange, initrange)
        self.l_out.bias.data.zero_()
        self.l_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.emb(x)
        x = self.pos(x)
        x = self.encoder(x)
        # x = x.mean(dim=1)
        # x = x[:,0]
        x = x[0]
        x = self.l_out(x)
        x = torch.sigmoid(x)
        x = x.squeeze()
        return x

def evaluate(loader, model, criterion):
    props = {'loss': 0, 'accuracy': 0}
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            x = data.text
            targets = data.label.float()
            preds = model(x)
            loss = criterion(preds, targets)
            acc = bce_accuracy(preds, targets)
            props['loss'] += loss.item()
            props['accuracy'] += acc.item()
    L = len(loader)
    props = {k:v/L for k,v in props.items()}
    return props

def train(loader, model, criterion, optimizer):
    props = {'loss': 0, 'accuracy': 0}
    model.train()
    for i, data in enumerate(loader):
        model.zero_grad()
        x = data.text
        targets = data.label.float()
        preds = model(x)
        loss = criterion(preds, targets)
        acc = bce_accuracy(preds, targets)
        loss.backward()
        # clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        props['loss'] += loss.item()
        props['accuracy'] += acc.item()
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
    train_loader, cv_loader, TEXT, LABEL = make_small_imdb()
    pad_idx = TEXT.vocab[TEXT.pad_token]
    vocab_sz = len(TEXT.vocab)
    print(vocab_sz)
    d_model = 20
    model = Model(d_model=d_model, vocab_sz=vocab_sz, emb_dim=d_model).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epochs = 300
    best_loss = np.inf

    for epoch in tqdm(range(epochs)):
        train_props = train(train_loader, model, criterion, optimizer)
        cv_props = evaluate(cv_loader, model, criterion)
        status(epoch, train_props, cv_props, epochs)

        # cv_loss = cv_props['loss']
        # if cv_loss < best_loss:
        #     print('Saving model..')
        #     torch.save(model, 'model.pt')
        #     best_loss = cv_loss


if __name__ == '__main__':
    main()