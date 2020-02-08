import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from loading import batch_size_fn, MyIterator
from utils import tokenize_sentence, tokenizer, accuracy
from tqdm import tqdm
import shutil

# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

class Embedder(nn.Module):
    def __init__(self, vocab_sz, dim):
        super().__init__()
        self.vocab_sz = vocab_sz
        self.dim = dim
        self.emb = nn.Embedding(vocab_sz, dim)

    def forward(self, x):
        return self.emb(x)

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        self.max_len = max_len
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        for pos in torch.arange(0, max_len):
            for i in torch.arange(0, dim, 2):
                pe[pos, i] = torch.sin(pos/(10000**(2*i/dim)))
                pe[pos, i+1] = torch.cos(pos/(10000**(2*(i+1)/dim)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        bs = x.size(0)
        sl = x.size(1)
        x = self.pos * torch.sqrt(self.dim)
        x = x + self.pe[:, :sl]
        return x

class DotProductAttention(nn.Module):
    '''Scaled dot product attention'''
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, q, k, v, mask, drop):
        scores = torch.matmul(q * k.transpose(-2, -1)) / torch.sqrt(self.d_k)
        return scores

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.dim = dim
        self.h = heads
        self.d_k = dim // heads

        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.attn = DotProductAttention(self.d_k)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(0.2)

    def forward(self, q, k, v, mask):
        bs = q.size(0)
        sl = q.size(1)

        q = self.k_lin(q).view(bs, -1, self.h, self.d_k)
        k = self.k_lin(k).view(bs, -1, self.h, self.d_k)
        v = self.k_lin(v).view(bs, -1, self.h, self.d_k)

        q.transpose_(1, 2)
        k.transpose_(1, 2)
        v.transpose_(1, 2)

        scores = self.attn(q, k, v, mask)
        scores = scores.transpose(1, 2).contiguous().view(bs, -1, self.dim)
        
        return outp