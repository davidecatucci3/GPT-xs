import torch.nn as nn
import torch
import math

from hyperparameters import hyperparams

# glossary
# B: batch_size
# T: ctx_lenght
# C: d_model
# hs: head_size
# nh: n_heads

# hyperparamters
vocab_size = hyperparams['vocab_size']
batch_size = hyperparams['batch_size']
ctx_length = hyperparams['ctx_length']
d_model = hyperparams['d_model']
dropout = hyperparams['dropout']
n_layers = hyperparams['n_layers']
head_size = hyperparams['head_size']
n_heads = hyperparams['n_heads']

# input pre-processing
def pe_table():
    pos = torch.arange(ctx_length)  # (T)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # (C)

    pe = torch.zeros(size=(ctx_length, d_model)) # (T, C)

    pe[:, 0::2] = torch.sin(pos * div_term) 
    pe[:, 1::2] = torch.cos(pos * div_term) 

    return pe  # (T, C)

class InputPreProcessing(nn.Module):
    def __init__(self):
        super().__init__()

        self.emb_table = nn.Embedding(vocab_size, d_model)

        self.register_buffer('pe_table', pe_table())

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        '''
        x: (B, T)
        '''

        assert x.size(1) <= ctx_length, "Input sequence length exceeds ctx_length"

        in_emb = self.emb_table(x) # (B, T, C)
        pe = self.pe_table # (T, C)

        x = in_emb + pe # (B, T, C) + (T, C) = (B, T, C)

        x = self.dropout(x)
      
        return x
    
# block
class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        '''
        x: (B, T, C)
        '''

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        assert d_model == head_size * n_heads, "d_model is not equal to head_size * n_heads"

        self.ll1 = nn.Linear(d_model, 3 * d_model)
        self.ll1 = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        x: (B, T, C)
        '''

        B, T, C = x.shape

        qkv = self.ll1(x) # (B, T, 3 * C)

        q, k, v = torch.split(qkv, dim=-1) # (B, T, C)

        q = q.view(B, T, n_heads, head_size).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, n_heads, head_size).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs)
        v = v.view(B, T, n_heads, head_size).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs)

        attn = (q @ k.tranpose(-2, -1)) * (1.0 / math.sqrt(head_size)) # (B, nh, T, hs) @ (B, nh, hs, T) = (B, nh, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) #  (B, nh, T, T)
        attn = nn.functional.softmax(attn, dim=-1) #  (B, nh, T, T)

        o = attn @ v #  (B, nh, T, T)  @ (B, nh, T, hs) = (B, nh, T, hs)
        o = o.transpose(1, 2).contiguious().view(B, T, C) # (B, T, C)

        o = self.ll2(o) # (B, T, C)
        
        return o

class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention()
        self.ln2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForwardNetwork()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        x: (B, T, C)
        '''

        x = x + self.dropout(self.mha(self.ln1(x))) # (B, T, C)

        x = x + self.dropout(self.ffwd(self.ln2(x))) # (B, T, C)

        return x

# model
class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_pre_processing = InputPreProcessing() # pre-process the input (B, T), transforming it in (B, T, C) before feedint it in the decoder
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
 
    def forward(self, x):
        '''
        x: (B, T)
        '''

        # pre-processing 
        x = self.in_pre_processing(x)

        # blocks
        for block in self.blocks:
            x = block(x) # (B, T, C)

        # output

model = GPT()

# test
xb, yb = torch.zeros(size=(batch_size, ctx_length), dtype=torch.long), torch.zeros(size=(batch_size, ctx_length), dtype=torch.long)

model(xb)