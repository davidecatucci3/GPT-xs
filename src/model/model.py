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
def pe_table(T):
    '''
    T: current length of the input, 1 <= T <= ctx_length
    '''

    pos = torch.arange(T).unsqueeze(1)  # (T, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # (C)

    pe = torch.zeros(size=(T, d_model)) # (T, C)

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

        _, T = x.shape

        assert T <= ctx_length, "Input sequence length exceeds ctx_length"

        in_emb = self.emb_table(x) # (B, T, C)
        pe = pe_table(T) # (T, C)

        x = in_emb + pe # (B, T, C) + (T, C) = (B, T, C)

        x = self.dropout(x)
      
        return x
    
# block
class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        '''
        x: (B, T, C)
        '''

        x = self.net(x) # (B, T, C)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        assert d_model == head_size * n_heads, "d_model is not equal to head_size * n_heads"

        self.ll1 = nn.Linear(d_model, 3 * d_model)
        self.ll2 = nn.Linear(d_model, d_model)

        self.register_buffer('mask', torch.tril(torch.ones(ctx_length, ctx_length).view(1, 1, ctx_length, ctx_length)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        x: (B, T, C)
        '''

        B, T, C = x.shape

        qkv = self.ll1(x) # (B, T, 3 * C)

        q, k, v = qkv.split(d_model, dim=-1) # (B, T, C)

        q = q.view(B, T, n_heads, head_size).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, n_heads, head_size).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs)
        v = v.view(B, T, n_heads, head_size).transpose(1, 2) # (B, T, nh, hs) -> (B, nh, T, hs)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size)) # (B, nh, T, hs) @ (B, nh, hs, T) = (B, nh, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) #  (B, nh, T, T)
        attn = nn.functional.softmax(attn, dim=-1) #  (B, nh, T, T)

        o = attn @ v #  (B, nh, T, T)  @ (B, nh, T, hs) = (B, nh, T, hs)
        o = o.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

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
    
# output
class OutputProbabilities(nn.Module):
    def __init__(self):
        super().__init__()

        self.ln = nn.LayerNorm(d_model)
        self.ll = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        '''
        x: (B, T, C)
        '''

        x = self.ln(x) # (B, T, C)

        logits = self.ll(x) # (B, T, V)

        probs = nn.functional.softmax(logits, dim=-1) # (B, T, V)

        return probs, logits

# model
class GPT(nn.Module):
    def __init__(self):
        super().__init__()

        self.in_pre_processing = InputPreProcessing() # pre-process the input (B, T), transforming it in (B, T, C) before feedint it in the decoder
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.output = OutputProbabilities()
 
    def forward(self, x, targets=None):
        '''
        x: (B, T)
        '''

        B, T = x.shape

        # pre-processing 
        x = self.in_pre_processing(x)

        # blocks
        for block in self.blocks:
            x = block(x) # (B, T, C)

        # output
        probs, logits = self.output(x) # (B, T, V), (B, T, C)
        
        if targets is None:
            loss = None
        else:
            logits = logits.view(-1, logits.size(-1)) #
            targets = targets.view(-1)

            loss = nn.functional.cross_entropy(logits, targets)

        return probs, loss
    
    def generate(self, x, max_tokens):
        '''
        x: (1, T)
        '''

        with torch.no_grad():
            for _ in range(max_tokens):
                x = x[:, -ctx_length:] # (1, T) truncate x.size(1) if exceed ctx_lenght

                probs = self(x) # (1, T, V)
        
                probs = probs[:, -1, :] # (1, V)

                next_tk = torch.multinomial(probs, num_samples=1) # (1, 1)

                x = torch.cat([x, next_tk], dim=-1) # (1, T + 1)

        return x

model = GPT()

# test
'''xb, yb = torch.zeros(size=(batch_size, ctx_length), dtype=torch.long), torch.zeros(size=(batch_size, ctx_length), dtype=torch.long)

model(xb)
'''
x = torch.ones(size=(1, 1), dtype=torch.long)

res = model.generate(x, max_tokens=1000)[0].tolist()
import sentencepiece as spm
sp = spm.SentencePieceProcessor()

sp.load('data/tokenizer/BPE-200-50527.model')

dec = sp.decode(res)

print(dec)