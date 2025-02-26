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
def pe_table(T, device):
    '''
    T: current length of the input, 1 <= T <= ctx_length
    '''

    pos = torch.arange(T).unsqueeze(1)  # (T, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # (C)

    pe = torch.zeros(size=(T, d_model), device=device) # (T, C)

    pe[:, 0::2] = torch.sin(pos * div_term) 
    pe[:, 1::2] = torch.cos(pos * div_term) 
    
    return pe  # (T, C)

class InputPreProcessing(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        self.emb_table = nn.Embedding(vocab_size, d_model)

        self.register_buffer('pe_table', pe_table(ctx_length, device))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        '''
        x: (B, T)
        '''

        _, T = x.shape

        assert T <= ctx_length, "Input sequence length exceeds ctx_length"

        in_emb = self.emb_table(x) # (B, T, C)
        pe = pe_table(T, self.device) # (T, C)

        x = in_emb + pe # (B, T, C) + (T, C) = (B, T, C)

        x = self.dropout(x)
      
        return x
    
# block
class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.ll1 = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU(approximate='tanh')
        self.ll2  = nn.Linear(4 * d_model, d_model)

        self.ll2.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        '''
        x: (B, T, C)
        '''

        x = self.ll1(x) # (B, T, 4 * C)

        x = self.gelu(x) # (B, T, C)

        x = self.ll2(x) # (B, T, C)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        assert d_model == head_size * n_heads, "d_model is not equal to head_size * n_heads"

        self.ll1 = nn.Linear(d_model, 3 * d_model)
        self.ll2 = nn.Linear(d_model, d_model)

        self.ll2.NANOGPT_SCALE_INIT = 1 

        #self.register_buffer('mask', torch.tril(torch.ones(ctx_length, ctx_length).view(1, 1, ctx_length, ctx_length)))

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

        #attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size)) # (B, nh, T, hs) @ (B, nh, hs, T) = (B, nh, T, T)
        #attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) #  (B, nh, T, T)
        #attn = nn.functional.softmax(attn, dim=-1) #  (B, nh, T, T)

        o = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention, (B, nh, T, hs)

        #o = attn @ v #  (B, nh, T, T)  @ (B, nh, T, hs) = (B, nh, T, hs)
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
    def __init__(self, device):
        super().__init__()

        self.in_pre_processing = InputPreProcessing(device) # pre-process the input (B, T), transforming it in (B, T, C) before feedint it in the decoder
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.output = OutputProbabilities()

        # init params
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02

            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * n_layers) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
 
    def forward(self, x, targets=None):
        '''
        x: (B, T)
        '''

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
            logits = logits.view(-1, logits.size(-1))
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

                probs, _ = self(x) # (1, T, V)
        
                probs = probs[:, -1, :] # (1, V)

                next_tk = torch.multinomial(probs, num_samples=1) # (1, 1)

                x = torch.cat([x, next_tk], dim=-1) # (1, T + 1)

        return x


