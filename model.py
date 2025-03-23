import torch.nn as nn
import inspect
import torch

from torch.nn import functional as F
from dataclasses import dataclass

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.d_model % config.n_heads == 0

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)

        self.n_heads = config.n_heads
        self.d_model = config.d_model

        self.c_proj.SCALE_INIT = 1

        #self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)

        #attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        #attn = F.softmax(attn, dim=-1)

        #y = attn @ v # (B, nh, T, T) x (B, nh, T, hs) = (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model)

        self.c_proj.SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.d_model) # layer normalization 1
        self.attn = CasualSelfAttention(config) # attention mechanism
        self.ln_2 = nn.LayerNorm(config.d_model) # layer normalization 2
        self.mlp = MLP(config) # multi layer perceptron

    def forward(self, x):
        # communication
        x = x + self.attn(self.ln_1(x))

        # computation 
        x = x + self.mlp(self.ln_2(x))

        return x
    
@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50304
    n_layers: int = 12
    n_heads: int = 12
    d_model = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model), # weights token embedding
            wpe = nn.Embedding(config.block_size, config.d_model), # weights positional encoding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]), # hidden blocks
            ln_f = nn.LayerNorm(config.d_model) # linear normalization function
        ))

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False) # language model head

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02

            if hasattr(module, 'SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        assert T <= self.config.block_size, f'Cannot forward sequence of lenght {T}, block_size is {self.config.block_size}'

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # (T, C)
        tok_emb = self.transformer.wte(idx) # (B, T, C)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x) # (B, T, voab_size)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, x, max_length):
        while x.size(1) < max_length:
            with torch.no_grad():
                logits, _ = self(x) # (B, T, vocab_size)

                logits = logits[:, -1, :] # (B, vocab_size)

                probs = F.softmax(logits, dim=-1) 

                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # take the top 50 probs, so it won't sample rare tokens

                ix = torch.multinomial(topk_probs, 1) # (B, 1)

                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)

                x = torch.cat((x, xcol), dim=1)
        
        return x

    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer