import torch.distributed as dist
import tiktoken
import torch
import time
import math
import os

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from data_loader import DataLoader
from model import GPT, GPTConfig

# setup DDP 
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    assert torch.cuda.is_available(), 'CUDA in not available'
    
    init_process_group(backend='nccl')

    ddp_rank = int(os.environ['RANK']) # rank goes from 0 to 7 that indicate gpus
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # equal to rank in this case
    ddp_world_size = int(os.environ['WORLD_SIZE']) # number of gpus, number of processes among all the nodes

    # local_rank: the rank of the process on the local machine.
    # rank: the rank of the process in the network.
    #                  |    Node1  |   Node2    |
    # processes        | p1 |  p2  |  p3  |  p4 | 
    # local_rank       | 0  |   1  |  0   |   1 |
    # rank             | 0  |   1  |  2   |   3 |

    # in our case: only one node so rank=loca_rank
    #                  |                    Node1                           |  
    # processes        | p1 |  p2  | p3  |  p4  |  p5  |  p6  |  p7  |  p8  |
    # local_rank       | 0  |   1  |  2  |  3   |  4   |   5  |  6   |   7  |
    # rank             | 0  |   1  |  2  |  3   |  4   |   5  |  6   |   7  |

    device = f'cuda:{ddp_local_rank}'

    torch.cuda.set_device(device)

    master_process = (ddp_rank == 0) # use for logging, printing stuff, etc...
else:
    print('DDP is not available because you are not using GPU')

    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1

    master_process = True

    # attempt to autodetect device
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        
print(f"Using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

model = GPT(GPTConfig())
model.to(device)

if device.startswith('cuda'):
    model = torch.compile(model)
elif device == 'mps':
    print("Skipping torch.compile on MPS due to Inductor backend limitations")
else:
    print("Running in eager mode on CPU")

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# data loader
total_batch_size = 524288 # 2 ** 19 ~ 0.5M, in number of tokens, so at each step 0.5M tokens are used so to finish all the tokens you need 10B / 0.5M = 20K ~ 19073

B = 64    # micro batch size, increase to make the traininig faster
T = 1024  # sequence length

assert total_batch_size % (B * T * ddp_world_size) == 0, 'make sure the total_batch_size is divisible by B * T * ddp_word_size'

grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train', master_process=master_process)
val_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val', master_process=master_process)

if master_process:
    print(f'total desired batch size: {total_batch_size}')
    print(f'=> calculated gradient accumulation steps: {grad_accum_steps}')


# FP32:  1 | 8 | 23  -> 1.0e-38-3.0e38  -> can represent same number of tf32 but with higher precision     | 91  TFLOPS
# TF32:  1 | 8 | 10  -> 1.0e-38-3.0e38  -> can represent same number of fp32 but with less precision       | 183 TFLOPS
# FP16:  1 | 5 | 10  -> 5.9e-8-6.5e4    ->  can represent smaller number then bf16 with higher precision   | 362 TFLOPS
# BF16:  1 | 8 | 7   -> 1.0e-38-3.0e38  -> can represent bigger number then fp16 with less precision       | 362 TFLOPS

'''
PyTorch does a mix of precision something is keeped in TF32 and other things in BF16, for example the weights are in TF32 
but the logits in BF16, 

CUDA Ops that can autocast to BF16:
__matmul__, addbmm, addmm, addmv, addr, baddbmm, bmm, chain_matmul, multi_dot, conv1d, conv2d, 
conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d, GRUCell, linear, LSTMCell, matmul, 
mm, mv, prelu, RNNCell

CUDA Ops that can autocast to TF32:
__pow__, __rdiv__, __rpow__, __rtruediv__, acos, asin, binary_cross_entropy_with_logits, cosh, 
cosine_embedding_loss, cdist, cosine_similarity, cross_entropy, cumprod, cumsum, dist, erfinv, 
exp, expm1, group_norm, hinge_embedding_loss, kl_div, l1_loss, layer_norm, log, log_softmax, 
log10, log1p, log2, margin_ranking_loss, mse_loss, multilabel_margin_loss, multi_margin_loss, 
nll_loss, norm, normalize, pdist, poisson_nll_loss, pow, prod, reciprocal, rsqrt, sinh, smooth_l1_loss, 
soft_margin_loss, softmax, softmin, softplus, sum, renorm, tan, triplet_margin_loss
'''

torch.set_float32_matmul_precision('high') # use TF32 (tensor float 32)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715 # 375e6 / 2 ** 19 = 715, after 375M tokens start decay
max_steps = 19073 # 10B tokens / 2 ** 19 = 19073

# log folder
log_dir = '/root/mark-124M-v1.0-ssh/log'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'log.txt')

with open(log_file, 'w') as f:
    pass

# tokenizer
tn = tiktoken.get_encoding('gpt2')

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps

    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)

    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff * (max_lr - min_lr)

optim = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type, master_process=master_process)

if master_process:
    print(f'Number of parameters: {(sum(n.numel() for n in model.parameters())) / 1e6}M')

# cmd launch with DDP:torchrun --standalone --nproc_per_node=8 train.py

for step in range(max_steps):
    t0 = time.time()

    # once in a while evaluate model
    if step % 500 == 0 or step == max_steps - 1:
        model.eval()

        val_loader.reset()

        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 15

            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)

                if 'cuda' in device or device == 'cpu':
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)
                
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
            
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            
        if master_process:
            print(f'Validation loss: {val_loss_accum:.3f}')

            with open(log_file, 'a') as f:
                f.write(f'{step} val {val_loss_accum.item():3f} \n')

    # once in a while we generate text
    if step > 0 and (step % 1000 == 0 or step == max_steps - 1):
        model.eval()

        num_return_sequences = 3
        max_length = 32

        tokens = tn.encode("Hello, I'm a Large Language Model")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        
        xgen = model.module.generate(xgen, max_length)

        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = tn.decode(tokens)

            print(f"rank: {ddp_rank} sample {i}: {decoded}")
    
    # once in a while eval hellaswag
    # ...
    
    # train loop
    model.train()

    optim.zero_grad()

    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        if 'cuda' in device or device == 'cpu':
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    lr = get_lr(step)
    
    for param_group in optim.param_groups:
        param_group['lr'] = lr

    optim.step()

    if 'cuda' in device:
        torch.cuda.synchronize()

    t1 = time.time()

    dt = (t1 - t0) * 1000

    tks_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1 - t0)

    if master_process:
        print(f'step {step} / {max_steps} | loss: {loss_accum.item():.3f} | lr: {lr:.6f} | dt: {dt:.2f}ms | tks/sec: {tks_sec:.2f}')

        if step % 5000 == 0 or step == max_steps - 1:
            print('SAVED CHECKPOINT!!!')

            checkpoint_path = f'/root/mark-124M-v1.0-ssh/log/model_{step:05d}.pt'
            checkpoint = {
                'model': model.state_dict(),
                'config': model.config,
                'step': step
            }

            torch.save(checkpoint, checkpoint_path)
     
        with open(log_file, 'a') as f:
            f.write(f'{step} train {loss_accum.item():4f} \n')

if ddp:
    destroy_process_group()