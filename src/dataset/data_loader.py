import numpy as np
import random
import torch

def load_shard(type_shard : bool = False):
    # select a random shard file
    x_shard = random.randrange(1, 100) # do not include shard 101 because there a too few tokens inside
    type_shard = random.choice(['en', 'it']) if not type_shard else type_shard
    path_name_shard = f'data/shard_{type_shard}_{x_shard:03d}.npy'
        
    tks_np = np.load(path_name_shard)

    return tks_np

def get_batch(batch_size: int, block_size: int, mix: bool = False, shuffle: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    get a batch of the dataset, each batch have T tokens

    mix: if mix is True, in one batch there are half samples in italian and the other half in english, mix=False for default
    shuffle: if shuffle is True, when you applty mix its gonna suffle the batches instead of be first half in en and other half in it, shuffle=False for default
    '''

    if not mix:
        # load random shard
        tks_np = load_shard()

        # create xb and yb
        idxs = torch.randint(0, len(tks_np) - block_size, size=(batch_size,))
        
        rows = np.arange(batch_size)[:, None]
        cols = np.arange(block_size)

        xb = torch.from_numpy(tks_np[idxs[rows] + cols]) # (B, T), B:batch_size, T:block_size
        yb = torch.from_numpy(tks_np[idxs[rows] + cols + 1]) # (B, T)
    else:
        assert batch_size % 2 == 0, 'batch_size has to be divisible by 2'

        half_batch = batch_size // 2

        xb = torch.zeros(batch_size, block_size, dtype=torch.int32)
        yb = torch.zeros(batch_size, block_size, dtype=torch.int32)
        
        for i, type_shard in enumerate(['en', 'it']):
            # load random shard
            tks_np = load_shard(type_shard)
            
            # create xb and yb for one language per time
            idxs = torch.randint(0, len(tks_np) - block_size, size=(half_batch,))

            rows = idxs[:, None] + torch.arange(block_size)
            start_idx = i * half_batch
            end_idx = (i + 1) * half_batch
            
            # fill the pre-allocated tensors directly
            xb[start_idx:end_idx] = torch.from_numpy(tks_np[rows]) # (B, T)
            yb[start_idx:end_idx] = torch.from_numpy(tks_np[rows + 1]) # (B, T)

        # instead of be the first half samples in english and the other in italian, they will be shuffled
        if shuffle:
            perm = torch.randperm(batch_size)

            return xb[perm], yb[perm]

    return xb, yb

xb, yb = get_batch(100_000, 1024, mix=True)