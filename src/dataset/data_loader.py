import numpy as np
import random
import torch

class DataLoader:
    def __init__(self, batch_size: int, block_size: int):
        self.B = batch_size
        self.T = block_size

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        get a batch of dataset and each batch has block_size tokens
        '''

        x_shard = random.randrange(0, 100) # do not include shard 101 because there a too few tokens inside
        type_shard = random.choice(['en', 'it'])
        path_name_shard = f'data/shard_{type_shard}_{x_shard:03d}.npy'
    
        tks_np = np.load(path_name_shard)
        idxs = torch.randint(0, len(tks_np) - self.T, size=(self.B,))

        # xb = torch.stack([torch.from_numpy(tks_np[idx:idx + block_size]) for idx in idxs], dim=0) 
        # yb = torch.stack([torch.from_numpy(tks_np[idx + 1:idx + block_size + 1]) for idx in idxs], dim=0)

        # this is much faster implementation of the one above, use less .from_numpy() and do not use for loops so it does not iterate B times, for very high batch_size can be exponentially faster
        rows = np.arange(self.B)[:, None]
        cols = np.arange(self.T)

        xb = torch.from_numpy(tks_np[idxs[rows] + cols]) # (B, T)
        yb = torch.from_numpy(tks_np[idxs[rows] + cols + 1]) # (B, T)

        return xb, yb