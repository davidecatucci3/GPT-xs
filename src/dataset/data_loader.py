import numpy as np
import torch
import os

class DataLoader:
    def __init__(self, B, T):
        self.B = B # batch_size
        self.T = T # ctx_lenght

        shards_path = 'data/dataset/'
        shards_subdir = os.listdir(shards_path)

        self.shards = [shards_path + shard for shard in shards_subdir]

        # initialize shard pointers and indices
        self.s_idx = 0 # starting index that slice curr_shard
        self.curr_pos = 0 # pointer shard pos
        self.curr_shard = self.load_shard(self.shards[self.curr_pos]) # current shard

    def load_shard(self, path: str) -> np.ndarray:
        '''
        load shard from data/dataset folder given the path
        '''

        tks = np.load(path)

        return tks

    def get_batch(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        get a batch of the dataset, each batch have T tokens

        device: on which hardware they will be processed, default is cpu
        mix: if mix is True, in one batch there are half samples in italian and the other half in english, mix=False for default
        shuffle: if shuffle is True, when you applty mix its gonna suffle the batches instead of be first half in en and other half in it, shuffle=False for default
        '''

        # check if current shard has enough indices left for a full batch
        if (self.B * self.T) + self.s_idx > len(self.curr_shard):
            # move to the next shard
            self.s_idx = 0
            self.curr_pos += 1
            self.curr_shard = self.load_shard(self.shards[self.curr_pos]) 

        # create xb and yb
        idxs = torch.randint(0, len(self.curr_shard) - self.B, size=(self.B,))
            
        rows = np.arange(self.B)[:, None]
        cols = np.arange(self.T)

        xb = torch.from_numpy(self.curr_shard[idxs[rows] + cols]).to(torch.long) # (B, T)
        yb = torch.from_numpy(self.curr_shard[idxs[rows] + cols + 1]).to(torch.long) # (B, T)

        self.curr_shard = self.curr_shard[(self.B * self.T) + self.s_idx:]
        self.s_idx += self.B * self.T

        perm = torch.randperm(self.B)

        xb, yb = xb[perm], yb[perm]

        return xb.to(device), yb.to(device)

