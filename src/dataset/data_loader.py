import numpy as np
import random
import torch
import os

class DataLoader:
    def __init__(self, B, T):
        self.B = B # batch_size
        self.T = T # ctx_lenght

        shards_path = 'data/dataset/'
        shards_subdir = os.listdir(shards_path)

        self.all_shards = [shards_path + shard for shard in shards_subdir] # list of path for all shards
        random.shuffle(self.all_shards) # shuffle en and it shards

        self.s_idx = 0 # starting index that slice curr_shard
        self.curr_pos = 0 # pointer shard pos
        self.curr_shard = self.load_shard(self.all_shards[self.curr_pos]) # current shard

    def load_shard(self, path):
        tks = np.load(path)

        return tks

    def get_batch(self, device: str, mix: bool = False, shuffle: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        get a batch of the dataset, each batch have T tokens

        device: on which hardware they will be processed, default is cpu
        mix: if mix is True, in one batch there are half samples in italian and the other half in english, mix=False for default
        shuffle: if shuffle is True, when you applty mix its gonna suffle the batches instead of be first half in en and other half in it, shuffle=False for default
        '''

        if not mix:
            # check if current shard has enough indices left for a full batch
            if (self.B * self.T) + self.s_idx > len(self.curr_shard):
                # move to the next shard
                self.s_idx = 0
                self.curr_pos += 1
                self.curr_shard = self.load_shard(self.all_shards[self.curr_pos]) 

            # create xb and yb
            idxs = torch.randint(0, len(self.curr_shard) - self.B, size=(self.B,))
            
            rows = np.arange(self.B)[:, None]
            cols = np.arange(self.T)

            xb = torch.from_numpy(self.curr_shard[idxs[rows] + cols]).to(torch.long) # (B, T)
            yb = torch.from_numpy(self.curr_shard[idxs[rows] + cols + 1]).to(torch.long) # (B, T)

            self.curr_shard = self.curr_shard[(self.B * self.T) + self.s_idx:]
            self.s_idx += self.B * self.T
        else:
            assert self.B % 2 == 0, 'batch_size has to be divisible by 2'

            half_batch = self.B // 2

            xb = torch.zeros(self.B, self.T, dtype=torch.long)
            yb = torch.zeros(self.B, self.T, dtype=torch.long)
            
            for i, type_shard in enumerate(['en', 'it']):
                # load random shard
                tks_np = self.load_shard(type_shard)
                
                # create xb and yb for one language per time
                idxs = torch.randint(0, len(tks_np) - self.T, size=(half_batch,))

                rows = idxs[:, None] + torch.arange(self.T)
                start_idx = i * half_batch
                end_idx = (i + 1) * half_batch
                
                # fill the pre-allocated tensors directly
                xb[start_idx:end_idx] = torch.from_numpy(tks_np[rows]) # (B, T)
                yb[start_idx:end_idx] = torch.from_numpy(tks_np[rows + 1]) # (B, T)

            # instead of be the first half samples in english and the other in italian, they will be shuffled
            if shuffle:
                perm = torch.randperm(self.B)

                xb_shuffled, yb_shuffled = xb[perm], yb[perm]

                return xb_shuffled.to(device), yb_shuffled.to(device)
        
        xb, yb = xb.to(device), yb.to(device) # (B, T)

        return xb, yb

