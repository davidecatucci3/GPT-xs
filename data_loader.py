import numpy as np
import torch
import os

def load_tokens(filename):
    npt = np.load(filename)

    npt = npt.astype(np.int32)

    ptt = torch.tensor(npt, dtype=torch.long)

    return ptt

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, split, master_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "/root/mark-124M-v1.0-ssh/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]

        self.shards = shards

        assert len(shards) > 0, f"no shards found for split {split}"

        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
    
        # if process_rank = 0 (gpu in use is 0) -> curr_pos = 0
        # if process_rank = 1 (gpu in use is 1) -> curr_pos = B * T
        # ... so each process elaborate different part of the dataset
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T

        buf = self.tokens[self.current_position:self.current_position + B * T + 1]

        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        
        return x, y

# EXAMPLE

# B = 4
# T = 8
# NP = 8

#    BUF 1                BUF 2             BUF N
# PR=0 -> 0:33     | PR=0 -> 96:129    | ...
# PR=1 -> 32:65    | PR=0 -> 128:161   | ...
# PR=2 -> 64: 97   | PR=0 -> 160:193   | ...
