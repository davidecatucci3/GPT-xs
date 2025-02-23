import numpy as np
import torch

from multiprocessing import Pool, cpu_count
from datasets import load_dataset
from tokenizers import Tokenizer

# tokenize text
tn = Tokenizer.from_pretrained('gpt2')

def tokenize(doc: dict) -> list[int]:
    '''
    doc: dictionary with inside the text (the text in doc is a single row of the dataset)
    '''

    txt = doc['text'] # string of text

    tks = tn.encode(txt).ids # str -> list of tokens

    tks_tensor = torch.tensor(data=tks, dtype=torch.uint16) # int16 is negative and postive, uint16 is only positive

    tks_np = tks_tensor.numpy()

    return tks_np

# remote dataset (with streaming=True i am not downloading the dataset in local, is in remote and ds now is an iterable)
ds = load_dataset(path="HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True, token=True) 

# load chunks of text from remote dataset, tokenize single text and each 100MT save this extended tokenized array in a .npy file
chunk_size = 16 # take 16 doc from remote dataset per time
n_procs = cpu_count() # how many cores you want to use
data_folder = 'data/' # where .npy files where will be stored
i = 1
s_idx = 0 # start index of chunk_all_tks array
output_file = f"data_mmap{i}.npy"
shard_size = 100_000_000
chunk_all_tks = np.memmap(filename=data_folder + output_file, dtype=np.uint16, mode="w+", shape=(shard_size,))

if __name__ == '__main__':
    # utilize multiprocessing to speed up
    with Pool(processes=n_procs) as pool:
        for tks in pool.imap(func=tokenize, iterable=ds, chunksize=chunk_size):
            x = tks.shape[0]
            
            # if there is place for the entire tks put it, if there is not put some in this one and if there is more in another chunk
            if s_idx + x < shard_size:
                chunk_all_tks[s_idx:s_idx + x] = tks

                s_idx += x
            else:
                e_idx = shard_size - s_idx # end_idx that is the next s_idx
       
                chunk_all_tks[s_idx:] = tks[:e_idx]

                i += 1

                output_file = f"data_mmap{i}.npy"
                chunk_all_tks = np.memmap(output_file, dtype=np.uint16, mode="w+", shape=(shard_size,))

                chunk_all_tks[:x - e_idx] = tks[e_idx:]

                s_idx = x - e_idx
        
        # if there are tokens left, put it in a new shard
        if s_idx != 0:
            pass
        

