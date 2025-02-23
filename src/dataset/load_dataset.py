import numpy as np
import tiktoken

from multiprocessing import Pool, cpu_count
from datasets import load_dataset
from tqdm import tqdm

tn = None

def init_tokenizer():
    '''
    initialize tokenizer (if you initialize the tokenizer before its gonna speed up the total process)
    '''

    global tn

    tn = tiktoken.get_encoding('gpt2')

def tokenize(doc: dict) -> list[int]:
    '''
    doc: dictionary with inside the text (the text in doc is a single row of the dataset)
    '''

    txt = doc['text'] # string of text

    tks = tn.encode_ordinary(txt) # str -> list of tokens

    tks_np = np.array(tks, dtype=np.uint16) # int16 is negative and postive, uint16 is only positive

    return tks_np

def save_file(file_path: str, shard: np.ndarray) -> None:
    '''
    save shard in data folder
    '''

    np.save(file_path, shard)

def save_shards(ds_path: str, ds_name: str, shard_name: str, token: bool = False):
    '''
    load chunks of text from remote dataset, tokenize single text and each 100MT save it into tokenized array in a .npy file
    '''

    # remote dataset (with streaming=True i am not downloading the dataset in local, is in remote and ds now is an iterable)
    ds = load_dataset(path=ds_path, name=ds_name, split="train", streaming=True, token=token) 

    # settings
    shard_size = int(1e8) # maximum number of tokens in one shard
    chunk_size = 16 # take a chunk of documents from the dataset per time
    data_folder = 'data/' # where .npy files where will be stored
    n_procs = cpu_count() # how many cpu cores you want to use
    n_shard = 1
    tot_tks = 0 # number of tks in a shard (the shard has 1e8 elements but the number of occupied ones is tot_tks)
    progress_bar = None
    max_tks = int(1e10)
    tot_shards_tks = 0 # keep track of the number of tokens in all the shards

    # create in-memory array
    shard = np.empty(shard_size, dtype=np.uint16)

    # utilize multiprocessing to speed up
    with Pool(processes=n_procs, initializer=init_tokenizer) as pool:
        for tks in pool.imap(func=tokenize, iterable=ds, chunksize=chunk_size):
            n_tks = len(tks)

            # for the it tokens dataset, the dataset contains 165BT, i have to stop at 10BT
            if shard_name == 'shard_it' and tot_shards_tks >= max_tks:
                break
                
            # if there is place for the entire tks put it, if there is not put some in this one and if there is more in another chunk
            if tot_tks + n_tks < shard_size:
                # put tks in shard
                shard[tot_tks:tot_tks + n_tks] = tks

                tot_tks += n_tks

                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'Shard {n_shard:03d}')
                    
                progress_bar.update(n_tks)
            else:
                available_elem = shard_size - tot_tks # available elemements in shard
        
                # put the tokens in tks that can fit in the available elemnts in the current shard
                shard[tot_tks:] = tks[:available_elem]

                progress_bar = None

                file_name_shard = f"{shard_name}_{n_shard:03d}.npy"

                save_file(data_folder + file_name_shard, shard[:tot_tks + available_elem])

                tot_shards_tks += int(1e8)

                n_shard += 1

                # creat next shard                
                shard = np.empty(shard_size, dtype=np.uint16)

                # put the tokens in tks that did not fit in the previous shard in the new one
                shard[:n_tks - available_elem] = tks[available_elem:]

                tot_tks = n_tks - available_elem

        # save the final shard if it has any tokens 
        if tot_tks != 0:
            file_name_shard = f"{shard_name}_{n_shard:03d}.npy"

            save_file(data_folder + file_name_shard, shard[:tot_tks])
        
if __name__ == '__main__':
    # save en tokens 
    save_shards("HuggingFaceFW/fineweb-edu", "sample-10BT", 'shard_en') # 9.944317243BT ~ 10BT | 100 shards | 19.8886GB ~ 20GB
    
    # save it tokens
    save_shards("uonlp/CulturaX", "it", 'shard_it', True) # 10.000001071BT ~ 10BT | 101 shards | 20.0002GB ~ 20GB
