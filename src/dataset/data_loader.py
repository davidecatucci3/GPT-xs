'''from datatrove.pipeline.readers import ParquetReader
from tokenizers import Tokenizer

data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT", limit=1) 

tn = Tokenizer.from_pretrained('gpt2')

max_tokens = 100_000_000

for document in data_reader():
    print(document.metadata['token_count'] == len(tn.encode(document.text).ids))
'''

import numpy as np
import tiktoken

tn = tiktoken.get_encoding('gpt2')

arr = np.load('data/shard_it_001.npy')

print(tn.decode(arr[:10]))

print(tn.decode(arr[-10:]))