import sentencepiece as spm
import numpy as np
import tiktoken

'''# train data
arr1 = np.load('data/shard_en_001.npy')
arr2 = np.load('data/shard_it_001.npy')

tn = tiktoken.get_encoding('gpt2')

str_arr1 = tn.decode(arr1)
str_arr2 = tn.decode(arr2)

input_file = 'corpus.txt'

with open(input_file, 'w', encoding='utf-8') as f:
    f.write(str_arr2 + '\n' + str_arr1)'''

# settings
input_file = 'src/tokenizer/corpus.txt'
model_prefix = 'BPE'
vocab_size = 50257        
model_type = 'bpe'       

# train the model
spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type=model_type
)

