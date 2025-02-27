import matplotlib.pyplot as plt
import sentencepiece as spm
import numpy as np
import tiktoken
import random

from collections import Counter

def calc_fertility():
    '''
    the fertility is the ratio between the number of tokens and the number of words in a document
    '''

    # tokenizer used to decode gpt2 tokens
    tn = tiktoken.get_encoding('gpt2')

    # tokenizer usede to encode
    sp = spm.SentencePieceProcessor()

    sp.load('data/tokenizer/BPE-200-50304.model')

    l_fertility_en, l_fertility_it = [], []

    i = random.randint(1, 100)

    arr = np.load(f'data/dataset/shard_en_{i:03d}.npy')
    arr2 = np.load(f'data/dataset/shard_it_{i:03d}.npy')

    for _ in range(100):
        # load tokenized data
        i = random.randint(1, 100)

        # generate random start indices
        chunk_size = 5000

        idxs = np.random.randint(0, len(arr) - chunk_size)
        idxs2 = np.random.randint(0, len(arr2) - chunk_size)  

        # extract chunks
        chunk_arr = arr[idxs:idxs + chunk_size]
        chunk_arr2 = arr2[idxs2:idxs2 + chunk_size]

        # decode token chunks to text
        txt = tn.decode(chunk_arr)
        txt2 = tn.decode(chunk_arr2)

        # count words using whitespace splitting
        n_words = len(txt.split())
        n_words2 = len(txt2.split())

        # re-encode text to get token counts
        enc_txt = sp.encode(txt, out_type=str)
        enc_txt2 = sp.encode(txt2, out_type=str)

        n_tks = len(enc_txt)
        n_tks2 = len(enc_txt2)

        # calculate fertility
        fertility_en = round(n_tks / n_words, 2)
        fertility_it = round(n_tks2 / n_words2, 2)

        l_fertility_en.append(fertility_en)
        l_fertility_it.append(fertility_it)

    return l_fertility_en, l_fertility_it

l_fertility_en, l_fertility_it = calc_fertility()

# take the most frequent fertility 
counter_en = Counter(l_fertility_en)
counter_it = Counter(l_fertility_it)

freq_en = max(counter_en, key=counter_en.get) # most frequent perplexity after tot iterations for en tokens
freq_it = max(counter_it, key=counter_it.get) # most frequent perplexity over tot iterations for it tokens
 
print(f'fertility en: {freq_en:.2f}') # 1.28
print(f'fertility it: {freq_it:.2f}') # 1.39

# plot 
plt.hist(l_fertility_en, label='EN')
plt.hist(l_fertility_it, label='IT')

plt.axvline(freq_en, color='blue', linestyle='--', label=f'Mode EN: {freq_en:.2f}')
plt.axvline(freq_it, color='orange', linestyle='--', label=f'Mode IT: {freq_it:.2f}')

plt.legend()

plt.show()

