import matplotlib.pyplot as plt
import sentencepiece as spm
import numpy as np
import random

from collections import Counter

def calc_fertility():
    '''
    the fertility is the ratio between the number of tokens and the number of words in a document
    '''

    # tokenizer usede to encode
    tn = spm.SentencePieceProcessor()

    tn.load('data/tokenizer/BPE-200-50304.model')

    l_fertility = []

    i = random.randint(1, 100)

    arr = np.load(f'data/dataset/shard_en_{i:03d}.npy')

    for _ in range(100):
        # load tokenized data
        i = random.randint(1, 100)

        # generate random start indices
        chunk_size = 5000

        idxs = np.random.randint(0, len(arr) - chunk_size)

        # extract chunks
        chunk_arr = arr[idxs:idxs + chunk_size].tolist()

        # decode token chunks to text
        txt = tn.decode(chunk_arr)

        # count words using whitespace splitting
        n_words = len(txt.split())

        # re-encode text to get token counts
        enc_txt = tn.encode(txt, out_type=str)

        n_tks = len(enc_txt)

        # calculate fertility
        fertility = round(n_tks / n_words, 2)

        l_fertility.append(fertility)

    return l_fertility

l_fertility = calc_fertility()

# take the most frequent fertility 
counter_en = Counter(l_fertility)

freq_en = max(counter_en, key=counter_en.get) # most frequent perplexity after tot iterations for en tokens
 
print(f'fertility en: {freq_en:.2f}') # 1.28

# plot 
plt.hist(l_fertility, label='EN')

plt.axvline(freq_en, color='blue', linestyle='--', label=f'Mode EN: {freq_en:.2f}')

plt.legend()

plt.show()

