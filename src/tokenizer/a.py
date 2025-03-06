import numpy as np
import tiktoken
import sentencepiece as spm

arr1 = np.load('data/dataset/shard_en_001.npy')

tn = spm.SentencePieceProcessor()

tn.load('BPE-200-50304.model')

print(tn.decode(arr1[:30].tolist()))