import sentencepiece as spm
import numpy as np
import tiktoken

# train data
arr1 = np.load('data/dataset/shard_en_001.npy')

tn = tiktoken.get_encoding('gpt2')

str_arr1 = tn.decode(arr1)

input_file = 'data/tokenizer/corpus.txt'

'''
with open(input_file, 'w', encoding='utf-8') as f:
    f.write(str_arr1)
'''

# settings
user_defined_symbols = ["<s>", "</s>", "<pad>", "<eod>"] + [f"<ph_{i}>" for i in range(1, 256)]
vocab_size = 50304
model_type = "bpe"
tot_tks = int(len(arr1) / 1e6)
model_prefix = f"{model_type.upper()}-{tot_tks}-{vocab_size}"
character_coverage = 0.9999
split_by_number = True
add_dummy_prefix = True
byte_fallback = True
normalization_rule_name = "nfkc"
remove_extra_whitespaces = True  

# train the model
spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type=model_type,
    character_coverage=character_coverage,
    split_by_number=split_by_number,
    add_dummy_prefix=add_dummy_prefix,
    user_defined_symbols=",".join(user_defined_symbols),  
    byte_fallback=byte_fallback,
    normalization_rule_name=normalization_rule_name,
    remove_extra_whitespaces=remove_extra_whitespaces
)