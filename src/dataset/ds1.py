from datatrove.pipeline.readers import ParquetReader
from tokenizers import Tokenizer

tk = Tokenizer.from_pretrained('gpt2')

data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT", limit=1) 

for document in data_reader():
    # do something with document
    print(document.metadata['token_count'] == len(tk.encode(document.text)))