import torch
from model import GPT, GPTConfig
import tiktoken

checkpoint1 = torch.load('log/model_00000.pt', map_location='cpu')
checkpoint = torch.load('log/model_05000.pt', map_location='cpu')

model = GPT(GPTConfig())

new_state_dict1 = {k.replace("_orig_mod.", ""): v for k, v in checkpoint1['model'].items()}
new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model'].items()}

model.load_state_dict(new_state_dict1)  # Load weights
model.eval()  # Set to evaluation mode
tn = tiktoken.get_encoding('gpt2')
seq = 'I am a student of a famous university and '

tks = torch.tensor(tn.encode(seq)).unsqueeze(0)

res = model.generate(tks, 128)

print(tn.decode(res[0].tolist()))