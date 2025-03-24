# GPT-xs

<p align="center">
  <img src="meme.gif" alt="plot">
  <br>
  I was searching an image to put and this just makes me laugh a lot, so i put it idkw
</p>

GPT-xs is an autoregressive text generation model designed to replicate the GPT-2 architecture originally developed by OpenAI. This implementation specifically recreates the "xs" (extra small) variant, featuring 124 million parameters. The project draws inspiration from a YouTube video by Andrej Karpathy and was undertaken as a personal exercise for educational purposes. Unlike commercially deployed models, GPT-xs is not intended for production use but serves as a personal exercise to improve my skills in this field

## Files
- data_loader.py: Implements functionality to efficiently load batches of data from preprocessed shards, preparing them for input into the model during training or inference
- dataset.py: Handles the retrieval of the dataset from Hugging Face, processes it into tokenized shards, and prepares the data for use in model training
- model.py: Defines the architecture and structure of the machine learning model, specifying layers, parameters, and configurations
- train.py: Orchestrates the training process, including loading the model, optimizing parameter and managing the training loop for effective learning
- eval.py: Used to use the model after training
- plot.py: Plot the data that have been collected during the training process
- hellaswag.py: Calculate the accuracy of the model using the HellaSwag dataset

## Dataset
The model was trained on the FineWeb-Edu dataset, which comprises 10 billion tokens. To facilitate training, I partitioned the dataset into 100 shards, with each shard containing 100 million tokens

## Architecture
As previously mentioned, this model is based on the GPT-2 architecture. I will outline the hyperparameters and methods employed in its design. For a more comprehensive and detailed explanation, please refer to the GPT-2 paper: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

The model adopts a dimensionality of d_model = 768, with n_heads = 12 attention heads and n_layers = 12 transformer layers. The context_length was set to 512 tokens. Notably, this value was reduced from the original 1024 specified in the GPT-2 paper due to GPU memory constraints, as the larger context size exceeded the available memory capacity. This adjustment, along with the vocabulary size that I will explain after, represents the only deviations from the standard GPT-2 hyperparameters

### Tokenizer
The tokenizer employed in this model is the GPT-2 tokenizer, consistent with the original paper. It uses a vocabulary size of vocab_size = 50304, which was increased from the original 50257 to ensure divisibility by a power of 2. This modification enhances token processing throughput and reduces latency. The GPT-2 tokenizer relies on the Byte Pair Encoding (BPE) algorithm. While it remains effective, it is worth noting that modern tokenizers have since surpassed it in efficiency. However, I retained the GPT-2 tokenizer to align closely with the methodology outlined in the original paper

### Optimizer
For optimization, I utilized the AdamW (Adam with Weight Decay) algorithm, paired with a cosine learning rate decay schedule. The maximum learning rate was set to max_lr = 6e-4, with the minimum learning rate defined as min_lr = max_lr * 0.1. The learning rate warmup phase spanned warmup_steps = 715, after which the cosine decay was initiated following the processing of 375 million tokens. Additional hyperparameters include a weight decay of weight_decay = 0.1, momentum parameters set to (β1, β2) = (0.9, 0.95), and an epsilon value of ε = 1e-8 to ensure numerical stability

## Training
The model was trained for 19,073 steps, equivalent to exactly one epoch. Training was conducted using eight L40S GPUs, each with 48 GB of memory, over a duration of 2.5 hours (150 minutes). This time includes 20 minutes dedicated to creating data shards from the dataset. The training process achieved an average latency of 400 milliseconds per step, with a token throughput of 1 million tokens per second (1M tokens/s)


